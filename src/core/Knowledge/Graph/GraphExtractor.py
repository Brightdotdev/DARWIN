"""
src/core/Knowledge/Graph/GraphExtractor.py
------------------------------------------
Unipartite concept graph extraction using chunk-context vectors.

Pipeline:
  1. Extract raw keywords per chunk (yake, regex, or spacy)
  2. Elevate to clean concepts (spaCy lemmatization + NER, or LLM)
  3. Build context vectors: each concept = mean of chunk embeddings it appears in
  4. Connect concepts by cosine similarity with auto-threshold (~3 edges/node)
  5. Render as unipartite concept graph (concepts only, no chunk nodes)

The graph targets Obsidian-like readability: clear clusters, visible bridges,
no hairballs. Query-time highlighting lights up relevant concepts.
"""

import re
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Tupler
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Chunking.Chunker import Chunk
from src.core.Knowledge.Graph.GraphUtils import (
    CONCEPT_STOPWORDS,
    extract_noun_phrases_regex,
    DEFAULT_MIN_CONCEPT_FREQ,
    DEFAULT_MAX_CONCEPTS,
    GraphData,
    ConceptNode,
    Relationship,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Raw keyword extraction (per chunk)
# ═══════════════════════════════════════════════════════════════════════════

def extract_keywords_yake(chunks: List[Chunk], max_per_chunk: int = 15) -> Dict[str, List[int]]:
    """
    YAKE keyword extraction. Fast, offline, handles multi-word phrases.
    Returns {keyword: [chunk_index, ...]}
    """
    try:
        import yake
    except ImportError:
        TerminalUI.warn("yake not installed — falling back to regex. Run: uv add yake")
        return extract_keywords_regex(chunks)

    kw_extractor = yake.KeywordExtractor(
        lan="en", n=3, dedupLim=0.7, top=max_per_chunk, stopwords=None,
    )

    keywords: Dict[str, List[int]] = defaultdict(list)

    for chunk in chunks:
        text = chunk.clean[:3000]
        if not text.strip():
            continue
        try:
            for kw, score in kw_extractor.extract_keywords(text):
                keyword = kw.lower().strip()
                if _is_valid_keyword(keyword):
                    keywords[keyword].append(chunk.index)
        except Exception:
            continue

    return dict(keywords)


def extract_keywords_regex(chunks: List[Chunk]) -> Dict[str, List[int]]:
    """
    Fallback regex noun phrase extraction.
    Returns {keyword: [chunk_index, ...]}
    """
    keywords: Dict[str, List[int]] = defaultdict(list)
    for chunk in chunks:
        for phrase in extract_noun_phrases_regex(chunk.clean):
            keyword = phrase.lower().strip()
            if _is_valid_keyword(keyword):
                keywords[keyword].append(chunk.index)
    return dict(keywords)


def extract_keywords_spacy(chunks: List[Chunk]) -> Dict[str, List[int]]:
    """spaCy keyword extraction via the ModelLoader."""
    try:
        from src.core.Models import get_loader, ModelCategory
        loader = get_loader()
        extractor = loader.get(ModelCategory.TOKENIZER)
        raw = extractor.extract_keywords_for_graph(chunks)
        # ── Apply stopword filter ──
        return {
            kw: indices for kw, indices in raw.items()
            if _is_valid_keyword(kw)
        }
    except Exception:
        return extract_keywords_yake(chunks)


def _is_valid_keyword(keyword: str) -> bool:
    """Filter noise: stopwords, too short, digits-only."""
    return (
        len(keyword) > 3
        and keyword not in CONCEPT_STOPWORDS
        and not keyword.isdigit()
        and any(c.isalpha() for c in keyword)
    )


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Concept elevation (keyword → clean concept)
# ═══════════════════════════════════════════════════════════════════════════

def _load_spacy():
    """Lazy-load spaCy model. Returns nlp or None."""
    try:
        import spacy
        for model in ["en_core_web_md", "en_core_web_sm"]:
            try:
                return spacy.load(model)
            except OSError:
                continue
        TerminalUI.warn("No spaCy model found. Run: python -m spacy download en_core_web_sm")
        return None
    except ImportError:
        TerminalUI.warn("spaCy not installed. Run: uv add spacy")
        return None


def elevate_concepts_spacy(
    raw_keywords: Dict[str, List[int]],
    max_concepts: int = 10,
) -> Dict[str, List[int]]:
    """Elevate keywords using the loaded spaCy model."""
    try:
        from src.core.Models import get_loader, ModelCategory
        loader = get_loader()
        extractor = loader.get(ModelCategory.TOKENIZER)
        return extractor.elevate_keywords(raw_keywords, max_concepts)
    except Exception:
        return raw_keywords  # fallback: return raw keywords unchanged


def elevate_concepts_llm(
    raw_keywords: Dict[str, List[int]],
    max_keywords: int = 100,
) -> Dict[str, List[int]]:
    """
    Elevate keywords using an LLM for deep semantic merging.
    Falls back to spaCy if LLM is unavailable.
    """
    import json

    sorted_keywords = sorted(
        raw_keywords.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )[:max_keywords]

    keyword_lines = "\n".join(
        f"- {kw} (in {len(indices)} chunks)"
        for kw, indices in sorted_keywords
    )

    prompt = f"""You are a knowledge graph concept extractor.

Take these raw keywords and identify true semantic concepts.
Rules:
- Merge related keywords into single meaningful concepts
- Name concepts abstractly and domain-agnostically
- Return 20-40 concepts — quality over quantity
- Prefer noun phrases: "Divine Authority" not "is divine"

Keywords:
{keyword_lines}

Return ONLY a JSON array:
[{{"concept": "Concept Name", "keywords": ["kw1", "kw2"], "frequency": 5}}]"""

    response = _call_llm(prompt)
    if not response:
        TerminalUI.warn("LLM unavailable — falling back to spaCy")
        return elevate_concepts_spacy(raw_keywords, max_keywords)

    cleaned = re.sub(r"```(?:json)?", "", response).strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        TerminalUI.warn("LLM response malformed — falling back to spaCy")
        return elevate_concepts_spacy(raw_keywords, max_keywords)

    try:
        elevated_list = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        TerminalUI.warn("LLM JSON parse failed — falling back to spaCy")
        return elevate_concepts_spacy(raw_keywords, max_keywords)

    keyword_to_indices = dict(sorted_keywords)
    result: Dict[str, List[int]] = {}

    for item in elevated_list:
        concept = item.get("concept", "").strip()
        keywords = item.get("keywords", [])
        if not concept:
            continue

        chunk_indices = []
        seen = set()
        for kw in keywords:
            for idx in keyword_to_indices.get(kw, []):
                if idx not in seen:
                    chunk_indices.append(idx)
                    seen.add(idx)

        if not chunk_indices:
            chunk_indices = list(range(item.get("frequency", 1)))

        result[concept] = sorted(chunk_indices)

    if not result:
        return elevate_concepts_spacy(raw_keywords, max_keywords)

    TerminalUI.success(
        f"Elevated {len(sorted_keywords)} keywords → {len(result)} concepts (LLM)"
    )
    return result


def _call_llm(prompt: str) -> Optional[str]:
    """Placeholder — replace with your agnostic LLM wrapper."""
    return None  # Offline by default


def elevate_concepts(
    raw_keywords: Dict[str, List[int]],
    max_keywords: int = 10,
    mode: str = "spacy",
) -> Dict[str, List[int]]:
    """
    Elevate raw keywords into clean semantic concepts.

    Args:
        raw_keywords: {keyword: [chunk_indices]} from yake/regex/spacy
        max_keywords: Max keywords to process
        mode:         "spacy" (offline, default) | "llm" (online) | "none"

    Returns:
        Elevated concepts dict, same shape as input.
        Falls back: llm → spacy → raw.
    """
    if not raw_keywords:
        return raw_keywords

    if mode == "none":
        return raw_keywords
    elif mode == "llm":
        return elevate_concepts_llm(raw_keywords, max_keywords)
    else:  # "spacy" (default)
        return elevate_concepts_spacy(raw_keywords, max_keywords)


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Build concept graph (unipartite, context-vector edges)
# ═══════════════════════════════════════════════════════════════════════════

def _build_context_vectors(
    sorted_concepts: List[Tuple[str, List[int]]],
    chunk_vectors: np.ndarray,
) -> np.ndarray:
    """
    Each concept gets a vector = mean of all chunk embeddings it appears in.
    Two concepts are similar if they appear in similar parts of the document.
    """
    n_chunks, dim = chunk_vectors.shape
    context_vecs = np.zeros((len(sorted_concepts), dim), dtype=np.float32)

    for i, (_, chunk_indices) in enumerate(sorted_concepts):
        valid = [idx for idx in chunk_indices if idx < n_chunks]
        if valid:
            context_vecs[i] = chunk_vectors[valid].mean(axis=0)

    return context_vecs


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. Returns (n, n) float array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norms + 1e-9)
    return (normalized @ normalized.T).astype(float)


def _auto_threshold(sim_matrix: np.ndarray, target_avg_degree: float = 3.0) -> float:
    """
    Find threshold producing ~target_avg_degree edges per node.
    3.0 = Obsidian-like: visible clusters with bridges, no hairball.
    """
    n = sim_matrix.shape[0]
    if n < 2:
        return 0.75

    upper = sorted(
        float(sim_matrix[i, j])
        for i in range(n) for j in range(i + 1, n)
    )

    if not upper:
        return 0.75

    target_edges = max(n - 1, min(int((n * target_avg_degree) / 2), len(upper) - 1))
    threshold = upper[-target_edges] if target_edges <= len(upper) else upper[0]
    threshold = max(0.45, min(0.85, threshold))

    TerminalUI.dim(f"Auto threshold: {threshold:.3f} → ~{target_avg_degree} edges/node")
    return threshold


def build_concept_graph(
    concepts: Dict[str, List[int]],
    chunks: List[Chunk],
    chunk_vectors: np.ndarray,
    min_freq: int = DEFAULT_MIN_CONCEPT_FREQ,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    similarity_threshold: Optional[float] = None,
) -> GraphData:
    """
    Build a unipartite concept graph.

    Nodes = clean concepts. Edges = they appear in similar chunks.
    No chunk nodes — just concepts connected by document context.

    Args:
        concepts:             {label: [chunk_indices]}
        chunks:               source Chunk objects
        chunk_vectors:        pre-computed embeddings (n_chunks, dim)
        min_freq:             minimum appearances to keep a concept
        max_concepts:         max nodes (keeps graph readable)
        similarity_threshold: cosine cutoff, or None for auto

    Returns:
        GraphData with concept nodes and context-similarity edges
    """
    # Filter by frequency
    filtered = {c: idx for c, idx in concepts.items() if len(idx) >= min_freq}
    if not filtered:
        TerminalUI.warn("No concepts passed frequency filter")
        return GraphData(filename="", chunk_count=len(chunks), total_concepts_extracted=len(concepts))

    # Rank by frequency, take top N
    sorted_concepts = sorted(filtered.items(), key=lambda x: len(x[1]), reverse=True)[:max_concepts]
    concept_labels = [c for c, _ in sorted_concepts]
    concept_to_id = {c: f"c{i}" for i, c in enumerate(concept_labels)}

    # Build nodes
    nodes = [
        ConceptNode(
            id=concept_to_id[c],
            label=c,
            frequency=len(idx),
            chunk_indices=idx,
        )
        for c, idx in sorted_concepts
    ]

    # Build context vectors from chunk embeddings
    TerminalUI.dim(f"Building context vectors for {len(concept_labels)} concepts...")
    context_vecs = _build_context_vectors(sorted_concepts, chunk_vectors)

    # Pairwise similarity → edges
    sim_matrix = _cosine_similarity_matrix(context_vecs)
    threshold = similarity_threshold or _auto_threshold(sim_matrix)

    edges = []
    n = len(concept_labels)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                edges.append(Relationship(
                    source=concept_to_id[concept_labels[i]],
                    target=concept_to_id[concept_labels[j]],
                    weight=round(sim, 4),
                    relation_type="context_similarity",
                ))

    TerminalUI.dim(
        f"Threshold: {threshold:.3f} → {len(edges)} edges "
        f"({len(edges)/max(len(nodes),1):.1f}/node)"
    )

    return GraphData(
        filename="",
        nodes=nodes,
        edges=edges,
        chunk_count=len(chunks),
        total_concepts_extracted=len(concepts),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Query highlighting
# ═══════════════════════════════════════════════════════════════════════════


def highlight_concepts(graph: GraphData, query: str) -> Dict:
    """Find concepts matching a query, with spaCy enhancement if available."""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    highlighted_ids = set()
    highlighted_labels = []

    for node in graph.nodes:
        label_lower = node.label.lower()
        if query_lower in label_lower:
            highlighted_ids.add(node.id)
            highlighted_labels.append(node.label)
        elif len(query_words & set(label_lower.split())) >= 2:
            highlighted_ids.add(node.id)
            highlighted_labels.append(node.label)
        elif any(w in label_lower for w in query_words if len(w) > 4):
            highlighted_ids.add(node.id)
            highlighted_labels.append(node.label)

    # spaCy-enhanced matching if no direct matches
    if not highlighted_ids:
        try:
            from src.core.Models import get_loader, ModelCategory
            loader = get_loader()
            extractor = loader.get(ModelCategory.TOKENIZER)
            query_concepts = set(extractor.extract_noun_chunks(query))
            
            for node in graph.nodes:
                label_concepts = set(extractor.extract_noun_chunks(node.label))
                if query_concepts & label_concepts:
                    highlighted_ids.add(node.id)
                    highlighted_labels.append(node.label)
        except Exception:
            pass

    paths = []
    if len(highlighted_ids) >= 2:
        for edge in graph.edges:
            if edge.source in highlighted_ids and edge.target in highlighted_ids:
                paths.append({
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                })

    return {
        "highlighted_nodes": list(highlighted_ids),
        "highlighted_labels": highlighted_labels,
        "paths": paths,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main extraction orchestration
# ═══════════════════════════════════════════════════════════════════════════

def extract_graph_from_chunks(
    chunks: List[Chunk],
    filename: str,
    min_freq: int = DEFAULT_MIN_CONCEPT_FREQ,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    chunk_vectors: Optional[np.ndarray] = None,
    similarity_threshold: Optional[float] = None,
    extraction_mode: str = "yake",
    elevation_mode: str = "spacy",
) -> GraphData:
    """
    Extract a unipartite concept graph from document chunks.

    Full pipeline:
      1. Extract raw keywords per chunk (yake/regex/spacy)
      2. Elevate to clean concepts (spacy lemmatization or LLM)
      3. Build context vectors from chunk embeddings
      4. Connect concepts by cosine similarity (auto-threshold)

    Args:
        chunks:               List of Chunk objects
        filename:             Source document name
        min_freq:             Minimum concept occurrences
        max_concepts:         Max nodes in graph (readability)
        chunk_vectors:        Pre-computed embeddings (n_chunks, dim)
        similarity_threshold: Cosine cutoff, None = auto
        extraction_mode:      "yake" | "regex" | "spacy"
        elevation_mode:       "spacy" | "llm" | "none"

    Returns:
        GraphData with concept nodes and context-similarity edges
    """
    TerminalUI.bold(f"Extracting concept graph: {filename} ({len(chunks)} chunks)")
    TerminalUI.dim(f"Extraction: {extraction_mode} | Elevation: {elevation_mode} | Max concepts: {max_concepts}")

    # ── Step 1: Extract raw keywords ──
    t = TerminalUI.timer("keyword_extraction")
    extractors = {
        "yake":  extract_keywords_yake,
        "regex": extract_keywords_regex,
        "spacy": extract_keywords_spacy,
    }
    extractor = extractors.get(extraction_mode, extract_keywords_yake)
    keywords = extractor(chunks)
    TerminalUI.dim(f"Extracted {len(keywords)} raw keywords")
    t.done()

    # ── Step 2: Elevate to concepts ──
    if elevation_mode != "none":
        t = TerminalUI.timer("concept_elevation")
        concepts = elevate_concepts(keywords, max_keywords=max_concepts, mode=elevation_mode)
        t.done()
    else:
        concepts = keywords

    # ── Step 3: Build graph ──
    t = TerminalUI.timer("graph_building")

    if chunk_vectors is not None:
        graph = build_concept_graph(
            concepts=concepts,
            chunks=chunks,
            chunk_vectors=chunk_vectors,
            min_freq=min_freq,
            max_concepts=max_concepts,
            similarity_threshold=similarity_threshold,
        )
    else:
        TerminalUI.warn("No chunk vectors — cannot build graph edges")
        graph = GraphData(
            filename=filename,
            nodes=[
                ConceptNode(id=f"c{i}", label=c, frequency=len(idx), chunk_indices=idx)
                for i, (c, idx) in enumerate(
                    sorted(concepts.items(), key=lambda x: len(x[1]), reverse=True)[:max_concepts]
                )
            ],
            chunk_count=len(chunks),
            total_concepts_extracted=len(concepts),
        )

    graph.filename = filename
    graph.total_concepts_extracted = len(concepts)
    t.done()

    TerminalUI.success(
        f"Graph: {len(graph.nodes)} concepts, {len(graph.edges)} edges "
        f"({len(graph.edges)/max(len(graph.nodes),1):.1f}/node)"
    )

    if graph.nodes:
        top = graph.nodes[:5]
        TerminalUI.dim(f"Top: {', '.join(n.label for n in top)}")

    return graph


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_graph_extraction_pipeline(
    filename: str,
    store: DarwinStore,
    min_freq: int = DEFAULT_MIN_CONCEPT_FREQ,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    similarity_threshold: Optional[float] = None,
    extraction_mode: str = "yake",
    elevation_mode: str = "spacy",
    visualize: bool = False,
    rerun: bool = False,
) -> Optional[GraphData]:
    """
    Full graph extraction pipeline for one document.
    Saves graph JSON + optional HTML visualization.
    """
    

    if not rerun and store.is_fresh(filename) and store.has_graph(filename):
        try:
            TerminalUI.dim(f"Skipping graph: fresh exists for {filename}")
            return store.load_graph(filename)
        except Exception as e:
            TerminalUI.warn(f"Cached graph corrupted ({e}) — re-extracting...")


    if store.has_embeddings(filename):
        vectors , chunks = store.load_embeddings(filename)
        if vector and chunks is None:
            darwin.log("No chunks and vectors provided for graph extraction", Logtype.ERROR)
        
        
    graph = extract_graph_from_chunks(
        chunks=chunks,
        filename=filename,
        min_freq=min_freq,
        max_concepts=max_concepts,
        chunk_vectors=vectors,
        similarity_threshold=similarity_threshold,
        extraction_mode=extraction_mode,
        elevation_mode=elevation_mode,
    )

    store.save_graph(filename, graph)

    if visualize and graph.nodes:
        try:
            from src.core.Knowledge.Graph.GraphVisualizer import render_concept_graph
            html = render_concept_graph(graph)
            if html:
                store.save_graph_html(filename, html)
                TerminalUI.success(f"Visualization saved for {filename}")
        except Exception as e:
            TerminalUI.error(f"Visualization error: {e}")

    return graph