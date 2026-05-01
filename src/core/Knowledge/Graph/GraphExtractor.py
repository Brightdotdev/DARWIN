# src/core/Knowledge/Graph/GraphExtractor.py - Fixed with darwin.log()

"""
src/core/Knowledge/Graph/GraphExtractor.py
------------------------------------------
Unipartite concept graph extraction using chunk-context vectors.

Pipeline:
  1. Extract raw keywords per chunk ( regex, or spacy)
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
from typing import List, Dict, Optional, Tuple
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




_tokenizer_cache = None

def _get_tokenizer():
    """Get or load the spaCy tokenizer once."""
    global _tokenizer_cache
    if _tokenizer_cache is not None:
        return _tokenizer_cache
    try:
        from src.core.Models import get_loader, ModelCategory
        loader = get_loader()
        _tokenizer_cache = loader.get(ModelCategory.TOKENIZER)
        return _tokenizer_cache
    except Exception:
        return None



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
    extractor = _get_tokenizer()
    if extractor is None:
        return extract_keywords_regex(chunks)
    
    try:
        raw = extractor.extract_keywords_for_graph(chunks)
        
        darwin.log(f"Sample raw keywords: {list(raw.keys())[:30]}", LogType.INFO)
        
        result = {
            kw: indices for kw, indices in raw.items()
            if _is_valid_keyword(kw)
        }
        darwin.log(f"Sample after filter: {list(result.keys())[:30]}", LogType.INFO)
        return result
    except Exception as e:
        darwin.log(f"spaCy extraction failed: {e}, falling back to regex", LogType.WARNING)
        return extract_keywords_regex(chunks)


def _is_valid_keyword(keyword: str) -> bool:
    import re
    clean = re.sub(r'^[^a-zA-Z]+', '', keyword)  # strip leading non-alpha chars
    clean = re.sub(r'[^a-zA-Z0-9\s]+$', '', clean)  # strip trailing non-alpha
    clean = clean.strip()
    
    words = clean.split()
    if len(words) < 2 or len(words) > 5:
        return False
    if len(clean) < 6:
        return False
    if clean in CONCEPT_STOPWORDS:
        return False
    if not any(c.isalpha() for c in clean):
        return False
    if re.search(r'[@\[\]"{}]|http|www\.', clean):
        return False
    if re.search(r'\s[-&]\s', clean):
        return False
    
    return True



def _score_concepts(concepts, total_chunks):
    import math
    scores = {}
    for concept, chunk_indices in concepts.items():
        df = len(set(chunk_indices))
        idf = math.log((total_chunks + 1) / (df + 1))
        scores[concept] = df * idf
    return scores


def _semantic_centrality_scores(
    concepts: Dict[str, List[int]],
    chunk_vectors: np.ndarray,
) -> Dict[str, float]:
    """
    Score each concept by how central it is to the document.
    Concept vector = mean of chunk embeddings it appears in.
    Score = cosine similarity to document centroid.
    Central concepts score high, peripheral examples score low.
    """
    doc_centroid = chunk_vectors.mean(axis=0)
    doc_centroid = doc_centroid / (np.linalg.norm(doc_centroid) + 1e-9)

    scores = {}
    for concept, chunk_indices in concepts.items():
        valid = [idx for idx in chunk_indices if idx < len(chunk_vectors)]
        if not valid:
            scores[concept] = 0.0
            continue
        concept_vec = chunk_vectors[valid].mean(axis=0)
        concept_vec = concept_vec / (np.linalg.norm(concept_vec) + 1e-9)
        scores[concept] = float(np.dot(concept_vec, doc_centroid))

    return scores

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Concept elevation (keyword → clean concept)
# ═══════════════════════════════════════════════════════════════════════════



def elevate_concepts_spacy(
    raw_keywords: Dict[str, List[int]],
    max_concepts: int = 10,
) -> Dict[str, List[int]]:
    """Elevate keywords using the loaded spaCy model."""
    extractor = _get_tokenizer()
    if extractor is None:
        return raw_keywords
    
    try:
        return extractor.elevate_keywords(raw_keywords, max_concepts)
    except Exception as e:
        darwin.log(f"spaCy elevation failed: {e}", LogType.WARNING)
        return raw_keywords



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
        darwin.log("LLM unavailable — falling back to spaCy", LogType.WARNING)
        return elevate_concepts_spacy(raw_keywords, max_keywords)

    cleaned = re.sub(r"```(?:json)?", "", response).strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        darwin.log("LLM response malformed — falling back to spaCy", LogType.WARNING)
        return elevate_concepts_spacy(raw_keywords, max_keywords)

    try:
        elevated_list = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        darwin.log("LLM JSON parse failed — falling back to spaCy", LogType.WARNING)
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

    darwin.log(f"Elevated {len(sorted_keywords)} keywords → {len(result)} concepts (LLM)", LogType.SUCCESS)
    return result


def _call_llm(prompt: str) -> Optional[str]:
    """Placeholder — replace with your agnostic LLM wrapper."""
    return None


def elevate_concepts(
    raw_keywords: Dict[str, List[int]],
    max_keywords: int = 10,
    mode: str = "spacy",
) -> Dict[str, List[int]]:
    """
    Elevate raw keywords into clean semantic concepts.

    Args:
        raw_keywords: {keyword: [chunk_indices]} from regex/spacy
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
    else:
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
    
    # Dynamic cap based on corpus size — small docs need lower threshold
    min_cap = 0.45
    max_cap = 0.85 if n >= 30 else 0.95  # ← tighter for small concept sets
    threshold = max(min_cap, min(max_cap, threshold))

    darwin.log(f"Auto threshold: {threshold:.3f} → ~{target_avg_degree} edges/node")
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
    
    tfidf_scores = _score_concepts(concepts, len(chunks))
    centrality_scores = _semantic_centrality_scores(concepts, chunk_vectors)

    # Normalize both to 0-1
    max_tfidf = max(tfidf_scores.values()) or 1.0
    max_cent = max(centrality_scores.values()) or 1.0

    combined = {
        c: (tfidf_scores[c] / max_tfidf) * 0.4 + (centrality_scores[c] / max_cent) * 0.6
        for c in concepts
    }

    sorted_scores = sorted(combined.values())
    cutoff_idx = int(len(sorted_scores) * 0.4)
    cutoff_score = sorted_scores[cutoff_idx]

    filtered = {
        c: idx for c, idx in concepts.items()
        if combined[c] >= cutoff_score and len(idx) >= min_freq}
    

    darwin.log(f"Top scored: {sorted(combined.items(), key=lambda x: x[1], reverse=True)[:10]}", LogType.INFO)

    if not filtered:
        darwin.log("No concepts passed frequency filter", LogType.WARNING)
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
    darwin.log(f"Building context vectors for {len(concept_labels)} concepts...")
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

    darwin.log(f"Threshold: {threshold:.3f} → {len(edges)} edges ({len(edges)/max(len(nodes),1):.1f}/node)")

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
        extractor = _get_tokenizer()
        if extractor is not None:
            try:
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
    extraction_mode: str = "spacy",
    elevation_mode: str = "none",
) -> GraphData:
    """
    Extract a unipartite concept graph from document chunks.

    Full pipeline:
      1. Extract raw keywords per chunk (regex/spacy)
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
        extraction_mode:     "regex" | "spacy"
        elevation_mode:       "spacy" | "llm" | "none"

    Returns:
        GraphData with concept nodes and context-similarity edges
    """
    darwin.log(f"Extracting concept graph: {filename} ({len(chunks)} chunks)", LogType.INFO)
    darwin.log(f"Extraction: {extraction_mode} | Elevation: {elevation_mode} | Max concepts: {max_concepts}")

    # ── Step 1: Extract raw keywords ──
    darwin.log("Keyword extraction...")
    extractors = {
        
        "regex": extract_keywords_regex,
        "spacy": extract_keywords_spacy,
    }
    extractor = extractors.get(extraction_mode, extract_keywords_spacy)

    keywords = extractor(chunks)
    darwin.log(f"Extracted {len(keywords)} raw keywords")

    # ── Step 2: Elevate to concepts ──
    if elevation_mode != "none":
        darwin.log("Concept elevation...")
        concepts = elevate_concepts(keywords, max_keywords=max_concepts, mode=elevation_mode)
    else:
        concepts = keywords

    # ── Step 3: Build graph ──
    darwin.log("Graph building...")

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
        darwin.log("No chunk vectors — cannot build graph edges", LogType.WARNING)
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

    darwin.log(f"Graph: {len(graph.nodes)} concepts, {len(graph.edges)} edges ({len(graph.edges)/max(len(graph.nodes),1):.1f}/node)", LogType.SUCCESS)

    if graph.nodes:
        top = graph.nodes[:5]
        darwin.log(f"Top: {', '.join(n.label for n in top)}")

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
    extraction_mode: str = "spacy",
    elevation_mode: str = "none",
    visualize: bool = False,
    rerun: bool = False,
) -> Optional[GraphData]:
    """
    Full graph extraction pipeline for one document.
    Saves graph JSON + optional HTML visualization.
    """
    if not rerun and store.is_fresh(filename) and store.has_graph(filename):
        try:
            darwin.log(f"Skipping graph: fresh exists for {filename}", LogType.INFO)
            return store.load_graph(filename)
        except Exception as e:
            darwin.log(f"Cached graph corrupted ({e}) — re-extracting...", LogType.WARNING)

    # BUG FIX: The original had incorrect logic that would never load vectors/chunks
    # Original: if vectors and chunks is None - WRONG - numpy array can't be in boolean context
    # Fixed: Load embeddings properly
    vectors = None
    chunks = None
    
    if store.has_embeddings(filename):
        try:
            vectors, chunks = store.load_embeddings(filename)
            if vectors is None or chunks is None:
                darwin.log(f"No chunk vectors found for {filename}", LogType.ERROR)
                return None
        except Exception as e:
            darwin.log(f"Failed to load embeddings for {filename}: {e}", LogType.ERROR)
            return None
    else:
        darwin.log(f"No embeddings found for {filename}. Run embedding first.", LogType.ERROR)
        return None

    # BUG: Original had wrong condition here - fixed
    if vectors is None or chunks is None:
        darwin.log("No chunks and vectors provided for graph extraction", LogType.ERROR)
        return None

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
                darwin.log(f"Visualization saved for {filename}", LogType.SUCCESS)
        except Exception as e:
            darwin.log(f"Visualization error: {e}", LogType.WARNING)

    return graph