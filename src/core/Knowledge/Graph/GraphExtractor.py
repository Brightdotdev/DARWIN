"""
src/core/Graph/GraphExtractor.py
--------------------------------
Extract concepts and relationships from document chunks using
semantic similarity rather than co-occurrence.

Pipeline
--------
chunks
    → extract keywords per chunk (yake)
    → filter by frequency
    → embed all concept labels (reuses your existing embedding model)
    → connect concepts by cosine similarity threshold
    → save via store.save_graph()

Modes:
- yake   : fast statistical keyword extraction (default, recommended)
- regex  : fallback if yake not installed
- spacy  : legacy, slow, not recommended
"""

from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import List, Dict, Set, Optional, Tuple, Callable
import re
import numpy as np

from src.core.Chunking.Chunker import Chunk
from src.core.Knowledge.Graph.GraphUtils import (
    CONCEPT_STOPWORDS,
    extract_noun_phrases_regex,
    DEFAULT_MIN_CONCEPT_FREQ,
    DEFAULT_MAX_CONCEPTS,
    DEFAULT_MIN_RELATION_FREQ,
    DEFAULT_SIMILARITY_THRESHOLD,
    GraphData,
    Relationship,
    ConceptNode,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


# ============================================================================
# Concept extraction
# ============================================================================

def extract_concepts_yake(chunks: List[Chunk]) -> Dict[str, List[int]]:
    """
    Extract keywords using YAKE (Yet Another Keyword Extractor).
    Fast, offline, no model needed. Handles multi-word phrases well.
    Install: uv add yake
    """
    try:
        import yake
    except ImportError:
        TerminalUI.warn("yake not installed — falling back to regex. Run: uv add yake")
        return extract_concepts_regex(chunks)

    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,         # up to 3-word phrases
        dedupLim=0.7,
        top=20,      # top 20 keywords per chunk
        stopwords=None,
    )

    concepts: Dict[str, List[int]] = defaultdict(list)

    for chunk in chunks:
        text = chunk.clean[:3000]
        if not text.strip():
            continue
        try:
            keywords = kw_extractor.extract_keywords(text)
            for kw, score in keywords:
                concept = kw.lower().strip()
                if (
                    len(concept) > 3
                    and concept not in CONCEPT_STOPWORDS
                    and not concept.isdigit()
                    and any(c.isalpha() for c in concept)
                ):
                    concepts[concept].append(chunk.index)
        except Exception:
            continue

    return dict(concepts)


def extract_concepts_regex(chunks: List[Chunk]) -> Dict[str, List[int]]:
    """Fallback concept extraction using regex noun phrases."""
    concepts: Dict[str, List[int]] = defaultdict(list)

    for chunk in chunks:
        phrases = extract_noun_phrases_regex(chunk.clean)
        for phrase in phrases:
            concept = phrase.lower().strip()
            if len(concept) > 3 and concept not in CONCEPT_STOPWORDS:
                concepts[concept].append(chunk.index)

    return dict(concepts)


def extract_concepts_spacy(chunks: List[Chunk]) -> Dict[str, List[int]]:
    """
    Extract noun phrases using spaCy.
    Requires: uv run python -m spacy download en_core_web_sm
    Slow on large documents — prefer yake.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError) as e:
        TerminalUI.warn(f"spaCy unavailable ({e}) — falling back to yake/regex")
        return extract_concepts_yake(chunks)

    concepts: Dict[str, List[int]] = defaultdict(list)

    for chunk in chunks:
        doc = nlp(chunk.clean[:5000])

        for np_chunk in doc.noun_chunks:
            concept = np_chunk.text.strip().lower()
            if (
                len(concept) > 3
                and concept not in CONCEPT_STOPWORDS
                and not concept.isdigit()
            ):
                concepts[concept].append(chunk.index)

        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"):
                concept = ent.text.strip().lower()
                if len(concept) > 3:
                    concepts[concept].append(chunk.index)

    return dict(concepts)


# ============================================================================
# Semantic graph building
# ============================================================================

def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix. Returns (n, n) float array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norms + 1e-9)
    return (normalized @ normalized.T).astype(float)


def build_semantic_graph(
    concepts: Dict[str, List[int]],
    chunks: List[Chunk],
    embed_fn: Callable[[List[str]], np.ndarray],
    min_freq: int = DEFAULT_MIN_CONCEPT_FREQ,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> GraphData:
    """
    Build a semantic similarity graph from extracted concepts.

    Concepts are connected when their embedding vectors exceed
    the similarity threshold — capturing semantic relationships
    even when concepts never appear in the same chunk.

    Args:
        concepts:             concept_label → list of chunk indices
        chunks:               source chunks (for metadata)
        embed_fn:             function (List[str]) → np.ndarray shape (n, dim)
        min_freq:             minimum chunk appearances to keep a concept
        max_concepts:         max concepts in the final graph
        similarity_threshold: cosine similarity cutoff for edges (0–1)

    Returns:
        GraphData with semantically connected concept nodes
    """
    # Filter by frequency
    filtered = {
        c: idx for c, idx in concepts.items()
        if len(idx) >= min_freq
    }

    if not filtered:
        TerminalUI.warn("No concepts passed frequency filter — try lowering --min-freq")
        return GraphData(filename="", chunk_count=len(chunks), total_concepts_extracted=len(concepts))

    # Rank by frequency, take top N
    sorted_concepts = sorted(
        filtered.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )[:max_concepts]

    concept_labels = [c for c, _ in sorted_concepts]
    concept_to_id  = {c: f"c{i}" for i, c in enumerate(concept_labels)}

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

    # Embed all concept labels in one batch (fast — short phrases)
    TerminalUI.dim(f"Embedding {len(concept_labels)} concept labels...")
    try:
        vectors = embed_fn(concept_labels)  # (n_concepts, dim)
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
    except Exception as e:
        TerminalUI.error(f"Embedding failed: {e} — falling back to co-occurrence graph")
        return build_cooccurrence_graph_fallback(concepts, chunks, min_freq, max_concepts)

    # Pairwise cosine similarity
    sim_matrix = _cosine_similarity_matrix(vectors)

    # Build edges above threshold
    edges = []
    n = len(concept_labels)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= similarity_threshold:
                edges.append(Relationship(
                    source=concept_to_id[concept_labels[i]],
                    target=concept_to_id[concept_labels[j]],
                    weight=round(sim, 4),
                    relation_type="semantic_similarity",
                ))

    return GraphData(
        filename="",
        nodes=nodes,
        edges=edges,
        chunk_count=len(chunks),
        total_concepts_extracted=len(concepts),
    )


# ============================================================================
# Co-occurrence fallback (original approach, kept as backup)
# ============================================================================

def build_cooccurrence_graph_fallback(
    concepts: Dict[str, List[int]],
    chunks: List[Chunk],
    min_freq: int = DEFAULT_MIN_CONCEPT_FREQ,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    min_rel_freq: int = DEFAULT_MIN_RELATION_FREQ,
) -> GraphData:
    """
    Legacy co-occurrence graph. Used as fallback if embedding fails.
    Two concepts are connected if they appear in the same chunk.
    """
    filtered = {c: idx for c, idx in concepts.items() if len(idx) >= min_freq}
    sorted_concepts = sorted(filtered.items(), key=lambda x: len(x[1]), reverse=True)[:max_concepts]

    concept_list  = [c for c, _ in sorted_concepts]
    concept_to_id = {c: f"c{i}" for i, c in enumerate(concept_list)}

    nodes = [
        ConceptNode(
            id=concept_to_id[c],
            label=c,
            frequency=len(idx),
            chunk_indices=idx,
        )
        for c, idx in sorted_concepts
    ]

    co_occurrence: Dict[Tuple[str, str], int] = Counter()
    for chunk_idx, chunk in enumerate(chunks):
        chunk_concepts = [c for c in concept_list if chunk_idx in filtered.get(c, [])]
        for i, c1 in enumerate(chunk_concepts):
            for c2 in chunk_concepts[i + 1:]:
                pair = tuple(sorted([c1, c2]))
                co_occurrence[pair] += 1

    edges = [
        Relationship(
            source=concept_to_id[c1],
            target=concept_to_id[c2],
            weight=weight,
            relation_type="co_occurrence",
        )
        for (c1, c2), weight in co_occurrence.items()
        if weight >= min_rel_freq
    ]

    return GraphData(
        filename="",
        nodes=nodes,
        edges=edges,
        chunk_count=len(chunks),
        total_concepts_extracted=len(concepts),
    )


# ============================================================================
# Main extraction function
# ============================================================================

def extract_graph_from_chunks(
    chunks: List[Chunk],
    filename: str,
    min_freq: int,
    max_concepts: int,
    min_rel_freq: int,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    extraction_mode: str = "yake",   # "yake" | "regex" | "spacy"
    embed_fn: Optional[Callable] = None,
    use_ai: bool = False,
) -> GraphData:
    """
    Extract a semantic knowledge graph from document chunks.

    Args:
        chunks:               List of Chunk objects
        filename:             Source document name
        min_freq:             Minimum concept occurrences
        max_concepts:         Max concepts to include
        min_rel_freq:         Min co-occurrences (fallback only)
        similarity_threshold: Cosine similarity cutoff for edges
        extraction_mode:      "yake" (default) | "regex" | "spacy"
        embed_fn:             Embedding function — if None, falls back to co-occurrence
        use_ai:               Elevate raw keywords to real concepts via LLM ✨

    Returns:
        GraphData with nodes and edges
    """
    TerminalUI.bold(f"Extracting concept graph from {filename} ({len(chunks)} chunks)")

    # ── Step 1: Extract raw keywords ─────────────────────────────────────────
    t = TerminalUI.timer("concept_extraction")

    if extraction_mode == "spacy":
        concepts = extract_concepts_spacy(chunks)
        mode_label = "spaCy"
    elif extraction_mode == "regex":
        concepts = extract_concepts_regex(chunks)
        mode_label = "regex"
    else:
        concepts = extract_concepts_yake(chunks)
        mode_label = "yake"

    TerminalUI.dim(f"Extracted {len(concepts)} raw keywords ({mode_label} mode)")
    t.done()

    # ── Step 1b: Elevate to real concepts via LLM (optional ✨) ──────────────
    if use_ai:
        t = TerminalUI.timer("concept_elevation")
        from src.core.Knowledge.Graph.ConceptElevator import elevate_concepts
        concepts = elevate_concepts(concepts, max_keywords=max_concepts)
        t.done()

    # ── Step 2: Build graph ──────────────────────────────────────────────────
    t = TerminalUI.timer("graph_building")

    if embed_fn is not None:
        graph = build_semantic_graph(
            concepts=concepts,
            chunks=chunks,
            embed_fn=embed_fn,
            min_freq=min_freq,
            max_concepts=max_concepts,
            similarity_threshold=similarity_threshold,
        )
        graph_type = "semantic"
    else:
        TerminalUI.warn("No embed_fn provided — using co-occurrence fallback")
        graph = build_cooccurrence_graph_fallback(
            concepts=concepts,
            chunks=chunks,
            min_freq=min_freq,
            max_concepts=max_concepts,
            min_rel_freq=min_rel_freq,
        )
        graph_type = "co-occurrence"

    graph.filename = filename
    graph.total_concepts_extracted = len(concepts)
    t.done()

    TerminalUI.success(
        f"Graph built ({graph_type}): {len(graph.nodes)} concepts, {len(graph.edges)} relationships"
    )

    if graph.nodes:
        top = graph.nodes[:5]
        TerminalUI.dim(f"Top concepts: {', '.join(n.label for n in top)}")

    return graph


# ============================================================================
# Pipeline entry point
# ============================================================================

def run_graph_extraction_pipeline(
    filename: str,
    chunks: List[Chunk],
    store: DarwinStore,
    min_freq: int = DEFAULT_MIN_CONCEPT_FREQ,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    min_rel_freq: int = DEFAULT_MIN_RELATION_FREQ,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    extraction_mode: str = "yake",
    embed_fn: Optional[Callable] = None,
    use_ai: bool = False,
    visualize: bool = False,
    rerun: bool = False,
) -> Optional[GraphData]:
    """
    Full graph extraction pipeline for one document.

    Args:
        filename:             Source document filename
        chunks:               List of Chunk objects
        store:                DarwinStore instance
        min_freq:             Minimum concept frequency
        max_concepts:         Maximum concepts to extract
        min_rel_freq:         Minimum co-occurrence (fallback only)
        similarity_threshold: Cosine similarity cutoff
        extraction_mode:      "yake" | "regex" | "spacy"
        embed_fn:             Embedding function for semantic graph
        use_ai:               Elevate keywords to real concepts via LLM ✨
        visualize:            Generate HTML visualization
        rerun:                Force re-extraction

    Returns:
        GraphData or None if skipped
    """
    if not chunks:
        TerminalUI.warn(f"No chunks for {filename}")
        return None

    # Check cache
    if not rerun and store.is_fresh(filename) and store.has_graph(filename):
        try:
            TerminalUI.dim(f"Skipping graph extraction: fresh graph exists for {filename}")
            return store.load_graph(filename)
        except Exception as e:
            TerminalUI.warn(f"Cached graph corrupted ({e}) — re-extracting...")

    # When use_ai=True the LLM handles concept quality,
    # so relax min_freq — don't filter keywords before LLM sees them
    effective_min_freq = 1 if use_ai else min_freq

    graph = extract_graph_from_chunks(
        chunks=chunks,
        filename=filename,
        min_freq=effective_min_freq,
        max_concepts=max_concepts,
        min_rel_freq=min_rel_freq,
        similarity_threshold=similarity_threshold,
        extraction_mode=extraction_mode,
        embed_fn=embed_fn,
        use_ai=use_ai,
    )

    # Save graph JSON
    store.save_graph(filename, graph, rerun=rerun)

    # Generate HTML visualization
    if visualize:
        try:
            from src.core.Knowledge.Graph.GraphUtils import export_graph_html
            html_graph = export_graph_html(graph)
            if html_graph:
                store.save_graph_html(filename, html_graph, rerun=rerun)
                TerminalUI.success(f"Graph visualization saved for {filename}")
            else:
                TerminalUI.error(f"Failed to generate visualization for {filename}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            TerminalUI.error(f"Visualization error: {e}")

    TerminalUI.success(f"Graph extracted for {filename}")
    return graph


# ============================================================================
# Legacy wrapper
# ============================================================================

def extract_concept_graph(
    chunks: List[Chunk],
    filename: str = "",
    **kwargs,
) -> GraphData:
    """Legacy wrapper for extract_graph_from_chunks."""
    return extract_graph_from_chunks(chunks, filename, **kwargs)