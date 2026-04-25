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
    # """
    # try:
    #     import spacy
    #     nlp = spacy.load("en_core_web_sm")
    # except (ImportError, OSError) as e:
    #     TerminalUI.warn(f"spaCy unavailable ({e}) — falling back to yake/regex")
    #     return extract_concepts_yake(chunks)

    # concepts: Dict[str, List[int]] = defaultdict(list)

    # for chunk in chunks:
    #     doc = nlp(chunk.clean[:5000])

    #     for np_chunk in doc.noun_chunks:
    #         concept = np_chunk.text.strip().lower()
    #         if (
    #             len(concept) > 3
    #             and concept not in CONCEPT_STOPWORDS
    #             and not concept.isdigit()
    #         ):
    #             concepts[concept].append(chunk.index)

    #     for ent in doc.ents:
    #         if ent.label_ in ("PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "WORK_OF_ART"):
    #             concept = ent.text.strip().lower()
    #             if len(concept) > 3:
    #                 concepts[concept].append(chunk.index)

    # return dict(concepts)
    pass


# ============================================================================
# Semantic graph building
# ============================================================================

def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix. Returns (n, n) float array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norms + 1e-9)
    return (normalized @ normalized.T).astype(float)


def _concept_context_vectors(
    sorted_concepts: List[Tuple[str, List[int]]],
    chunk_vectors: np.ndarray,
) -> np.ndarray:
    """
    Build one context vector per concept by averaging the embeddings
    of all chunks the concept appears in.

    This measures how a concept is *used* in the document rather than
    how its label sounds — so "god" and "covenant" get different vectors
    based on which chunks they appear in, not because their names differ.

    Args:
        sorted_concepts:  [(label, [chunk_indices]), ...]
        chunk_vectors:    shape (n_chunks, dim) — pre-computed from .npz

    Returns:
        np.ndarray shape (n_concepts, dim)
    """
    n_chunks, dim = chunk_vectors.shape
    context_vecs  = np.zeros((len(sorted_concepts), dim), dtype=np.float32)

    for i, (_, chunk_indices) in enumerate(sorted_concepts):
        # Keep only valid indices
        valid = [idx for idx in chunk_indices if idx < n_chunks]
        if valid:
            context_vecs[i] = chunk_vectors[valid].mean(axis=0)

    return context_vecs


def _auto_threshold(sim_matrix: np.ndarray, target_avg_degree: float = 3.0) -> float:
    """
    Find a threshold that gives roughly target_avg_degree edges per node.

    Instead of a fixed cutoff, we pick the threshold that produces a
    graph with meaningful structure — connected enough to show clusters
    but sparse enough that nodes are clearly separated.

    target_avg_degree=3.0 means each concept connects to ~3 others on average,
    which produces the obsidian-graph feel: clusters with bridges, not a hairball.
    """
    n = sim_matrix.shape[0]
    if n < 2:
        return 0.75

    # Collect upper triangle similarities (excluding self)
    upper = []
    for i in range(n):
        for j in range(i + 1, n):
            upper.append(float(sim_matrix[i, j]))

    if not upper:
        return 0.75

    upper.sort(reverse=True)

    # Target number of edges for desired avg degree
    target_edges = int((n * target_avg_degree) / 2)
    target_edges = max(n - 1, min(target_edges, len(upper)))  # clamp

    # Threshold is just below the target_edges-th highest similarity
    threshold = upper[min(target_edges, len(upper) - 1)]

    # Hard clamp: never go below 0.50 (too noisy) or above 0.90 (too sparse)
    threshold = max(0.50, min(0.90, threshold))

    TerminalUI.dim(f"Auto threshold: {threshold:.3f} (targeting ~{target_avg_degree} edges/node)")
    return threshold


def build_semantic_graph(
    concepts: Dict[str, List[int]],
    chunks: List[Chunk],
    chunk_vectors: np.ndarray,
    min_freq: int = DEFAULT_MIN_CONCEPT_FREQ,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    similarity_threshold: Optional[float] = None,
) -> GraphData:
    """
    Build a concept graph using context vectors from chunk embeddings.

    Each concept gets a vector = mean of all chunk embeddings it appears in.
    Two concepts connect when their context vectors are similar — meaning
    they appear in similar parts of the document, not just that their
    names sound alike.

    If similarity_threshold is None, auto-threshold is used to target
    ~3 edges per node (good obsidian-graph structure).

    Args:
        concepts:             concept_label → list of chunk indices
        chunks:               source Chunk objects
        chunk_vectors:        pre-computed embeddings shape (n_chunks, dim)
        min_freq:             minimum chunk appearances to keep a concept
        max_concepts:         max concepts in final graph
        similarity_threshold: cosine similarity cutoff, or None for auto

    Returns:
        GraphData with context-vector-connected concept nodes
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

    # Build context vectors from chunk embeddings
    TerminalUI.dim(f"Building context vectors for {len(concept_labels)} concepts...")
    context_vecs = _concept_context_vectors(sorted_concepts, chunk_vectors)

    # Pairwise cosine similarity in chunk-embedding space
    sim_matrix = _cosine_similarity_matrix(context_vecs)

    # Auto threshold if not specified
    if similarity_threshold is None:
        threshold = _auto_threshold(sim_matrix, target_avg_degree=3.0)
    else:
        threshold = similarity_threshold

    # Build edges above threshold
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

    TerminalUI.dim(f"Threshold: {threshold:.3f} → {len(edges)} edges")

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
    chunk_vectors: Optional[np.ndarray] = None,
    similarity_threshold: Optional[float] = None,
    extraction_mode: str = "yake",
    use_ai: bool = False,
) -> GraphData:
    """
    Extract a concept graph from document chunks.

    Concepts are connected by context similarity — the mean of the
    chunk embeddings they appear in — not by label name similarity.
    This means "atonement" and "sacrifice" connect because they appear
    in similar chunks, not because the words sound related.

    If similarity_threshold is None, auto-threshold picks a value that
    produces ~3 edges per node (good obsidian-graph structure).

    Args:
        chunks:               List of Chunk objects
        filename:             Source document name
        min_freq:             Minimum concept occurrences
        max_concepts:         Max concepts to include
        min_rel_freq:         Min co-occurrences (fallback only)
        chunk_vectors:        Pre-computed chunk embeddings (n_chunks, dim).
                              If None, falls back to co-occurrence.
        similarity_threshold: Cosine similarity cutoff, or None for auto
        extraction_mode:      "yake" (default) | "regex" | "spacy"
        use_ai:               Elevate raw keywords to real concepts via LLM ✨

    Returns:
        GraphData with nodes and edges
    """
    TerminalUI.bold(f"Extracting concept graph from {filename} ({len(chunks)} chunks)")

    # ── Step 1: Extract raw keywords ────────────────────────────────────────
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

    # ── Step 1b: Elevate to real concepts via LLM (optional ✨) ─────────────
    if use_ai:
        t = TerminalUI.timer("concept_elevation")
        from src.core.Knowledge.Graph.ConceptElevator import elevate_concepts
        concepts = elevate_concepts(concepts, max_keywords=max_concepts)
        t.done()

    # ── Step 2: Build graph ──────────────────────────────────────────────────
    t = TerminalUI.timer("graph_building")

    if chunk_vectors is not None:
        graph = build_semantic_graph(
            concepts=concepts,
            chunks=chunks,
            chunk_vectors=chunk_vectors,
            min_freq=min_freq,
            max_concepts=max_concepts,
            similarity_threshold=similarity_threshold,  # None = auto
        )
        graph_type = "context-vector"
    else:
        TerminalUI.warn("No chunk vectors available — using co-occurrence fallback")
        TerminalUI.warn("Run the embedding pipeline first for better graph quality")
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

    # Load pre-computed chunk vectors from store (.npz bundle)
    chunk_vectors = None
    if store.has_embeddings(filename):
        try:
            embeddings   = store.load_embeddings(filename)
            chunk_vectors = embeddings["vectors"]  # shape (n_chunks, dim)
            TerminalUI.dim(f"Loaded chunk vectors: {chunk_vectors.shape}")
        except Exception as e:
            TerminalUI.warn(f"Could not load chunk embeddings ({e}) — using co-occurrence fallback")
    else:
        TerminalUI.warn(f"No embeddings found for {filename} — using co-occurrence fallback")
        TerminalUI.warn("Run: uv run src/core/Embedding/main.py to generate embeddings first")

    graph = extract_graph_from_chunks(
        chunks=chunks,
        filename=filename,
        min_freq=effective_min_freq,
        max_concepts=max_concepts,
        min_rel_freq=min_rel_freq,
        chunk_vectors=chunk_vectors,
        similarity_threshold=similarity_threshold if similarity_threshold != DEFAULT_SIMILARITY_THRESHOLD else None,
        extraction_mode=extraction_mode,
        use_ai=use_ai,
    )

    # Save graph JSON
    store.save_graph(filename, graph)

    # Generate HTML visualization
    if visualize:
        try:
            from src.core.Knowledge.Graph.GraphUtils import export_graph_html
            html_graph = export_graph_html(graph)
            if html_graph:
                store.save_graph_html(filename, html_graph)
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