import numpy as np
from typing import List, Dict, Optional

from src.features.Ingestion.Chunking.lib import Chunk
from src.features.Knowledge.Graph.GraphUtils import GraphData, ConceptNode, Relationship
from src.features.Knowledge.Wiki.WikiUtils import (
    WikiPage,
    DocumentWiki,
    DEFAULT_CHUNKS_PER_CONCEPT,
)
from src.features.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType


# ═══════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════

_PAGE_PROMPT = """You are summarizing one concept from a document.

Concept: {concept}
Related concepts in this document: {related}

Source passages:
{context}

Write a 3-5 sentence summary of what this document says about "{concept}".
Be specific to this document. Do not add outside knowledge.
Return only the summary, no preamble."""

_MANIFEST_PROMPT = """You are writing a one-paragraph overview of a document.

The document covers these concepts:
{concept_summaries}

Write a single paragraph describing what this document is about.
Themes and purpose only. No author names, dates, or specific claims.
Return only the paragraph, no preamble."""


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_edge_map(graph: GraphData) -> Dict[str, List[str]]:
    """Build {node_id: [neighbor_labels]} from graph edges."""
    node_labels = {n.id: n.label for n in graph.nodes}
    from collections import defaultdict
    neighbors: Dict[str, set] = defaultdict(set)
    for edge in graph.edges:
        neighbors[edge.source].add(edge.target)
        neighbors[edge.target].add(edge.source)
    return {
        nid: [node_labels[t] for t in targets if t in node_labels]
        for nid, targets in neighbors.items()
    }


def _extract_keywords(
    chunk_indices: List[int],
    chunks: List[Chunk],
    spacy_extractor,
    max_keywords: int = 5,
) -> List[str]:
    """
    Extract top keywords from a cluster's chunks.
    Uses spaCy if available, falls back to empty list — never blocks.
    """
    if spacy_extractor is None:
        return []
    try:
        from collections import Counter
        counts: Counter = Counter()
        for idx in chunk_indices[:10]:
            if idx < len(chunks):
                phrases = spacy_extractor.extract_noun_chunks(chunks[idx].clean)
                for p in phrases:
                    counts[p.lower().strip()] += 1
        return [p for p, _ in counts.most_common(max_keywords)]
    except Exception:
        return []


def _call_llm(prompt: str, model: str) -> Optional[str]:
    """Call LLM via litellm. Returns None on any failure — never raises."""
    try:
        from litellm import completion
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        darwin.log(f"LLM call failed: {e}", LogType.WARNING)
        return None


def _build_context(
    chunk_indices: List[int],
    chunks: List[Chunk],
    chunks_per_concept: int,
    max_chars: int = 4000,
) -> str:
    """Assemble chunk text into LLM context string, respecting char budget."""
    parts = []
    total = 0
    for idx in chunk_indices[:chunks_per_concept]:
        if idx >= len(chunks):
            continue
        c = chunks[idx]
        part = f"[Page {c.page}] {c.heading or ''}\n{c.clean[:600]}".strip()
        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return "\n\n---\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — Build wiki from graph (offline, no LLM)
# ═══════════════════════════════════════════════════════════════════════════

def build_wiki_from_graph(
    graph: GraphData,
    centroids: np.ndarray,
    chunks: List[Chunk],
    spacy_extractor=None,
) -> DocumentWiki:
    """
    Build DocumentWiki directly from GraphData. No LLM calls.

    Args:
        graph:            Extracted concept graph
        centroids:        Cluster centroids (n_nodes, dim) — for query-time retrieval
        chunks:           Source chunks — for keyword extraction
        spacy_extractor:  Optional SpacyConceptExtractor — pass None to skip keywords

    Returns:
        DocumentWiki with all pages populated, summary=None, manifest=None
    """
    darwin.log(f"Building wiki from graph: {len(graph.nodes)} concepts", LogType.INFO)

    edge_map = _build_edge_map(graph)

    pages = []
    for i, node in enumerate(graph.nodes):
        centroid = (
            centroids[i].tolist()
            if i < len(centroids)
            else []
        )

        related = edge_map.get(node.id, [])

        keywords = _extract_keywords(
            node.chunk_indices, chunks, spacy_extractor
        )

        pages.append(WikiPage(
            concept       = node.label,
            chunk_indices = node.chunk_indices,
            related       = related,
            centroid      = centroid,
            frequency     = node.frequency,
            keywords      = keywords,
        ))

    darwin.log(
        f"Wiki built — {len(pages)} pages, "
        f"{sum(1 for p in pages if p.keywords)} with keywords",
        LogType.SUCCESS,
    )

    return DocumentWiki(filename=graph.filename, pages=pages)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — LLM enrichment (opt-in)
# ═══════════════════════════════════════════════════════════════════════════

def generate_wiki_summaries(
    wiki: DocumentWiki,
    chunks: List[Chunk],
    llm_model: str,
    chunks_per_concept: int = DEFAULT_CHUNKS_PER_CONCEPT,
    max_context_chars: int = 4000,
) -> DocumentWiki:
    """
    Fill in WikiPage.summary and DocumentWiki.manifest via LLM.
    Operates in-place on the wiki object and returns it.
    Skips pages that already have a summary (idempotent).
    Never raises — failed LLM calls leave summary=None and continue.

    Args:
        wiki:               DocumentWiki from build_wiki_from_graph()
        chunks:             Source chunks for context assembly
        llm_model:          litellm model string e.g. "ollama/llama3.2"
        chunks_per_concept: Max chunks to include per concept context
        max_context_chars:  Hard cap on context chars per LLM call

    Returns:
        The same DocumentWiki with summaries filled where LLM succeeded
    """
    pending = [p for p in wiki.pages if p.summary is None]
    darwin.log(
        f"Generating summaries: {len(pending)} pending / "
        f"{len(wiki.pages)} total [{llm_model}]",
        LogType.INFO,
    )

    for i, page in enumerate(pending):
        darwin.log(
            f"  [{i+1}/{len(pending)}] {page.concept}", LogType.INFO
        )

        context = _build_context(
            page.chunk_indices, chunks, chunks_per_concept, max_context_chars
        )
        if not context:
            darwin.log(f"  No context for '{page.concept}' — skipping", LogType.WARNING)
            continue

        related_str = ", ".join(page.related[:5]) if page.related else "none"
        prompt = _PAGE_PROMPT.format(
            concept=page.concept,
            related=related_str,
            context=context,
        )

        summary = _call_llm(prompt, llm_model)
        if summary:
            page.summary = summary.strip()
        else:
            darwin.log(f"  LLM failed for '{page.concept}' — leaving pending", LogType.WARNING)

    # ── Manifest — generated once, after all pages ──
    if wiki.manifest is None:
        completed = [p for p in wiki.pages if p.summary]
        if completed:
            darwin.log("Generating document manifest...", LogType.INFO)
            concept_summaries = "\n".join(
                f"- {p.concept}: {p.summary[:120]}..."
                if p.summary and len(p.summary) > 120
                else f"- {p.concept}: {p.summary or '(no summary)'}"
                for p in completed[:20]
            )
            manifest = _call_llm(
                _MANIFEST_PROMPT.format(concept_summaries=concept_summaries),
                llm_model,
            )
            if manifest:
                wiki.manifest = manifest.strip()
                darwin.log("Manifest generated", LogType.SUCCESS)
            else:
                darwin.log("Manifest generation failed — leaving None", LogType.WARNING)

    completed_count = sum(1 for p in wiki.pages if p.summary)
    darwin.log(
        f"Summaries complete: {completed_count}/{len(wiki.pages)} pages",
        LogType.SUCCESS,
    )
    return wiki


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_wiki_pipeline(
    filename: str,
    store: DarwinStore,
    mode: str = "index",
    llm_model: Optional[str] = None,
    chunks_per_concept: int = DEFAULT_CHUNKS_PER_CONCEPT,
    rerun: bool = False,
    completed_files : Optional[List[str]] = None,
) -> Optional[DocumentWiki]:
    """
    Full wiki generation pipeline for one document.

    Mode "index":   build_wiki_from_graph() only — offline, fast
    Mode "summary": build_wiki_from_graph() + generate_wiki_summaries()

    Always loads graph and centroids from store.
    Saves DocumentWiki via store.save_wiki().
    """
    if not store.has_graph(filename):
        darwin.log(
            f"No graph found for '{filename}'. Run graph extraction first.",
            LogType.ERROR,
        )
        return None

    # ── Cache check ──
    if not rerun and store.has_wiki(filename):
        darwin.log(f"Skipping wiki: fresh exists for {filename}", LogType.INFO)
        completed_files.append(filename)

        try:
            return DocumentWiki.from_dict(store.load_wiki(filename))
        except Exception:
            darwin.log("Cached wiki corrupted — regenerating...", LogType.WARNING)

    # ── Load graph ──
    graph_dict = store.load_graph(filename)
    nodes = [
        ConceptNode(
            id=n["id"], label=n["label"],
            frequency=n["freq"], chunk_indices=n["chunks"],
        )
        for n in graph_dict.get("nodes", [])
    ]
    edges = [
        Relationship(
            source=e["source"], target=e["target"],
            weight=e["weight"], relation_type=e.get("type", "context_similarity"),
        )
        for e in graph_dict.get("edges", [])
    ]
    graph = GraphData(
        filename=filename, nodes=nodes, edges=edges,
        chunk_count=graph_dict.get("chunk_count", 0),
        total_concepts_extracted=graph_dict.get("total_concepts_extracted", 0),
    )

    if not graph.nodes:
        darwin.log(f"Graph is empty for '{filename}'", LogType.ERROR)
        return None

    # ── Load centroids ──
    centroid_data = store.load_graph_centroids(filename)
    if centroid_data:
        # centroid_data is [{concept, centroid}, ...] — align by node order
        centroid_map = {c["concept"]: c["centroid"] for c in centroid_data}
        centroids = np.array([
            centroid_map.get(node.label, [])
            for node in graph.nodes
        ], dtype=object)
    else:
        # Fallback: recompute from embeddings if centroids not saved
        darwin.log("No centroids found — recomputing from embeddings", LogType.WARNING)
        vectors, emb_chunks = store.load_embeddings(filename)
        if vectors is None:
            darwin.log(f"No embeddings for '{filename}' — cannot build wiki", LogType.ERROR)
            return None
        centroids = np.array([
            vectors[[i for i in node.chunk_indices if i < len(vectors)]].mean(axis=0).tolist()
            if any(i < len(vectors) for i in node.chunk_indices)
            else []
            for node in graph.nodes
        ], dtype=object)

    # ── Load chunks ──
    chunks = store.load_chunks(filename)
    if not chunks:
        darwin.log(f"No chunks for '{filename}'", LogType.ERROR)
        return None

    # Validate chunk indices
    max_chunk_idx = max(
        (idx for node in nodes for idx in node.chunk_indices),
        default=0,
    )
    if max_chunk_idx >= len(chunks):
        darwin.log(
            f"Graph indices stale — references chunk {max_chunk_idx} "
            f"but only {len(chunks)} exist. Re-run graph extraction.",
            LogType.ERROR,
        )
        return None

    # ── Get spaCy extractor for keyword extraction ──
    spacy_extractor = None
    try:
        from src.Models import ModelManager, ModelCategory
        loader = ModelManager.get_loader()
        spacy_extractor = loader.get(ModelCategory.TOKENIZER)
    except Exception:
        pass

    # ── Stage 1: build structure (always) ──
    wiki = build_wiki_from_graph(
        graph=graph,
        centroids=centroids,
        chunks=chunks,
        spacy_extractor=spacy_extractor,
    )

    # ── Stage 2: LLM summaries (opt-in) ──
    if mode == "summary":
        if not llm_model:
            darwin.log(
                "Summary mode requested but no LLM model specified — "
                "defaulting to ollama/llama3.2",
                LogType.WARNING,
            )
            llm_model = "ollama/llama3.2"
        wiki = generate_wiki_summaries(
            wiki=wiki,
            chunks=chunks,
            llm_model=llm_model,
            chunks_per_concept=chunks_per_concept,
        )

    store.save_wiki(filename, wiki)
    darwin.log(
        f"Wiki saved — {len(wiki.pages)} pages, "
        f"manifest={'yes' if wiki.manifest else 'pending'}",
        LogType.SUCCESS,
    )
    return wiki