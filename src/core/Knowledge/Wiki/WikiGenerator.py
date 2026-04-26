"""
src/core/Knowledge/Wiki/WikiGenerator.py
-----------------------------------------
Wikipedia-style documentation from document knowledge graphs.

Two modes:
  - index (offline):   Phonebook-style concept listing with citations
  - summary (LLM):     Wikipedia prose generated from chunk context

Uses the same spaCy/yake/regex extraction pipeline as the graph module.
LLM calls are opt-in via --mode summary. Everything else runs offline.

Pipeline
--------
Graph + Chunks
    → Select top concepts (by frequency)
    → For each concept: gather relevant chunks
    → Index mode: format as markdown with snippets + page numbers
    → Summary mode: LLM writes prose from chunk context
    → Save via store.save_wiki()
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

from src.core.Chunking.Chunker import Chunk
from src.core.Knowledge.Graph.GraphUtils import GraphData, ConceptNode, Relationship
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WikiPage:
    title: str
    content: str
    concept: Optional[str] = None
    chunk_count: int = 0
    related: List[str] = field(default_factory=list)
    concept_count: int = 0


@dataclass
class WikiIndex:
    filename: str
    mode: str
    pages: List[WikiPage]
    concept_count: int


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_MAX_CONCEPTS = 30
DEFAULT_CHUNKS_PER_CONCEPT = 8

SUMMARY_PROMPT = """Write a Wikipedia-style encyclopedia entry for "{concept}" 
based on the document "{filename}".

Rules:
- Write in neutral, factual tone
- Start with a clear definition
- Include 2-4 paragraphs with key details
- Every claim must have a citation like [Page X]
- Keep it under 300 words

Context from the document:
{context}

Wikipedia entry for "{concept}":
"""

INDEX_PROMPT = """Create a master index for the document "{filename}".

Below are the top concepts with their frequencies and relationships.
Group related concepts into 3-6 thematic sections. For each section,
write a 1-sentence description of what ties these concepts together.

Concepts:
{concepts_list}

Master Index (markdown format):
"""


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_snippet(text: str, concept: str, max_length: int = 200) -> str:
    """Extract a text snippet centered on the concept mention."""
    import re
    pattern = re.compile(re.escape(concept), re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return text[:max_length] + "..." if len(text) > max_length else text
    start = max(0, match.start() - 100)
    end = min(len(text), match.end() + 100)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet += "..."
    return snippet


def _build_neighbor_map(graph: GraphData) -> Dict[str, List[str]]:
    """Build {node_id: [neighbor_node_ids]} from graph edges."""
    node_labels = {n.id: n.label for n in graph.nodes}
    neighbors = defaultdict(set)
    for edge in graph.edges:
        neighbors[edge.source].add(edge.target)
        neighbors[edge.target].add(edge.source)
    return {
        nid: [node_labels[t] for t in targets if t in node_labels]
        for nid, targets in neighbors.items()
    }


def _call_llm(prompt: str, model: str = "ollama/llama3.2") -> Optional[str]:
    """Call an LLM and return the response text. Returns None on failure."""
    try:
        from litellm import completion
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        TerminalUI.warn(f"LLM call failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Index mode (offline — no LLM)
# ═══════════════════════════════════════════════════════════════════════════

def generate_concept_index(
    graph: GraphData,
    chunks: List[Chunk],
    filename: str,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    chunks_per_concept: int = DEFAULT_CHUNKS_PER_CONCEPT,
) -> WikiIndex:
    """
    Phonebook-style concept index. Fully offline.
    Each concept gets a page with chunk references + snippets.
    """
    TerminalUI.bold(f"Generating concept index: {filename}")

    sorted_nodes = sorted(graph.nodes, key=lambda n: n.frequency, reverse=True)
    top_concepts = sorted_nodes[:max_concepts]
    neighbors = _build_neighbor_map(graph)

    pages: List[WikiPage] = []

    # ── Master index page ──
    lines = [
        f"# {filename} — Concept Index\n",
        f"*{len(top_concepts)} concepts extracted from {graph.chunk_count} chunks*\n",
        "\n## Concepts by frequency\n",
    ]
    for node in top_concepts:
        related = neighbors.get(node.id, [])[:5]
        related_str = ", ".join(related) if related else "—"
        lines.append(
            f"- **[{node.label}](#{node.label.replace(' ', '-').lower()})**  "
            f"→ {node.frequency} mentions  |  Related: {related_str}\n"
        )
    pages.append(WikiPage(
        title=f"{filename} — Index",
        content="".join(lines),
        concept_count=len(top_concepts),
    ))

    # ── Individual concept pages ──
    for node in top_concepts:
        TerminalUI.dim(f"  {node.label}")

        concept_chunks = [chunks[i] for i in node.chunk_indices[:chunks_per_concept] if i < len(chunks)]
        related = neighbors.get(node.id, [])[:10]

        lines = [
            f"# {node.label}\n",
            f"*Mentioned in {node.frequency} chunks across the document*\n",
        ]
        if related:
            lines.append(f"\n**See also:** {', '.join(related)}\n")
        lines.append("\n## References\n")

        for chunk in concept_chunks:
            snippet = _extract_snippet(chunk.clean, node.label)
            heading = chunk.heading or "Untitled"
            lines.append(f"### Page {chunk.page} — {heading}\n")
            lines.append(f"> {snippet}\n")
            lines.append(f"*Chunk {chunk.index}*\n\n")

        pages.append(WikiPage(
            title=node.label,
            content="".join(lines),
            concept=node.label,
            chunk_count=len(node.chunk_indices),
            related=related,
        ))

    TerminalUI.success(f"Generated {len(pages)} pages (index mode)")
    return WikiIndex(filename=filename, mode="index", pages=pages, concept_count=len(top_concepts))


# ═══════════════════════════════════════════════════════════════════════════
# Summary mode (LLM — opt-in)
# ═══════════════════════════════════════════════════════════════════════════

def generate_concept_summaries(
    graph: GraphData,
    chunks: List[Chunk],
    filename: str,
    llm_model: str = "ollama/llama3.2",
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    chunks_per_concept: int = DEFAULT_CHUNKS_PER_CONCEPT,
) -> WikiIndex:
    """
    Wikipedia-style summaries using an LLM.
    Falls back to index mode if LLM is unavailable.
    """
    TerminalUI.bold(f"Generating Wikipedia summaries: {filename}")
    TerminalUI.dim(f"LLM: {llm_model}")

    sorted_nodes = sorted(graph.nodes, key=lambda n: n.frequency, reverse=True)
    top_concepts = sorted_nodes[:max_concepts]
    neighbors = _build_neighbor_map(graph)

    pages: List[WikiPage] = []

    # ── LLM-generated master index ──
    concept_lines = []
    for node in top_concepts[:30]:
        related = neighbors.get(node.id, [])[:3]
        concept_lines.append(
            f"- {node.label} ({node.frequency} mentions)"
            + (f" — related: {', '.join(related)}" if related else "")
        )

    index_text = _call_llm(
        INDEX_PROMPT.format(filename=filename, concepts_list="\n".join(concept_lines)),
        model=llm_model,
    )
    if not index_text:
        index_text = f"# {filename} — Concept Index\n\n" + "\n".join(concept_lines)

    pages.append(WikiPage(
        title=f"{filename} — Index",
        content=index_text,
        concept_count=len(top_concepts),
    ))

    # ── LLM-generated concept summaries ──
    for i, node in enumerate(top_concepts):
        TerminalUI.dim(f"  [{i+1}/{len(top_concepts)}] {node.label}")

        concept_chunks = [chunks[j] for j in node.chunk_indices[:chunks_per_concept] if j < len(chunks)]
        context = "\n\n---\n\n".join(
            f"[Page {c.page}] {c.heading or ''}\n{c.clean[:600]}"
            for c in concept_chunks
        )

        summary = _call_llm(
            SUMMARY_PROMPT.format(concept=node.label, filename=filename, context=context),
            model=llm_model,
        )

        if not summary:
            # Fallback: plain snippet collection
            summary = f"# {node.label}\n\n*Mentioned in {node.frequency} chunks.*\n\n"
            for c in concept_chunks[:3]:
                summary += f"- Page {c.page}: {_extract_snippet(c.clean, node.label, 150)}\n"

        related = neighbors.get(node.id, [])[:8]
        if related:
            summary += f"\n\n**See also:** {', '.join(related)}"

        pages.append(WikiPage(
            title=node.label,
            content=summary,
            concept=node.label,
            chunk_count=len(node.chunk_indices),
            related=related,
        ))

    TerminalUI.success(f"Generated {len(pages)} pages (summary mode)")
    return WikiIndex(filename=filename, mode="summary", pages=pages, concept_count=len(top_concepts))


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def generate_wiki(
    graph: GraphData,
    chunks: List[Chunk],
    filename: str,
    mode: str = "index",
    llm_model: Optional[str] = None,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    chunks_per_concept: int = DEFAULT_CHUNKS_PER_CONCEPT,
) -> WikiIndex:
    """
    Generate wiki documentation from a concept graph.

    Args:
        graph:             Extracted concept graph
        chunks:            Original document chunks
        filename:          Source document name
        mode:              "index" (offline) or "summary" (LLM)
        llm_model:         LLM model for summary mode
        max_concepts:      Number of top concepts to include
        chunks_per_concept: Max chunks per concept for context

    Returns:
        WikiIndex with all generated pages
    """
    if mode == "summary":
        return generate_concept_summaries(
            graph, chunks, filename,
            llm_model=llm_model or "ollama/llama3.2",
            max_concepts=max_concepts,
            chunks_per_concept=chunks_per_concept,
        )
    else:
        return generate_concept_index(
            graph, chunks, filename,
            max_concepts=max_concepts,
            chunks_per_concept=chunks_per_concept,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_wiki_pipeline(
    filename: str,
    store: DarwinStore,
    mode: str = "index",
    llm_model: Optional[str] = None,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    chunks_per_concept: int = DEFAULT_CHUNKS_PER_CONCEPT,
    rerun: bool = False,
) -> Optional[WikiIndex]:
    """
    Full wiki pipeline for one document.
    Loads graph + chunks, generates wiki, saves to store.
    """
    if not store.has_graph(filename):
        TerminalUI.error(f"No graph found for '{filename}'. Run graph extraction first.")
        return None

    if not store.has_chunks(filename):
        TerminalUI.error(f"No chunks found for '{filename}'")
        return None

    if not rerun and store.has_wiki(filename):
        TerminalUI.dim(f"Skipping wiki: fresh exists for {filename}")
        try:
            wiki_dict = store.load_wiki(filename)
            # Reconstruct WikiIndex from dict
            pages = [
                WikiPage(
                    title=p.get("title", ""),
                    content=p.get("content", ""),
                    concept=p.get("concept"),
                    chunk_count=p.get("chunk_count", 0),
                    related=p.get("related", []),
                )
                for p in wiki_dict.get("pages", [])
            ]
            return WikiIndex(
                filename=wiki_dict.get("filename", filename),
                mode=wiki_dict.get("mode", "index"),
                pages=pages,
                concept_count=wiki_dict.get("concept_count", 0),
            )
        except Exception:
            TerminalUI.warn("Cached wiki corrupted — regenerating...")

    # ── Load graph ──
    graph_dict = store.load_graph(filename)
    nodes = [
        ConceptNode(
            id=n["id"], label=n["label"], frequency=n["freq"],
            chunk_indices=n["chunks"],
        )
        for n in graph_dict.get("nodes", [])
    ]
    edges = [
        Relationship(
            source=e["source"], target=e["target"], weight=e["weight"],
            relation_type=e.get("type", "context_similarity"),
        )
        for e in graph_dict.get("edges", [])
    ]
    graph = GraphData(
        filename=filename, nodes=nodes, edges=edges,
        chunk_count=graph_dict.get("chunk_count", 0),
        total_concepts_extracted=graph_dict.get("total_concepts_extracted", 0),
    )

    chunks = store.load_chunks(filename)
    if not chunks:
        TerminalUI.error(f"No chunks for '{filename}'")
        return None

    # ── Generate ──
    wiki = generate_wiki(
        graph=graph, chunks=chunks, filename=filename,
        mode=mode, llm_model=llm_model,
        max_concepts=max_concepts, chunks_per_concept=chunks_per_concept,
    )

    # ── Save ──
    store.save_wiki(filename, wiki)
    return wiki