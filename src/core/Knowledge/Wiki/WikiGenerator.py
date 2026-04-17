"""
src/core/Wiki/WikiGenerator.py
------------------------------
Generate Wikipedia-style documentation from document knowledge graphs.

This module uses the extracted concept graph and chunks to produce
human-readable documentation in two modes:

- Index mode (no LLM): Phonebook-style listing of concepts with page numbers
- Summary mode (LLM): Wikipedia-style prose with citations

Pipeline
--------
Graph + Chunks
    → Select top concepts (by frequency)
    → For each concept: gather relevant chunks
    → Index mode: format as markdown with snippets
    → Summary mode: LLM generates prose from chunks
    → Save as interlinked .md files
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

from src.core.Chunking.Chunker import Chunk
from src.core.Knowledge.Graph.GraphExtractor import GraphData
from src.core.Knowledge.Wiki.WikiUtils import (
    INDEX_TEMPLATE,
    CONCEPT_SECTION_TEMPLATE,
    CHUNK_REFERENCE_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    INDEX_PROMPT_TEMPLATE,
    extract_snippet,
    save_wiki_files,
    DEFAULT_MODE,
    DEFAULT_MAX_CONCEPTS,
    DEFAULT_CHUNKS_PER_CONCEPT,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


# ============================================================================
# Data models
# ============================================================================

@dataclass
class WikiPage:
    """A single wiki page (concept or index)."""
    title: str
    content: str
    concept: Optional[str] = None
    chunk_count: int = 0
    related: List[str] = field(default_factory=list)
    concept_count: int = 0


@dataclass
class WikiIndex:
    """Complete wiki output for a document."""
    filename: str
    pages: List[WikiPage]
    mode: str
    concept_count: int


# ============================================================================
# Index mode (no LLM)
# ============================================================================

def generate_concept_index(
    graph: GraphData,
    chunks: List[Chunk],
    filename: str,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    chunks_per_concept: int = DEFAULT_CHUNKS_PER_CONCEPT,
) -> WikiIndex:
    """
    Generate phonebook-style concept index (no LLM required).
    
    Args:
        graph: Extracted knowledge graph
        chunks: Original document chunks
        filename: Source document name
        max_concepts: Number of top concepts to include
        chunks_per_concept: Max chunk references per concept
    
    Returns:
        WikiIndex with index page + concept pages
    """
    TerminalUI.bold(f"Generating concept index for {filename}")
    
    # Sort concepts by frequency
    sorted_nodes = sorted(graph.nodes, key=lambda n: n.frequency, reverse=True)
    top_concepts = sorted_nodes[:max_concepts]
    
    # Build node lookup for relationships
    node_labels = {n.id: n.label for n in graph.nodes}
    edges_by_source = defaultdict(set)
    for edge in graph.edges:
        edges_by_source[edge.source].add(edge.target)
        edges_by_source[edge.target].add(edge.source)
    
    pages = []
    
    # Build index page
    index_content = []
    index_content.append(f"# {filename} — Concept Index\n")
    index_content.append(f"*{len(top_concepts)} concepts extracted*\n")
    index_content.append("\n## All Concepts\n")
    
    for node in top_concepts:
        related = [node_labels[t] for t in edges_by_source.get(node.id, set())]
        related_str = ", ".join(related[:5]) if related else "none"
        index_content.append(f"- **[{node.label}](#{node.label.replace(' ', '-').lower()})** — {node.frequency} mentions | Related: {related_str}\n")
    
    pages.append(WikiPage(
        title=f"{filename} — Index",
        content="".join(index_content),
        concept_count=len(top_concepts)
    ))
    
    # Build individual concept pages
    for node in top_concepts:
        TerminalUI.dim(f"  Building page: {node.label}")
        
        # Get chunks that mention this concept
        concept_chunks = [chunks[i] for i in node.chunk_indices[:chunks_per_concept]]
        
        # Get related concepts
        related_ids = edges_by_source.get(node.id, set())
        related = [node_labels[rid] for rid in related_ids if rid in node_labels]
        
        # Build page content
        content = []
        content.append(f"# {node.label}\n")
        content.append(f"*Frequency: {node.frequency} mentions across {len(node.chunk_indices)} chunks*\n")
        
        if related:
            content.append(f"\n**Related concepts:** {', '.join(related[:10])}\n")
        
        content.append("\n## Document References\n")
        
        for chunk in concept_chunks:
            snippet = extract_snippet(chunk.clean, node.label, 200)
            heading = chunk.heading or "Untitled section"
            
            content.append(f"### Page {chunk.page} — {heading}\n")
            content.append(f"> {snippet}\n")
            content.append(f"*Chunk {chunk.index}*\n")
        
        pages.append(WikiPage(
            title=node.label,
            content="".join(content),
            concept=node.label,
            chunk_count=len(node.chunk_indices),
            related=related
        ))
    
    TerminalUI.success(f"Generated {len(pages)} wiki pages")
    
    return WikiIndex(
        filename=filename,
        pages=pages,
        mode="index",
        concept_count=len(top_concepts)
    )


# ============================================================================
# Summary mode (LLM required)
# ============================================================================

def generate_concept_summary(
    graph: GraphData,
    chunks: List[Chunk],
    filename: str,
    llm_model: str = "ollama/llama3.2",
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    chunks_per_concept: int = DEFAULT_CHUNKS_PER_CONCEPT,
) -> WikiIndex:
    """
    Generate Wikipedia-style summaries using LLM.
    
    Args:
        graph: Extracted knowledge graph
        chunks: Original document chunks
        filename: Source document name
        llm_model: LLM model for generation
        max_concepts: Number of top concepts to include
        chunks_per_concept: Max chunks for context
    
    Returns:
        WikiIndex with LLM-generated summary pages
    """
    try:
        from litellm import completion
    except ImportError:
        TerminalUI.error("litellm not installed. Run: pip install litellm")
        TerminalUI.warn("Falling back to index mode")
        return generate_concept_index(graph, chunks, filename, max_concepts, chunks_per_concept)
    
    TerminalUI.bold(f"Generating Wikipedia-style summaries for {filename}")
    TerminalUI.dim(f"Using LLM: {llm_model}")
    
    # Sort concepts by frequency
    sorted_nodes = sorted(graph.nodes, key=lambda n: n.frequency, reverse=True)
    top_concepts = sorted_nodes[:max_concepts]
    
    # Build node lookup for relationships
    node_labels = {n.id: n.label for n in graph.nodes}
    edges_by_source = defaultdict(set)
    for edge in graph.edges:
        edges_by_source[edge.source].add(edge.target)
        edges_by_source[edge.target].add(edge.source)
    
    pages = []
    
    # Generate master index with LLM
    TerminalUI.dim("Generating master index...")
    concepts_list = "\n".join([f"- {n.label} ({n.frequency} mentions)" for n in top_concepts[:30]])
    
    index_prompt = INDEX_PROMPT_TEMPLATE.format(
        filename=filename,
        concepts_list=concepts_list
    )
    
    try:
        response = completion(
            model=llm_model,
            messages=[{"role": "user", "content": index_prompt}],
            temperature=0.3,
        )
        index_content = response.choices[0].message.content
    except Exception as e:
        TerminalUI.warn(f"LLM failed for index: {e}. Using fallback.")
        index_content = f"# {filename} — Concept Index\n\n" + concepts_list
    
    pages.append(WikiPage(
        title=f"{filename} — Index",
        content=index_content,
        concept_count=len(top_concepts)
    ))
    
    # Generate summary for each concept
    for i, node in enumerate(top_concepts):
        TerminalUI.dim(f"  [{i+1}/{len(top_concepts)}] Summarizing: {node.label}")
        
        # Gather context chunks
        concept_chunks = [chunks[i] for i in node.chunk_indices[:chunks_per_concept]]
        
        # Build context
        context = "\n\n---\n\n".join([
            f"[Page {c.page}] {c.heading}\n{c.clean[:500]}"
            for c in concept_chunks
        ])
        
        # Get related concepts
        related_ids = edges_by_source.get(node.id, set())
        related = [node_labels[rid] for rid in related_ids if rid in node_labels]
        
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            concept=node.label,
            filename=filename,
            context=context
        )
        
        try:
            response = completion(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            summary = response.choices[0].message.content
            
            # Add related concepts footer
            if related:
                summary += f"\n\n**See also:** {', '.join(related[:10])}"
            
            pages.append(WikiPage(
                title=node.label,
                content=summary,
                concept=node.label,
                chunk_count=len(node.chunk_indices),
                related=related
            ))
            
        except Exception as e:
            TerminalUI.error(f"Failed to summarize {node.label}: {e}")
            continue
    
    TerminalUI.success(f"Generated {len(pages)} wiki pages")
    
    return WikiIndex(
        filename=filename,
        pages=pages,
        mode="summary",
        concept_count=len(pages) - 1
    )


# ============================================================================
# Main generator
# ============================================================================

def generate_wiki(
    graph: GraphData,
    chunks: List[Chunk],
    filename: str,
    mode: str = DEFAULT_MODE,
    llm_model: Optional[str] = None,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    chunks_per_concept: int = DEFAULT_CHUNKS_PER_CONCEPT,
    output_dir: Optional[Path] = None,
) -> WikiIndex:
    """
    Generate wiki documentation from graph and chunks.
    
    Args:
        graph: Extracted knowledge graph
        chunks: Original document chunks
        filename: Source document name
        mode: "index" (no LLM) or "summary" (LLM)
        llm_model: LLM model for summary mode
        max_concepts: Number of top concepts to include
        chunks_per_concept: Max chunks per concept for context
        output_dir: Optional directory to save .md files
    
    Returns:
        WikiIndex with all generated pages
    """
    if mode == "summary":
        wiki = generate_concept_summary(
            graph, chunks, filename, llm_model, max_concepts, chunks_per_concept
        )
    else:
        wiki = generate_concept_index(
            graph, chunks, filename, max_concepts, chunks_per_concept
        )
    
    # Save to files if output directory provided
    if output_dir:
        saved = save_wiki_files(wiki.pages, output_dir, filename.replace('.', '_'))
        TerminalUI.success(f"Saved {len(saved)} files to {output_dir}")
    
    return wiki


# ============================================================================
# Pipeline entry point
# ============================================================================

def run_wiki_pipeline(
    filename: str,
    store: DarwinStore,
    mode: str = DEFAULT_MODE,
    llm_model: Optional[str] = None,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    chunks_per_concept: int = DEFAULT_CHUNKS_PER_CONCEPT,
    output_dir: Optional[Path] = None,
    rerun: bool = False,
) -> Optional[WikiIndex]:
    """
    Full wiki generation pipeline for one document.
    
    Args:
        filename: Source document filename
        store: DarwinStore instance
        mode: "index" or "summary"
        llm_model: LLM model for summary mode
        max_concepts: Max concepts to include
        chunks_per_concept: Max chunks per concept
        output_dir: Directory to save .md files
        rerun: Force regeneration
    
    Returns:
        WikiIndex or None if skipped
    """
    # Check prerequisites
    if not store.has_graph(filename):
        TerminalUI.error(f"No graph found for {filename}. Run graph extraction first.")
        return None
    
    if not store.has_chunks(filename):
        TerminalUI.error(f"No chunks found for {filename}")
        return None
    
    # Load data
    graph_dict = store.load_graph(filename)
    chunks = store.load_chunks(filename)
    
    if not chunks:
        TerminalUI.error(f"Failed to load chunks for {filename}")
        return None
    
    # Reconstruct GraphData from dict
    from src.core.Knowledge.Graph.GraphExtractor import GraphData, ConceptNode, Relationship
    
    nodes = [
        ConceptNode(
            id=n["id"],
            label=n["label"],
            frequency=n["frequency"],
            chunk_indices=n["chunk_indices"]
        )
        for n in graph_dict.get("nodes", [])
    ]
    
    edges = [
        Relationship(
            source=e["source"],
            target=e["target"],
            weight=e["weight"],
            relation_type=e.get("relation_type", "co_occurrence")
        )
        for e in graph_dict.get("edges", [])
    ]
    
    graph = GraphData(
        filename=filename,
        nodes=nodes,
        edges=edges,
        chunk_count=graph_dict.get("chunk_count", 0),
        total_concepts_extracted=graph_dict.get("total_concepts_extracted", 0)
    )
    output_dir = store._ensure_wiki_dir(filename)
    # Generate wiki
    wiki = generate_wiki(
        graph=graph,
        chunks=chunks,
        filename=filename,
        mode=mode,
        llm_model=llm_model,
        max_concepts=max_concepts,
        chunks_per_concept=chunks_per_concept,
        output_dir=output_dir,
    )
    
    # Save wiki to store
    store.save_wiki(filename, wiki, rerun=rerun)
    
    return wiki