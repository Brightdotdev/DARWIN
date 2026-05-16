"""
src/core/Knowledge/Wiki/WikiGenerator.py
----------------------------------------
Generates DocumentWiki from GraphData.

Document-agnostic structural chunk filtering via content analysis.
"""

import re
import numpy as np
from typing import List, Dict, Optional, Callable
from collections import Counter

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
# Document-agnostic structural chunk detection
# ═══════════════════════════════════════════════════════════════════════════

STRUCTURAL_SIGNALS = {
    "cover", "title", "copyright", "table of contents", "contents",
    "acknowledgments", "preface", "foreword", "introduction", "about",
    "appendix", "index", "glossary", "bibliography", "references",
    "page", "chapter", "book", "part", "section", "volume",
    "ebook", "edition", "published", "publisher", "isbn",
    "all rights reserved", "printed in", "first edition",
}


def _is_structural_chunk(chunk: Chunk, all_chunks: List[Chunk]) -> bool:
    """Detect structural chunks by content, not index."""
    text = chunk.clean.lower()
    words = set(re.findall(r'\b[a-z]+\b', text))

    # Very short chunks
    if len(chunk.clean) < 200:
        return True

    # High structural signal ratio
    structural_found = words & STRUCTURAL_SIGNALS
    if len(words) > 0 and len(structural_found) / len(words) > 0.3:
        return True

    # TOC pattern: many short numbered lines
    lines = [l.strip() for l in chunk.clean.split('\n') if l.strip()]
    if len(lines) > 3:
        short_numbered = sum(1 for l in lines if len(l) < 60 and re.search(r'\d', l))
        if short_numbered / len(lines) > 0.5:
            return True

    # Repetitive text (headers/footers)
    text_hash = hash(text[:200])
    repeat_count = sum(1 for c in all_chunks if hash(c.clean.lower()[:200]) == text_hash)
    if repeat_count > 2:
        return True

    # First 5% of doc with very low word count
    if chunk.index <= len(all_chunks) * 0.05 and len(words) < 20:
        return True

    return False


def _filter_page_chunks(page_chunk_indices: List[int], all_chunks: List[Chunk]) -> List[int]:
    """Filter a page's chunks to exclude structural ones."""
    structural = {c.index for c in all_chunks if _is_structural_chunk(c, all_chunks)}
    filtered = [idx for idx in page_chunk_indices if idx not in structural]

    # Fallback: if all removed, keep the longest non-structural or original
    if not filtered and page_chunk_indices:
        candidates = [c for c in all_chunks if c.index in page_chunk_indices]
        if candidates:
            non_struct = [c for c in candidates if not _is_structural_chunk(c, all_chunks)]
            if non_struct:
                longest = max(non_struct, key=lambda c: len(c.clean))
            else:
                longest = max(candidates, key=lambda c: len(c.clean))
            return [longest.index]

    return filtered


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
# Anchor chunk selection
# ═══════════════════════════════════════════════════════════════════════════

def _find_best_anchor_chunk(page: WikiPage, chunks: List[Chunk]) -> Optional[Chunk]:
    """Text overlap — no embeddings needed."""
    concept_words = set(page.concept.lower().split())

    best_chunk = None
    best_score = -1

    for idx in page.chunk_indices:
        chunk = next((c for c in chunks if c.index == idx), None)
        if not chunk:
            continue

        chunk_text = chunk.clean.lower()
        chunk_words = set(re.findall(r'\b[a-z]{3,}\b', chunk_text))
        overlap = len(concept_words & chunk_words)

        if page.concept.lower() in chunk_text:
            overlap += 10
        elif any(w in chunk_text for w in concept_words if len(w) > 4):
            overlap += 3

        if overlap > best_score:
            best_score = overlap
            best_chunk = chunk

    return best_chunk


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_edge_map(graph: GraphData) -> Dict[str, List[str]]:
    """Build {node_id: [neighbor_labels]} from graph edges."""
    node_labels = {n.id: n.label for n in graph.nodes}

    semantic_neighbors: Dict[str, set] = {}
    cooc_neighbors: Dict[str, set] = {}

    for edge in graph.edges:
        src, tgt = edge.source, edge.target
        if edge.relation_type != "co_occurrence":
            semantic_neighbors.setdefault(src, set()).add(tgt)
            semantic_neighbors.setdefault(tgt, set()).add(src)
        else:
            cooc_neighbors.setdefault(src, set()).add(tgt)
            cooc_neighbors.setdefault(tgt, set()).add(src)

    result = {}
    for nid, label in node_labels.items():
        related = []
        for target in semantic_neighbors.get(nid, set()):
            tlabel = node_labels.get(target)
            if tlabel and tlabel not in related:
                related.append(tlabel)
        for target in cooc_neighbors.get(nid, set()):
            tlabel = node_labels.get(target)
            if tlabel and tlabel not in related:
                related.append(tlabel)
        result[nid] = related

    return result


# Stopwords for keyword extraction — expanded from original
_KEYWORD_STOPS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her",
    "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "man",
    "new", "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let",
    "put", "say", "she", "too", "use", "than", "them", "then", "they", "this",
    "that", "with", "have", "from", "know", "want", "been", "good", "much", "some",
    "time", "very", "when", "come", "here", "just", "like", "long", "make", "many",
    "over", "such", "take", "well", "were", "what", "will", "would", "there", "their",
    "said", "each", "which", "could", "other", "after", "first", "never", "these",
    "think", "where", "being", "every", "great", "might", "shall", "still", "those",
    "while", "about", "before", "right", "should", "through", "years", "place",
    "world", "people", "between", "another", "because", "without", "against",
    "nothing", "himself", "something", "anything", "everything", "someone",
    "anyone", "everyone", "book", "chapter", "part", "section", "page",
}


def _extract_keywords(
    chunk_indices: List[int],
    chunks: List[Chunk],
    spacy_extractor=None,
    max_keywords: int = 5,
) -> List[str]:
    """Extract top keywords from a cluster's chunks."""
    counts: Counter = Counter()

    for idx in chunk_indices[:10]:
        if idx >= len(chunks):
            continue

        text = chunks[idx].clean.lower()

        # Try spaCy noun chunks
        if spacy_extractor and hasattr(spacy_extractor, 'nlp'):
            try:
                doc = spacy_extractor.nlp(text[:3000])
                for nc in doc.noun_chunks:
                    phrase = " ".join(
                        t.text for t in nc
                        if not t.is_stop and len(t.text) > 2
                    )
                    if len(phrase) > 3:
                        counts[phrase.lower().strip()] += 1
            except Exception:
                pass

        # Fallback: simple bigram extraction
        words = re.findall(r'\b[a-z]{3,}\b', text)
        words = [w for w in words if w not in _KEYWORD_STOPS]

        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i + 1]}"
            # Reject stopword-only bigrams that slipped through
            if words[i] not in _KEYWORD_STOPS and words[i + 1] not in _KEYWORD_STOPS:
                counts[bigram] += 1

    return [k for k, _ in counts.most_common(max_keywords)]


def _call_llm(client, prompt: str) -> Optional[str]:
    """Call the LLM via the provided LLMClient."""
    if client is None:
        return None

    try:
        from src.features.LLM.LLMTypes import LLMResponse
        resp: LLMResponse = client.call_small(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        if resp.error or not resp.text:
            darwin.log(f"LLM call failed: {resp.error}", LogType.WARNING)
            return None
        return resp.text.strip()
    except Exception as e:
        darwin.log(f"LLM call exception: {e}", LogType.WARNING)
        return None


def _build_context(
    chunk_indices: List[int],
    chunks: List[Chunk],
    chunks_per_concept: int,
    max_chars: int = 4000,
) -> str:
    """Assemble chunk text into LLM context string."""
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


def _is_structural_node(node: ConceptNode) -> bool:
    """Detect structural/TOC noise by concept label."""
    label = node.label.lower()

    structural_fragments = {
        "cover", "copyright", "title page", "table of contents",
        "acknowledgments", "preface", "foreword", "introduction - about",
        "appendix", "index of", "glossary", "bibliography",
    }
    if any(frag in label for frag in structural_fragments):
        return True

    if label.count(".") >= 3 or label.count(" - ") >= 2:
        return True
    if label.count(" ch") >= 2 or label.count("book ") >= 2:
        return True

    if len(node.label) > 100:
        return True

    if node.first_mention_chunk <= 1 and node.frequency > 20:
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════
# Schema guard for GraphData deserialization
# ═══════════════════════════════════════════════════════════════════════════

class GraphDataSchema:
    """Minimal schema validation to catch drift between graph extractor and wiki builder."""
    REQUIRED_NODE_FIELDS = {"id", "label", "freq", "chunks"}
    REQUIRED_EDGE_FIELDS = {"source", "target", "weight"}

    @classmethod
    def validate(cls, graph_dict: dict) -> None:
        if not isinstance(graph_dict, dict):
            raise ValueError("Graph data must be a dict")

        nodes = graph_dict.get("nodes", [])
        edges = graph_dict.get("edges", [])

        if nodes and isinstance(nodes[0], dict):
            missing = cls.REQUIRED_NODE_FIELDS - set(nodes[0].keys())
            if missing:
                darwin.log(
                    f"Graph node schema drift: missing {missing}. "
                    "Wiki builder may drop fields silently.",
                    LogType.WARNING,
                )

        if edges and isinstance(edges[0], dict):
            missing = cls.REQUIRED_EDGE_FIELDS - set(edges[0].keys())
            if missing:
                darwin.log(
                    f"Graph edge schema drift: missing {missing}.",
                    LogType.WARNING,
                )


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — Build wiki from graph
# ═══════════════════════════════════════════════════════════════════════════

def build_wiki_from_graph(
    graph: GraphData,
    chunks: List[Chunk],
    spacy_extractor=None,
) -> DocumentWiki:
    """Build DocumentWiki directly from GraphData."""
    darwin.log(f"Building wiki from graph: {len(graph.nodes)} concepts", LogType.INFO)

    edge_map = _build_edge_map(graph)

    pages = []
    for node in graph.nodes:
        if _is_structural_node(node):
            continue

        centroid = node.centroid if hasattr(node, 'centroid') else []
        related = edge_map.get(node.id, [])

        # Filter structural chunks by content, not index
        valid_chunks = _filter_page_chunks(node.chunk_indices, chunks)

        # Fallback: if all filtered out, keep original
        if not valid_chunks:
            valid_chunks = node.chunk_indices

        keywords = _extract_keywords(valid_chunks, chunks, spacy_extractor)

        # Fallback: if spaCy keywords empty, use concept words + related
        if not keywords:
            concept_words = [
                w for w in node.label.lower().split()
                if len(w) > 2 and w not in _KEYWORD_STOPS
            ]
            keywords = concept_words[:5]

        keywords = list(set(keywords + [r.lower() for r in related[:3]]))[:10]

        pages.append(WikiPage(
            concept=node.label,
            chunk_indices=valid_chunks,
            related=related,
            centroid=centroid,
            frequency=node.frequency,
            keywords=keywords,
        ))

    # Sort by connectivity-weighted frequency
    node_degrees = Counter()
    for e in graph.edges:
        node_degrees[e.source] += 1
        node_degrees[e.target] += 1

    for page in pages:
        node_id = next((n.id for n in graph.nodes if n.label == page.concept), None)
        deg = node_degrees.get(node_id, 0)
        page._sort_score = page.frequency * (1 + deg / 10)

    pages.sort(key=lambda p: getattr(p, '_sort_score', p.frequency), reverse=True)

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
    client,
    chunks_per_concept: int = DEFAULT_CHUNKS_PER_CONCEPT,
    max_context_chars: int = 4000,
) -> DocumentWiki:
    """Fill in WikiPage.summary and DocumentWiki.manifest via LLM."""
    pending = [p for p in wiki.pages if p.summary is None]

    for i, page in enumerate(pending):
        darwin.log(f"  [{i+1}/{len(pending)}] {page.concept}", LogType.INFO)

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

        summary = _call_llm(client, prompt)
        if summary:
            page.summary = summary.strip()
        else:
            darwin.log(f"  LLM failed for '{page.concept}' — leaving pending", LogType.WARNING)

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
            manifest = _call_llm(client,
                _MANIFEST_PROMPT.format(concept_summaries=concept_summaries)
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
    provider: Optional[str] = None,
    chunks_per_concept: int = DEFAULT_CHUNKS_PER_CONCEPT,
    rerun: bool = False,
    completed_files: Optional[List[str]] = None,
) -> Optional[DocumentWiki]:
    """Full wiki generation pipeline for one document."""
    if not store.has_graph(filename):
        darwin.log(
            f"No graph found for '{filename}'. Run graph extraction first.",
            LogType.ERROR,
        )
        return None

    if store.has_wiki(filename) and not rerun:
        darwin.log(f"Skipping wiki: fresh exists for {filename}", LogType.INFO)
        if completed_files is not None:
            completed_files.append(filename)
        try:
            return store.load_wiki(filename)
        except Exception:
            darwin.log("Cached wiki corrupted — regenerating...", LogType.WARNING)

    graph_dict = store.load_graph(filename)

    # Schema guard
    try:
        GraphDataSchema.validate(graph_dict)
    except ValueError as e:
        darwin.log(f"Graph schema validation failed: {e}", LogType.ERROR)
        return None

    nodes = [
        ConceptNode(
            id=n["id"],
            label=n["label"],
            display_name=n.get("display_name", n["label"]),
            entity_type=n.get("type", "UNKNOWN"),
            frequency=n["freq"],
            chunk_indices=n["chunks"],
            first_mention_chunk=n.get("first_mention_chunk", 0),
            mention_forms=n.get("mention_forms", [n["label"]]),
            centroid=n.get("centroid", []),
        )
        for n in graph_dict.get("nodes", [])
    ]
    edges = [
        Relationship(
            source=e["source"],
            target=e["target"],
            weight=e["weight"],
            relation_type=e.get("relation_type", e.get("type", "context_similarity")),
            evidence_chunks=e.get("evidence_chunks", []),
            surface_forms=e.get("surface_forms", []),
        )
        for e in graph_dict.get("edges", [])
    ]
    graph = GraphData(
        filename=filename, nodes=nodes, edges=edges,
        chunk_count=graph_dict.get("chunk_count", 0),
        total_entities_extracted=graph_dict.get("total_entities_extracted", 0),
        total_relations_extracted=graph_dict.get("total_relations_extracted", 0),
        processing_mode=graph_dict.get("processing_mode", "unknown"),
    )

    if not graph.nodes:
        darwin.log(f"Graph is empty for '{filename}'", LogType.ERROR)
        return None

    darwin.log(
        f"Using {sum(1 for n in graph.nodes if n.centroid)} node centroids from graph",
        LogType.INFO,
    )

    chunks = store.load_chunks(filename)
    if not chunks:
        darwin.log(f"No chunks for '{filename}'", LogType.ERROR)
        return None

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

    spacy_extractor = None
    try:
        from src.Models import ModelManager, ModelCategory
        loader = ModelManager.get_loader()
        spacy_extractor = loader.get(ModelCategory.TOKENIZER)
    except Exception:
        pass

    wiki = build_wiki_from_graph(
        graph=graph,
        chunks=chunks,
        spacy_extractor=spacy_extractor,
    )

    if mode == "summary":
        darwin.log("Initializing LLMManager for wiki summary...", LogType.INFO)
        from src.features.LLM.LLMManager import LLMManager
        LLMManager.initialize()

        if provider and LLMManager._config:
            try:
                LLMManager.switch_default_provider(provider)
                darwin.log(f"Using provider: {provider}", LogType.INFO)
            except Exception as e:
                darwin.log(str(e), LogType.ERROR)
                return None

        client = LLMManager.get_client()

        wiki = generate_wiki_summaries(
            wiki=wiki,
            chunks=chunks,
            client=client,
            chunks_per_concept=chunks_per_concept,
        )

    store.save_wiki(filename, wiki)
    darwin.log(
        f"Wiki saved — {len(wiki.pages)} pages, "
        f"manifest={'yes' if wiki.manifest else 'pending'}",
        LogType.SUCCESS,
    )
    return wiki