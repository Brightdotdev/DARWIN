"""
src/core/Knowledge/Wiki/WikiGenerator.py
----------------------------------------
Generates DocumentWiki from GraphData.

Optimizations:
  1. O(n) precomputed structural mask (was O(n²) via repeat-scan).
  2. Batched spaCy via nlp.pipe() with dynamic batch size.
  3. Cross-node spaCy cache so shared chunks are processed once.
  4. O(1) label→id lookup + precomputed degrees for sorting.
  5. Compiled regexes & frozenset for faster set ops.
  6. Ordered dedup with dict.fromkeys() instead of list(set()).
"""

import re
from typing import List, Dict, Optional, Counter as CounterType
from collections import Counter, defaultdict

from src.features.Ingestion.Chunking.lib import Chunk
from src.features.Knowledge.Graph.GraphUtils import GraphData, ConceptNode, Relationship
from src.features.Knowledge.Wiki.WikiUtils import WikiPage, DocumentWiki
from src.features.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType


# ═══════════════════════════════════════════════════════════════════════════
# Compiled patterns & constants
# ═══════════════════════════════════════════════════════════════════════════

_WORD_RE = re.compile(r'\b[a-z]+\b')
_BIGRAM_WORD_RE = re.compile(r'\b[a-z]{3,}\b')
_TOC_DIGIT_RE = re.compile(r'\d')

STRUCTURAL_SIGNALS = frozenset({
    "cover", "title", "copyright", "table of contents", "contents",
    "acknowledgments", "preface", "foreword", "introduction", "about",
    "appendix", "index", "glossary", "bibliography", "references",
    "page", "chapter", "book", "part", "section", "volume",
    "ebook", "edition", "published", "publisher", "isbn",
    "all rights reserved", "printed in", "first edition",
})

_KEYWORD_STOPS = frozenset({
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
})


# ═══════════════════════════════════════════════════════════════════════════
# O(n) structural chunk detection
# ═══════════════════════════════════════════════════════════════════════════

def _precompute_structural_mask(chunks: List[Chunk]) -> set:
    """One-pass structural detection. Replaces the old O(n²) repeat-scan."""
    if not chunks:
        return set()

    total = len(chunks)
    five_pct = total * 0.05
    mask: set = set()

    # 1. Precompute fingerprints for repeat detection
    fingerprints: CounterType[str] = Counter(c.clean.lower()[:200] for c in chunks)

    # 2. Single-pass evaluation
    for chunk in chunks:
        text = chunk.clean.lower()
        words = set(_WORD_RE.findall(text))

        # Very short chunks
        if len(chunk.clean) < 200:
            mask.add(chunk.index)
            continue

        # High structural signal ratio
        if words:
            hits = len(words & STRUCTURAL_SIGNALS)
            if hits / len(words) > 0.3:
                mask.add(chunk.index)
                continue

        # TOC pattern: many short numbered lines
        lines = [l.strip() for l in chunk.clean.split('\n') if l.strip()]
        if len(lines) > 3:
            short_numbered = sum(
                1 for l in lines if len(l) < 60 and _TOC_DIGIT_RE.search(l)
            )
            if short_numbered / len(lines) > 0.5:
                mask.add(chunk.index)
                continue

        # Repetitive text (headers/footers)
        if fingerprints[text[:200]] > 2:
            mask.add(chunk.index)
            continue

        # First 5% of doc with very low word count
        if chunk.index <= five_pct and len(words) < 20:
            mask.add(chunk.index)
            continue

    return mask


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
# Schema guard
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
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_edge_map(graph: GraphData) -> Dict[str, List[str]]:
    """Build {node_id: [neighbor_labels]} from graph edges — O(edges)."""
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

    result: Dict[str, List[str]] = {}
    for nid, label in node_labels.items():
        related: List[str] = []
        seen: set = set()
        for target in semantic_neighbors.get(nid, ()):
            tlabel = node_labels.get(target)
            if tlabel and tlabel not in seen:
                seen.add(tlabel)
                related.append(tlabel)
        for target in cooc_neighbors.get(nid, ()):
            tlabel = node_labels.get(target)
            if tlabel and tlabel not in seen:
                seen.add(tlabel)
                related.append(tlabel)
        result[nid] = related

    return result


def _extract_keywords(
    chunk_indices: List[int],
    chunks: List[Chunk],
    spacy_extractor=None,
    max_keywords: int = 5,
    batch_size: int = 32,
    spacy_cache: Optional[Dict[int, CounterType[str]]] = None,
) -> List[str]:
    """
    Extract top keywords with batched spaCy and cross-node chunk caching.
    If multiple nodes reference the same chunk, spaCy processes it once.
    """
    counts: CounterType[str] = Counter()

    # Deduplicate while preserving order
    unique_indices = list(dict.fromkeys(idx for idx in chunk_indices[:10] if idx < len(chunks)))
    if not unique_indices:
        return []

    # ── Batched spaCy with cache ──────────────────────────────────────────
    if spacy_extractor and hasattr(spacy_extractor, 'nlp'):
        try:
            nlp = spacy_extractor.nlp
            cache = spacy_cache if spacy_cache is not None else {}

            # Only process chunks we haven't seen yet
            uncached = [idx for idx in unique_indices if idx not in cache]
            if uncached:
                texts = [chunks[idx].clean[:3000] for idx in uncached]
                for idx, doc in zip(uncached, nlp.pipe(texts, batch_size=batch_size)):
                    idx_counts: CounterType[str] = Counter()
                    for nc in doc.noun_chunks:
                        phrase = " ".join(
                            t.text for t in nc
                            if not t.is_stop and len(t.text) > 2
                        )
                        if len(phrase) > 3:
                            idx_counts[phrase.lower().strip()] += 1
                    cache[idx] = idx_counts

            # Merge cached results for this request
            for idx in unique_indices:
                counts.update(cache.get(idx, Counter()))

        except Exception:
            pass  # Fall through to bigram fallback

    # ── Fallback bigram extraction ────────────────────────────────────────
    if not counts:
        for idx in unique_indices:
            text = chunks[idx].clean.lower()
            words = _BIGRAM_WORD_RE.findall(text)
            words = [w for w in words if w not in _KEYWORD_STOPS]

            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                if w1 not in _KEYWORD_STOPS and w2 not in _KEYWORD_STOPS:
                    counts[f"{w1} {w2}"] += 1

    return [k for k, _ in counts.most_common(max_keywords)]


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

    # ── Precompute O(1) lookups & masks ─────────────────────────────────
    edge_map = _build_edge_map(graph)
    structural_mask = _precompute_structural_mask(chunks)

    # Precompute node degrees for sorting
    node_degrees: CounterType[str] = Counter()
    for e in graph.edges:
        node_degrees[e.source] += 1
        node_degrees[e.target] += 1

    # Fast label → node_id lookup (was O(n) per page)
    label_to_id = {n.label: n.id for n in graph.nodes}

    # Dynamic batch size: scales with doc size, bounded for memory safety
    n_chunks = len(chunks)
    spacy_batch_size = min(128, max(16, n_chunks // 50))

    # Shared spaCy cache across all nodes
    spacy_cache: Dict[int, CounterType[str]] = {}

    pages: List[WikiPage] = []
    for node in graph.nodes:
        if _is_structural_node(node):
            continue

        centroid = getattr(node, 'centroid', [])
        related = edge_map.get(node.id, [])

        # Filter structural chunks using precomputed mask
        valid_chunks = [idx for idx in node.chunk_indices if idx not in structural_mask]

        # Fallback: if all filtered out, keep the longest non-structural or original
        if not valid_chunks and node.chunk_indices:
            candidates = [chunks[idx] for idx in node.chunk_indices if idx < len(chunks)]
            if candidates:
                non_struct = [c for c in candidates if c.index not in structural_mask]
                if non_struct:
                    longest = max(non_struct, key=lambda c: len(c.clean))
                else:
                    longest = max(candidates, key=lambda c: len(c.clean))
                valid_chunks = [longest.index]

        keywords = _extract_keywords(
            valid_chunks,
            chunks,
            spacy_extractor,
            batch_size=spacy_batch_size,
            spacy_cache=spacy_cache,
        )

        # Fallback: if spaCy keywords empty, use concept words + related
        if not keywords:
            concept_words = [
                w for w in node.label.lower().split()
                if len(w) > 2 and w not in _KEYWORD_STOPS
            ]
            keywords = concept_words[:5]

        # Ordered dedup without set() shuffle
        keywords = list(dict.fromkeys(keywords + [r.lower() for r in related[:3]]))[:10]

        pages.append(WikiPage(
            concept=node.label,
            chunk_indices=valid_chunks,
            related=related,
            centroid=centroid,
            frequency=node.frequency,
            keywords=keywords,
        ))

    # Sort by connectivity-weighted frequency — O(pages) now
    for page in pages:
        node_id = label_to_id.get(page.concept)
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
# Pipeline entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_wiki_pipeline(
    filename: str,
    store: DarwinStore,
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

    store.save_wiki(filename, wiki)
    darwin.log(
        f"Wiki saved — {len(wiki.pages)} pages",
        LogType.SUCCESS,
    )
    return wiki