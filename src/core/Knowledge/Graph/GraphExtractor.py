"""
src/core/Knowledge/Graph/GraphExtractor.py
------------------------------------------
Unipartite concept graph extraction using cluster-first, label-second architecture.

Pipeline:
  1. Cluster chunks in embedding space (HDBSCAN → k-means fallback)
  2. Name clusters from noun phrases scoped to cluster chunks
  3. Score and prune clusters
  4. Build edges between cluster centroids (top-3 per node, floor 0.3)
  5. Render as unipartite concept graph + attach cluster_map for wiki consumer
"""

import re
import numpy as np
from collections import Counter
from typing import List, Dict, Optional, Tuple

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Chunking.Chunker import Chunk
from src.core.Knowledge.Graph.ConceptClustering import cluster_chunks
from src.core.Knowledge.Graph.GraphUtils import (
    GraphData,
    ConceptNode,
    Relationship,
    DEFAULT_MIN_CONCEPT_FREQ,
    DEFAULT_MAX_CONCEPTS,
    extract_noun_phrases_regex,
)
from src.core.Storage.DarwinStore import DarwinStore

_tokenizer_cache = None
_embedder_cache = None


# ═══════════════════════════════════════════════════════════════════════════
# Model loaders — ALL defined here, no missing methods
# ═══════════════════════════════════════════════════════════════════════════

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


def _get_embedder():
    """Get or load the embedding model once."""
    global _embedder_cache
    if _embedder_cache is not None:
        return _embedder_cache
    try:
        from src.core.Models import get_loader, ModelCategory
        loader = get_loader()
        _embedder_cache = loader.get(ModelCategory.EMBEDDING)
        return _embedder_cache
    except Exception:
        return None


def _embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    """Embed a list of short texts using the loaded embedding model."""
    if not texts:
        return None
    embedder = _get_embedder()
    if embedder is None:
        return None
    try:
        vecs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        if isinstance(vecs, list):
            vecs = np.array(vecs)
        return vecs.astype(np.float32)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Aggressive phrase filtering — domain-agnostic noise rejection
# ═══════════════════════════════════════════════════════════════════════════

_BAD_PREFIXES = {"'s", "'", "-", "—", "–", ":", ";", ",", "."}
_MSS_ABBREVS = {"mss", "lxx", "hb", "nt", "ot", "cf", "lit", "or", "vv", "v", "ch", "chs"}
_GENERIC_WORDS = {
    "process", "method", "system", "type", "form", "way", "thing", "part",
    "section", "chapter", "verse", "page", "note", "text", "passage",
    "people", "person", "man", "men", "woman", "women", "one", "two",
    "time", "day", "year", "place", "land", "word", "words",
}


def _is_valid_phrase(phrase: str) -> bool:
    """Reject noisy phrases: possessives, abbreviations, too-short, too-generic."""
    phrase = phrase.strip().lower()
    if not phrase or len(phrase) < 6 or len(phrase) > 60:
        return False

    words = phrase.split()
    if len(words) < 2:
        if words[0] in _GENERIC_WORDS or len(words[0]) < 6:
            return False
    if re.search(r'[=()\[\]{}]', phrase):
        return False
    # Reject possessive fragments and punctuation-led strings
    if phrase.startswith("'s ") or phrase.startswith("' "):
        return False
    if any(phrase.startswith(p + " ") for p in _BAD_PREFIXES):
        return False

    # Reject manuscript abbreviations
    if any(w in _MSS_ABBREVS for w in words):
        return False

    # Reject abbreviation-heavy phrases
    short_words = sum(1 for w in words if len(w) <= 2)
    if short_words > len(words) // 2:
        return False

    # Must have at least one content word
    content_words = [w for w in words if w not in _GENERIC_WORDS]
    if not content_words:
        return False

    # Reject URLs, markup
    if re.search(r'[@\[\]"{}]|http|www\.|\.com', phrase):
        return False

    if re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', phrase):   # dates
        return False
    if re.search(r'[=_]{1}', phrase):                     # code/config
        return False
    if sum(1 for w in words if w.isdigit()) > 0:          # numeric tokens
        return False


        
    return True


def _clean_phrase(phrase: str) -> Optional[str]:
    # Check possessive BEFORE stripping punctuation
    stripped = phrase.strip()
    if stripped.startswith("'s ") or stripped.lower().startswith("s ") and len(stripped) < 15:
        return None
    
    cleaned = re.sub(r"^[^a-zA-Z0-9]+", "", stripped)
    cleaned = re.sub(r"[^a-zA-Z0-9]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned.strip().lower())
    
    # Reject if first word is a single letter (stripped possessive)
    words = cleaned.split()
    if words and len(words[0]) == 1:
        return None
        
    return cleaned if _is_valid_phrase(cleaned) else None



# ═══════════════════════════════════════════════════════════════════════════
# Distinctive phrase extraction — what makes THIS cluster different?
# ═══════════════════════════════════════════════════════════════════════════

def _extract_distinctive_phrases(
    cluster_chunks: List[Chunk],
    global_phrase_counts: Counter,
) -> List[Tuple[str, float]]:
    """
    Extract phrases distinctive to this cluster using TF-IDF-style scoring.
    A phrase is distinctive if frequent here but rare globally.
    """
    local_counts = Counter()
    for chunk in cluster_chunks:
        words = re.findall(r"\b[a-z]{3,}\b", chunk.clean.lower())
        # Bigrams
        for i in range(len(words) - 1):
            bg = f"{words[i]} {words[i+1]}"
            if _is_valid_phrase(bg):
                local_counts[bg] += 1
        # Trigrams
        for i in range(len(words) - 2):
            tg = f"{words[i]} {words[i+1]} {words[i+2]}"
            if _is_valid_phrase(tg):
                local_counts[tg] += 1

    scored = []
    for phrase, local_freq in local_counts.most_common(30):
        global_freq = global_phrase_counts.get(phrase, 1)
        score = local_freq / (global_freq ** 0.5)
        scored.append((phrase, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _extract_phrases_from_cluster(
    cluster_chunks: List[Chunk],
    tokenizer,
) -> List[Tuple[str, int]]:
    """Extract noun phrases from cluster chunks. Returns (phrase, freq)."""
    phrase_counts = Counter()

    if tokenizer is not None:
        try:
            for chunk in cluster_chunks:
                phrases = tokenizer.extract_noun_chunks(chunk.clean)
                for p in phrases:
                    cleaned = _clean_phrase(p)
                    if cleaned:
                        phrase_counts[cleaned] += 1
        except Exception:
            tokenizer = None

    if tokenizer is None:
        for chunk in cluster_chunks:
            for p in extract_noun_phrases_regex(chunk.clean):
                cleaned = _clean_phrase(p)
                if cleaned:
                    phrase_counts[cleaned] += 1

    return phrase_counts.most_common()


def _most_frequent_content_word(cluster_chunks: List[Chunk]) -> str:
    """Fallback: most frequent non-generic word in cluster."""
    words = Counter()
    for chunk in cluster_chunks:
        for w in re.findall(r"\b[a-z]{4,}\b", chunk.clean.lower()):
            if w not in _GENERIC_WORDS and w not in _MSS_ABBREVS:
                words[w] += 1
    return words.most_common(1)[0][0] if words else "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Smart deduplication — never produce "david (1)", "david (2)"
# ═══════════════════════════════════════════════════════════════════════════

def _disambiguate_label(
    proposed_label: str,
    cluster_chunks: List[Chunk],
    existing_labels: Dict[str, Dict],
    global_phrase_counts: Counter,
    all_chunks: List[Chunk],
    scored_candidates: List[Tuple[str, float]] = None,  # NEW
) -> str:
    # Try runner-up candidates from original scoring first
    if scored_candidates:
        for phrase, score in scored_candidates[1:6]:  # skip [0], that's the collision
            cleaned = _clean_phrase(phrase)
            if cleaned and cleaned not in existing_labels:
                return cleaned

    # Then try distinctive phrases
    distinctive = _extract_distinctive_phrases(cluster_chunks, global_phrase_counts)
    for phrase, _ in distinctive:
        if phrase not in existing_labels and phrase != proposed_label:
            return phrase

    # Find words that appear much more in this cluster than the existing one
    this_words = Counter()
    for chunk in cluster_chunks:
        for w in re.findall(r"\b[a-z]{4,}\b", chunk.clean.lower()):
            if w not in _GENERIC_WORDS and w not in _MSS_ABBREVS:
                this_words[w] += 1

    other_words = Counter()
    if proposed_label in existing_labels:
        for idx in existing_labels[proposed_label]["chunk_indices"]:
            if idx < len(all_chunks):
                for w in re.findall(r"\b[a-z]{4,}\b", all_chunks[idx].clean.lower()):
                    if w not in _GENERIC_WORDS:
                        other_words[w] += 1

    # Find distinctive words for this cluster
    for word, count in this_words.most_common(20):
        other_count = other_words.get(word, 0)
        if count > other_count * 2 and word != proposed_label.split()[0]:
            return f"{proposed_label} ({word})"

    # Last resort: use second most common distinctive word
    second_words = [w for w, _ in this_words.most_common(5) if w not in proposed_label]
    if second_words:
        return f"{proposed_label} ({second_words[0]})"

    return f"{proposed_label} (variant)"


def _call_llm(prompt: str) -> Optional[str]:
    """Placeholder — replace with your agnostic LLM wrapper."""
    return None


def _llm_name_cluster(cluster_chunks: List[Chunk], candidates: List[str]) -> Optional[str]:
    """Ask LLM for a context-aware concept name."""
    sample_text = " ".join([c.clean[:300] for c in cluster_chunks[:5]])[:1500]

    prompt = f"""You are a knowledge graph concept extractor.

Given this sample text from a document cluster and candidate phrases, provide a single clean concept name (2-4 lowercase words).

Rules:
- 2-4 lowercase words
- Specific to this cluster's theme, not generic
- If about a biblical figure in a specific context, include the context: "davidic covenant" not just "david"

Sample text:
{sample_text}

Candidates:
{chr(10).join(f"- {c}" for c in candidates[:5])}

Return ONLY the concept name, nothing else."""

    response = _call_llm(prompt)
    if response:
        cleaned = response.strip().lower().split("\n")[0].strip('-"\'')
        if 4 <= len(cleaned) <= 60 and " " in cleaned:
            return cleaned
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — Name clusters with semantic distinctiveness
# ═══════════════════════════════════════════════════════════════════════════


def _name_clusters(
    cluster_map: Dict[int, List[int]],
    chunks: List[Chunk],
    chunk_vectors: np.ndarray,
    elevation_mode: str = "none",
) -> Dict[str, Dict]:
    tokenizer = _get_tokenizer()
    named: Dict[str, Dict] = {}

    # ── Extract all phrases once — no double pass ──
    cluster_phrases: Dict[int, List[Tuple[str, int]]] = {}
    global_phrase_counts: Counter = Counter()

    for cluster_id, chunk_indices in cluster_map.items():
        cluster_chunk_objs = [chunks[i] for i in chunk_indices if i < len(chunks)]
        if not cluster_chunk_objs:
            cluster_phrases[cluster_id] = []
            continue
        phrases = _extract_phrases_from_cluster(cluster_chunk_objs, tokenizer)
        cluster_phrases[cluster_id] = phrases
        for phrase, count in phrases:
            global_phrase_counts[phrase] += count

    # ── Batch embed all candidate phrases in ONE call ──
    # Collect top-5 candidates per cluster for embedding
    cluster_top_candidates: Dict[int, List[str]] = {}
    all_phrases_to_embed: List[str] = []
    phrase_to_idx: Dict[str, int] = {}

    for cluster_id, candidates in cluster_phrases.items():
        top = [p for p, _ in candidates[:10]]
        cluster_top_candidates[cluster_id] = top
        for p in top:
            if p not in phrase_to_idx:
                phrase_to_idx[p] = len(all_phrases_to_embed)
                all_phrases_to_embed.append(p)

    # Single encode call for all phrases across all clusters
    all_vecs = _embed_texts(all_phrases_to_embed) if all_phrases_to_embed else None

    # ── Name each cluster using pre-computed embeddings ──
    for cluster_id, chunk_indices in cluster_map.items():
        cluster_chunk_objs = [chunks[i] for i in chunk_indices if i < len(chunks)]
        if not cluster_chunk_objs:
            continue

        candidates = cluster_phrases[cluster_id]
        top_phrases = cluster_top_candidates.get(cluster_id, [])
        best_label = None

        if top_phrases:
            centroid = chunk_vectors[chunk_indices].mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

            if all_vecs is not None:
                scored = []
                for phrase, freq in candidates[:10]:
                    idx = phrase_to_idx.get(phrase)
                    if idx is not None and idx < len(all_vecs):
                        vec = all_vecs[idx]
                        vec = vec / (np.linalg.norm(vec) + 1e-9)
                        sim = float(np.dot(vec, centroid))
                        scored.append((phrase, float(freq * max(0.0, sim))))
                    else:
                        scored.append((phrase, float(freq)))
                scored.sort(key=lambda x: x[1], reverse=True)
            else:
                scored = [(p, float(f)) for p, f in candidates[:10]]

            if elevation_mode == "llm" and scored:
                llm_name = _llm_name_cluster(
                    cluster_chunk_objs, [p for p, _ in scored[:5]]
                )
                if llm_name:
                    best_label = llm_name

            if not best_label and scored:
                best_label = scored[0][0]

        if not best_label:
            distinctive = _extract_distinctive_phrases(
                cluster_chunk_objs, global_phrase_counts
            )
            if distinctive:
                best_label = distinctive[0][0]

        if not best_label:
            best_label = _most_frequent_content_word(cluster_chunk_objs)

        best_label = best_label.lower().strip()

        # Dedup only logs on actual collision
        if best_label in named:
            original = best_label
            best_label = _disambiguate_label(
                best_label, cluster_chunk_objs, named, global_phrase_counts, chunks,
                scored_candidates=scored  
            )
            darwin.log(
                f"Cluster {cluster_id}: '{original}' → '{best_label}' (disambiguated)",
                LogType.INFO,
            )

        named[best_label] = {
            "chunk_indices": chunk_indices,
            "centroid": chunk_vectors[chunk_indices].mean(axis=0),
        }

    return named

def _is_likely_references_cluster(chunk_indices: List[int], chunks: List[Chunk]) -> bool:
    """
    Detect bibliography/references clusters.
    These tend to be dense with short chunks — citation entries, author lists.
    """
    if not chunk_indices:
        return False
    sample = [chunks[i] for i in chunk_indices[:20] if i < len(chunks)]
    if not sample:
        return False
    short_chunks = sum(1 for c in sample if len(c.clean) < 200)
    return (short_chunks / len(sample)) > 0.5

# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 — Score and prune clusters
# ═══════════════════════════════════════════════════════════════════════════

def _prune_clusters(
    named_clusters: Dict[str, Dict],
    chunk_vectors: np.ndarray,
    chunks: List[Chunk],
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
) -> Dict[str, Dict]:
    """
    Score clusters by size and centroid proximity to document centroid.
    Penalises likely reference/bibliography clusters.
    Drop bottom 30%, then apply hard cap (max 20 for readability).
    """
    if not named_clusters:
        return {}

    doc_centroid = chunk_vectors.mean(axis=0)
    doc_centroid = doc_centroid / (np.linalg.norm(doc_centroid) + 1e-9)

    sizes = [len(v["chunk_indices"]) for v in named_clusters.values()]
    max_size = max(sizes) if sizes else 1

    scored = []
    for label, data in named_clusters.items():
        size = len(data["chunk_indices"])
        centroid = data["centroid"]
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        sim_to_doc = float(np.dot(centroid, doc_centroid))

        score = (size / max_size) * 0.4 + sim_to_doc * 0.6

        # Penalise reference/bibliography clusters heavily
        if _is_likely_references_cluster(data["chunk_indices"], chunks):
            darwin.log(f"References cluster detected: '{label}' — penalising", LogType.INFO)
            score *= 0.3

        scored.append((label, score, data))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Drop bottom 30%
    keep_idx = max(1, int(len(scored) * 0.7))
    scored = scored[:keep_idx]

    # Hard cap at 20
    hard_cap = min(max_concepts, 20)
    if len(scored) > hard_cap:
        scored = scored[:hard_cap]
        darwin.log(f"Pruned to top {hard_cap} clusters", LogType.INFO)

    return {label: data for label, _, data in scored}
# ═══════════════════════════════════════════════════════════════════════════
# Stage 4 — Build edges
# ═══════════════════════════════════════════════════════════════════════════

def _build_graph_from_clusters(
    pruned_clusters: Dict[str, Dict],
    chunk_vectors: np.ndarray,
    similarity_threshold: Optional[float] = None,
) -> GraphData:
    """Build GraphData from pruned clusters. Edges = top-3 centroid similarities."""
    labels = list(pruned_clusters.keys())
    n = len(labels)
    if n == 0:
        return GraphData(filename="", chunk_count=len(chunk_vectors))

    # Normalize centroids
    centroids = np.stack([pruned_clusters[l]["centroid"] for l in labels])
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
    centroids = centroids / norms

    sim_matrix = (centroids @ centroids.T).astype(float)
    floor = similarity_threshold if similarity_threshold is not None else 0.3

    concept_to_id = {c: f"c{i}" for i, c in enumerate(labels)}

    nodes = [
        ConceptNode(
            id=concept_to_id[label],
            label=label,
            frequency=len(pruned_clusters[label]["chunk_indices"]),
            chunk_indices=pruned_clusters[label]["chunk_indices"],
        )
        for label in labels
    ]

    edges = []
    for i in range(n):
        sims = [
            (j, float(sim_matrix[i, j]))
            for j in range(n)
            if i != j and sim_matrix[i, j] >= floor
        ]
        sims.sort(key=lambda x: x[1], reverse=True)
        for j, sim in sims[:3]:
            edges.append(
                Relationship(
                    source=concept_to_id[labels[i]],
                    target=concept_to_id[labels[j]],
                    weight=round(sim, 4),
                    relation_type="context_similarity",
                )
            )

    # Deduplicate bidirectional edges
    deduped = {}
    for e in edges:
        key = tuple(sorted((e.source, e.target)))
        if key not in deduped or e.weight > deduped[key].weight:
            deduped[key] = e

    final_edges = list(deduped.values())
    darwin.log(
        f"Graph: {n} nodes, {len(final_edges)} edges ({len(final_edges) / max(n, 1):.1f}/node)"
    )

    return GraphData(
        filename="",
        nodes=nodes,
        edges=final_edges,
        chunk_count=len(chunk_vectors),
        total_concepts_extracted=n,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def build_concept_graph(
    concepts: Dict[str, List[int]],
    chunks: List[Chunk],
    chunk_vectors: np.ndarray,
    min_freq: int = DEFAULT_MIN_CONCEPT_FREQ,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    similarity_threshold: Optional[float] = None,
) -> GraphData:
    """
    Build a unipartite concept graph using the new cluster-first pipeline.

    Args:
        concepts:             Ignored (kept for backward-compatible signature).
        chunks:               Source Chunk objects.
        chunk_vectors:        Pre-computed embeddings (n_chunks, dim).
        min_freq:             Minimum cluster size filter.
        max_concepts:         Hard cap on nodes (clamped to 20).
        similarity_threshold: Cosine floor for edges (default 0.3).
    """
    n_chunks = len(chunks)
    if chunk_vectors is None or len(chunk_vectors) == 0:
        darwin.log("No chunk vectors provided", LogType.WARNING)
        return GraphData(filename="", chunk_count=n_chunks)

    darwin.log(f"Clustering {n_chunks} chunks...", LogType.INFO)
    labels, cluster_map = cluster_chunks(chunk_vectors)

    if not cluster_map:
        darwin.log("No clusters found", LogType.WARNING)
        return GraphData(filename="", chunk_count=n_chunks)

    if min_freq > 1:
        cluster_map = {k: v for k, v in cluster_map.items() if len(v) >= min_freq}
        darwin.log(f"After min_freq filter: {len(cluster_map)} clusters", LogType.INFO)

    darwin.log("Naming clusters...", LogType.INFO)
    named = _name_clusters(cluster_map, chunks, chunk_vectors, elevation_mode="none")

    darwin.log("Scoring and pruning clusters...", LogType.INFO)
    pruned = _prune_clusters(named, chunk_vectors, max_concepts)

    graph = _build_graph_from_clusters(pruned, chunk_vectors, similarity_threshold)
    graph.chunk_count = n_chunks
    graph.total_concepts_extracted = len(named)
    return graph


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
                paths.append(
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "weight": edge.weight,
                    }
                )

    return {
        "highlighted_nodes": list(highlighted_ids),
        "highlighted_labels": highlighted_labels,
        "paths": paths,
    }


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

    New pipeline:
      1. Cluster chunks in embedding space
      2. Name clusters from scoped noun phrases
      3. Score / prune clusters
      4. Build edges between centroids
    """
    darwin.log(
        f"Extracting concept graph: {filename} ({len(chunks)} chunks)", LogType.INFO
    )
    darwin.log(f"Elevation: {elevation_mode} | Max concepts: {max_concepts}")

    if chunk_vectors is None:
        darwin.log("No chunk vectors — cannot build graph", LogType.WARNING)
        return GraphData(
            filename=filename,
            chunk_count=len(chunks),
            total_concepts_extracted=0,
        )

    n_chunks = len(chunks)

    # Stage 1
    darwin.log(f"[1/4] Clustering {n_chunks} chunks...", LogType.INFO)
    labels, cluster_map = cluster_chunks(chunk_vectors)
    if not cluster_map:
        darwin.log("No clusters found", LogType.WARNING)
        return GraphData(filename=filename, chunk_count=n_chunks)

    if min_freq > 1:
        cluster_map = {k: v for k, v in cluster_map.items() if len(v) >= min_freq}

    darwin.log(f"[1/4] Done — {len(cluster_map)} clusters", LogType.SUCCESS)


    darwin.log(f"[2/4] Naming {len(cluster_map)} clusters (elevation={elevation_mode})...", LogType.INFO)
    named = _name_clusters(cluster_map, chunks, chunk_vectors, elevation_mode=elevation_mode)
    darwin.log(f"[2/4] Done — {len(named)} named clusters", LogType.SUCCESS)

    # Add this summary so you can see what labels came out:
    if named:
        preview = ", ".join(list(named.keys())[:8])
        darwin.log(f"Labels: {preview}{'...' if len(named) > 8 else ''}", LogType.INFO)


    # Stage 3
    darwin.log(f"[3/4] Scoring and pruning clusters...", LogType.INFO)
    pruned = _prune_clusters(named, chunk_vectors, chunks, max_concepts)
    darwin.log(f"[3/4] Done — {len(pruned)} clusters after pruning", LogType.SUCCESS)

    # Stage 4
    darwin.log(f"[4/4] Building graph...", LogType.INFO)
    graph = _build_graph_from_clusters(pruned, chunk_vectors, similarity_threshold)
    graph.filename = filename
    graph.chunk_count = n_chunks
    graph.total_concepts_extracted = len(named)

    # Attach cluster_map for wiki consumer (drop-in addition, no dataclass change)
    graph.cluster_map = {node.label: node.chunk_indices for node in graph.nodes}

    darwin.log(
        f"[4/4] Done — {len(graph.nodes)} nodes, {len(graph.edges)} edges", LogType.SUCCESS
    )

    if graph.nodes:
        darwin.log(f"Top: {', '.join(n.label for n in graph.nodes[:5])}")

    return graph



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
    if not rerun and store.is_fresh(filename) and store.has_graph(filename):
        try:
            darwin.log(f"Skipping graph: fresh exists for {filename}", LogType.INFO)
            return store.load_graph(filename)
        except Exception as e:
            darwin.log(f"Cached graph corrupted ({e}) — re-extracting...", LogType.WARNING)

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
    darwin.log(f"Graph saved for {filename}", LogType.SUCCESS)

    # Save centroids alongside graph for wiki pipeline
    try:
        centroids = []
        for node in graph.nodes:
            valid = [idx for idx in node.chunk_indices if idx < len(vectors)]
            centroid = vectors[valid].mean(axis=0).tolist() if valid else []
            centroids.append({"concept": node.label, "centroid": centroid})
        store.save_graph_centroids(filename, centroids)
    except Exception as e:
        darwin.log(f"Could not save centroids: {e}", LogType.WARNING)

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