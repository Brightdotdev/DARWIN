"""
src/core/Knowledge/Graph/GraphUtils.py
--------------------------------------
Data models, constants, and helpers for knowledge graph extraction.
Updated for spaCy-based entity-relationship extraction + community detection.
Optimized for new DarwinStore: accepts dict-based chunk rows.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
import re
import math
from collections import Counter, defaultdict

import numpy as np

# Community detection — best-in-class Louvain algorithm
try:
    import networkx as nx
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConceptNode:
    """A canonical entity in the knowledge graph."""
    id: str
    label: str
    display_name: str
    entity_type: str
    frequency: int
    chunk_indices: List[int]
    first_mention_chunk: int
    mention_forms: List[str]
    centroid: List[float] = field(default_factory=list)


@dataclass
class Relationship:
    """A typed, directed edge between two entities."""
    source: str
    target: str
    weight: float
    relation_type: str
    evidence_chunks: List[int]
    surface_forms: List[str]


@dataclass
class CommunityCluster:
    """
    A detected community of related entities that forms one wiki theme.
    """
    community_id: int
    node_ids: List[str]
    node_labels: List[str]
    chunk_indices: List[int]
    centroid: List[float] = field(default_factory=list)
    best_phrase: str = ""
    keywords: List[str] = field(default_factory=list)
    related_communities: List[str] = field(default_factory=list)


@dataclass
class GraphData:
    """Complete knowledge graph for a document."""
    filename: str
    nodes: List[ConceptNode] = field(default_factory=list)
    edges: List[Relationship] = field(default_factory=list)
    chunk_count: int = 0
    total_entities_extracted: int = 0
    total_relations_extracted: int = 0
    cluster_map: Optional[Dict[str, List[int]]] = None
    config_snapshot: Optional[Dict] = None
    processing_mode: str = "unknown"

    @property
    def total_concepts_extracted(self) -> int:
        return self.total_entities_extracted

    @property
    def total_relations(self) -> int:
        return self.total_relations_extracted

    def to_dict(self) -> Dict:
        result = {
            "filename": self.filename,
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "display_name": n.display_name,
                    "entity_type": n.entity_type,
                    "frequency": n.frequency,
                    "first_mention_chunk": n.first_mention_chunk,
                    "chunk_indices": n.chunk_indices,
                    "mention_forms": n.mention_forms,
                    "centroid": n.centroid,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "weight": e.weight,
                    "relation_type": e.relation_type,
                    "evidence_chunks": e.evidence_chunks,
                    "surface_forms": e.surface_forms,
                }
                for e in self.edges
            ],
            "chunk_count": self.chunk_count,
            "total_entities_extracted": self.total_entities_extracted,
            "total_relations_extracted": self.total_relations_extracted,
            "total_concepts_extracted": self.total_concepts_extracted,
            "processing_mode": self.processing_mode,
        }
        if self.cluster_map:
            result["cluster_map"] = self.cluster_map
        if self.config_snapshot:
            result["config"] = self.config_snapshot
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_MIN_CONCEPT_FREQ = 2
DEFAULT_MAX_CONCEPTS = 75


@dataclass
class GraphExtractionArgs:
    """Legacy CLI args - maps to GraphExtractionConfig."""
    files: list[str] = field(default_factory=list)
    min_freq: int = DEFAULT_MIN_CONCEPT_FREQ
    max_concepts: int = DEFAULT_MAX_CONCEPTS
    similarity_threshold: Optional[float] = None
    visualize: bool = False
    inspect: bool = False
    json_output: bool = False
    rerun: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Domain-agnostic helpers
# ═══════════════════════════════════════════════════════════════════════════

def is_generic_word(token_text: str, spacy_token=None) -> bool:
    if spacy_token is not None:
        return spacy_token.is_stop or spacy_token.is_punct or spacy_token.is_space
    text = token_text.strip().lower()
    if len(text) <= 2:
        return True
    basic_stops = {
        "the", "a", "an", "this", "that", "these", "those",
        "he", "she", "it", "they", "we", "you", "i", "me",
        "his", "her", "its", "their", "our", "your", "my",
        "and", "or", "but", "nor", "for", "yet", "so",
        "of", "in", "on", "at", "to", "from", "by", "with",
        "as", "is", "was", "are", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall",
    }
    return text in basic_stops


def normalize_entity(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lemmatize_verb(verb_text: str, spacy_doc=None) -> str:
    if spacy_doc is not None and len(spacy_doc) > 0:
        for token in spacy_doc:
            if token.pos_ == "VERB" or token.dep_.startswith("ROOT"):
                return token.lemma_.lower()
    verb = verb_text.strip().lower()
    if verb.endswith("ing"):
        return verb[:-3]
    if verb.endswith("ed"):
        return verb[:-2]
    if verb.endswith("es"):
        return verb[:-2]
    if verb.endswith("s"):
        return verb[:-1]
    return verb


def canonicalize_svo(subject: str, verb: str, obj: str) -> tuple:
    s = normalize_entity(subject)
    v = lemmatize_verb(verb)
    o = normalize_entity(obj)
    return (s, v, o)


# ═══════════════════════════════════════════════════════════════════════════
# Community Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_communities_louvain(graph: GraphData, resolution: float = 1.0) -> List[CommunityCluster]:
    """
    Run Louvain community detection on entity graph.
    Falls back to greedy modularity if python-louvain not installed.
    """
    if not graph.nodes:
        return []

    if LOUVAIN_AVAILABLE:
        G = nx.Graph()
        node_map = {n.id: n for n in graph.nodes}

        for n in graph.nodes:
            G.add_node(n.id, label=n.label, chunks=n.chunk_indices)

        for e in graph.edges:
            G.add_edge(e.source, e.target, weight=e.weight, relation=e.relation_type)

        partition = community_louvain.best_partition(G, resolution=resolution, weight='weight')

        # Group by community
        comm_groups: Dict[int, List[str]] = defaultdict(list)
        for node_id, comm_id in partition.items():
            comm_groups[comm_id].append(node_id)

    else:
        # Fallback: simple connected-component + greedy merge by edge density
        comm_groups = _fallback_community_detection(graph)

    clusters = []
    for comm_id, node_ids in comm_groups.items():
        labels = [node_map[nid].label for nid in node_ids if nid in node_map]
        all_chunks = sorted(set(
            idx for nid in node_ids for idx in node_map[nid].chunk_indices
        ))
        # Compute centroid from node centroids
        centroids = [node_map[nid].centroid for nid in node_ids if node_map[nid].centroid]
        if centroids:
            mean_centroid = np.mean(centroids, axis=0).tolist()
        else:
            mean_centroid = []

        clusters.append(CommunityCluster(
            community_id=comm_id,
            node_ids=node_ids,
            node_labels=labels,
            chunk_indices=all_chunks,
            centroid=mean_centroid,
        ))

    # Build cross-community links
    comm_id_map = {}
    for c in clusters:
        for nid in c.node_ids:
            comm_id_map[nid] = c.community_id

    for c in clusters:
        related = set()
        for e in graph.edges:
            if e.source in c.node_ids:
                target_comm = comm_id_map.get(e.target)
                if target_comm is not None and target_comm != c.community_id:
                    related.add(target_comm)
            elif e.target in c.node_ids:
                source_comm = comm_id_map.get(e.source)
                if source_comm is not None and source_comm != c.community_id:
                    related.add(source_comm)
        c.related_communities = [f"community_{r}" for r in related]

    # Sort by chunk count (largest communities first)
    clusters.sort(key=lambda c: len(c.chunk_indices), reverse=True)
    return clusters


def _fallback_community_detection(graph: GraphData) -> Dict[int, List[str]]:
    """Greedy fallback when Louvain is unavailable."""
    node_map = {n.id: n for n in graph.nodes}
    adjacency: Dict[str, Set[str]] = {n.id: set() for n in graph.nodes}

    for e in graph.edges:
        adjacency[e.source].add(e.target)
        adjacency[e.target].add(e.source)

    visited = set()
    communities = {}
    comm_counter = 0

    for start in node_map:
        if start in visited:
            continue
        # BFS to find connected component
        queue = [start]
        component = set()
        while queue:
            curr = queue.pop(0)
            if curr in component:
                continue
            component.add(curr)
            for neighbor in adjacency.get(curr, []):
                if neighbor not in component:
                    queue.append(neighbor)
        for nid in component:
            visited.add(nid)
        communities[comm_counter] = list(component)
        comm_counter += 1

    return communities


# ═══════════════════════════════════════════════════════════════════════════
# TextRank-style Keyphrase Extraction (document-agnostic)
# ═══════════════════════════════════════════════════════════════════════════

def extract_narrative_keyphrase(
    text: str,
    spacy_extractor,
    candidate_labels: List[str],
    max_phrase_len: int = 4,
) -> str:
    """
    Extract the best narrative keyphrase from community text.
    Uses spaCy noun chunks + TextRank-like co-occurrence scoring.
    """
    if not spacy_extractor or not hasattr(spacy_extractor, 'nlp'):
        # Fallback: pick longest candidate label that appears in text
        text_lower = text.lower()
        best = ""
        for label in candidate_labels:
            if label.lower() in text_lower and len(label) > len(best):
                best = label
        return best if best else (candidate_labels[0] if candidate_labels else "unknown theme")

    doc = spacy_extractor.nlp(text[:8000])  # cap for speed

    # Collect candidate phrases: noun chunks + multi-word entities
    candidates = []
    seen = set()

    for nc in doc.noun_chunks:
        tokens = [t for t in nc if not t.is_stop and not t.is_punct and len(t.text) > 2]
        phrase = " ".join(t.lemma_.lower() for t in tokens)
        if len(phrase) < 3:
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        candidates.append({
            "text": phrase,
            "tokens": tokens,
            "pos": [t.pos_ for t in tokens],
        })

    # Also add multi-word entities
    for ent in doc.ents:
        if ent.label_ in {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}:
            continue
        phrase = " ".join(t.lemma_.lower() for t in ent if not t.is_stop and len(t.text) > 2)
        if len(phrase) < 3 or phrase in seen:
            continue
        seen.add(phrase)
        candidates.append({
            "text": phrase,
            "tokens": list(ent),
            "pos": [t.pos_ for t in ent],
        })

    if not candidates:
        return candidate_labels[0] if candidate_labels else "unknown theme"

    # Build word graph for TextRank
    word_counts: Counter = Counter()
    word_cooccurrence: Dict[str, Set[str]] = defaultdict(set)

    for cand in candidates:
        words = [t.lemma_.lower() for t in cand["tokens"] if not t.is_stop and len(t.text) > 2]
        for w in words:
            word_counts[w] += 1
        for i, w1 in enumerate(words):
            for w2 in words[i+1:]:
                if w1 != w2:
                    word_cooccurrence[w1].add(w2)
                    word_cooccurrence[w2].add(w1)

    # PageRank on words
    damping = 0.85
    max_iter = 30
    tol = 1e-4

    vocab = list(word_cooccurrence.keys())
    if not vocab:
        return candidates[0]["text"] if candidates else "unknown theme"

    scores = {w: 1.0 for w in vocab}

    for _ in range(max_iter):
        new_scores = {}
        for w in vocab:
            s = (1 - damping)
            neighbors = word_cooccurrence.get(w, set())
            if neighbors:
                s += damping * sum(scores[n] / max(len(word_cooccurrence.get(n, set())), 1) for n in neighbors)
            new_scores[w] = s
        # Normalize
        max_score = max(new_scores.values()) if new_scores else 1
        for w in new_scores:
            new_scores[w] /= max_score
        # Check convergence
        delta = sum(abs(scores[w] - new_scores[w]) for w in vocab)
        scores = new_scores
        if delta < tol:
            break

    # Score phrases by sum of word scores + specificity bonuses
    def score_phrase(cand):
        words = [t.lemma_.lower() for t in cand["tokens"] if not t.is_stop and len(t.text) > 2]
        if not words:
            return 0.0

        base = sum(scores.get(w, 0) for w in words) / len(words)

        # Length bonus: prefer 2-4 word phrases
        n_words = len(words)
        if n_words == 2:
            base *= 1.2
        elif n_words == 3:
            base *= 1.3
        elif n_words >= 4:
            base *= 1.1

        # Proper noun / entity bonus
        if any(pos in ("PROPN", "NOUN") for pos in cand["pos"]):
            base *= 1.15

        # Candidate label overlap bonus (tie to graph entities)
        cand_text = cand["text"]
        label_overlap = sum(1 for lab in candidate_labels if lab.lower() in cand_text)
        base *= (1 + 0.1 * label_overlap)

        return base

    candidates.sort(key=score_phrase, reverse=True)

    best = candidates[0]["text"] if candidates else "unknown theme"

    # Clean up: title case, remove excessive whitespace
    best = " ".join(best.split())
    best = best.title() if len(best) < 40 else best

    return best


def extract_community_keywords(
    chunk_indices: List[int],
    chunk_rows: List[Dict[str, Any]],
    spacy_extractor,
    max_keywords: int = 8,
) -> List[str]:
    """
    Extract top keywords for a community from its chunk text.
    
    chunk_rows: list of dicts from store.query_records() — NOT Chunk objects.
    Each row must have at least a "clean" or "text" key.
    """
    from collections import Counter as _Counter

    # Build text from chunk rows using dict access
    texts = []
    for row in chunk_rows:
        idx = row.get("index", -1)
        if idx in chunk_indices:
            text = row.get("clean", "") or row.get("text", "")
            if text:
                texts.append(text)

    text = " ".join(texts[:20])  # cap at 20 chunks for speed

    if spacy_extractor and hasattr(spacy_extractor, 'nlp'):
        doc = spacy_extractor.nlp(text[:5000])
        noun_chunks = []
        for nc in doc.noun_chunks:
            phrase = " ".join(t.lemma_.lower() for t in nc if not t.is_stop and len(t.text) > 2)
            if len(phrase) > 3:
                noun_chunks.append(phrase)
        if noun_chunks:
            counts = _Counter(noun_chunks)
            return [k for k, _ in counts.most_common(max_keywords)]

    # Fallback
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    stops = {
        "that", "with", "have", "from", "they", "will", "would", "there",
        "their", "what", "said", "each", "which", "could", "other", "after",
        "first", "never", "these", "think", "where", "being", "every", "great",
        "might", "shall", "still", "those", "while", "about", "before",
        "right", "should", "through", "book", "chapter", "part", "section",
    }
    words = [w for w in words if w not in stops]
    counts = _Counter(words)
    return [k for k, _ in counts.most_common(max_keywords)]


# ═══════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════

def export_graph_json(graph: GraphData) -> str:
    import json
    return json.dumps(graph.to_dict(), indent=2)