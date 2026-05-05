"""
src/core/Knowledge/Graph/GraphUtils.py
--------------------------------------
Data models, constants, and helpers for knowledge graph extraction.

Updated for spaCy-based entity-relationship extraction.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import re


# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConceptNode:
    """A canonical entity in the knowledge graph."""
    id: str
    label: str                    # canonical name (e.g., "steve jobs")
    display_name: str             # original form for display (e.g., "Steve Jobs")
    entity_type: str              # spaCy NER type (e.g., "PERSON", "ORG")
    frequency: int              # number of chunks mentioning this entity
    chunk_indices: List[int]     # which chunks contain this entity
    first_mention_chunk: int     # earliest chunk index (structural ordering)
    mention_forms: List[str]     # all surface forms seen (e.g., ["Steve Jobs", "Jobs"])
    centroid: List[float] = field(default_factory=list)  # mean chunk embedding for search


@dataclass
class Relationship:
    """A typed, directed edge between two entities."""
    source: str                   # source node id
    target: str                   # target node id
    weight: float                 # confidence (0.0-1.0), based on chunk votes
    relation_type: str            # lemmatized verb (e.g., "acquire", "found")
    evidence_chunks: List[int]    # chunk indices where this relation was extracted
    surface_forms: List[str]      # original verb forms seen (e.g., ["acquired", "bought"])


@dataclass
class GraphData:
    """Complete knowledge graph for a document."""
    filename: str
    nodes: List[ConceptNode] = field(default_factory=list)
    edges: List[Relationship] = field(default_factory=list)
    chunk_count: int = 0
    total_entities_extracted: int = 0
    total_relations_extracted: int = 0

    # For wiki section generation
    cluster_map: Optional[Dict[str, List[int]]] = None

    # Processing metadata
    config_snapshot: Optional[Dict] = None
    processing_mode: str = "unknown"

    # Backward-compatible aliases for legacy consumers
    @property
    def total_concepts_extracted(self) -> int:
        """Alias for total_entities_extracted (legacy compatibility)."""
        return self.total_entities_extracted

    @property
    def total_relations(self) -> int:
        """Alias for total_relations_extracted."""
        return self.total_relations_extracted

    def to_dict(self) -> Dict:
        result = {
            "filename": self.filename,
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "display_name": n.display_name,
                    "type": n.entity_type,
                    "freq": n.frequency,
                    "first_mention_chunk": n.first_mention_chunk,
                    "chunks": n.chunk_indices,
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
            "total_concepts_extracted": self.total_concepts_extracted,  # legacy
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


# ═══════════════════════════════════════════════════════════════════════════
# CLI Args (legacy compatibility)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GraphExtractionArgs:
    """Legacy CLI args - maps to GraphExtractionConfig."""
    files: list[str] = field(default_factory=list)
    min_freq: int = DEFAULT_MIN_CONCEPT_FREQ
    max_concepts: int = DEFAULT_MAX_CONCEPTS
    similarity_threshold: Optional[float] = None
    elevation_mode: str = "none"
    visualize: bool = False
    inspect: bool = False
    json_output: bool = False
    rerun: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Domain-agnostic helpers
# ═══════════════════════════════════════════════════════════════════════════

def is_generic_word(token_text: str, spacy_token=None) -> bool:
    """
    Check if a word is generic/stopword.
    Uses spaCy's is_stop if available, else basic heuristics.
    """
    if spacy_token is not None:
        return spacy_token.is_stop or spacy_token.is_punct or spacy_token.is_space

    text = token_text.strip().lower()
    if len(text) <= 2:
        return True
    # Basic pronouns/determiners that might slip through
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
    """
    Normalize entity text for canonicalization.
    Lowercase, strip punctuation, collapse whitespace.
    """
    text = text.strip().lower()
    # Keep alphanumeric, spaces, and hyphens (for compound entities)
    text = re.sub(r"[^a-z0-9\s\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lemmatize_verb(verb_text: str, spacy_doc=None) -> str:
    """
    Lemmatize a verb to its base form.
    Uses spaCy if available, else simple heuristics.
    """
    if spacy_doc is not None and len(spacy_doc) > 0:
        for token in spacy_doc:
            if token.pos_ == "VERB" or token.dep_.startswith("ROOT"):
                return token.lemma_.lower()

    # Simple suffix stripping fallback
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
    """
    Canonicalize an SVO triple for deduplication.

    Returns (canonical_subject, canonical_verb, canonical_object)
    where subject and object are sorted alphabetically for undirected collapse
    (if dedup_strategy is "svo_canonical").
    """
    s = normalize_entity(subject)
    v = lemmatize_verb(verb)
    o = normalize_entity(obj)
    return (s, v, o)


# ═══════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════

def export_graph_json(graph: GraphData) -> str:
    import json
    return json.dumps(graph.to_dict(), indent=2)