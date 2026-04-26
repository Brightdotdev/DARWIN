"""
src/core/Knowledge/Graph/GraphUtils.py
--------------------------------------
Data models, constants, and CLI args for concept graph extraction.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re


# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConceptNode:
    id: str
    label: str
    frequency: int
    chunk_indices: List[int]


@dataclass
class Relationship:
    source: str
    target: str
    weight: float
    relation_type: str  # "context_similarity"


@dataclass
class GraphData:
    filename: str
    nodes: List[ConceptNode] = field(default_factory=list)
    edges: List[Relationship] = field(default_factory=list)
    chunk_count: int = 0
    total_concepts_extracted: int = 0

    def to_dict(self) -> Dict:
        return {
            "filename": self.filename,
            "nodes": [
                {"id": n.id, "label": n.label, "freq": n.frequency, "chunks": n.chunk_indices}
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "weight": e.weight, "type": e.relation_type}
                for e in self.edges
            ],
            "chunk_count": self.chunk_count,
            "total_concepts_extracted": self.total_concepts_extracted,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_MIN_CONCEPT_FREQ = 10
DEFAULT_MAX_CONCEPTS     = 10


# ═══════════════════════════════════════════════════════════════════════════
# CLI Args
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GraphExtractionArgs:
    target: str
    file: Optional[str] = None
    project: Optional[str] = None
    min_freq: int = DEFAULT_MIN_CONCEPT_FREQ
    max_concepts: int = DEFAULT_MAX_CONCEPTS
    similarity_threshold: Optional[float] = None
    extraction_mode: str = "yake"
    elevation_mode: str = "spacy"
    visualize: bool = False
    inspect: bool = False
    json_output: bool = False
    rerun: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Stopwords
# ═══════════════════════════════════════════════════════════════════════════

CONCEPT_STOPWORDS = {
    "the", "this", "that", "these", "those", "a", "an",
    "he", "she", "it", "we", "they", "them", "him", "her", "his",
    "its", "our", "their", "you", "your", "who", "whom", "whose",
    "and", "but", "or", "nor", "for", "yet", "so", "both",
    "either", "neither", "not", "also", "even", "just",
    "about", "above", "after", "against", "among", "around",
    "before", "behind", "below", "beneath", "beside", "between",
    "beyond", "despite", "down", "during", "except", "from",
    "into", "near", "off", "onto", "out", "outside", "over",
    "past", "since", "than", "through", "until", "upon", "with",
    "was", "were", "are", "is", "has", "have", "had", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "been", "being", "said", "says", "like", "come", "came",
    "went", "going", "made", "make", "take", "took", "give",
    "very", "then", "when", "where", "how", "what", "why",
    "more", "most", "much", "many", "some", "any", "all",
    "each", "every", "few", "less", "only", "other",
    "same", "such", "too", "well", "back", "still", "own",
    "however", "therefore", "thus", "hence",
    "again", "already", "another", "because", "during", "enough",
    "first", "following", "found", "further", "including",
    "itself", "likely", "never", "none", "nothing", "often",
    "once", "ones", "otherwise", "particular", "perhaps",
    "possible", "present", "rather", "second", "several",
    "something", "sometimes", "third", "though", "together",
    "unless", "using", "various", "whether", "whilst",
}


# ═══════════════════════════════════════════════════════════════════════════
# Regex fallback
# ═══════════════════════════════════════════════════════════════════════════

_NOUN_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,})*)\b|"
    r"\b(?:[a-z]{4,}(?:\s+[a-z]{4,}){1,3})\b",
    re.MULTILINE,
)


def extract_noun_phrases_regex(text: str) -> List[str]:
    return [m.group().strip() for m in _NOUN_PHRASE_RE.finditer(text)
            if len(m.group().strip()) > 3]


# ═══════════════════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════════════════

def export_graph_json(graph: GraphData) -> str:
    import json
    return json.dumps(graph.to_dict(), indent=2)