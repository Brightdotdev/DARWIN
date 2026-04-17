"""
src/core/Graph/GraphUtils.py
----------------------------
Utilities for graph extraction: data models, constants, concept indexing,
and export formats for visualization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter
import re


# ============================================================================
# Data models
# ============================================================================

@dataclass
class ConceptNode:
    """A concept extracted from the document."""
    id: str                     # Normalized identifier e.g. "c0"
    label: str                  # Display name e.g. "holy spirit"
    frequency: int              # Number of chunks it appears in
    chunk_indices: List[int]    # Which chunks mention this concept


@dataclass
class Relationship:
    """A relationship between two concepts."""
    source: str                 # Source concept ID
    target: str                 # Target concept ID
    weight: float               # Cosine similarity (0–1) or co-occurrence count
    relation_type: str          # "semantic_similarity" | "co_occurrence"


@dataclass
class GraphData:
    """Complete knowledge graph for a document."""
    filename: str
    nodes: List[ConceptNode] = field(default_factory=list)
    edges: List[Relationship] = field(default_factory=list)
    chunk_count: int = 0
    total_concepts_extracted: int = 0

    def to_dict(self) -> Dict:
        return {
            "filename": self.filename,
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "freq": n.frequency,
                    "chunks": n.chunk_indices,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "weight": e.weight,
                    "type": e.relation_type,
                }
                for e in self.edges
            ],
            "chunk_count": self.chunk_count,
            "total_concepts_extracted": self.total_concepts_extracted,
        }


# ============================================================================
# Constants
# ============================================================================

DEFAULT_MIN_CONCEPT_FREQ    = 3      # Minimum chunk appearances
DEFAULT_MAX_CONCEPTS        = 100    # Max concepts per document
DEFAULT_MIN_RELATION_FREQ   = 2      # Min co-occurrences (fallback only)
DEFAULT_SIMILARITY_THRESHOLD = 0.45  # Cosine similarity cutoff for edges


# ============================================================================
# Args
# ============================================================================

@dataclass
class GraphExtractionArgs:
    target: str
    file: Optional[str] = None
    project: Optional[str] = None
    min_freq: int = DEFAULT_MIN_CONCEPT_FREQ
    max_concepts: int = DEFAULT_MAX_CONCEPTS
    min_rel_freq: int = DEFAULT_MIN_RELATION_FREQ
    similarity_threshold: Optional[float] = None  # None = auto
    extraction_mode: str = "yake"   # "yake" | "regex" | "spacy"
    visualize: bool = False
    inspect: bool = False
    json_output: bool = False
    rerun: bool = False
    use_ai: bool = False


# ============================================================================
# Stopwords
# ============================================================================

CONCEPT_STOPWORDS: Set[str] = {
    # articles / determiners
    "the", "this", "that", "these", "those", "a", "an",
    # pronouns
    "he", "she", "it", "we", "they", "them", "him", "her", "his",
    "its", "our", "their", "you", "your", "yours", "who", "whom",
    "whose", "there", "here",
    # conjunctions / prepositions
    "and", "but", "or", "nor", "for", "yet", "so", "both",
    "either", "neither", "not", "also", "even", "just",
    "about", "above", "after", "against", "among", "around",
    "before", "behind", "below", "beneath", "beside", "between",
    "beyond", "despite", "down", "during", "except", "from",
    "into", "near", "off", "onto", "out", "outside", "over",
    "past", "since", "than", "through", "throughout", "till",
    "toward", "under", "until", "unto", "upon", "with", "within",
    "without",
    # common verbs
    "was", "were", "are", "is", "has", "have", "had", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "been", "being", "said", "says", "like", "come", "came",
    "went", "going", "made", "make", "take", "took", "give",
    "gave", "know", "knew", "say", "see", "saw", "get", "got",
    # adverbs / misc
    "very", "then", "when", "where", "how", "what", "why",
    "more", "most", "much", "many", "some", "any", "all",
    "each", "every", "few", "less", "least", "only", "other",
    "same", "such", "too", "well", "back", "still", "own",
    "however", "therefore", "thus", "hence", "whereas",
    # noise
    "ing", "tion", "ness", "able", "ible", "ment", "ful",
    # original set
    "again", "already", "also", "another", "because", "been",
    "both", "during", "either", "enough", "first", "following",
    "found", "further", "including", "itself", "likely",
    "might", "moreover", "moreover", "never", "none", "nor",
    "nothing", "often", "once", "ones", "otherwise",
    "particular", "perhaps", "possible", "present", "rather",
    "second", "several", "since", "something", "sometimes",
    "third", "though", "together", "unless", "using", "various",
    "want", "whether", "whilst", "yourself", "yourselves",
}


# ============================================================================
# Regex noun phrase extraction (fallback)
# ============================================================================

# Require 4+ char words; common nouns must be multi-word to reduce noise
NOUN_PHRASE_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,})*)\b|"   # Proper noun phrases
    r"\b(?:[a-z]{4,}(?:\s+[a-z]{4,}){1,3})\b",           # Multi-word common only
    re.MULTILINE,
)


def extract_noun_phrases_regex(text: str) -> List[str]:
    """Fallback noun phrase extraction using regex."""
    phrases = []
    for match in NOUN_PHRASE_PATTERN.finditer(text):
        phrase = match.group().strip()
        if (
            len(phrase) > 3
            and phrase.lower() not in CONCEPT_STOPWORDS
            and any(c.isalpha() for c in phrase)
        ):
            phrases.append(phrase)
    return phrases


# ============================================================================
# Concept index (human-readable phonebook)
# ============================================================================

def build_concept_index(
    concepts: Dict[str, List[int]],
    chunks: List,
    max_per_concept: int = 10,
) -> str:
    """Build a markdown concept index for debugging/inspection."""
    lines = ["# Document Concept Index\n"]

    sorted_concepts = sorted(concepts.items(), key=lambda x: len(x[1]), reverse=True)

    for concept, chunk_indices in sorted_concepts:
        lines.append(f"## {concept}")
        lines.append(f"*Appears in {len(chunk_indices)} chunks*\n")

        for idx in chunk_indices[:max_per_concept]:
            chunk = chunks[idx]
            page_info    = f"Page {chunk.page}"
            heading_info = f" — {chunk.heading}" if chunk.heading else ""
            lines.append(f"- {page_info}{heading_info}")

            snippet = chunk.clean[:150].replace('\n', ' ')
            if len(chunk.clean) > 150:
                snippet += "..."
            lines.append(f"  > {snippet}\n")

        if len(chunk_indices) > max_per_concept:
            lines.append(f"*...and {len(chunk_indices) - max_per_concept} more*\n")

        lines.append("")

    return "\n".join(lines)


# ============================================================================
# Export formats
# ============================================================================

def export_graph_json(graph_data) -> str:
    """Export graph as JSON string."""
    import json
    return json.dumps({
        "nodes": [
            {
                "id": node.id,
                "label": node.label,
                "frequency": node.frequency,
                "chunk_count": len(node.chunk_indices),
            }
            for node in graph_data.nodes
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "weight": edge.weight,
                "relation_type": edge.relation_type,
            }
            for edge in graph_data.edges
        ],
    }, indent=2)


def export_graph_html(graph_data) -> Optional[str]:
    """
    Build an interactive graph visualization using pyvis.
    Edge thickness and opacity reflect semantic similarity weight.
    Returns HTML string or None if pyvis not available.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        return None

    net = Network(
        height="800px",
        width="100%",
        bgcolor="#0f0f0f",
        font_color="#e0e0e0",
    )

    # Valid JSON — double quotes, no JS wrapper
    net.set_options("""{
        "nodes": {
            "shape": "dot",
            "scaling": {
                "min": 10,
                "max": 50,
                "label": {
                    "enabled": true,
                    "min": 12,
                    "max": 28
                }
            },
            "font": {
                "size": 14,
                "face": "monospace"
            }
        },
        "edges": {
            "smooth": {
                "type": "continuous"
            },
            "arrows": {
                "to": {
                    "enabled": false
                }
            },
            "color": {
                "inherit": false
            }
        },
        "physics": {
            "stabilization": {
                "iterations": 150
            },
            "barnesHut": {
                "gravitationalConstant": -8000,
                "springLength": 160,
                "springConstant": 0.04
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100
        }
    }""")

    # Add nodes — size reflects frequency
    for node in graph_data.nodes:
        size  = min(50, max(10, node.frequency * 3))
        color = _frequency_color(node.frequency, graph_data.nodes)
        net.add_node(
            node.id,
            label=node.label,
            title=f"<b>{node.label}</b><br>Frequency: {node.frequency}<br>Chunks: {len(node.chunk_indices)}",
            size=size,
            color=color,
        )

    # Add edges — thickness and opacity reflect weight
    for edge in graph_data.edges:
        w = float(edge.weight)
        # Map weight 0.45–1.0 to opacity 0.2–1.0
        opacity = round(0.2 + (w - 0.45) / 0.55 * 0.8, 2) if w <= 1.0 else 1.0
        width   = round(1 + w * 4, 2)
        net.add_edge(
            edge.source,
            edge.target,
            value=w,
            width=width,
            color={"color": f"rgba(120,180,255,{opacity})", "highlight": "#ffffff"},
            title=f"Similarity: {w:.3f}" if edge.relation_type == "semantic_similarity" else f"Co-occurrence: {int(w)}",
        )

    return net.generate_html()


def _frequency_color(freq: int, all_nodes: list) -> str:
    """Map node frequency to a color on a warm gradient."""
    if not all_nodes:
        return "#4a9eff"
    max_freq = max(n.frequency for n in all_nodes)
    ratio    = freq / max_freq if max_freq > 0 else 0
    # low freq → cool blue, high freq → warm orange
    r = int(74  + ratio * (255 - 74))
    g = int(158 + ratio * (140 - 158))
    b = int(255 + ratio * (50  - 255))
    return f"#{r:02x}{g:02x}{b:02x}"