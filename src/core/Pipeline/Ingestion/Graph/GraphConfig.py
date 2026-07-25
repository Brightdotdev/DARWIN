from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GraphConfig:
    """Configuration for entity graph extraction."""

    min_mentions: int = 2
    min_salience: float = 0.1
    co_occurrence_window: int = 1
    include_noun_chunks: bool = True
    include_dependency_edges: bool = True
    wiki_tier: int = 2
    max_wiki_entities: int = 100
    visualization: bool = True
    entity_types: Optional[set] = field(default_factory=lambda: {
        "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT",
        "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
    })
    label_filter: Optional[set] = None
    custom_blacklist: Optional[set] = None
    custom_whitelist: Optional[set] = None
