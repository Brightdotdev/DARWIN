from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from enum import Enum


class OverlapDirection(str, Enum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


@dataclass
class ChunkLink:
    url: str
    text: str = ""


@dataclass
class SearchAudit:
    """Audit trail attached to search results explaining ranking decisions."""
    raw_score: float = 0.0
    final_score: float = 0.0
    structural_type: Optional[str] = None
    penalty_factor: float = 1.0
    was_penalized: bool = False
    penalty_reason: Optional[str] = None
    merge_reason: Optional[str] = None
    gap_filled: bool = False
    concept_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_score": self.raw_score,
            "final_score": self.final_score,
            "structural_type": self.structural_type,
            "penalty_factor": self.penalty_factor,
            "was_penalized": self.was_penalized,
            "penalty_reason": self.penalty_reason,
            "merge_reason": self.merge_reason,
            "gap_filled": self.gap_filled,
            "concept_source": self.concept_source,
        }


@dataclass
class HybridSearchResult:
    text: str
    clean: str
    rrf_score: float = 0.0
    final_score: float = 0.0
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    source: str = ""
    page: int = 0
    page_end: int = 0
    heading: str = ""
    chunk_index: int = 0
    links: List[ChunkLink] = field(default_factory=list)
    matched_terms: Set[str] = field(default_factory=set)
    overlap_direction: OverlapDirection = OverlapDirection.NONE
    audit: Optional[SearchAudit] = None


@dataclass
class SemanticSearchResult:
    text: str
    clean: str
    score: float
    source: str
    page: int
    page_end: int = 0
    heading: str = ""
    chunk_index: int = 0
    links: List[ChunkLink] = field(default_factory=list)
    matched_terms: Set[str] = field(default_factory=set)
    overlap_direction: OverlapDirection = OverlapDirection.NONE
    audit: Optional[SearchAudit] = None


@dataclass
class KeywordSearchResult:
    text: str
    clean: str
    score: float
    source: str
    page: int
    page_end: int = 0
    heading: str = ""
    chunk_index: int = 0
    links: List[ChunkLink] = field(default_factory=list)
    matched_terms: Set[str] = field(default_factory=set)
    overlap_direction: OverlapDirection = OverlapDirection.NONE
    audit: Optional[SearchAudit] = None