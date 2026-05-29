"""
src/features/Retrival/RetrievalResult.py
-----------------------------------------
Universal result type for all retrieval paths.

Every search method (hybrid, semantic, keyword) returns List[RetrievalResult].
Fields that don't apply to a given path are None.

ResultType tells you which path produced the result:

    HYBRID   → text, clean, source, page, page_end, heading, chunk_index,
               score (= final_score), rrf_score, semantic_score,
               keyword_score, matched_terms, overlap_direction, links

    SEMANTIC → text, clean, source, page, page_end, heading, chunk_index,
               score, overlap_direction, links
               (rrf_score, keyword_score, matched_terms all None)

    KEYWORD  → text, clean, source, page, page_end, heading, chunk_index,
               score, matched_terms, overlap_direction, links
               (rrf_score, semantic_score all None)

Reranking fields (any path when rerank=True):
    rerank_score, rerank_original_score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Dict


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class ResultType(str, Enum):
    HYBRID   = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD  = "keyword"


class OverlapDirection(str, Enum):
    NONE  = "none"
    LEFT  = "left"
    RIGHT = "right"
    BOTH  = "both"

# ═══════════════════════════════════════════════════════════════════════════
# Supporting types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkLink:
    url:  str
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


# ═══════════════════════════════════════════════════════════════════════════
# Universal result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """
    Single result from any retrieval path.

    Always populated:
        result_type  — which path produced this result
        score        — primary confidence score
        source       — filename this result came from
        text         — passage text (for LLM context)

    HYBRID only:
        rrf_score, semantic_score, keyword_score

    KEYWORD / HYBRID:
        matched_terms

    When rerank=True (any path):
        rerank_score, rerank_original_score
    """

    # ── universal ──────────────────────────────────────────────────────────
    result_type:            ResultType
    score:                  float
    source:                 str

    # ── chunk fields ───────────────────────────────────────────────────────
    text:                   Optional[str]           = None
    clean:                  Optional[str]           = None
    page:                   Optional[int]           = None
    page_end:               Optional[int]           = None
    heading:                Optional[str]           = None
    chunk_index:            Optional[int]           = None
    overlap_direction:      OverlapDirection        = OverlapDirection.NONE
    links:                  List[ChunkLink]         = field(default_factory=list)

    # ── score breakdown ────────────────────────────────────────────────────
    rrf_score:              Optional[float]         = None   # HYBRID only
    semantic_score:         Optional[float]         = None   # HYBRID only
    keyword_score:          Optional[float]         = None   # HYBRID only
    matched_terms:          Optional[List[str]]     = None   # KEYWORD / HYBRID

    # ── reranking ──────────────────────────────────────────────────────────
    rerank_score:           Optional[float]         = None
    rerank_original_score:  Optional[float]         = None

    # ── audit ──────────────────────────────────────────────────────────────
    audit:                  Optional[SearchAudit]   = None

    # ── helpers ────────────────────────────────────────────────────────────

    @property
    def is_hybrid(self) -> bool:
        return self.result_type == ResultType.HYBRID

    @property
    def is_semantic(self) -> bool:
        return self.result_type == ResultType.SEMANTIC

    @property
    def is_keyword(self) -> bool:
        return self.result_type == ResultType.KEYWORD

    @property
    def was_reranked(self) -> bool:
        return self.rerank_score is not None

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(type={self.result_type.value.upper()}, "
            f"score={self.score:.3f}, source='{self.source}', page={self.page})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Converters
# ═══════════════════════════════════════════════════════════════════════════

def from_hybrid(r: Any) -> RetrievalResult:
    """Convert a HybridSearchResult → RetrievalResult."""
    score = getattr(r, "final_score", getattr(r, "rrf_score", 0.0))
    return RetrievalResult(
        result_type           = ResultType.HYBRID,
        score                 = score,
        source                = r.source,
        text                  = r.text,
        clean                 = getattr(r, "clean", None),
        page                  = r.page,
        page_end              = getattr(r, "page_end", None),
        heading               = getattr(r, "heading", None),
        chunk_index           = getattr(r, "chunk_index", None),
        overlap_direction     = getattr(r, "overlap_direction", OverlapDirection.NONE),
        links                 = getattr(r, "links", []),
        rrf_score             = getattr(r, "rrf_score", None),
        semantic_score        = getattr(r, "semantic_score", None),
        keyword_score         = getattr(r, "keyword_score", None),
        matched_terms         = getattr(r, "matched_terms", None),
        rerank_score          = getattr(r, "rerank_score", None),
        rerank_original_score = getattr(r, "original_score", None),
        audit                 = getattr(r, "audit", None),
    )


def from_semantic(r: Any) -> RetrievalResult:
    """Convert a SemanticSearchResult → RetrievalResult."""
    return RetrievalResult(
        result_type           = ResultType.SEMANTIC,
        score                 = getattr(r, "score", 0.0),
        source                = r.source,
        text                  = r.text,
        clean                 = getattr(r, "clean", None),
        page                  = r.page,
        page_end              = getattr(r, "page_end", None),
        heading               = getattr(r, "heading", None),
        chunk_index           = getattr(r, "chunk_index", None),
        overlap_direction     = getattr(r, "overlap_direction", OverlapDirection.NONE),
        links                 = getattr(r, "links", []),
        rerank_score          = getattr(r, "rerank_score", None),
        rerank_original_score = getattr(r, "original_score", None),
        audit                 = getattr(r, "audit", None),
    )


def from_keyword(r: Any) -> RetrievalResult:
    """Convert a KeywordSearchResult → RetrievalResult."""
    return RetrievalResult(
        result_type           = ResultType.KEYWORD,
        score                 = getattr(r, "score", 0.0),
        source                = r.source,
        text                  = r.text,
        clean                 = getattr(r, "clean", None),
        page                  = r.page,
        page_end              = getattr(r, "page_end", None),
        heading               = getattr(r, "heading", None),
        chunk_index           = getattr(r, "chunk_index", None),
        overlap_direction     = getattr(r, "overlap_direction", OverlapDirection.NONE),
        links                 = getattr(r, "links", []),
        matched_terms         = getattr(r, "matched_terms", None),
        rerank_score          = getattr(r, "rerank_score", None),
        rerank_original_score = getattr(r, "original_score", None),
        audit                 = getattr(r, "audit", None),
    )