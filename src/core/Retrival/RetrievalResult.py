from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Set

from core.Retrival.lib import (
    OverlapDirection,
    ChunkLink,
    SearchAudit,
)
from core.Retrival.retrieval_utils import normalize_links, ensure_overlap_direction


class ResultType(str, Enum):
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"


@dataclass
class RetrievalResult:
    result_type: ResultType
    score: float
    source: str

    text: Optional[str] = None
    clean: Optional[str] = None
    page: Optional[int] = None
    page_end: Optional[int] = None
    heading: Optional[str] = None
    chunk_index: Optional[int] = None
    overlap_direction: OverlapDirection = OverlapDirection.NONE
    links: List[ChunkLink] = field(default_factory=list)

    rrf_score: Optional[float] = None
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    matched_terms: Optional[Set[str]] = None

    rerank_score: Optional[float] = None
    rerank_original_score: Optional[float] = None

    structural_type: Optional[str] = None

    audit: Optional[SearchAudit] = None

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


def from_hybrid(r: Any) -> RetrievalResult:
    score = getattr(r, "final_score", 0.0)
    return RetrievalResult(
        result_type=ResultType.HYBRID,
        score=score,
        source=getattr(r, "source", ""),
        text=getattr(r, "text", ""),
        clean=getattr(r, "clean", None),
        page=getattr(r, "page", 0),
        page_end=getattr(r, "page_end", None),
        heading=getattr(r, "heading", None),
        chunk_index=getattr(r, "chunk_index", None),
        overlap_direction=ensure_overlap_direction(
            getattr(r, "overlap_direction", OverlapDirection.NONE)
        ),
        links=normalize_links(getattr(r, "links", [])),
        rrf_score=getattr(r, "rrf_score", None),
        semantic_score=getattr(r, "semantic_score", None),
        keyword_score=getattr(r, "keyword_score", None),
        matched_terms=getattr(r, "matched_terms", None),
        structural_type=getattr(r, "structural_type", None),
        audit=getattr(r, "audit", None),
    )


def from_semantic(r: Any) -> RetrievalResult:
    return RetrievalResult(
        result_type=ResultType.SEMANTIC,
        score=getattr(r, "score", 0.0),
        semantic_score=getattr(r, "score", None),
        source=getattr(r, "source", ""),
        text=getattr(r, "text", ""),
        clean=getattr(r, "clean", None),
        page=getattr(r, "page", 0),
        page_end=getattr(r, "page_end", None),
        heading=getattr(r, "heading", None),
        chunk_index=getattr(r, "chunk_index", None),
        overlap_direction=ensure_overlap_direction(
            getattr(r, "overlap_direction", OverlapDirection.NONE)
        ),
        links=normalize_links(getattr(r, "links", [])),
        matched_terms=getattr(r, "matched_terms", None),
        structural_type=getattr(r, "structural_type", None),
        audit=getattr(r, "audit", None),
    )


def from_keyword(r: Any) -> RetrievalResult:
    return RetrievalResult(
        result_type=ResultType.KEYWORD,
        score=getattr(r, "score", 0.0),
        keyword_score=getattr(r, "score", None),
        source=getattr(r, "source", ""),
        text=getattr(r, "text", ""),
        clean=getattr(r, "clean", None),
        page=getattr(r, "page", 0),
        page_end=getattr(r, "page_end", None),
        heading=getattr(r, "heading", None),
        chunk_index=getattr(r, "chunk_index", None),
        overlap_direction=ensure_overlap_direction(
            getattr(r, "overlap_direction", OverlapDirection.NONE)
        ),
        links=normalize_links(getattr(r, "links", [])),
        matched_terms=getattr(r, "matched_terms", None),
        structural_type=getattr(r, "structural_type", None),
        audit=getattr(r, "audit", None),
    )
