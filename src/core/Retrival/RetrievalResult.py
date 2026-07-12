from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Set

from src.core.Retrival.lib import (
    OverlapDirection,
    ChunkLink,
    SearchAudit,
)


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


def _normalize_links(links_raw: Any) -> List[ChunkLink]:
    out = []
    for l in links_raw or []:
        if isinstance(l, dict):
            out.append(ChunkLink(url=l.get("url", ""), text=l.get("text", "")))
        elif isinstance(l, ChunkLink):
            out.append(l)
        else:
            out.append(ChunkLink(url=str(l), text=""))
    return out


def from_hybrid(r: Any) -> RetrievalResult:
    score = getattr(r, "final_score", None)
    if score is None:
        score = getattr(r, "rrf_score", 0.0)
    return RetrievalResult(
        result_type=ResultType.HYBRID,
        score=score,
        source=r.source,
        text=r.text,
        clean=getattr(r, "clean", None),
        page=r.page,
        page_end=getattr(r, "page_end", None),
        heading=getattr(r, "heading", None),
        chunk_index=getattr(r, "chunk_index", None),
        overlap_direction=_ensure_overlap_direction(
            getattr(r, "overlap_direction", OverlapDirection.NONE)
        ),
        links=_normalize_links(getattr(r, "links", [])),
        rrf_score=getattr(r, "rrf_score", None),
        semantic_score=getattr(r, "semantic_score", None),
        keyword_score=getattr(r, "keyword_score", None),
        matched_terms=getattr(r, "matched_terms", None),
        rerank_score=getattr(r, "rerank_score", None),
        rerank_original_score=getattr(r, "original_score", None),
        audit=getattr(r, "audit", None),
    )


def from_semantic(r: Any) -> RetrievalResult:
    return RetrievalResult(
        result_type=ResultType.SEMANTIC,
        score=getattr(r, "score", 0.0),
        source=r.source,
        text=r.text,
        clean=getattr(r, "clean", None),
        page=r.page,
        page_end=getattr(r, "page_end", None),
        heading=getattr(r, "heading", None),
        chunk_index=getattr(r, "chunk_index", None),
        overlap_direction=_ensure_overlap_direction(
            getattr(r, "overlap_direction", OverlapDirection.NONE)
        ),
        links=_normalize_links(getattr(r, "links", [])),
        matched_terms=getattr(r, "matched_terms", None),
        rerank_score=getattr(r, "rerank_score", None),
        rerank_original_score=getattr(r, "original_score", None),
        audit=getattr(r, "audit", None),
    )


def _ensure_overlap_direction(val: Any) -> OverlapDirection:
    if isinstance(val, OverlapDirection):
        return val
    if isinstance(val, str):
        try:
            return OverlapDirection(val)
        except ValueError:
            pass
    return OverlapDirection.NONE


def from_keyword(r: Any) -> RetrievalResult:
    return RetrievalResult(
        result_type=ResultType.KEYWORD,
        score=getattr(r, "score", 0.0),
        source=r.source,
        text=r.text,
        clean=getattr(r, "clean", None),
        page=r.page,
        page_end=getattr(r, "page_end", None),
        heading=getattr(r, "heading", None),
        chunk_index=getattr(r, "chunk_index", None),
        overlap_direction=_ensure_overlap_direction(
            getattr(r, "overlap_direction", OverlapDirection.NONE)
        ),
        links=_normalize_links(getattr(r, "links", [])),
        matched_terms=getattr(r, "matched_terms", None),
        rerank_score=getattr(r, "rerank_score", None),
        rerank_original_score=getattr(r, "original_score", None),
        audit=getattr(r, "audit", None),
    )
