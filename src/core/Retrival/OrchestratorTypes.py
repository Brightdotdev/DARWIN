"""
OrchestratorTypes.py -- Data structures for the Retrieval Orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class MergeStrategy(str, Enum):
    NONE = "none"
    ADJACENT = "adjacent"
    AGGRESSIVE = "aggressive"


class FallbackStrategy(str, Enum):
    EXPAND = "expand"
    RELAX = "relax"
    BOTH = "both"


@dataclass
class OrchestratorConfig:
    query: str = ""
    method: str = "hybrid"
    files: Optional[List[str]] = None
    top_k: int = 5

    max_chars: int = 6000
    max_passages: int = 5
    min_passage_score: float = 0.3

    merge_strategy: MergeStrategy = MergeStrategy.ADJACENT
    max_merge_gap: int = 2
    fill_gaps: bool = False

    max_per_source: int = 2
    prefer_diverse_sources: bool = True
    require_diverse_sources: bool = False
    prefer_sources: Optional[List[str]] = None
    exclude_sources: Optional[List[str]] = None

    filter_structural: bool = True
    structural_penalty_factor: float = 0.5

    fallback_strategy: FallbackStrategy = FallbackStrategy.BOTH
    fallback_top_k: int = 20
    fallback_max_chars: int = 12000
    fallback_min_score: float = 0.1
    fallback_min_passages: int = 2

    include_metadata: bool = True
    include_audit: bool = False
    include_raw_results: bool = False
    return_empty_on_fallback: bool = False

    rerank: bool = False
    rerank_weight: float = 0.3
    rerank_top_k: int = 100

    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "method": self.method,
            "files": self.files,
            "top_k": self.top_k,
            "max_chars": self.max_chars,
            "max_passages": self.max_passages,
            "min_passage_score": self.min_passage_score,
            "merge_strategy": self.merge_strategy.value,
            "max_merge_gap": self.max_merge_gap,
            "fill_gaps": self.fill_gaps,
            "max_per_source": self.max_per_source,
            "prefer_diverse_sources": self.prefer_diverse_sources,
            "require_diverse_sources": self.require_diverse_sources,
            "prefer_sources": self.prefer_sources,
            "exclude_sources": self.exclude_sources,
            "filter_structural": self.filter_structural,
            "structural_penalty_factor": self.structural_penalty_factor,
            "fallback_strategy": self.fallback_strategy.value,
            "fallback_top_k": self.fallback_top_k,
            "fallback_max_chars": self.fallback_max_chars,
            "fallback_min_score": self.fallback_min_score,
            "fallback_min_passages": self.fallback_min_passages,
            "include_metadata": self.include_metadata,
            "include_audit": self.include_audit,
            "include_raw_results": self.include_raw_results,
            "return_empty_on_fallback": self.return_empty_on_fallback,
            "rerank": self.rerank,
            "rerank_weight": self.rerank_weight,
            "rerank_top_k": self.rerank_top_k,
            "log_level": self.log_level,
        }


@dataclass
class PassageMetadata:
    source: str
    page_start: int
    page_end: int
    chunk_indices: List[int]
    chunk_count: int
    is_merged: bool
    gap_filled: bool
    structural_type: Optional[str] = None
    penalty_factor: float = 1.0
    was_penalized: bool = False

    avg_score: float = 0.0
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None

    matched_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_indices": self.chunk_indices,
            "chunk_count": self.chunk_count,
            "is_merged": self.is_merged,
            "gap_filled": self.gap_filled,
            "structural_type": self.structural_type,
            "penalty_factor": self.penalty_factor,
            "was_penalized": self.was_penalized,
            "avg_score": self.avg_score,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "rrf_score": self.rrf_score,
            "rerank_score": self.rerank_score,
            "matched_terms": self.matched_terms,
        }


@dataclass
class ResolvedPassage:
    text: str
    metadata: PassageMetadata
    audit: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict(),
            "audit": self.audit,
        }


@dataclass
class RetrievalContext:
    query: str
    method: str
    passages: List[ResolvedPassage]
    total_chars: int
    total_passages: int
    elapsed_ms: int = 0
    candidates_found: int = 0
    penalized_count: int = 0
    max_score: float = 0.0
    avg_score: float = 0.0
    sources_used: List[str] = field(default_factory=list)
    source_diversity: float = 0.0
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    raw_results: List[Any] = field(default_factory=list)
    audit_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "method": self.method,
            "passages": [p.to_dict() for p in self.passages],
            "total_chars": self.total_chars,
            "total_passages": self.total_passages,
            "elapsed_ms": self.elapsed_ms,
            "candidates_found": self.candidates_found,
            "penalized_count": self.penalized_count,
            "max_score": self.max_score,
            "avg_score": self.avg_score,
            "sources_used": self.sources_used,
            "source_diversity": self.source_diversity,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "audit_summary": self.audit_summary,
            "raw_results": [
                r.__dict__ if hasattr(r, "__dataclass_fields__") else str(r)
                for r in self.raw_results
            ],
        }

    def is_empty(self) -> bool:
        return len(self.passages) == 0
