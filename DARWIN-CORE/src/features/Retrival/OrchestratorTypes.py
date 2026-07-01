"""
OrchestratorTypes.py — Data structures for the Retrieval Orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from enum import Enum


class MergeStrategy(str, Enum):
    """Strategy for merging adjacent chunks."""
    NONE = "none"           # Never merge, treat each result separately
    ADJACENT = "adjacent"   # Only merge if indices are consecutive
    AGGRESSIVE = "aggressive"  # Merge if within gap (legacy, not recommended)


class FallbackStrategy(str, Enum):
    """Strategy when no results are found."""
    EXPAND = "expand"       # Increase top_k and max_chars
    RELAX = "relax"         # Lower min_score threshold
    BOTH = "both"           # Both expand and relax


@dataclass
class OrchestratorConfig:
    """
    Configuration for the RetrievalOrchestrator.
    
    All parameters are tunable for different use cases.
    """
    
    # ── Core Retrieval ──────────────────────────────────────────────
    query: str = ""
    method: str = "hybrid"  # hybrid | semantic | keyword
    files: Optional[List[str]] = None
    top_k: int = 5
    
    # ── Token Budgeting ──────────────────────────────────────────────
    max_chars: int = 6000
    max_passages: int = 5
    min_passage_score: float = 0.3
    
    # ── Merging Strategy ──────────────────────────────────────────────
    merge_strategy: MergeStrategy = MergeStrategy.ADJACENT
    max_merge_gap: int = 2
    fill_gaps: bool = False
    
    # ── Source Diversity & Filtering ──────────────────────────────
    max_per_source: int = 2
    prefer_diverse_sources: bool = True      # if True, require ≥2 sources
    require_diverse_sources: bool = False    # if True, fallback if <2 sources (overrides above)
    prefer_sources: Optional[List[str]] = None   # if set, only these sources are used
    exclude_sources: Optional[List[str]] = None  # if set, these sources are excluded
    
    # ── Penalties & Filters ──────────────────────────────────────────
    filter_structural: bool = True
    structural_penalty_factor: float = 0.5   # only applied to non‑body structural types
    
    # ── Fallback Behavior ────────────────────────────────────────────
    fallback_strategy: FallbackStrategy = FallbackStrategy.BOTH
    fallback_top_k: int = 20
    fallback_max_chars: int = 12000
    fallback_min_score: float = 0.1
    fallback_min_passages: int = 2           # minimum passages to consider success
    
    # ── Output ──────────────────────────────────────────────────────
    include_metadata: bool = True
    include_audit: bool = False
    include_raw_results: bool = False
    return_empty_on_fallback: bool = False
    
    # ── Debugging ────────────────────────────────────────────────────
    inspect: bool = False
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
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
            "include_metadata": self.include_metadata,
            "include_audit": self.include_audit,
        }


@dataclass
class PassageMetadata:
    """Rich metadata for a resolved passage."""
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
    
    # Score breakdown
    avg_score: float = 0.0
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    
    # Terms
    matched_terms: List[str] = field(default_factory=list)


@dataclass
class ResolvedPassage:
    """A fully resolved passage ready for LLM consumption."""
    text: str
    metadata: PassageMetadata
    audit: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": {
                "source": self.metadata.source,
                "page_start": self.metadata.page_start,
                "page_end": self.metadata.page_end,
                "chunk_indices": self.metadata.chunk_indices,
                "chunk_count": self.metadata.chunk_count,
                "is_merged": self.metadata.is_merged,
                "gap_filled": self.metadata.gap_filled,
                "structural_type": self.metadata.structural_type,
                "penalty_factor": self.metadata.penalty_factor,
                "was_penalized": self.metadata.was_penalized,
                "avg_score": self.metadata.avg_score,
                "semantic_score": self.metadata.semantic_score,
                "keyword_score": self.metadata.keyword_score,
                "rrf_score": self.metadata.rrf_score,
                "rerank_score": self.metadata.rerank_score,
                "matched_terms": self.metadata.matched_terms,
            },
            "audit": self.audit,
        }


@dataclass
class RetrievalContext:
    """Complete retrieval context with passages and metadata."""
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
        }
    
    def is_empty(self) -> bool:
        return len(self.passages) == 0