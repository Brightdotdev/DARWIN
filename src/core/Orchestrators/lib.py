"""
lib.py — Central parameter library for Darwin orchestrators.
Single source of truth for all parameter defaults with Literal types
for IDE intellisense and MCP schema generation.

FIXES APPLIED (2026-07-19):
  1. Added fallback_min_passages, fallback_top_k, fallback_max_chars
     to RetrievalParameters (previously hardcoded in OrchestratorConfig)
  2. Added diversity_rerank to RetrievalParameters (tunable from public API)
  3. build() now maps new fields to OrchestratorConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

# ═══════════════════════════════════════════════════════════════════
# Type aliases — Literal = IDE intellisense + MCP JSON Schema enum
# ═══════════════════════════════════════════════════════════════════

Method = Literal["hybrid", "semantic", "keyword"]
MergeStrategy = Literal["none", "adjacent", "aggressive"]
FallbackStrategy = Literal["expand", "relax", "both"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

# ═══════════════════════════════════════════════════════════════════
# Store resolver
# ═══════════════════════════════════════════════════════════════════


def resolve_store(store_ref: Optional[str] = None):
    """Resolve a store identifier to a DarwinStore instance.

    Accepts (in order of attempt):
      - Store ID    (UUID)   → DarwinStore.load_store()
      - Store name           → DarwinStore.load_store_by_name()
      - Filesystem path      → DarwinStore(path)
      - None or error        → DarwinStore() (default)

    Always returns a valid DarwinStore — logs a warning on fallback.
    """
    from src.core.Storage.DarwinStore import DarwinStore
    from src.core.Logger.DarwinLogger import darwin

    if store_ref is None:
        return DarwinStore()

    # 1. Try as store ID (UUID)
    try:
        return DarwinStore.load_store(store_ref)
    except Exception:
        pass

    # 2. Try as store name
    try:
        return DarwinStore.load_store_by_name(store_ref)
    except Exception:
        pass

    # 3. Try as filesystem path
    path = Path(store_ref)
    if path.exists():
        return DarwinStore(path)

    # 4. Fallback to default
    darwin.warning(f"Store '{store_ref}' not resolved — using default store")
    return DarwinStore()


# ═══════════════════════════════════════════════════════════════════
# Retrieval Parameters
# ═══════════════════════════════════════════════════════════════════


@dataclass
class RetrievalParameters:
    """Parameters for RetrievalOrchestrator.retrieve_and_resolve().

    All defaults defined here — single source of truth.
    Pass only what you need; defaults fill the rest.

    Type hints use Literal so your IDE shows valid options.

    Examples:
        RetrievalParameters(query="hello")
        RetrievalParameters(query="hello", method="keyword", top_k=10)
        RetrievalParameters(query="hello", merge_strategy="aggressive")
    """

    query: str = ""
    method: Method = "hybrid"
    store: Optional[str] = None
    """Store ID, name, or path. Resolved via resolve_store(). None = default."""
    files: Optional[list[str]] = None
    top_k: int = 5
    max_passages: int = 5
    max_chars: int = 6000
    min_passage_score: float = 0.15
    merge_strategy: MergeStrategy = "adjacent"
    fill_gaps: bool = False
    max_per_source: int = 2
    prefer_sources: Optional[list[str]] = None
    exclude_sources: Optional[list[str]] = None
    filter_structural: bool = True
    structural_penalty_factor: float = 0.5
    """Penalty multiplier applied to non-body structural types (headers, tables, etc.)."""
    diversity_rerank: bool = True
    """Apply diversity re-ranking to penalize repeated sources. Disabled automatically 
    in fallback mode or when only one source is available."""
    fallback_strategy: FallbackStrategy = "both"
    fallback_top_k: int = 20
    """Number of results to fetch in fallback mode."""
    fallback_max_chars: int = 12000
    """Max characters to return in fallback mode."""
    fallback_min_score: float = 0.05
    """Minimum score threshold in fallback mode."""
    fallback_min_passages: int = 1
    """Minimum number of qualifying passages required to avoid fallback."""
    rerank: bool = False
    rerank_weight: float = 0.3
    """Weight for reranking score in final score computation."""
    rerank_top_k: int = 100
    """Number of top results to pass to reranker."""
    include_audit: bool = False
    """Include per-passage audit details in results."""
    log_level: LogLevel = "INFO"
    save: bool = False
    """Save retrieval results as JSON to the results directory."""
    results_dir: Optional[str] = None
    """Custom directory for saved results. None = default ({base}/search-results/)."""

    def __post_init__(self):
        from src.core.Logger.DarwinLogger import darwin

        # ── Critical: raise (can't proceed without valid values) ──
        if not isinstance(self.query, str):
            raise TypeError(f"query must be str, got {type(self.query).__name__}")
        self.query = self.query.strip()
        if not self.query:
            raise ValueError("query must not be empty")
        valid_methods = {"hybrid", "semantic", "keyword"}
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method!r}")
        valid_merges = {"none", "adjacent", "aggressive"}
        if self.merge_strategy not in valid_merges:
            raise ValueError(f"merge_strategy must be one of {valid_merges}, got {self.merge_strategy!r}")
        valid_fallback = {"expand", "relax", "both"}
        if self.fallback_strategy not in valid_fallback:
            raise ValueError(f"fallback_strategy must be one of {valid_fallback}, got {self.fallback_strategy!r}")

        # ── Numeric bounds: warn + clamp to valid range ──
        def _clamp_int(val, name, lo, hi, default):
            if val < lo:
                darwin.warning(f"{name}={val} below minimum, clamping to {lo}")
                return lo
            if val > hi:
                darwin.warning(f"{name}={val} above maximum, clamping to {hi}")
                return hi
            return val

        def _clamp_float(val, name, lo, hi, default):
            if val < lo:
                darwin.warning(f"{name}={val} below range [{lo}, {hi}], clamping to {lo}")
                return lo
            if val > hi:
                darwin.warning(f"{name}={val} above range [{lo}, {hi}], clamping to {hi}")
                return hi
            return val

        self.top_k = _clamp_int(self.top_k, "top_k", 1, 1000, 5)
        self.max_passages = _clamp_int(self.max_passages, "max_passages", 1, 500, 5)
        self.max_chars = _clamp_int(self.max_chars, "max_chars", 100, 500_000, 6000)
        self.min_passage_score = _clamp_float(self.min_passage_score, "min_passage_score", 0.0, 1.0, 0.15)
        self.max_per_source = _clamp_int(self.max_per_source, "max_per_source", 1, 500, 2)
        self.structural_penalty_factor = _clamp_float(self.structural_penalty_factor, "structural_penalty_factor", 0.0, 1.0, 0.5)
        self.fallback_top_k = _clamp_int(self.fallback_top_k, "fallback_top_k", 1, 1000, 20)
        self.fallback_max_chars = _clamp_int(self.fallback_max_chars, "fallback_max_chars", 100, 500_000, 12000)
        self.fallback_min_score = _clamp_float(self.fallback_min_score, "fallback_min_score", 0.0, 1.0, 0.05)
        self.fallback_min_passages = _clamp_int(self.fallback_min_passages, "fallback_min_passages", 0, 100, 1)
        self.rerank_weight = _clamp_float(self.rerank_weight, "rerank_weight", 0.0, 1.0, 0.3)
        self.rerank_top_k = _clamp_int(self.rerank_top_k, "rerank_top_k", 1, 10_000, 100)

    def build(self):
        """Convert to internal OrchestratorConfig for the orchestrator."""
        from src.core.Retrival.OrchestratorTypes import (
            OrchestratorConfig as _OC,
            MergeStrategy as _MS,
            FallbackStrategy as _FS,
        )

        return _OC(
            query=self.query,
            method=self.method,
            files=self.files,
            top_k=self.top_k,
            max_passages=self.max_passages,
            max_chars=self.max_chars,
            min_passage_score=self.min_passage_score,
            merge_strategy=_MS(self.merge_strategy),
            fill_gaps=self.fill_gaps,
            max_per_source=self.max_per_source,
            prefer_sources=self.prefer_sources,
            exclude_sources=self.exclude_sources,
            filter_structural=self.filter_structural,
            structural_penalty_factor=self.structural_penalty_factor,
            # FIX: Map new diversity_rerank field
            diversity_rerank=self.diversity_rerank,
            fallback_strategy=_FS(self.fallback_strategy),
            # FIX: Map new fallback tuning fields
            fallback_top_k=self.fallback_top_k,
            fallback_max_chars=self.fallback_max_chars,
            fallback_min_score=self.fallback_min_score,
            fallback_min_passages=self.fallback_min_passages,
            rerank=self.rerank,
            rerank_weight=self.rerank_weight,
            rerank_top_k=self.rerank_top_k,
            include_audit=self.include_audit,
            log_level=self.log_level,
        )


# ═══════════════════════════════════════════════════════════════════
# Pipeline Parameters
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PipelineParameters:
    """Parameters for PipelineOrchestrator.build_pipeline().

    All defaults defined here — single source of truth.
    Pass only what you need; defaults fill the rest.

    Examples:
        PipelineParameters(target="./docs")
        PipelineParameters(target="./docs", ocr=True, recursive=True)
        PipelineParameters(stages=["indexing"], threshold=0.7)
    """

    target: str = ""
    """Local path or static URL. Only static HTML pages are scraped
    (JavaScript-heavy frameworks like Vite/React SPAs are not supported)."""
    store: Optional[str] = None
    """Store ID, name, or path. Resolved via resolve_store(). None = default."""
    stages: list[str] = field(default_factory=lambda: ["ingestion", "indexing"])
    files: Optional[list[str]] = None
    ocr: bool = False
    recursive: bool = False
    precise: bool = False
    threshold: float = 0.5
    idf_smoothing: float = 1.5
    only_kw: bool = False
    only_semantic: bool = False
    rerun: bool = False
    strict: bool = True

    def __post_init__(self):
        from src.core.Logger.DarwinLogger import darwin

        # ── Critical: raise (can't proceed without valid values) ──
        if not isinstance(self.target, str):
            raise TypeError(f"target must be str, got {type(self.target).__name__}")
        valid_stages = {"ingestion", "indexing"}
        if not isinstance(self.stages, list) or not self.stages:
            raise ValueError("stages must be a non-empty list")
        bad = set(self.stages) - valid_stages
        if bad:
            raise ValueError(f"Invalid stage(s): {bad}. Valid: {sorted(valid_stages)}")
        if self.only_kw and self.only_semantic:
            raise ValueError("only_kw and only_semantic are mutually exclusive")

        # ── Numeric bounds: warn + clamp to valid range ──
        def _clamp_float(val, name, lo, hi, default):
            if val < lo:
                darwin.warning(f"{name}={val} below range [{lo}, {hi}], clamping to {lo}")
                return lo
            if val > hi:
                darwin.warning(f"{name}={val} above range [{lo}, {hi}], clamping to {hi}")
                return hi
            return val

        self.threshold = _clamp_float(self.threshold, "threshold", 0.0, 1.0, 0.5)
        self.idf_smoothing = _clamp_float(self.idf_smoothing, "idf_smoothing", 0.0, 10.0, 1.5)

    def build(self):
        """Convert to internal (StageParameters, stages) for the orchestrator."""
        from src.core.Orchestrators.PipelineOrchestrator import StageParameters
        from src.core.Pipeline.Ingestion.Parsing.lib import ParsingArgs
        from src.core.Pipeline.Indexing.lib import IndexingArgs

        valid_stages = {"ingestion", "indexing"}
        bad = set(self.stages) - valid_stages
        if bad:
            raise ValueError(f"Invalid stage(s): {bad}. Valid: {sorted(valid_stages)}")

        internal = StageParameters()
        if "ingestion" in self.stages:
            internal.parsing = ParsingArgs(
                target=self.target,
                files=self.files,
                ocr=self.ocr,
                recursive=self.recursive,
                precise=self.precise,
                rerun=self.rerun,
            )
        if "indexing" in self.stages:
            internal.indexing = IndexingArgs(
                threshold=self.threshold,
                idf_smoothing=self.idf_smoothing,
                only_kw=self.only_kw,
                only_semantic=self.only_semantic,
                files=self.files,
                rerun=self.rerun,
            )
        return internal, self.stages