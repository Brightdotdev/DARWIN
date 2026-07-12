"""
lib.py — Central parameter library for Darwin orchestrators.
Single source of truth for all parameter defaults with Literal types
for IDE intellisense and MCP schema generation.

Usage:
    from src.core.Orchestrators.lib import RetrievalParameters, PipelineParameters

    # Retrieval — defaults fill what you omit
    params = RetrievalParameters(query="hello", method="hybrid")
    cfg = params.build()
    ctx = orchestrator.retrieve_and_resolve(cfg)

    # Pipeline — defaults fill what you omit  
    params = PipelineParameters(target="./docs", ocr=True)
    internal_params, stages = params.build()
    result = orchestrator.build_pipeline(internal_params, stages)

Store resolution — every parameter set accepts an optional store identifier
that is resolved to a DarwinStore instance at call time:

    params = RetrievalParameters(query="hello", store="work_store")
    store = resolve_store(params.store)
    orch = RetrievalOrchestrator(store=store)
    ctx = orch.retrieve_and_resolve(params)

Resolve priority:
    1. Store ID (UUID) via DarwinStore.load_store()
    2. Store name via DarwinStore.load_store_by_name()
    3. Filesystem path via DarwinStore(path)
    4. Default store (DarwinStore()) on any error or None

Target (PipelineParameters) accepts both local paths and static URLs.
Only static HTML pages are scraped — JavaScript-heavy frameworks
(e.g. Vite, React SPAs) are not supported.
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
PipelineMode = Literal["quiet", "normal"]

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
    min_passage_score: float = 0.3
    merge_strategy: MergeStrategy = "adjacent"
    fill_gaps: bool = False
    max_per_source: int = 2
    prefer_sources: Optional[list[str]] = None
    exclude_sources: Optional[list[str]] = None
    filter_structural: bool = True
    fallback_strategy: FallbackStrategy = "both"
    rerank: bool = False
    log_level: LogLevel = "INFO"

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
            fallback_strategy=_FS(self.fallback_strategy),
            rerank=self.rerank,
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
    log_mode: PipelineMode = "quiet"

    def build(self):
        """Convert to internal (PipelineParameters, stages) for the orchestrator."""
        from src.core.Orchestrators.PipelineOrchestrator import (
            PipelineParameters as _PP,
            ParsingArgs as _ParsingArgs,
            ChunkingArgs as _ChunkingArgs,
            IndexingArgs as _IndexingArgs,
        )

        internal = _PP()
        file_list = self.files or []
        if "ingestion" in self.stages:
            internal.parsing = _ParsingArgs(
                target=self.target,
                files=file_list,
                ocr=self.ocr,
                recursive=self.recursive,
                precise=self.precise,
                rerun=self.rerun,
            )
            internal.chunking = _ChunkingArgs(files=file_list, rerun=self.rerun)
        if "indexing" in self.stages:
            internal.indexing = _IndexingArgs(
                threshold=self.threshold,
                idf_smoothing=self.idf_smoothing,
                only_kw=self.only_kw,
                only_semantic=self.only_semantic,
                files=file_list,
                rerun=self.rerun,
            )
        return internal, self.stages
