"""
core.api — Public API surface for darwin-rag.

    from core import Darwin

    store = Darwin("/path/to/my/data")
    store.ingest("papers/")
    results = store.search("what is this paper about")
    records = store.search_records(filters={"year": 2024})
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Union


class Darwin:
    """One-stop interface to a Darwin RAG store.

    Wraps StoreOrchestrator, RetrievalOrchestrator, PipelineOrchestrator,
    and LoggerOrchestrator behind a flat, documented surface.

    Args:
        store_path: Filesystem path to the store directory.
                    None = use the registered default store.
        name:       Optional human-readable name for the store.
        quiet:      Suppress dashboard output on bootstrap (default True
                    for library use; set False for CLI-style startup).

    Examples:
        # Default store
        d = Darwin()
        d.ingest("./papers")
        hits = d.search("hybrid search theory")

        # Named store
        d = Darwin("/data/my-project", name="project-alpha")
        d.ingest("./docs", recursive=True)
        records = d.search_records(filters={"status": "active"})

        # Async (inside FastAPI, etc.)
        hits = await d.search_async("query", method="semantic")
    """

    def __init__(
        self,
        store_path: Optional[str | Path] = None,
        name: Optional[str] = None,
        quiet: bool = True,
    ):
        from core.Orchestrators.StoreOrchestrator import StoreOrchestrator

        path = Path(store_path) if store_path else None
        self._orch = StoreOrchestrator(darwin_dir=path, name=name)
        self._orch.bootstrap(initialize_models=True, quiet=quiet)

        # Lazy-init orchestrators (created once, reused)
        self._retrieval = None
        self._pipeline = None

    # ═══════════════════════════════════════════════════════════════
    # Properties
    # ═══════════════════════════════════════════════════════════════

    @property
    def path(self) -> str:
        """Absolute path to the store directory."""
        return str(self._orch.store.darwin_dir)

    @property
    def store_id(self) -> str:
        """Unique store identifier."""
        return self._orch.store.store_id

    @property
    def store_name(self) -> str:
        """Human-readable store name."""
        return self._orch.store.store_name

    @property
    def _store(self):
        """Direct access to the underlying DarwinStore (escape hatch)."""
        return self._orch.store

    def _get_retrieval(self):
        if self._retrieval is None:
            from core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
            self._retrieval = RetrievalOrchestrator(store=self._orch.store)
        return self._retrieval

    def _get_pipeline(self):
        if self._pipeline is None:
            from core.Orchestrators.PipelineOrchestrator import PipelineOrchestrator
            self._pipeline = PipelineOrchestrator(store=self._orch.store)
        return self._pipeline

    # ═══════════════════════════════════════════════════════════════
    # Core — the 90% use case
    # ═══════════════════════════════════════════════════════════════

    def ingest(
        self,
        target: str,
        *,
        files: Optional[List[str]] = None,
        ocr: bool = False,
        recursive: bool = False,
        rerun: bool = False,
        strict: bool = True,
        stages: Optional[List[str]] = None,
    ) -> dict:
        """Ingest files into the store (parse + chunk + index).

        Args:
            target:   Path to a file, directory, or URL.
            files:    Subset of filenames within target to ingest.
            ocr:      Run OCR on scanned documents.
            recursive: recurse into subdirectories.
            rerun:    Re-process files that already have artifacts.
            strict:   Halt on first failure (default True).
            stages:   Pipeline stages to run. Default: ["ingestion", "indexing"].

        Returns:
            dict with keys: success, stages_completed, stages_failed, summary.
        """
        from core.Orchestrators.lib import PipelineParameters

        params = PipelineParameters(
            target=target,
            files=files,
            ocr=ocr,
            recursive=recursive,
            rerun=rerun,
            strict=strict,
            stages=stages or ["ingestion", "indexing"],
        )
        result = self._get_pipeline().build_pipeline(params)
        return {
            "success": result.success,
            "stages_completed": [s.value for s in result.stages_completed],
            "stages_failed": result.stages_failed,
            "summary": result.summary,
        }

    def search(
        self,
        query: str,
        *,
        method: str = "hybrid",
        top_k: int = 5,
        max_passages: int = 5,
        max_chars: int = 6000,
        min_passage_score: float = 0.15,
        merge_strategy: str = "adjacent",
        max_per_source: int = 2,
        diversity_rerank: bool = True,
        prefer_sources: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        files: Optional[List[str]] = None,
        save: bool = False,
    ) -> dict:
        """Search the knowledge base (hybrid / semantic / keyword).

        Args:
            query:  Search query string.
            method: "hybrid" (default), "semantic", or "keyword".
            top_k:  Max results to return.
            save:   Persist results to disk for later retrieval.

        Returns:
            dict with keys: passages (list), total, sources, max_score,
            avg_score, elapsed_ms, fallback_used.
        """
        from core.Orchestrators.lib import RetrievalParameters

        params = RetrievalParameters(
            query=query,
            method=method,
            top_k=top_k,
            max_passages=max_passages,
            max_chars=max_chars,
            min_passage_score=min_passage_score,
            merge_strategy=merge_strategy,
            max_per_source=max_per_source,
            diversity_rerank=diversity_rerank,
            prefer_sources=prefer_sources,
            exclude_sources=exclude_sources,
            files=files,
            save=save,
        )
        ctx = self._get_retrieval().retrieve_and_resolve(params)
        if ctx is None or ctx.is_empty():
            return {"passages": [], "total": 0, "sources": [], "elapsed_ms": 0}

        passages = []
        for p in ctx.passages:
            meta = p.metadata
            passages.append({
                "text": p.text,
                "score": meta.avg_score,
                "source": meta.source,
                "heading": meta.heading or "",
                "page": meta.page_start,
            })

        return {
            "passages": passages,
            "total": ctx.total_passages,
            "total_chars": ctx.total_chars,
            "sources": ctx.sources_used,
            "max_score": ctx.max_score,
            "avg_score": ctx.avg_score,
            "elapsed_ms": ctx.elapsed_ms,
            "fallback_used": ctx.fallback_used,
            "fallback_reason": ctx.fallback_reason,
        }

    def search_records(
        self,
        *,
        filters: Optional[Dict[str, Any]] = None,
        query: str = "",
        mode: str = "auto",
        top_k: int = 5,
        files: Optional[List[str]] = None,
        rerank: bool = False,
        save: bool = False,
    ) -> dict:
        """Search structured data (CSVs, JSON) by field values.

        Filters use Django-style operators:
            {"status": "failed"}                 → exact match
            {"latency__gt": 100}                 → greater than
            {"name__contains": "test"}           → substring
            {"year__in": [2023, 2024]}           → value in list
            {"archived__exists": false}          → field existence

        Args:
            filters: Dict of field-value filter pairs.
            query:   Text query for hybrid fallback (optional).
            mode:    "auto" (default), "structured", or "text".
            top_k:   Max results.

        Returns:
            dict with keys: records (list of {fields, score, source}),
            total, mode, warning.
        """
        from core.Orchestrators.lib import StructuredSearchParams

        params = StructuredSearchParams(
            query=query,
            filters=filters,
            mode=mode,
            top_k=top_k,
            files=files,
            rerank=rerank,
            save=save,
        )
        ctx = self._get_retrieval().search_lists(params)
        if ctx is None or ctx.is_empty():
            return {"records": [], "passages": [], "total": 0, "mode": mode}

        if "structured" in (ctx.method or ""):
            records = []
            for p in ctx.passages:
                meta = p.metadata
                fields = getattr(meta, "_structured_fields", {})
                records.append({
                    "fields": fields,
                    "score": meta.avg_score,
                    "source": meta.source,
                    "page": meta.page_start,
                })
            return {
                "records": records,
                "passages": [],
                "total": ctx.total_passages,
                "mode": "structured",
                "warning": ctx.fallback_reason,
            }
        else:
            passages = [
                {"text": p.text, "score": p.metadata.avg_score, "source": p.metadata.source}
                for p in ctx.passages
            ]
            return {
                "records": [],
                "passages": passages,
                "total": ctx.total_passages,
                "mode": "text",
                "warning": ctx.fallback_reason,
            }

    def schema(
        self,
        *,
        file: Optional[str] = None,
        files: Optional[List[str]] = None,
        list_all: bool = False,
    ) -> dict:
        """Get schema information for structured files.

        Args:
            file:     Single filename to inspect.
            files:    Multiple filenames; also returns overlapping keys.
            list_all: Return schemas for ALL indexed structured files.

        Returns:
            dict with keys: schemas, overlapping_keys, total_files.
        """
        store = self._orch.store

        if list_all:
            schemas = store.get_all_schemas()
            overlapping = store.find_overlapping_keys()
            return {
                "schemas": schemas,
                "total_files": len(schemas),
                "overlapping_keys": overlapping,
            }

        if files:
            result_schemas = {}
            missing = []
            for f in files:
                s = store.load_schema(f)
                if s is None:
                    missing.append(f)
                else:
                    result_schemas[f] = s
            overlapping = store.find_overlapping_keys()
            all_keys = set()
            for s in result_schemas.values():
                all_keys.update(s.get("keys", []))
            relevant = {k: v for k, v in overlapping.items() if k in all_keys}
            result = {
                "schemas": result_schemas,
                "found": len(result_schemas),
                "missing": missing,
                "overlapping_keys": relevant,
            }
            if missing:
                result["warning"] = f"No schemas found for: {missing}"
            return result

        if file:
            s = store.load_schema(file)
            if s is None:
                return {"error": f"No schema for '{file}' — not indexed as structured."}
            return s

        return {"error": "Provide file, files, or list_all=True"}

    # ═══════════════════════════════════════════════════════════════
    # Admin — status, logs, pipeline report
    # ═══════════════════════════════════════════════════════════════

    def status(self) -> dict:
        """Store identity and health overview."""
        return self._orch.info()

    def list_files(self, filename: Optional[str] = None) -> Union[List[dict], dict]:
        """Pipeline report for all files (or a single file).

        Returns a list of dicts, each with: filename, stages, progress_percent,
        chunk_count, completed, missing, core_complete.
        """
        return self._orch.pipeline_report(filename)

    def file_status(self, filename: str) -> dict:
        """Detailed pipeline status for a single file."""
        return self._orch.file_status(filename)

    def logs(
        self,
        *,
        session: Optional[str] = None,
        level: str = "verbose",
        limit: int = 100,
    ) -> dict:
        """Query session logs.

        Args:
            session: "PIPELINE", "RETRIEVAL", "Ingestion", or "Indexing".
                     None = all sessions.
            level:   "verbose" (SUCCESS+WARNING+ERROR) or "quiet" (SUCCESS+ERROR).
            limit:   Max entries (1–1000).

        Returns:
            dict with keys: logs (list), total, session, level.
        """
        from core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

        levels = "SUCCESS,WARNING,ERROR" if level == "verbose" else "SUCCESS,ERROR"
        logs = LoggerOrchestrator.get_logs(
            session=session, levels=levels, limit=limit,
        )
        return {"logs": logs, "total": len(logs), "session": session, "level": level}

    # ═══════════════════════════════════════════════════════════════
    # Advanced — filters, keys, purge
    # ═══════════════════════════════════════════════════════════════

    def validate_filters(
        self,
        filters: Dict[str, Any],
        files: Optional[List[str]] = None,
    ) -> dict:
        """Validate filter keys/types against schemas before searching.

        Returns dict with keys: valid, warnings.
        """
        return self._orch.store.validate_filters(filters, files)

    def overlapping_keys(self) -> Dict[str, List[str]]:
        """Find keys that appear across multiple structured files."""
        return self._orch.store.find_overlapping_keys()

    def purge(
        self,
        stages: List[str],
        filenames: List[str],
        dry_run: bool = False,
    ) -> dict:
        """Delete pipeline artifacts for specific files.

        Args:
            stages:    "ingestion" (markdown/records) or "indexing"
                       (embeddings/vector/bm25).
            filenames: Exact filenames as used during ingestion.
            dry_run:   Preview without deleting.

        Returns:
            dict with keys: success, message.
        """
        result = self._get_pipeline().purge(stages, filenames)
        return {"success": True, "message": "Purge completed"}

    # ═══════════════════════════════════════════════════════════════
    # Config — setup
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def setup():
        """Run the interactive setup wizard (CLI only).

        Configures API keys, embedding models, and provider settings.
        This is a blocking interactive call — use from a terminal, not
        inside an async framework.
        """
        from core.Models.ModelManager import modelManager
        modelManager.initialize()

    # ═══════════════════════════════════════════════════════════════
    # Store lifecycle (static)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def create_store(path: str, name: Optional[str] = None) -> dict:
        """Create a new Darwin store at the given path.

        Returns dict with: store_id, store_name, darwin_dir, status.
        """
        from core.Orchestrators.StoreOrchestrator import StoreOrchestrator
        return StoreOrchestrator.create_store(path, name)

    @staticmethod
    def list_stores() -> List[dict]:
        """List all registered stores."""
        from core.Orchestrators.StoreOrchestrator import StoreOrchestrator
        return StoreOrchestrator.list_stores()

    # ═══════════════════════════════════════════════════════════════
    # Async variants — for FastAPI, agent loops, etc.
    # ═══════════════════════════════════════════════════════════════

    async def search_async(self, query: str, **kwargs) -> dict:
        """Async version of search()."""
        return await asyncio.to_thread(self.search, query, **kwargs)

    async def search_records_async(self, **kwargs) -> dict:
        """Async version of search_records()."""
        return await asyncio.to_thread(self.search_records, **kwargs)

    async def ingest_async(self, target: str, **kwargs) -> dict:
        """Async version of ingest()."""
        return await asyncio.to_thread(self.ingest, target, **kwargs)

    async def schema_async(self, **kwargs) -> dict:
        """Async version of schema()."""
        return await asyncio.to_thread(self.schema, **kwargs)

    async def list_files_async(self, filename: Optional[str] = None) -> Union[List[dict], dict]:
        """Async version of list_files()."""
        return await asyncio.to_thread(self.list_files, filename)

    async def file_status_async(self, filename: str) -> dict:
        """Async version of file_status()."""
        return await asyncio.to_thread(self.file_status, filename)

    async def logs_async(self, **kwargs) -> dict:
        """Async version of logs()."""
        return await asyncio.to_thread(self.logs, **kwargs)

    # ═══════════════════════════════════════════════════════════════
    # Repr
    # ═══════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        return (
            f"Darwin(name={self.store_name!r}, "
            f"id={self.store_id[:8]}..., "
            f"path={self.path!r})"
        )
