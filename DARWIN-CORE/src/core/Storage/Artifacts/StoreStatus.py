from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.Storage.Artifacts.StoreCore import get_current_embedding_model, get_current_model_dimensions
from src.core.Logger.DarwinLogger import darwin, LogType


class StoreStatus:
    """File-level and aggregate status, search readiness, and pipeline reports."""

    # ── Legacy status ───────────────────────────────────────────────────

    def file_status(self, file_path: str | Path) -> dict:
        path = Path(file_path)
        name = path.name
        source = self._sources.find_by_name(name)
        if not source:
            source, _ = self._sources.register(path)

        status = self._contents.get_status(source.content_hash)
        return {
            "source_uuid": source.uuid,
            "file_hash": source.content_hash,
            "original_name": source.display_name,
            "last_seen_path": source.current_path,
            "file_size_mb": round(source.file_size / (1024 * 1024), 1),
            "page_count": status.page_count,
            "is_parsed": status.stages.get("markdown", False),
            "is_chunked": status.stages.get("chunks", False),
            "is_embedded": status.stages.get("embeddings", False),
            "is_kw_searchable": status.is_kw_searchable,
            "is_embedding_searchable": status.is_embedding_searchable,
            "is_hybrid_searchable": status.is_hybrid_searchable,
            "has_graph": status.stages.get("graph", False) and status.stages.get("graph_vis", False),
            "has_wiki": status.stages.get("wiki", False),
            "progress_percent": status.progress_percent,
            "status_summary": status.status_summary,
            "missing_stages": status.missing_stages,
            "embedding_model": status.embedding_model,
            "chunk_count": status.chunk_count,
            "created_at": source.created_at,
            "updated_at": source.updated_at,
        }

    def status(self) -> dict:
        all_sources = self._sources.list_all()
        files: dict[str, dict] = {}
        total_chunks = 0
        total_pages = 0
        kw_searchable = 0

        for source in all_sources:
            st = self._contents.get_status(source.content_hash)
            total_chunks += st.chunk_count
            total_pages += st.page_count
            if st.is_kw_searchable:
                kw_searchable += 1
            files[source.display_name] = {
                "file_hash": source.content_hash,
                "original_name": source.display_name,
                "last_seen_path": source.current_path,
                "file_size_mb": round(source.file_size / (1024 * 1024), 1),
                "page_count": st.page_count,
                "is_parsed": st.stages.get("markdown", False),
                "is_chunked": st.stages.get("chunks", False),
                "is_embedded": st.stages.get("embeddings", False),
                "is_kw_searchable": st.is_kw_searchable,
                "is_embedding_searchable": st.is_embedding_searchable,
                "is_hybrid_searchable": st.is_hybrid_searchable,
                "has_graph": st.stages.get("graph", False) and st.stages.get("graph_vis", False),
                "has_wiki": st.stages.get("wiki", False),
                "progress_percent": st.progress_percent,
                "status_summary": st.status_summary,
                "missing_stages": st.missing_stages,
                "embedding_model": st.embedding_model,
                "chunk_count": st.chunk_count,
                "created_at": source.created_at,
                "updated_at": source.updated_at,
            }

        stats = {
            "total_files": len(all_sources),
            "files_needing_work": sum(
                1 for s in files
                if not self._contents.get_status(self._sources.find_by_name(s).content_hash).is_core_complete
            ),
            "keyword_searchable_files": kw_searchable,
            "total_chunks": total_chunks,
            "total_pages": total_pages,
        }

        return {
            "darwin_dir": str(self._darwin_dir),
            "stats": stats,
            "files": files,
        }

    def get_searchables(self, filename: str) -> dict:
        source = self._sources.find_by_name(filename)
        if not source:
            return {
                "filename": filename,
                "kw_searchable": False,
                "semantic_searchable": False,
                "hybrid_searchable": False,
                "missing_stages": ["not_registered"],
            }
        status = self._contents.get_status(source.content_hash)
        missing = status.missing_stages
        kw = status.is_kw_searchable
        sem = status.is_embedding_searchable
        return {
            "filename": filename,
            "kw_searchable": kw,
            "semantic_searchable": sem,
            "hybrid_searchable": kw and sem,
            "missing_stages": missing,
        }

    # ── Search readiness helpers ────────────────────────────────────────

    def get_search_ready_files(
        self,
        filenames: str | list[str] | None = None,
        mode: str = "hybrid",
    ) -> list[str]:
        targets = []
        if filenames is None:
            targets = [s.display_name for s in self._sources.list_all()]
        elif isinstance(filenames, str):
            targets = [filenames]
        else:
            targets = filenames

        ready = []
        for name in targets:
            if self.is_search_ready(name, mode):
                ready.append(name)
        return ready

    def is_search_ready(self, filename: str, mode: str = "hybrid") -> bool:
        try:
            source = self._sources.find_by_name(filename)
            if not source:
                return False
            status = self._contents.get_status(source.content_hash)
        except Exception:
            return False

        if mode == "kw":
            return status.is_kw_searchable
        elif mode == "semantic":
            return status.is_embedding_searchable
        elif mode == "hybrid":
            return status.is_hybrid_searchable
        return False

    # ── Embedding model drift report ────────────────────────────────────

    def get_files_needing_reembed(
        self,
        filenames: str | list[str] | None = None,
    ) -> list[dict]:
        """Read-only report. Does NOT delete or re-embed anything."""
        current_model = get_current_embedding_model()
        current_dim = get_current_model_dimensions()

        targets = []
        if filenames is None:
            targets = [s.display_name for s in self._sources.list_all()]
        elif isinstance(filenames, str):
            targets = [filenames]
        else:
            targets = filenames

        result = []
        for name in targets:
            source = self._sources.find_by_name(name)
            if not source:
                continue
            content = self._contents.get(source.content_hash)
            if not content:
                continue

            stored_model = content.embedding_model
            status = "ok"
            if not stored_model:
                status = "never_embedded"
            elif stored_model != current_model:
                status = "model_mismatch"

            meta = self._meta.load(source.content_hash)
            stored_dim = meta.get("embedding_dimensions", 0)
            if current_dim and stored_dim and stored_dim != current_dim:
                status = "dimension_mismatch"

            result.append({
                "filename": name,
                "hash": source.content_hash[:16] + "…",
                "stored_model": stored_model,
                "current_model": current_model,
                "stored_dimensions": stored_dim,
                "current_dimensions": current_dim,
                "status": status,
                "needs_reembed": status != "ok",
            })

        return result

    # ── Pipeline report ─────────────────────────────────────────────────

    def get_pipeline_report(self, filename: str | None = None) -> dict | list[dict]:
        def _make(source) -> dict:
            st = self._contents.get_status(source.content_hash)
            return {
                "filename": source.display_name,
                "hash": source.content_hash[:16] + "…",
                "last_seen": source.current_path,
                "stages": {
                    "markdown": st.stages.get("markdown", False),
                    "chunks": st.stages.get("chunks", False),
                    "embeddings": st.stages.get("embeddings", False),
                    "vector": st.stages.get("vector", False),
                    "bm25": st.stages.get("bm25", False),
                    "stopwords": st.stages.get("stopwords", False),
                    "graph": st.stages.get("graph", False),
                    "graph_vis": st.stages.get("graph_vis", False),
                    "wiki": st.stages.get("wiki", False),
                },
                "completed": [name for name, done in st.stages.items() if done],
                "missing": st.missing_stages,
                "core_complete": st.is_core_complete,
                "progress_percent": st.progress_percent,
                "chunk_count": st.chunk_count,
                "embedding_model": st.embedding_model,
            }

        if filename:
            src = self._sources.find_by_name(filename)
            if not src:
                raise KeyError(f"File not registered: {filename}")
            return _make(src)

        return [_make(s) for s in self._sources.list_all()]

    def print_pipeline_report(self, filename: str | None = None) -> None:

        data = self.get_pipeline_report(filename)
        files = [data] if filename else data

        darwin.log("[PIPELINE REPORT]")
 

        for f in files:
            icon = "✓" if f["core_complete"] else "○"
            darwin.log(f"{icon} {f['filename']}")
            darwin.log(
                f"   Hash {f['hash']}  |  "
                f"Progress {f['progress_percent']}%  |  "
                f"Chunks {f['chunk_count']}"
            )

            stage_str = " | ".join(
                f"{'✓' if done else '·'} {name}"
                for name, done in f["stages"].items()
            )
            darwin.log(f"   {stage_str}")

            if f["missing"]:
                darwin.log(f"   Missing: {', '.join(f['missing'])}", LogType.WARNING)