from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from core.Storage.DarwinStore import DarwinStore
from core.Storage.StoreRegistry import StoreRegistry


class StoreInspector:
    """Thin wrapper around DarwinStore for status/health/integrity queries.

    Provides a stable API surface for the MCP layer and other consumers
    that need read-only store introspection.
    """

    def __init__(self, store: DarwinStore):
        self._store = store

    # ── Store Lifecycle ──

    def info(self) -> dict:
        """Summary of the current store."""
        status = self._store.status()
        return {
            "store_id": self._store.store_id,
            "store_name": self._store.store_name,
            "darwin_dir": str(self._store.darwin_dir),
            "total_files": status["stats"]["total_files"],
            "files_needing_work": status["stats"]["files_needing_work"],
            "keyword_searchable_files": status["stats"]["keyword_searchable_files"],
            "total_chunks": status["stats"]["total_chunks"],
            "total_pages": status["stats"]["total_pages"],
        }

    # ── Source Management ──

    def file_status(self, filename: str) -> dict:
        return self._store.file_status(filename)

    def store_status(self) -> dict:
        return self._store.status()

    # ── Pipeline Reports ──

    def pipeline_report(self, filename: Optional[str] = None) -> dict | list[dict]:
        return self._store.get_pipeline_report(filename)

    def search_readiness(self, filename: str, mode: str = "hybrid") -> dict:
        return self._store.get_searchables(filename)

    def ready_files(self, filenames: Optional[list[str]] = None, mode: str = "hybrid") -> list[str]:
        return self._store.get_search_ready_files(filenames, mode)

    # ── Integrity ──

    def check_integrity(self, content_hash: str) -> dict:
        return self._store.check_integrity(content_hash)

    def repair(self, content_hash: Optional[str] = None, dry_run: bool = False) -> dict:
        return self._store.repair(content_hash, dry_run)

    def audit(self) -> dict:
        return self._store.full_audit()

    # ── Embedding Health ──

    def embedding_health(self, model_id: str) -> dict:
        return self._store.get_embedding_health(model_id)

    def reembed_check(self, filenames: Optional[list[str]] = None) -> list[dict]:
        return self._store.get_files_needing_reembed(filenames)

    # ── Static / Class-level ──

    @staticmethod
    def list_stores() -> list[dict]:
        return StoreRegistry().list_stores()

    @staticmethod
    def get_default() -> Optional[StoreInspector]:
        info = StoreRegistry().get_default_store()
        if not info:
            return None
        store = DarwinStore(Path(info["darwin_dir"]))
        return StoreInspector(store)
