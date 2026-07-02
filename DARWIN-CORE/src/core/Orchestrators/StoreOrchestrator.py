from __future__ import annotations

from pathlib import Path

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Storage.StoreRegistry import DarwinStoreRegistry


class StoreOrchestrator:
    """
    Lean orchestrator over DarwinStore.
    High-level overview, status, and lifecycle ONLY.
    All artifact I/O is delegated directly to DarwinStore.
    """

    def __init__(self, darwin_dir: Path | None = None, name: str | None = None):
        self._store = DarwinStore(darwin_dir, name)



    # ------------------------------------------------------------------
    # Store lifecycle
    # ------------------------------------------------------------------
    def info(self) -> dict:
        return {
            "store_id": self._store.store_id,
            "store_name": self._store.store_name,
            "darwin_dir": str(self._store.darwin_dir),
        }

    def switch(self, darwin_dir: Path | None = None, name: str | None = None) -> dict:
        self._store = DarwinStore(darwin_dir, name)
        return self.info()

    # ------------------------------------------------------------------
    # Status & health (overview only — no artifact proxying)
    # ------------------------------------------------------------------
    def file_status(self, filename: str) -> dict:
        return self._store.file_status(filename)

    def store_status(self) -> dict:
        return self._store.status()

    def pipeline_report(self, filename: str | None = None) -> dict | list[dict]:
        return self._store.get_pipeline_report(filename)

    def search_readiness(self, filename: str, mode: str = "hybrid") -> dict:
        return self._store.get_searchables(filename)

    def ready_files(self, filenames: list[str] | None = None, mode: str = "hybrid") -> list[str]:
        return self._store.get_search_ready_files(filenames, mode)

    # ------------------------------------------------------------------
    # Integrity & health
    # ------------------------------------------------------------------
    def check_integrity(self, content_hash: str) -> dict:
        return self._store.check_integrity(content_hash)

    def repair(self, content_hash: str | None = None, dry_run: bool = False) -> dict:
        return self._store.repair(content_hash, dry_run)

    def audit(self) -> dict:
        return self._store.full_audit()

    def embedding_health(self, model_id: str) -> dict:
        return self._store.get_embedding_health(model_id)

    def reembed_check(self, filenames: list[str] | None = None) -> list[dict]:
        return self._store.get_files_needing_reembed(filenames)

    # ------------------------------------------------------------------
    # Registry helpers (thin delegates — persistent lookup via DarwinStore)
    # ------------------------------------------------------------------
    @staticmethod
    def list_stores() -> list[dict]:
        return list(DarwinStoreRegistry.list_stores().values())

    @staticmethod
    def load_store(store_id: str) -> StoreOrchestrator:
        ds = DarwinStore.load_store(store_id)
        return StoreOrchestrator(ds.darwin_dir, ds.store_name)

    @staticmethod
    def get_default() -> StoreOrchestrator | None:
        ds = DarwinStore.get_default_store()
        return StoreOrchestrator(ds.darwin_dir, ds.store_name) if ds else None

    @staticmethod
    def load_by_name(name: str) -> StoreOrchestrator:
        ds = DarwinStore.load_store_by_name(name)
        return StoreOrchestrator(ds.darwin_dir, ds.store_name)