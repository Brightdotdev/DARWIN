from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Storage.DataModels import Source


class StoreOrchestrator:
    """
    Lean orchestrator over DarwinStore.
    Returns plain dicts/lists so it is naturally MCP-consumable.
    """

    def __init__(self, darwin_dir: Path | None = None, name: str | None = None):
        self._store = DarwinStore(darwin_dir, name)

    # ------------------------------------------------------------------
    # Store lifecycle
    # ------------------------------------------------------------------
    def info(self) -> dict:
        return {
            "store_id": self._store._store_id,
            "store_name": self._store._store_name,
            "darwin_dir": str(self._store._darwin_dir),
        }

    def switch(self, darwin_dir: Path | None = None, name: str | None = None) -> dict:
        self._store = DarwinStore(darwin_dir, name)
        return self.info()

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------
    def register(self, file_path: str, page_count: int = 0, file_size: int = 0) -> dict:
        src = self._store.register(file_path, page_count, file_size)
        return self._source_to_dict(src)

    def register_url(self, slug: str, url: str, content_hash: str, page_count: int = 0) -> dict:
        src = self._store.register_url(slug, url, content_hash, page_count)
        return self._source_to_dict(src)

    def remove(self, filename: str) -> dict:
        ok = self._store.remove_file(filename)
        return {"removed": ok, "filename": filename}

    def list_files(self) -> list[dict]:
        status = self._store.status()
        return [{"name": k, **v} for k, v in status.get("files", {}).items()]

    def get_file(self, filename: str) -> dict | None:
        src = self._store.get_source(filename)
        return self._source_to_dict(src) if src else None

    # ------------------------------------------------------------------
    # Artifacts — pipeline stages
    # ------------------------------------------------------------------
    def save_markdown(self, filename: str, display_text: str, embedding_text: str = "",
                      extras: dict | None = None) -> dict:
        doc = {
            "display_text": display_text,
            "embedding_text": embedding_text or display_text,
            **(extras or {}),
        }
        self._store.save_markdown(filename, doc)
        return {"saved": True, "stage": "markdown", "filename": filename}

    def load_markdown(self, filename: str) -> str | None:
        return self._store.load_markdown(filename)

    def delete_markdown(self, filename: str) -> dict:
        ok = self._store.delete_markdown(filename)
        return {"deleted": ok, "stage": "markdown", "filename": filename}

    def save_chunks(self, filename: str, records: list[dict]) -> dict:
        self._store.save_records(filename, records)
        return {"saved": True, "stage": "chunks", "count": len(records), "filename": filename}

    def load_chunks(self, filename: str, columns: list[str] | None = None) -> list[dict]:
        table = self._store.load_records(filename, columns)
        if table.num_rows == 0:
            return []
        pyd = table.to_pydict()
        return [{k: pyd[k][i] for k in pyd} for i in range(table.num_rows)]

    def delete_chunks(self, filename: str) -> dict:
        ok = self._store.delete_records(filename)
        return {"deleted": ok, "stage": "chunks", "filename": filename}

    def save_embeddings(self, filename: str, vectors: list[list[float]],
                        indices: list[int] | None = None) -> dict:
        arr = np.array(vectors, dtype=np.float32)
        self._store.save_embeddings(filename, arr, indices)
        return {
            "saved": True,
            "stage": "embeddings",
            "count": len(vectors),
            "dimensions": arr.shape[1],
            "filename": filename,
        }

    def load_embeddings(self, filename: str) -> list[list[float]] | None:
        arr = self._store.load_embeddings(filename)
        if arr is None:
            return None
        return arr.astype(np.float32).tolist()

    def delete_embeddings(self, filename: str) -> dict:
        ok = self._store.delete_embeddings(filename)
        return {"deleted": ok, "stage": "embeddings", "filename": filename}

    def save_vector(self, filename: str, vectors: list[list[float]]) -> dict:
        arr = np.array(vectors, dtype=np.float32)
        self._store.save_vector(filename, arr)
        return {"saved": True, "stage": "vector", "count": len(vectors), "filename": filename}

    def delete_vector(self, filename: str) -> dict:
        ok = self._store.delete_vector(filename)
        return {"deleted": ok, "stage": "vector", "filename": filename}

    def save_bm25(self, filename: str, data: dict) -> dict:
        self._store.save_bm25(filename, data)
        return {"saved": True, "stage": "bm25", "filename": filename}

    def load_bm25(self, filename: str) -> dict | None:
        return self._store.load_bm25(filename)

    def delete_bm25(self, filename: str) -> dict:
        ok = self._store.delete_bm25(filename)
        return {"deleted": ok, "stage": "bm25", "filename": filename}

    def save_graph(self, filename: str, nodes: list[dict], edges: list[dict]) -> dict:
        self._store.save_graph(filename, nodes, edges)
        return {
            "saved": True,
            "stage": "graph",
            "node_count": len(nodes),
            "edge_count": len(edges),
            "filename": filename,
        }

    def save_graph_html(self, filename: str, html: str) -> dict:
        self._store.save_graph_html(filename, html)
        return {"saved": True, "stage": "graph_vis", "filename": filename}

    def load_graph(self, filename: str) -> dict:
        return self._store.load_graph(filename)

    def delete_graph(self, filename: str) -> dict:
        ok = self._store.delete_graph(filename)
        return {"deleted": ok, "stage": "graph", "filename": filename}

    def delete_graph_html(self, filename: str) -> dict:
        ok = self._store.delete_graph_html(filename)
        return {"deleted": ok, "stage": "graph_vis", "filename": filename}

    def save_wiki(self, filename: str, wiki: dict) -> dict:
        self._store.save_wiki(filename, wiki)
        return {"saved": True, "stage": "wiki", "page_count": len(wiki), "filename": filename}

    def load_wiki(self, filename: str) -> dict | None:
        return self._store.load_wiki(filename)

    def delete_wiki(self, filename: str) -> dict:
        ok = self._store.delete_wiki(filename)
        return {"deleted": ok, "stage": "wiki", "filename": filename}

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, filename: str, query_vec: list[float], k: int = 10,
               min_score: float = 0.0, columns: list[str] | None = None,
               force_numpy: bool = False) -> list[dict]:
        vec = np.array(query_vec, dtype=np.float32)
        return self._store.search_single(
            vec, filename, k=k, min_score=min_score,
            columns=columns, force_numpy=force_numpy,
        )

    def search_multi(self, filenames: list[str], query_vec: list[float], top_k: int = 10,
                     min_score: float = 0.0, columns: list[str] | None = None,
                     force_numpy: bool = False) -> list[dict]:
        vec = np.array(query_vec, dtype=np.float32)
        return self._store.search_multi(
            vec, filenames, top_k=top_k, min_score=min_score,
            columns=columns, force_numpy=force_numpy,
        )

    # ------------------------------------------------------------------
    # Status & health
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
    # Integrity
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
    # Registry
    # ------------------------------------------------------------------
    @staticmethod
    def list_stores() -> list[dict]:
        return DarwinStore.list_stores()

    @staticmethod
    def load_store(store_id: str) -> StoreOrchestrator:
        ds = DarwinStore.load_store(store_id)
        return StoreOrchestrator(ds._darwin_dir, ds._store_name)

    @staticmethod
    def get_default() -> StoreOrchestrator | None:
        ds = DarwinStore.get_default_store()
        return StoreOrchestrator(ds._darwin_dir, ds._store_name) if ds else None

    @staticmethod
    def load_by_name(name: str) -> StoreOrchestrator:
        ds = DarwinStore.load_store_by_name(name)
        return StoreOrchestrator(ds._darwin_dir, ds._store_name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _source_to_dict(src: Source) -> dict:
        return {
            "uuid": src.uuid,
            "display_name": src.display_name,
            "content_hash": src.content_hash,
            "current_path": src.current_path,
            "file_size": src.file_size,
            "page_count": src.page_count,
            "created_at": src.created_at,
            "updated_at": src.updated_at,
        }


# Module-level singleton for MCP consumption
storeOrchestrator: StoreOrchestrator = StoreOrchestrator()