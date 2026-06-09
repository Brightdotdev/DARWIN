from __future__ import annotations

from typing import Any

import numpy as np

from src.core.Storage.Artifacts.StoreCore import get_current_embedding_model, get_current_model_dimensions
from src.core.Storage.StoreUtils import now, read_json, write_json
from src.core.Logger.DarwinLogger import darwin, LogType


class ArtifactEmbeddings:
    """Numpy embeddings, USearch vector index, and deletion."""

    def has_embeddings(self, filename: str) -> bool:
        return self._artifacts.has_artifact(self._resolve_hash(filename), "embeddings")

    def save_embeddings(self, filename: str, vectors: np.ndarray, indices: list[int] | None = None) -> None:
        if vectors is None or len(vectors) == 0:
            darwin.log(f"No vectors to save for {filename}", LogType.ERROR)
            return

        h = self._resolve_hash(filename)
        self._artifacts.ensure_dir(h)

        vectors = np.ascontiguousarray(vectors.astype(np.float16))
        emb_path = self.paths.embedding_path(h)
        emb_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(str(emb_path), vectors)

        dim = vectors.shape[1]
        count = vectors.shape[0]
        model_name = get_current_embedding_model()

        if indices is not None:
            map_path = self.paths.embedding_index_map_path(h)
            map_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(map_path, indices)

        self._meta.update(h, "embeddings", extras={
            "embedding_model": model_name,
            "embedding_dimensions": dim,
            "embedding_count": count,
        })
        self._contents.set_stage(h, "embeddings")
        self._contents.update_meta(h, embedding_model=model_name)
        self._cache.invalidate(f"embeddings:{h}")

        darwin.log(f"Embeddings saved: {count} vectors, {dim}d, float16", LogType.SUCCESS)

    def load_embeddings(self, filename: str, mmap: bool = True) -> np.ndarray | None:
        h = self._resolve_hash(filename)
        cache_key = f"embeddings:{h}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        emb_path = self.paths.embedding_path(h)
        if not emb_path.exists():
            return None

        meta = self._meta.load(h)
        stored_model = meta.get("embedding_model", "unknown")
        current_model = get_current_embedding_model()
        if stored_model not in ("unknown", current_model):
            darwin.log(f"Model mismatch for '{filename}': {stored_model} vs {current_model}", LogType.WARNING)
            return None

        mode = "r" if mmap else None
        vectors = np.load(str(emb_path), mmap_mode=mode)

        if mmap:
            self._cache.put(cache_key, vectors)
        return vectors

    def load_embedding_index_map(self, filename: str) -> list[int] | None:
        h = self._resolve_hash(filename)
        path = self.paths.embedding_index_map_path(h)
        if not path.exists():
            return None
        return read_json(path)

    # ── Vector (USearch) ────────────────────────────────────────────────

    def has_vector(self, filename: str) -> bool:
        return self._artifacts.has_artifact(self._resolve_hash(filename), "vector")

    def save_vector(self, filename: str, vectors: np.ndarray) -> None:
        h = self._resolve_hash(filename)
        self._artifacts.ensure_dir(h)

        from usearch.index import Index
        n, dim = vectors.shape
        index = Index(ndim=dim, metric="cos", dtype="f32")
        keys = np.arange(n, dtype=np.int64)
        index.add(keys=keys, vectors=vectors.astype(np.float32))

        idx_path = self.paths.vector_index_path(h)
        meta_path = self.paths.vector_json_path(h)

        idx_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        index.save(str(idx_path))

        write_json(meta_path, {
            "dimension": dim,
            "count": n,
            "metric": "cos",
            "dtype": "f32",
            "created_at": now(),
        })

        self._meta.update(h, "vector", extras={"vector_dimension": dim, "vector_count": n})
        self._contents.set_stage(h, "vector")
        self._cache.invalidate(f"vector:{h}")

        darwin.log(f"Vector index saved: {n} vectors, {dim}d", LogType.SUCCESS)

    def load_vector(self, filename: str):
        h = self._resolve_hash(filename)
        cache_key = f"vector:{h}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        idx_path = self.paths.vector_index_path(h)
        meta_path = self.paths.vector_json_path(h)
        if not idx_path.exists() or not meta_path.exists():
            return None

        meta = self._meta.load(h)
        stored_model = meta.get("embedding_model", "unknown")
        current_model = get_current_embedding_model()
        if stored_model not in ("unknown", current_model):
            darwin.log(f"Vector model mismatch for '{filename}'", LogType.WARNING)
            return None

        vector_meta = read_json(meta_path) or {}
        stored_dim = vector_meta.get("dimension", 0)
        expected_dim = get_current_model_dimensions()
        if expected_dim and stored_dim != expected_dim:
            darwin.log(f"Vector dimension mismatch: {stored_dim} vs {expected_dim}", LogType.WARNING)
            return None

        from usearch.index import Index
        index = Index.restore(str(idx_path))
        self._cache.put(cache_key, index)
        return index

    def search_vector(self, filename: str, query_vec: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        index = self.load_vector(filename)
        if index is None:
            return []

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        matches = index.search(query_vec.astype(np.float32), k)
        if matches is None:
            return []

        index_map = self.load_embedding_index_map(filename)
        results: list[tuple[int, float]] = []

        if hasattr(matches, "keys") and hasattr(matches, "distances"):
            for key, dist in zip(matches.keys, matches.distances):
                row_idx = int(key)
                chunk_idx = index_map[row_idx] if index_map else row_idx
                results.append((chunk_idx, 1.0 - float(dist)))
        else:
            for match in matches:
                row_idx = int(match.key)
                chunk_idx = index_map[row_idx] if index_map else row_idx
                results.append((chunk_idx, 1.0 - float(match.distance)))

        return results

    # ── Deletion ────────────────────────────────────────────────────────

    def delete_embeddings(self, filename: str) -> bool:
        """Remove .npy embeddings + index map and clear the registry flag."""
        h = self._resolve_hash(filename)
        removed = False
        for p in (self.paths.embedding_path(h), self.paths.embedding_index_map_path(h)):
            if p.exists():
                p.unlink()
                removed = True
        self._contents.set_stage(h, "embeddings", complete=False)
        self._contents.update_meta(h, embedding_model=None)
        self._cache.invalidate(f"embeddings:{h}")
        return removed

    def delete_vector(self, filename: str) -> bool:
        """Remove USearch index + meta and clear the registry flag."""
        h = self._resolve_hash(filename)
        removed = False
        for p in (self.paths.vector_index_path(h), self.paths.vector_json_path(h)):
            if p.exists():
                p.unlink()
                removed = True
        self._contents.set_stage(h, "vector", complete=False)
        self._cache.invalidate(f"vector:{h}")
        return removed