# src/core/Storage/RetrievalArtifacts.py
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Storage.StoreUtils import now, read_json, write_json
from src.core.Storage.PathResolverMixin import PathResolverMixin


# ── Model helpers ─────────────────────────────────────────────────────────

def _get_current_embedding_model() -> str:
    from src.core.Models import ModelManager, ModelCategory
    loader = ModelManager.get_loader()
    return loader.get_model_key(ModelCategory.EMBEDDING)


def _get_current_model_dimensions() -> int:
    """Expected embedding dimensions for the active model. Safe fallback."""
    try:
        
        from src.core.Models.EmbeddingConfig import get_preset
        key = _get_current_embedding_model()
        presets = get_preset(key)
        if presets:
            return presets.dimensions
        # Fallback lookup table — should migrate to ModelRegistry specs
        _DIMENSIONS = {
            "intfloat/e5-small-v2": 384,
            "BAAI/bge-m3": 1024,
            "GritLM/GritLM-7B": 4096,
        }
        return _DIMENSIONS.get(key, 0)
    except Exception:
        return 0



# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Embeddings (float16 .npy, zero metadata duplication)
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingsMixin(PathResolverMixin):
    """
    Pure numerical embedding storage.
    * ONLY float16 vectors on disk.
    * NO text, NO metadata, NO chunk ids inside the .npy file.
    * Metadata lives in meta.json (single source of truth).
    * mmap_mode="r" keeps RAM overhead near zero until vectors are accessed.
    """

    def has_embeddings(self, filename: str) -> bool:
        return self._has_artifact(filename, "embeddings")


    def save_embeddings(
        self,
        filename: str,
        vectors: np.ndarray,
        indices: Optional[List[int]] = None,
    ) -> None:
        if vectors is None or len(vectors) == 0:
            darwin.log(f"No vectors to save for {filename}", LogType.ERROR)
            return

        d = self._dir_for(filename)
        d.mkdir(parents=True, exist_ok=True)

        fh = self._hash_for(filename)
        embedding_path = self.paths.embedding_path(fh)
        embedding_path.parent.mkdir(parents=True, exist_ok=True)

        # C-contiguous float16: optimal for both disk layout and mmap paging
        vectors = np.ascontiguousarray(vectors.astype(np.float16))
        np.save(embedding_path, vectors)

        dim = vectors.shape[1]
        count = vectors.shape[0]
        model_name = _get_current_embedding_model()

        # Save chunk index map alongside embeddings
        if indices is not None:
            map_path = self.paths.embedding_index_map_path(self._hash_for(filename))
            map_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(map_path, indices)

        # Global meta.json is the ONLY place embedding metadata lives
        self._update_meta(filename, "embeddings", extras={
            "embedding_model": model_name,
            "embedding_dimensions": dim,
            "embedding_count": count,
        })

        record = self._record_for(filename)
        if record:
            try:
                self.registry.update_pipeline(
                    file_hash=record.file_hash,
                    stage="embeddings",
                )
            except Exception as reg_err:
                darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)

        if hasattr(self, "invalidate"):
            self.invalidate(filename)

        darwin.log(
            f"Embeddings saved: {embedding_path.name} ({count} vectors, {dim}d, float16)",
            LogType.SUCCESS,
        )

    def load_embeddings(self, filename: str, mmap: bool = True) -> np.ndarray | None:
        """
        Load embeddings as a float16 array.
        With mmap=True the OS pages vectors in on demand — essential for large
        corpora that would otherwise OOM on eager loading.
        """
        cache_key = f"embeddings:{filename}"
        if hasattr(self, "_artifact_cache") and cache_key in self._artifact_cache:
            return self._artifact_cache[cache_key]

        d = self._dir_for(filename)
        if not d.exists():
            darwin.log(
                f"{filename} does not exist on disk or is not indexed. Cannot load embeddings.",
                LogType.ERROR,
            )
            return None

        fh = self._hash_for(filename)
        embedding_path = self.paths.embedding_path(fh)

        if not embedding_path.exists():
            meta = self.load_meta(filename)
            stored_model = meta.get("embedding_model", "unknown")
            current_model = _get_current_embedding_model()
            if stored_model not in ("unknown", current_model):
                darwin.log(
                    f"Embeddings for '{filename}' were built with '{stored_model}', "
                    f"but current model is '{current_model}'. Re-embedding required.",
                    LogType.ERROR,
                )
            else:
                darwin.log(
                    f"No embeddings found for '{filename}'. Run embedding pipeline first.",
                    LogType.ERROR,
                )
            return None

        # Model compatibility check
        meta = self.load_meta(filename)
        stored_model = meta.get("embedding_model", "unknown")
        current_model = _get_current_embedding_model()
        if stored_model not in ("unknown", current_model):
            darwin.log(
                f"Embeddings for '{filename}' were built with '{stored_model}' "
                f"but current model is '{current_model}'. Using a different embedding "
                f"model breaks cosine-similarity search. Re-embedding required.",
                LogType.WARNING,
            )
            return None

        # Dimension sanity check — peek header without full load
        expected_dim = meta.get("embedding_dimensions")
        if expected_dim:
            header = np.load(embedding_path, mmap_mode="r")
            if header.shape[1] != expected_dim:
                darwin.log(
                    f"Embedding dimension mismatch on disk: expected {expected_dim}, got {header.shape[1]}.",
                    LogType.WARNING,
                )
            del header  # Release mmap peek immediately

        # mmap_mode='r' → read-only memory map. No RAM consumed until touched.
        mode = "r" if mmap else None
        vectors = np.load(embedding_path, mmap_mode=mode)

        if hasattr(self, "_artifact_cache") and mmap:
            self._artifact_cache[cache_key] = vectors

        darwin.log(
            f"Loaded embeddings: {vectors.shape[0]} vectors, {vectors.shape[1]}d (mmap={mmap})",
            LogType.INFO,
        )
        return vectors

    def load_embedding_index_map(self, filename: str) -> Optional[List[int]]:
        """Return the chunk index map for embedding vectors, or None for identity mapping."""
        map_path = self.paths.embedding_index_map_path(self._hash_for(filename))
        if not map_path.exists():
            return None
        return read_json(map_path)


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Vector (USearch) — row index == embedding row (with index map translation)
# ═══════════════════════════════════════════════════════════════════════════

class VectorMixin(PathResolverMixin):
    """
    ANN index via USearch.
    Keys are row indices (0..N-1) aligned with embeddings.f16.npy.
    Chunk index mapping is handled via embeddings.index_map.json.
    """

    def has_vector(self, filename: str) -> bool:
        return self._has_artifact(filename, "vector")

    def save_vector(self, filename: str, vectors: np.ndarray) -> None:
        d = self._dir_for(filename)
        d.mkdir(parents=True, exist_ok=True)

        from usearch.index import Index

        n, dim = vectors.shape
        # USearch index built as f32 for broad compatibility; source vectors may be f16
        index = Index(ndim=dim, metric="cos", dtype="f32")
        keys = np.arange(n, dtype=np.int64)
        index.add(keys=keys, vectors=vectors.astype(np.float32))

        fh = self._hash_for(filename)
        index_path = self.paths.vector_index_path(fh)
        meta_path = self.paths.vector_json_path(fh)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        index.save(str(index_path))

        # vector.meta.json: index-only metadata (no embedding model info — that's in meta.json)
        write_json(meta_path, {
            "dimension": dim,
            "count": n,
            "metric": "cos",
            "dtype": "f32",
            "created_at": now(),
        })

        self._update_meta(filename, "vector", extras={
            "vector_dimension": dim,
            "vector_count": n,
        })

        record = self._record_for(filename)
        if record:
            try:
                self.registry.update_pipeline(
                    file_hash=record.file_hash,
                    stage="vector",
                )
            except Exception as reg_err:
                darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)

        if hasattr(self, "invalidate"):
            self.invalidate(filename)

        darwin.log(f"Vector index saved: {n} vectors, {dim}d", LogType.SUCCESS)

    def load_vector(self, filename: str):
        cache_key = f"vector:{filename}"
        if hasattr(self, "_artifact_cache") and cache_key in self._artifact_cache:
            return self._artifact_cache[cache_key]

        d = self._dir_for(filename)
        if not d.exists():
            return None

        fh = self._hash_for(filename)
        index_path = self.paths.vector_index_path(fh)
        meta_path = self.paths.vector_json_path(fh)

        if not index_path.exists() or not meta_path.exists():
            return None

        # Model compatibility check
        global_meta = self.load_meta(filename)
        stored_model = global_meta.get("embedding_model", "unknown")
        current_model = _get_current_embedding_model()
        if stored_model not in ("unknown", current_model):
            darwin.log(
                f"Vector index for '{filename}' was built with '{stored_model}' "
                f"but current model is '{current_model}'. Using a different embedding "
                f"model breaks ANN search. Re-embedding required.",
                LogType.WARNING,
            )
            return None

        # Dimension compatibility check
        vector_meta = read_json(meta_path)
        stored_dim = vector_meta.get("dimension", 0)
        expected_dim = _get_current_model_dimensions()
        if expected_dim and stored_dim != expected_dim:
            darwin.log(
                f"Vector dimension mismatch: index has {stored_dim}d "
                f"but current model expects {expected_dim}d. Re-embedding required.",
                LogType.WARNING,
            )
            return None

        from usearch.index import Index
        index = Index.restore(str(index_path))

        if hasattr(self, "_artifact_cache"):
            self._artifact_cache[cache_key] = index
        return index

    def search_vector(
        self,
        filename: str,
        query_vec: np.ndarray,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Return (chunk_index, cosine_similarity) tuples.
        Row indices are translated through embeddings.index_map.json to original chunk indices.
        """
        index = self.load_vector(filename)
        if index is None:
            return []

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        matches = index.search(query_vec.astype(np.float32), k)
        results: list[tuple[int, float]] = []

        if matches is None:
            return results

        # Load index map to translate row positions → chunk indices
        index_map = self.load_embedding_index_map(filename)

        if hasattr(matches, "keys") and hasattr(matches, "distances"):
            for key, dist in zip(matches.keys, matches.distances):
                row_idx = int(key)
                chunk_idx = index_map[row_idx] if index_map else row_idx
                score = 1.0 - float(dist)
                results.append((chunk_idx, score))
        else:
            for match in matches:
                row_idx = int(match.key)
                chunk_idx = index_map[row_idx] if index_map else row_idx
                score = 1.0 - float(match.distance)
                results.append((chunk_idx, score))

        return results

    def load_vector_metadata(self, filename: str) -> Optional[dict]:
        fh = self._hash_for(filename)
        meta_path = self.paths.vector_json_path(fh)
        if not meta_path.exists():
            return None
        return read_json(meta_path)

    def is_vector_compatible(self, filename: str) -> bool:
        try:
            index = self.load_vector(filename)
            return index is not None
        except FileNotFoundError:
            return False


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Search — high-level semantic retrieval (USearch + numpy fallback)
# ═══════════════════════════════════════════════════════════════════════════

class SearchMixin(PathResolverMixin):
    """
    High-level semantic search operations.
    Combines USearch ANN and numpy fallback with lazy record fetching.

    Depends on VectorMixin, RecordsMixin, and EmbeddingsMixin being present
    in the same class (resolved via MRO — typically via DarwinStore).
    """

    def retrieve_by_vector(
        self,
        filename: str,
        query_vec: np.ndarray,
        k: int = 10,
        columns: list[str] | None = None,
        cls: Any = None,
    ) -> list[Any]:
        """
        USearch ANN path: query vector → chunk indices → parquet fetch → reconstruction.
        """
        matches = self.search_vector(filename, query_vec, k=k)
        if not matches:
            return []

        indices, _scores = zip(*matches)

        # Fetch by chunk index using index__in filter
        rows = self.query_records(
            filename,
            columns=columns,
            index__in=list(indices),
        )

        # Attach scores by chunk_index lookup
        score_by_index = {idx: score for idx, score in matches}
        for row in rows:
            row["_score"] = score_by_index.get(row.get("index", -1), 0.0)

        if cls is not None:
            from src.core.Storage.StoreUtils import reconstruct_records
            return reconstruct_records(rows, cls=cls)
        return rows

    def retrieve_by_numpy(
        self,
        filename: str,
        query_vec: np.ndarray,
        k: int = 10,
        columns: list[str] | None = None,
        cls: Any = None,
    ) -> list[Any]:
        """
        Numpy fallback: exhaustive dot-product over mmap'd float16 embeddings.
        No USearch index required.
        """
        vectors = self.load_embeddings(filename, mmap=True)
        if vectors is None or vectors.shape[0] == 0:
            return []

        # Load index map to translate positions → chunk indices
        index_map = self.load_embedding_index_map(filename)

        # Cosine similarity via dot product (assumes L2-normalized embeddings)
        similarities = np.dot(vectors.astype(np.float32), query_vec.astype(np.float32))

        n = len(similarities)
        if k >= n:
            top_positions = np.argsort(similarities)[::-1][:k]
        else:
            top_positions = np.argpartition(similarities, -k)[-k:]
            top_positions = top_positions[np.argsort(similarities[top_positions])[::-1]]

        # Translate positions to chunk indices
        chunk_indices = [
            index_map[pos] if index_map else int(pos)
            for pos in top_positions.tolist()
        ]
        scores = similarities[top_positions].tolist()

        # Fetch by chunk index
        rows = self.query_records(
            filename,
            columns=columns,
            index__in=chunk_indices,
        )

        score_by_index = {idx: score for idx, score in zip(chunk_indices, scores)}
        for row in rows:
            row["_score"] = score_by_index.get(row.get("index", -1), 0.0)

        if cls is not None:
            from src.core.Storage.StoreUtils import reconstruct_records
            return reconstruct_records(rows, cls=cls)
        return rows

    def search_single(
        self,
        query_vec: np.ndarray,
        filename: str,
        k: int = 10,
        min_score: float = 0.0,
        columns: list[str] | None = None,
        force_numpy: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Single-document search with automatic method selection.
        Returns plain dicts with '_score' attached. Reconstruction is left to
        the caller so that score-based filtering/sorting happens before object
        instantiation overhead.
        """
        rows: list[dict] = []

        if not force_numpy and self.has_vector(filename):
            try:
                rows = self.retrieve_by_vector(
                    filename, query_vec, k=k, columns=columns, cls=None,
                )
            except Exception as e:
                darwin.log(f"USearch failed for {filename}: {e}", LogType.WARNING)

        if not rows and self.has_embeddings(filename):
            try:
                rows = self.retrieve_by_numpy(
                    filename, query_vec, k=k, columns=columns, cls=None,
                )
            except Exception as e:
                darwin.log(f"Numpy fallback failed for {filename}: {e}", LogType.WARNING)

        return [r for r in rows if r.get("_score", 0.0) >= min_score]

    def search_multi(
        self,
        query_vec: np.ndarray,
        filenames: list[str],
        top_k: int = 10,
        min_score: float = 0.0,
        columns: list[str] | None = None,
        force_numpy: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Multi-document semantic search. Collects per-doc results, sorts globally
        by score, and returns the top-k. Always returns dicts.
        """
        if not filenames:
            return []

        # Deduplicate filenames to avoid redundant work
        unique_filenames = list(dict.fromkeys(filenames))

        all_results: list[dict] = []
        per_doc_k = max(5, top_k // len(unique_filenames) + 1)

        for filename in unique_filenames:
            rows = self.search_single(
                query_vec, filename, k=per_doc_k,
                min_score=min_score, columns=columns,
                force_numpy=force_numpy,
            )
            all_results.extend(rows)

        all_results.sort(key=lambda r: r.get("_score", 0.0), reverse=True)
        return all_results[:top_k]