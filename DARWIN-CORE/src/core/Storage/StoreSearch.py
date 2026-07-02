from __future__ import annotations

from typing import Any

import numpy as np

from src.core.Logger.DarwinLogger import darwin, LogType


class StoreSearch:
    """Unified retrieval: vector, numpy fallback, and multi-file search.

    Intended as a mixin for DarwinStore. Requires the host class to implement:
    - query_records(filename, columns, **filters)
    - search_vector(filename, query_vec, k)
    - load_embeddings(filename, mmap)
    - load_embedding_index_map(filename)
    - has_vector(filename)
    - has_embeddings(filename)
    """

    def _finalize_matches(
        self,
        filename: str,
        matches: list[tuple[int, float]],
        columns: list[str] | None,
        cls: Any,
    ) -> list[Any]:
        """Shared tail: query records, attach scores, optional reconstruct."""
        if not matches:
            return []

        indices, scores = zip(*matches)
        rows = self.query_records(filename, columns=columns, index__in=list(indices))
        score_map = {idx: sc for idx, sc in matches}
        for row in rows:
            row["_score"] = score_map.get(row.get("index", -1), 0.0)

        if cls is not None:
            try:
                from src.core.Storage.StoreUtils import reconstruct_records
                return reconstruct_records(rows, cls=cls)
            except Exception:
                # If reconstruction fails, fall back to raw dicts
                pass
        return rows

    def retrieve_by_vector(
        self,
        filename: str,
        query_vec: np.ndarray,
        k: int = 10,
        columns: list[str] | None = None,
        cls: Any = None,
    ) -> list[Any]:
        matches = self.search_vector(filename, query_vec, k=k)
        return self._finalize_matches(filename, matches, columns, cls)

    def retrieve_by_numpy(
        self,
        filename: str,
        query_vec: np.ndarray,
        k: int = 10,
        columns: list[str] | None = None,
        cls: Any = None,
    ) -> list[Any]:
        vectors = self.load_embeddings(filename, mmap=True)
        if vectors is None or vectors.shape[0] == 0:
            return []

        index_map = self.load_embedding_index_map(filename)

        # Cosine similarity via normalized dot product (matches USearch "cos" metric)
        v = vectors.astype(np.float32)
        q = query_vec.astype(np.float32)
        v_norms = np.linalg.norm(v, axis=1, keepdims=True)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        v_unit = np.divide(v, v_norms, out=np.zeros_like(v), where=v_norms != 0)
        q_unit = q / q_norm
        similarities = np.dot(v_unit, q_unit)

        n = len(similarities)
        if k >= n:
            top_positions = np.argsort(similarities)[::-1][:k]
        else:
            top_positions = np.argpartition(similarities, -k)[-k:]
            top_positions = top_positions[np.argsort(similarities[top_positions])[::-1]]

        chunk_indices = [index_map[pos] if index_map else int(pos) for pos in top_positions.tolist()]
        scores = similarities[top_positions].tolist()
        matches = list(zip(chunk_indices, scores))

        return self._finalize_matches(filename, matches, columns, cls)

    def search_single(
        self,
        query_vec: np.ndarray,
        filename: str,
        k: int = 10,
        min_score: float = 0.0,
        columns: list[str] | None = None,
        force_numpy: bool = False,
    ) -> list[dict[str, Any]]:
        rows: list[dict] = []
        if not force_numpy and self.has_vector(filename):
            try:
                rows = self.retrieve_by_vector(filename, query_vec, k=k, columns=columns, cls=None)
            except Exception as e:
                darwin.log(f"USearch failed for {filename}: {e}", LogType.WARNING)

        if not rows and self.has_embeddings(filename):
            try:
                rows = self.retrieve_by_numpy(filename, query_vec, k=k, columns=columns, cls=None)
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
        if not filenames:
            return []
        unique = list(dict.fromkeys(filenames))
        per_doc_k = max(5, top_k // len(unique) + 1)
        all_results: list[dict] = []

        for filename in unique:
            rows = self.search_single(
                query_vec, filename, k=per_doc_k,
                min_score=min_score, columns=columns,
                force_numpy=force_numpy,
            )
            all_results.extend(rows)

        all_results.sort(key=lambda r: r.get("_score", 0.0), reverse=True)
        return all_results[:top_k]
