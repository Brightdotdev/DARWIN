from __future__ import annotations

from typing import Any

import numpy as np

from src.core.Logger.DarwinLogger import darwin, LogType


def reconstruct_records(rows: list[dict], cls: Any) -> list[Any]:
    """
    Reconstruct dataclass/class instances from dictionary rows.
    If cls is None or reconstruction fails, return rows as-is.
    """
    if cls is None:
        return rows

    try:
        # Check if cls is a dataclass or has a from_dict method
        if hasattr(cls, "from_dict"):
            return [cls.from_dict(row) for row in rows]
        elif hasattr(cls, "__dataclass_fields__"):
            # Dataclass: instantiate with field values
            results = []
            for row in rows:
                # Filter to only fields that exist in the dataclass
                field_names = set(cls.__dataclass_fields__.keys())
                filtered = {k: v for k, v in row.items() if k in field_names}
                results.append(cls(**filtered))
            return results
        else:
            # Try direct instantiation with all fields
            return [cls(**row) for row in rows]
    except Exception as e:
        darwin.warning(f"Record reconstruction failed: {e}")
        return rows


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
            # Try both 'index' and 'chunk_index' fields
            idx = row.get("index")
            if idx is None:
                idx = row.get("chunk_index")

            if idx is not None and idx in score_map:
                row["_score"] = score_map[idx]
            else:
                row["_score"] = 0.0
                darwin.warning(
                    f"Row index {idx} not in score map for {filename}; "
                    f"available indices: {list(score_map.keys())[:10]}..."
                )

        if cls is not None:
            try:
                return reconstruct_records(rows, cls=cls)
            except Exception as e:
                darwin.warning(f"Record reconstruction failed for {filename}: {e}")
                # Fall through to return raw rows
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
        # Avoid copying if already float32; use astype with copy=False when possible
        v = vectors.astype(np.float32, copy=False)
        q = query_vec.astype(np.float32).ravel()

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

        # Map positions back to original chunk indices
        if index_map is None:
            darwin.warning(
                f"No index map for {filename}; using position-based mapping, "
                "which may be incorrect after reordering or deletion"
            )
            chunk_indices = [int(pos) for pos in top_positions.tolist()]
        else:
            # index_map is a list where position -> original index
            try:
                chunk_indices = [index_map[pos] for pos in top_positions.tolist()]
            except IndexError as e:
                darwin.error(
                    f"Index map lookup failed for {filename}: {e}; "
                    "falling back to position mapping"
                )
                chunk_indices = [int(pos) for pos in top_positions.tolist()]

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
        error_msg = None

        if not force_numpy and self.has_vector(filename):
            try:
                darwin.info(f"Searching USearch index for {filename}")
                rows = self.retrieve_by_vector(
                    filename, query_vec, k=k, columns=columns, cls=None
                )
                if rows:
                    darwin.info(
                        f"USearch returned {len(rows)} results for {filename}"
                    )
            except Exception as e:
                error_msg = f"USearch failed for {filename}: {e}"
                darwin.warning(error_msg)

        if not rows and self.has_embeddings(filename):
            try:
                darwin.warning(
                    f"Falling back to numpy search for {filename}"
                )
                rows = self.retrieve_by_numpy(
                    filename, query_vec, k=k, columns=columns, cls=None
                )
                if rows:
                    darwin.info(
                        f"Numpy search returned {len(rows)} results for {filename}"
                    )
            except Exception as e:
                error_msg = f"Numpy fallback failed for {filename}: {e}"
                darwin.warning(error_msg)

        if not rows and error_msg:
            darwin.warning(f"No results for {filename}: {error_msg}")
        elif not rows:
            darwin.info(f"No results found for {filename}")

        # Filter by min_score and ensure _score exists
        filtered = []
        for r in rows:
            score = r.get("_score", 0.0)
            if score >= min_score:
                filtered.append(r)
            else:
                darwin.debug(
                    f"Result filtered out: score {score} below threshold {min_score} for {filename}"
                )

        return filtered

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
        # Fetch enough candidates per file to meet top_k after merging
        per_doc_k = max(1, top_k // len(unique) + 1)
        all_results: list[dict] = []
        errors: list[str] = []

        for filename in unique:
            try:
                rows = self.search_single(
                    query_vec,
                    filename,
                    k=per_doc_k,
                    min_score=min_score,
                    columns=columns,
                    force_numpy=force_numpy,
                )
                all_results.extend(rows)
                if rows:
                    darwin.info(f"Got {len(rows)} results from {filename}")
            except Exception as e:
                error_msg = f"Search failed for {filename}: {e}"
                errors.append(error_msg)
                darwin.error(error_msg)

        if errors:
            darwin.warning(
                f"{len(errors)} file(s) failed during multi-file search"
            )

        # Sort by _score (highest first)
        all_results.sort(key=lambda r: r.get("_score", 0.0), reverse=True)

        results = all_results[:top_k]
        darwin.info(
            f"Multi-search: {len(results)} results across {len(unique)} files"
        )
        return results
