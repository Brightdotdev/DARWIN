from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from src.core.Storage.StoreUtils import read_json, write_json
from src.core.Logger.DarwinLogger import darwin, LogType


class ArtifactBM25:
    """BM25 index I/O and deletion."""

    def has_bm25(self, filename: str) -> bool:
        if self._resolve_hash(filename) is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return False
        return self._artifacts.has_artifact(self._resolve_hash(filename), "bm25")

    def save_bm25(self, filename: str, data: Any) -> None:
        if isinstance(data, dict):
            tokens = data.get("tokens")
            doc_lengths = data.get("doc_lengths")
            doc_freqs = data.get("doc_freqs")
            avgdl = data.get("avgdl")
            chunk_count = data.get("chunk_count", len(tokens) if tokens else 0)
            vocab_size = data.get("vocab_size", len(doc_freqs) if doc_freqs else 0)
            threshold = data.get("threshold", 0.0)
            idf_smoothing = data.get("idf_smoothing", 0.0)
            domain_stopwords = data.get("domain_stopwords", [])
            precomputed_idfs = data.get("precomputed_idfs")
        else:
            tokens = data.tokens
            doc_lengths = data.doc_lengths
            doc_freqs = data.doc_freqs
            avgdl = data.avgdl
            chunk_count = getattr(data, "chunk_count", len(tokens) if tokens else 0)
            vocab_size = getattr(data, "vocab_size", len(doc_freqs) if doc_freqs else 0)
            threshold = getattr(data, "threshold", 0.0)
            idf_smoothing = getattr(data, "idf_smoothing", 0.0)
            domain_stopwords = getattr(data, "domain_stopwords", [])
            precomputed_idfs = getattr(data, "precomputed_idfs", None)

        if not tokens or not doc_lengths:
            darwin.log(f"No BM25 data to save for {filename}", LogType.ERROR)
            return None

        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return None
        self._artifacts.ensure_dir(h)

        npz_path = self.paths.bm25_path(h)
        meta_path = self.paths.bm25_meta_path(h)
        tokens_path = self.paths.bm25_tokens_path(h)

        npz_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        tokens_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(npz_path, doc_lengths=np.array(doc_lengths, dtype=np.int32))

        tokens_records = [{"tokens": list(tok_list)} for tok_list in tokens]
        tokens_table = pa.Table.from_pylist(tokens_records)
        pq.write_table(tokens_table, tokens_path, compression="zstd")

        write_json(meta_path, {
            "avgdl": avgdl,
            "chunk_count": chunk_count,
            "vocab_size": vocab_size,
            "threshold": threshold,
            "idf_smoothing": idf_smoothing,
            "domain_stopwords": list(domain_stopwords),
            "doc_freqs": doc_freqs,
            "precomputed_idfs": precomputed_idfs or {},
        }, compact=True)

        self._meta.update(h, "bm25")
        self._contents.set_stage(h, "bm25")
        self._cache.invalidate(f"bm25:{h}")

        darwin.log(f"Saved BM25 index: {npz_path.name} + {tokens_path.name}", LogType.SUCCESS)

    def load_bm25(self, filename: str) -> dict | None:
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return None
        cache_key = f"bm25:{h}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        npz_path = self.paths.bm25_path(h)
        meta_path = self.paths.bm25_meta_path(h)
        tokens_path = self.paths.bm25_tokens_path(h)

        if not npz_path.exists() or not meta_path.exists() or not tokens_path.exists():
            return None

        try:
            import numpy as np
            raw = np.load(npz_path, allow_pickle=False)
            meta = read_json(meta_path) or {}

            tokens_df = pl.read_parquet(tokens_path)
            tokens = tokens_df["tokens"].to_list()

            result = {
                "tokens": tokens,
                "doc_freqs": meta.get("doc_freqs", {}),
                "avgdl": meta.get("avgdl", 0.0),
                "doc_lengths": raw["doc_lengths"].tolist(),
                "domain_stopwords": set(meta.get("domain_stopwords", [])),
                "threshold": meta.get("threshold", 0.0),
                "chunk_count": meta.get("chunk_count", len(tokens)),
                "vocab_size": meta.get("vocab_size", 0),
                "idf_smoothing": meta.get("idf_smoothing", 0.0),
                "precomputed_idfs": meta.get("precomputed_idfs") or None,
            }

            self._cache.put(cache_key, result)
            return result
        except Exception as e:
            darwin.log(f"Failed to load BM25 for {filename}: {e}", LogType.ERROR)
            return None

    def delete_bm25(self, filename: str) -> bool:
        """Remove BM25 artifacts and clear the registry flag."""
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return False
        removed = False
        for p in (self.paths.bm25_path(h), self.paths.bm25_meta_path(h), self.paths.bm25_tokens_path(h)):
            if p.exists():
                p.unlink()
                removed = True
        self._contents.set_stage(h, "bm25", complete=False)
        self._cache.invalidate(f"bm25:{h}")
        return removed