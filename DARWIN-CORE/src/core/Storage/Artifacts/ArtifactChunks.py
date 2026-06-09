from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from src.core.Storage.StoreUtils import normalize_record, reconstruct_records
from src.core.Logger.DarwinLogger import darwin, LogType


class ArtifactChunks:
    """Parquet chunk records I/O, querying, and deletion."""

    def has_records(self, filename: str) -> bool:
        if self._resolve_hash(filename) is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return False
        return self._artifacts.has_artifact(self._resolve_hash(filename), "chunks")

    def save_records(self, filename: str, records: list[Any]) -> None:
        if not records:
            darwin.log(f"No records to save for {filename}", LogType.ERROR)
            return

        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return None
        self._artifacts.ensure_dir(h)

        dicts = [normalize_record(r) for r in records]
        table = pa.Table.from_pylist(dicts)
        path = self.paths.chunks_path(h)
        path.parent.mkdir(parents=True, exist_ok=True)

        pq.write_table(table, path, compression="zstd", use_dictionary=True, write_statistics=True)

        self._meta.update(h, "chunks", extras={"chunk_count": len(records)})
        self._contents.set_stage(h, "chunks")
        self._contents.update_meta(h, chunk_count=len(records))
        self._cache.invalidate(f"records:{h}")

        darwin.log(f"Records saved: {path.name} ({len(records)} rows)", LogType.SUCCESS)

    def load_records(self, filename: str, columns: list[str] | None = None) -> pa.Table:
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return pa.table({})
        cache_key = f"records:{h}"
        cached = self._cache.get(cache_key)
        if cached is not None and columns is None:
            return cached

        path = self.paths.chunks_path(h)
        if not path.exists():
            return pa.table({})

        table = pq.read_table(path, columns=columns)
        if columns is None:
            self._cache.put(cache_key, table)
        return table

    def query_records(self, filename: str, columns: list[str] | None = None, **filters) -> list[dict[str, Any]]:
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return []
        path = self.paths.chunks_path(h)
        if not path.exists():
            return []

        lf = pl.scan_parquet(path)
        if columns:
            lf = lf.select(columns)

        exprs = self._build_filter_exprs(**filters)
        if exprs:
            combined = exprs[0]
            for e in exprs[1:]:
                combined = combined & e
            lf = lf.filter(combined)

        return lf.collect().to_dicts()

    def iter_records(self, filename: str, columns: list[str] | None = None, batch_size: int = 1000) -> Iterator[dict[str, Any]]:
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return iter([])
        path = self.paths.chunks_path(h)
        if not path.exists():
            return iter([])

        import pyarrow.dataset as ds
        dataset = ds.dataset(path, format="parquet")
        scanner = dataset.scanner(columns=columns, batch_size=batch_size)
        for batch in scanner.to_batches():
            yield from batch.to_pylist()

    def reconstruct_records(self, filename: str, cls: Any = None, columns: list[str] | None = None, **filters) -> list[Any]:
        rows = self.query_records(filename, columns=columns, **filters)
        return reconstruct_records(rows, cls=cls)

    def _build_filter_exprs(self, **filters) -> list[pl.Expr]:
        exprs: list[pl.Expr] = []
        for key, value in filters.items():
            if "__" in key:
                field, op = key.rsplit("__", 1)
                col = pl.col(field)
                if op == "range":
                    start, end = value
                    exprs.append(col.is_between(start, end, closed="both"))
                elif op == "gt":
                    exprs.append(col > value)
                elif op == "gte":
                    exprs.append(col >= value)
                elif op == "lt":
                    exprs.append(col < value)
                elif op == "lte":
                    exprs.append(col <= value)
                elif op == "contains":
                    exprs.append(col.str.contains(value))
                elif op == "startswith":
                    exprs.append(col.str.starts_with(value))
                elif op == "endswith":
                    exprs.append(col.str.ends_with(value))
                elif op == "in":
                    exprs.append(col.is_in(list(value)))
                elif op == "ne":
                    exprs.append(col != value)
                else:
                    raise ValueError(f"Unknown filter operator: {op!r}")
            else:
                col = pl.col(key)
                if callable(value) and not isinstance(value, (list, tuple, set)):
                    exprs.append(value(col))
                elif isinstance(value, (list, tuple, set)):
                    exprs.append(col.is_in(list(value)))
                else:
                    exprs.append(col == value)
        return exprs

    def delete_records(self, filename: str) -> bool:
        """Remove chunk parquet and clear the registry flag."""
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return False
        path = self.paths.chunks_path(h)
        removed = False
        if path.exists():
            path.unlink()
            removed = True
        self._contents.set_stage(h, "chunks", complete=False)
        self._contents.update_meta(h, chunk_count=0)
        self._cache.invalidate(f"records:{h}")
        return removed