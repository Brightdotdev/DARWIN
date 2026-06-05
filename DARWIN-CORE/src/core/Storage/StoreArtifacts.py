"""
src/core/Storage/StoreArtifacts.py
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

    
from src.core.Storage.PathResolverMixin import PathResolverMixin
from src.core.Storage.StoreUtils import (
    reconstruct_records,
    read_json,
    write_json,
)
from src.core.Logger.DarwinLogger import darwin, LogType



def normalize_record(obj: Any) -> dict[str, Any]:
    """
    Convert any object to a plain dict without importing Chunk, dataclasses,
    pydantic, or attrs.  Duck-typing keeps the storage layer fully decoupled.

    FIX: Added handling for objects with __slots__ (no __dict__) by iterating
    over all slot names and extracting values.
    """
    if isinstance(obj, dict):
        return dict(obj)
    
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()

    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()

    # Pydantic v1
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()

    # Dataclass (duck-typed via __dataclass_fields__)
    dc_fields = getattr(obj, "__dataclass_fields__", None)
    if dc_fields is not None:
        return {
            k: getattr(obj, k)
            for k in dc_fields
            if not k.startswith("_")
        }

    # attrs (duck-typed via __attrs_attrs__)
    attrs_attrs = getattr(obj, "__attrs_attrs__", None)
    if attrs_attrs is not None:
        return {
            a.name: getattr(obj, a.name)
            for a in attrs_attrs
        }

    # FIX: Handle objects with __slots__ (no __dict__)
    if hasattr(obj, "__slots__"):
        slots = obj.__slots__
        # __slots__ can be a string or a list/tuple of strings
        if isinstance(slots, str):
            slots = [slots]
        result = {}
        for slot in slots:
            if not slot.startswith("_"):
                try:
                    result[slot] = getattr(obj, slot)
                except AttributeError:
                    pass
        if result:
            return result

    # Generic Python object with __dict__
    if hasattr(obj, "__dict__"):
        return vars(obj)

    raise TypeError(
        f"Cannot normalize object of type {type(obj).__name__} to dict. "
        "Expected dict, dataclass, pydantic model, attrs, or object with __dict__ or __slots__."
    )


def _deep_serialize(obj: Any) -> Any:
    """Recursively convert any object to a JSON‑serializable type."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_deep_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _deep_serialize(v) for k, v in obj.items()}
    try:
        # Try duck‑typing with normalize_record (handles dataclasses, etc.)
        return _deep_serialize(normalize_record(obj))
    except TypeError:
        # Last resort: string representation
        return str(obj)


def records_to_table(records: list[dict[str, Any]]) -> pa.Table:
    """
    Convert plain dicts to an Arrow Table.
    If schema inference fails, convert **every** value to a JSON string.
    """
    if not records:
        return pa.Table.from_pydict({})
    try:
        return pa.Table.from_pylist(records)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        darwin.log(
            f"Parquet schema inference failed ({exc}); coercing ALL fields to JSON strings",
            LogType.WARNING,
        )

        def to_json_string(obj: Any) -> str:
            return json.dumps(_deep_serialize(obj))

        normalized = [{k: to_json_string(v) for k, v in record.items()} for record in records]
        return pa.Table.from_pylist(normalized)


def table_to_records(table: pa.Table) -> list[dict[str, Any]]:
    """Arrow Table → list[dict].  Symmetric with records_to_table."""
    return table.to_pylist()




# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Markdown
# ═══════════════════════════════════════════════════════════════════════════

class MarkdownMixin(PathResolverMixin):

    def has_markdown(self, filename: str) -> bool:
        return self._has_artifact(filename, "markdown")

    def has_display_markdown(self, filename: str) -> bool:
        return self._has_artifact(filename, "display-markdown")

    def save_markdown(self, filename: str, doc: Any) -> None:
        # Duck typing: accept ParsedDocument-like object or plain dict
        if isinstance(doc, dict):
            error = doc.get("error")
            display_text = doc.get("display_text", "")
            embedding_text = doc.get("embedding_text", "")
        else:
            error = getattr(doc, "error", None)
            display_text = getattr(doc, "display_text", "")
            embedding_text = getattr(doc, "embedding_text", "")

        if error:
            darwin.log(f"Skipping markdown save for {filename}: {error}", LogType.INFO)
            return

        d = self._dir_for(filename)
        if d is None:
            return

        md_path = self.paths.markdown_path(self._hash_for(filename))
        display_md_path = self.paths.display_markdown_path(self._hash_for(filename))



        try:
            display_md_path.write_text(display_text, encoding="utf-8")
            md_path.write_text(embedding_text, encoding="utf-8")

            # Schema-agnostic metadata extraction (no dataclasses.asdict)
            doc_dict = normalize_record(doc)
            skip_fields = {"text", "error", "filename", "images", "embedding_text", "display_text"}
            extras = {k: v for k, v in doc_dict.items() if k not in skip_fields}

            self._update_meta(filename, "markdown", extras=extras)

            record = self._record_for(filename)
            if record:
                try:
                    self.registry.update_pipeline(
                        record.last_seen_path or filename,
                        "markdown"
                    )
                except Exception as reg_err:
                    darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)

            self.invalidate(filename)
            darwin.log(f"Saved markdown for {filename}", LogType.SUCCESS)
        except Exception as e:
            darwin.log(f"Failed to save markdown for {filename}: {e}", LogType.ERROR)

    def load_markdown(self, filename: str) -> str | None:
        
        md_path = self.paths.markdown_path(self._hash_for(filename))

        if not md_path.exists():
            darwin.log(f"Markdown not found for {filename}", LogType.ERROR)
            return None
        try:
            return md_path.read_text(encoding="utf-8")
        except Exception as e:
            darwin.log(f"Error reading markdown for {filename}: {e}", LogType.ERROR)
            return None


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Records 
# ═══════════════════════════════════════════════════════════════════════════


class RecordsMixin(PathResolverMixin):
    """
    Parquet-backed schema-agnostic record storage.
    All chunk text and metadata live in chunks.parquet — the single source of truth.

    Filter syntax (query_records & reconstruct_records):
    ─────────────────────────────────────────────────────
    Exact match:           page=1
    Multiple values:       page=[1, 2, 3]
    Range (inclusive):     page__range=(5, 10)
    Greater than:          page__gt=5
    Greater/equal:         page__gte=5
    Less than:             page__lt=10
    Less/equal:            page__lte=10
    String contains:       heading__contains="Intro"
    String starts with:    heading__startswith="Chapter"
    String ends with:      heading__endswith="Summary"
    In list:               page__in=[1, 3, 5]
    Not equal:             page__ne=0
    Raw Polars expr:       page=lambda col: col.is_between(5, 10)

    Combined filters are AND-ed together.
    """

    def has_records(self, filename: str) -> bool:
        return self._has_artifact(filename, "chunks")

    def save_records(self, filename: str, records: list[Any]) -> None:
        if not records:
            darwin.log(f"No records to save for {filename}", LogType.ERROR)
            return

        d = self._dir_for(filename)
        if d is None:
            return

        dicts = [normalize_record(r) for r in records]
        table = records_to_table(dicts)

        path = self.paths.chunks_path(self._hash_for(filename))

        pq.write_table(
            table,
            path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )

        extras = {"chunk_count": len(records)}
        self._update_meta(filename, "chunks", extras=extras)

        record = self._record_for(filename)
        if record:
            try:
                self.registry.update_pipeline(
                    record.last_seen_path or filename,
                    "chunks"
                )
            except Exception as reg_err:
                darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)

        self.invalidate(filename)
        darwin.log(f"Records saved: {path.name} ({len(records)} rows)", LogType.SUCCESS)

    def load_records(self, filename: str, columns: list[str] | None = None) -> pa.Table:
        """
        Load records as an Arrow Table.  Partial column projection avoids reading
        unused columns from disk — critical for large corpora with wide schemas.
        """
        cache_key = f"records:{filename}"
        if hasattr(self, "_artifact_cache") and cache_key in self._artifact_cache:
            cached = self._artifact_cache[cache_key]
            if columns is None:
                return cached
            available = cached.column_names
            if set(columns).issubset(set(available)):
                return cached.select(columns)


        path = self.paths.chunks_path(self._hash_for(filename))

        if not path.exists():
            return pa.table({})

        table = pq.read_table(path, columns=columns)

        if hasattr(self, "_artifact_cache") and columns is None:
            self._artifact_cache[cache_key] = table
        return table

    def _parse_filter_expr(self, col_name: str, value: Any) -> pl.Expr:
        """
        Convert a filter kwarg into a Polars expression.

        Syntax:
            page=1              → col == 1
            page=[1,2]          → col.is_in([1,2])
            page__range=(5,10)  → col.is_between(5, 10, closed="both")
            page__gt=5          → col > 5
            page__gte=5         → col >= 5
            page__lt=10         → col < 10
            page__lte=10        → col <= 10
            heading__contains="x" → col.str.contains("x")
            heading__startswith="x" → col.str.starts_with("x")
            heading__endswith="x"   → col.str.ends_with("x")
            page__in=[1,2]      → col.is_in([1,2])
            page__ne=0          → col != 0
            page=lambda c: c>5  → raw lambda (passthrough)
        """
        col = pl.col(col_name)

        # Raw lambda — pass through directly
        if callable(value) and not isinstance(value, (list, tuple, set)):
            return value(col)

        # List/tuple/set → IN clause
        if isinstance(value, (list, tuple, set)):
            return col.is_in(list(value))

        # Exact match
        return col == value

    def _build_filter_exprs(self, **filters) -> list[pl.Expr]:
        """Parse all filter kwargs into Polars expressions."""
        exprs: list[pl.Expr] = []

        for key, value in filters.items():
            # Check for operator suffix: page__range, heading__contains, etc.
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
                    raise ValueError(f"Unknown filter operator: {op!r} in {key!r}")
            else:
                # No operator — exact match, list, or raw lambda
                exprs.append(self._parse_filter_expr(key, value))

        return exprs

    def query_records(
        self,
        filename: str,
        columns: list[str] | None = None,
        **filters,
    ) -> list[dict[str, Any]]:
        """
        Filter records with expressive syntax. Only matching rows are materialized.

        Examples:
            store.query_records("doc.pdf", page=1)
            store.query_records("doc.pdf", page__range=(5, 10))
            store.query_records("doc.pdf", index__gte=100, index__lte=200)
            store.query_records("doc.pdf", heading__contains="Introduction")
            store.query_records("doc.pdf", page__in=[1, 3, 5])
            store.query_records("doc.pdf", overlap_direction="left")
        """
        
        path = self.paths.chunks_path(self._hash_for(filename))

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

    def iter_records(
        self,
        filename: str,
        columns: list[str] | None = None,
        batch_size: int = 1000,
    ) -> Iterator[dict[str, Any]]:
        """
        Memory-efficient iterator over Parquet batches.
        """
        path = self.paths.chunks_path(self._hash_for(filename))
        if not path.exists():
            return iter([])

        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        scanner = dataset.scanner(columns=columns, batch_size=batch_size)
        for batch in scanner.to_batches():
            yield from batch.to_pylist()

    def reconstruct_records(
        self,
        filename: str,
        cls: Any = None,
        columns: list[str] | None = None,
        **filters,
    ) -> list[Any]:
        """
        Fetch rows (optionally filtered) and reconstruct into objects.

        Supports the same filter syntax as query_records.
        NOTE: If cls requires fields not in `columns`, pass columns=None
        to load all fields, or make those fields optional in your dataclass.
        """
        rows = self.query_records(filename, columns=columns, **filters)
        return reconstruct_records(rows, cls=cls)




# ═══════════════════════════════════════════════════════════════════════════
# Mixin: BM25
# ═══════════════════════════════════════════════════════════════════════════

class BM25Mixin(PathResolverMixin):
    """
    BM25 storage split across three files to respect the "no object arrays" rule:
    * bm25.npz        — numerical arrays (doc_lengths as int32)
    * bm25_tokens.parquet — tokens as a list<string> column (one row per chunk)
    * bm25_meta.json  — scalar metadata and doc_freqs dict
    """

    def has_bm25(self, filename: str) -> bool:
        return self._has_artifact(filename, "bm25")

    def save_bm25(self, filename: str, data) -> None:
        """Accepts dict or duck-typed BM25Data object."""
        # Duck-type extraction — storage layer never imports BM25Data
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
            return

        d = self._dir_for(filename)
        if d is None:
            return

        try:
            
            
            npz_path = self.paths.bm25_path(self._hash_for(filename))
            meta_path = self.paths.bm25_meta_path(self._hash_for(filename))
            tokens_path = self.paths.bm25_tokens_path(self._hash_for(filename))


            np.savez_compressed(
                npz_path,
                doc_lengths=np.array(doc_lengths, dtype=np.int32),
            )
            

            tokens_records = [{"tokens": list(tok_list)} for tok_list in tokens]
            tokens_table = records_to_table(tokens_records)
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

            self._update_meta(filename, "bm25")
            record = self._record_for(filename)
            if record:
                try:
                    self.registry.update_pipeline(
                        record.last_seen_path or filename,
                        "bm25"
                    )
                except Exception as reg_err:
                    darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)

            self.invalidate(filename)
            darwin.log(f"Saved BM25 index: {npz_path.name} + {tokens_path.name}", LogType.SUCCESS)

        except Exception as e:
            darwin.log(f"Failed to save BM25 for {filename}: {e}", LogType.ERROR)

    def load_bm25(self, filename: str) -> dict | None:
        """
        Load BM25 data as a plain dict.
        Reconstruction into BM25Data objects happens outside the storage layer.
        """
        cache_key = f"bm25:{filename}"
        if hasattr(self, "_artifact_cache") and cache_key in self._artifact_cache:
            return self._artifact_cache[cache_key]

        d = self._dir_for(filename)
        if d is None:
            return None

        npz_path = self.paths.bm25_path(self._hash_for(filename))
        meta_path = self.paths.bm25_meta_path(self._hash_for(filename))
        tokens_path = self.paths.bm25_tokens_path(self._hash_for(filename))

        if not npz_path.exists() or not meta_path.exists() or not tokens_path.exists():
            return None

        try:
            # Security: allow_pickle=False is mandatory for untrusted local files
            raw = np.load(npz_path, allow_pickle=False)
            meta = read_json(meta_path)

            # Load tokens via columnar parquet — fast and memory-efficient
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

            if hasattr(self, "_artifact_cache"):
                self._artifact_cache[cache_key] = result
            return result

        except Exception as e:
            darwin.log(f"Failed to load BM25 for {filename}: {e}", LogType.ERROR)
            return None


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Stopwords
# ═══════════════════════════════════════════════════════════════════════════

class StopwordsMixin(PathResolverMixin):

    def has_stopwords(self, filename: str) -> bool:
        return self._has_artifact(filename, "stopwords")

    def save_stopwords(self, filename: str, stopwords: set[str]) -> None:
        if not stopwords:
            darwin.log(f"No stopwords to save for {filename}", LogType.ERROR)
            return
        d = self._dir_for(filename)
        if d is None:
            darwin.log(f"Cannot save stopwords for {filename}: source file not found", LogType.ERROR) 
            return
        write_json(self.paths.stopwords_path(self._hash_for(filename)), {"stopwords": sorted(stopwords)})
        self._update_meta(filename, "stopwords")
        record = self._record_for(filename)
        if record:
            try:
                self.registry.update_pipeline(record.last_seen_path or filename, "stopwords")
            except Exception as reg_err:
                darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)
        self.invalidate(filename)

    def load_stopwords(self, filename: str) -> set[str]:
        path = self.paths.stopwords_path(self._hash_for(filename))
        if not path.exists():
            darwin.log(f"Stopwords not found for {filename}", LogType.ERROR)
            return set()
        return set(read_json(path).get("stopwords", []))

