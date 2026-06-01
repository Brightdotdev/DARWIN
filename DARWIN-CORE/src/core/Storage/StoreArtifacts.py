"""
src/core/Storage/StoreArtifacts.py
-----------------------------------
Production-grade local-first RAG storage subsystem.

Design principles
-----------------
* Schema-agnostic: storage never imports Chunk, dataclasses, pydantic, or attrs.
  Normalization and reconstruction are duck-typed.
* Zero duplication: chunk text/metadata lives ONLY in chunks.parquet.
  Vectors live ONLY in embeddings.f16.npy.
* Mmap-friendly: embeddings load with mmap_mode="r" for near-zero RAM overhead
  until a vector is actually touched by the CPU.
* Content-addressed: all artifacts live under store/<file_hash>/.
* Lazy reconstruction: vector search returns row indices; Parquet rows are fetched
  on demand; object reconstruction happens outside the storage layer.
* Columnar I/O: Parquet + zstd gives fast partial-column reads and small disk
  footprint. Polars lazy scans keep large-corpora queries memory-efficient.
* Backward migration: old chunks.json, embeddings.npz, graph.json, and wiki.json
  are detected and upgraded automatically on first touch.

Storage layout
--------------
store/<file_hash>/
    meta.json               # global metadata, timestamps, pipeline state, embedding_model
    chunks.parquet          # SINGLE SOURCE OF TRUTH for all chunk records
    embeddings.f16.npy      # pure float16 matrix, no metadata, no text
    vector.usearch          # ANN index; keys == row indices (0..N-1)
    vector.meta.json        # index metadata only (dimension, count, metric)
    bm25.npz                # BM25 numerical arrays (doc_lengths, etc.)
    bm25_meta.json          # BM25 scalar metadata + doc_freqs
    bm25_tokens.parquet     # BM25 tokens as list<string> column (no object arrays)
    graph/
        nodes.parquet
        edges.parquet
    wiki/
        wiki.parquet
    markdown.md
    display-markdown.md
    stopwords.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from src.core.Storage.StoreUtils import (
    file_hash,
    now,
    read_json,
    write_json,
    get_current_model_name,
)
from src.core.Logger.DarwinLogger import darwin, LogType


# ═══════════════════════════════════════════════════════════════════════════
# Artifact manifest — single source of truth for integrity checks
# ═══════════════════════════════════════════════════════════════════════════

ARTIFACT_MANIFEST = {
    "markdown": ["markdown.md", "display-markdown.md"],
    "chunks": ["chunks.parquet"],           # upgraded from chunks.json
    "vector": ["vector.usearch", "vector.meta.json"],
    "embeddings": ["embeddings.f16.npy"],   # upgraded from *.npz
    "bm25": ["bm25.npz", "bm25_meta.json", "bm25_tokens.parquet"],
    "stopwords": ["stopwords.json"],
    "graph": ["graph/nodes.parquet", "graph/edges.parquet"],  # upgraded from graph.json
    "graph_vis": ["graph/graph.html"],
    "wiki": ["wiki/wiki.parquet"],          # upgraded from wiki.json
    "meta" : ["meta.json"],
}


# ═══════════════════════════════════════════════════════════════════════════
# Schema-agnostic normalization helpers
# ═══════════════════════════════════════════════════════════════════════════


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


def reconstruct_records(rows: list[dict[str, Any]], cls: Any = None) -> list[Any]:
    """
    Reconstruct objects from plain dict rows using duck typing.
    Runtime reconstruction happens OUTSIDE the storage layer, but this helper
    is provided for convenience in the retrieval pipeline.
    """
    if cls is None or not rows:
        return rows

    # Pydantic v2
    if hasattr(cls, "model_validate") and callable(cls.model_validate):
        return [cls.model_validate(row) for row in rows]

    # Pydantic v1
    if hasattr(cls, "parse_obj") and callable(cls.parse_obj):
        return [cls.parse_obj(row) for row in rows]

    # Dataclass
    dc_fields = getattr(cls, "__dataclass_fields__", None)
    if dc_fields is not None:
        valid = set(dc_fields)
        return [
            cls(**{k: v for k, v in row.items() if k in valid})
            for row in rows
        ]

    # attrs
    attrs_attrs = getattr(cls, "__attrs_attrs__", None)
    if attrs_attrs is not None:
        valid = {a.name for a in attrs_attrs}
        return [
            cls(**{k: v for k, v in row.items() if k in valid})
            for row in rows
        ]

    # Generic fallback: inspect __init__ signature to avoid passing unknown kwargs
    import inspect
    try:
        sig = inspect.signature(cls.__init__)
        valid = {
            p.name for p in sig.parameters.values()
            if p.name != "self" and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        return [
            cls(**{k: v for k, v in row.items() if k in valid})
            for row in rows
        ]
    except (ValueError, TypeError):
        pass

    # Last resort: pass all kwargs and hope for the best
    return [cls(**row) for row in rows]


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
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_model_name(name: str) -> str:
    """Convert any model identifier to the canonical model_id."""
    from src.core.Models.ModelRegistry import REGISTRY

    if name in REGISTRY:
        return REGISTRY[name].model_id

    id_to_key = {spec.model_id: key for key, spec in REGISTRY.items()}
    if name in id_to_key:
        return name

    return name


def _get_current_model_dimensions() -> int:
    """Expected embedding dimensions for the active model. Safe fallback."""
    try:
        from src.core.Models import ModelManager, ModelCategory
        loader = ModelManager.get_loader()
        key = loader.get_model_key(ModelCategory.EMBEDDING)
        dims = {
            "intfloat/e5-small-v2": 384,
            "BAAI/bge-m3": 1024,
            "GritLM/GritLM-7B": 4096,
        }
        return dims.get(key, 0)
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# Base mixin — filename → hash → directory resolution + artifact checks
# ═══════════════════════════════════════════════════════════════════════════

class _PathResolverMixin:
    """Bridge between calling code (bare filenames) and StorePaths (file hashes)."""

    # ── Path resolution ──────────────────────────────────────────────────

    def _record_for(self, filename: str):
        """Return the FileRecord for a bare filename. Raises KeyError if missing."""
        record = self.registry._find_by_name(filename)
        if record is None:
            darwin.log(
                f"File '{filename}' is not registered. Register it before saving artifacts.",
                LogType.ERROR,
            )
            return None
        return record

    def _resolve(self, filename: str) -> tuple[str, Path]:
        """Resolve a bare filename to (file_hash, index_dir) in one registry lookup."""
        record = self._record_for(filename)
        if record is None:
            return "", None
        fh = record.file_hash
        return fh, self.paths.dir_for_hash(fh)

    def _hash_for(self, filename: str) -> str:
        """Resolve a bare filename to its file hash."""
        fh, _ = self._resolve(filename)
        return fh

    def _dir_for(self, filename: str) -> Path | None:
        """Return (and create) the index directory for a bare filename."""
        _, d = self._resolve(filename)
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def _source_path_for(self, filename: str) -> Optional[Path]:
        """Return the last-known source Path for a file (may not exist if moved)."""
        record = self._record_for(filename)
        if record and record.last_seen_path:
            return Path(record.last_seen_path)
        return None

    # ── Unified artifact existence check ─────────────────────────────────

    def _has_artifact(self, filename: str, stage: str) -> bool:
        """
        Check whether all required artifact files for a pipeline stage exist.
        Uses the ARTIFACT_MANIFEST as the single source of truth.
        """
        dir_path = self._dir_for(filename)
        if dir_path is None:
            return False

        # Embeddings: static filename in the new layout
        if stage == "embeddings":
            return (dir_path / "embeddings.f16.npy").exists()

        file_list = ARTIFACT_MANIFEST.get(stage, [])
        if not file_list:
            return True
        return all((dir_path / fname).exists() for fname in file_list)

    def validate_artifact_integrity(self, filename: str) -> dict:
        """
        Full integrity check for one file.
        """
        record = self._record_for(filename)
        if record is None:
            return {
                "filename": filename,
                "error": "Not registered",
                "is_healthy": False,
            }

        registry_done = []
        registry_missing = []
        actually_missing = []
        actually_present = []

        for stage in ARTIFACT_MANIFEST:
            registry_says_done = record.stage_completed(stage)
            files_exist = self._has_artifact(filename, stage)

            if registry_says_done:
                registry_done.append(stage)
                if not files_exist:
                    actually_missing.append(stage)
            else:
                registry_missing.append(stage)
                if files_exist:
                    actually_present.append(stage)

        return {
            "filename": filename,
            "registry_says_done": registry_done,
            "actually_missing": actually_missing,
            "registry_says_missing": registry_missing,
            "actually_present": actually_present,
            "is_healthy": len(actually_missing) == 0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Meta / freshness
# ═══════════════════════════════════════════════════════════════════════════

class MetaMixin(_PathResolverMixin):
    """Handles meta.json read/write and file freshness checks."""

    def _update_meta(self, filename: str, component: str, extras: dict | None = None) -> None:
        fh, d = self._resolve(filename)
        if d is None:
            return
        meta_path = d / "meta.json"
        meta = read_json(meta_path) or {
            "filename": filename,
            "created_at": now(),
            "updates": {},
        }

        meta.setdefault("updates", {})
        timestamp = now()
        meta["updates"][component] = timestamp
        meta["updated_at"] = timestamp

        if extras:
            for key, value in extras.items():
                if isinstance(value, dict) and isinstance(meta.get(key), dict):
                    meta[key].update(value)
                else:
                    meta[key] = value

        write_json(meta_path, meta)

    def _batch_meta_update(self, filename: str, updates: dict[str, dict]) -> None:
        """Write meta.json once for multiple component updates."""
        fh, d = self._resolve(filename)
        if d is None:
            return
        meta_path = d / "meta.json"
        meta = read_json(meta_path) or {
            "filename": filename,
            "created_at": now(),
            "updates": {},
        }
        meta.setdefault("updates", {})
        timestamp = now()

        for component, extras in updates.items():
            meta["updates"][component] = timestamp
            if extras:
                for key, value in extras.items():
                    if isinstance(value, dict) and isinstance(meta.get(key), dict):
                        meta[key].update(value)
                    else:
                        meta[key] = value

        meta["updated_at"] = timestamp
        write_json(meta_path, meta)

    def load_meta(self, filename: str) -> dict:
        return read_json(self._dir_for(filename) / "meta.json") or {}

    def is_fresh(self, filename: str) -> bool:
        fh, d = self._resolve(filename)
        if d is None:
            return False
        meta_path = d / "meta.json"
        if not meta_path.exists():
            return False

        source = self._source_path_for(filename)
        if source is None or not source.exists():
            return False

        return read_json(meta_path).get("file_hash", "") == file_hash(source)



# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Markdown
# ═══════════════════════════════════════════════════════════════════════════

class MarkdownMixin(_PathResolverMixin):

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
        md_path = d / "markdown.md"
        display_md_path = d / "display-markdown.md"

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
                        "markdown",
                        **extras,
                    )
                except Exception as reg_err:
                    darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)

            self.invalidate(filename)
            darwin.log(f"Saved markdown for {filename}", LogType.SUCCESS)
        except Exception as e:
            darwin.log(f"Failed to save markdown for {filename}: {e}", LogType.ERROR)

    def load_markdown(self, filename: str) -> str | None:
        md_path = self._dir_for(filename) / "markdown.md"
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


class RecordsMixin(_PathResolverMixin):
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

        path = d / "chunks.parquet"
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
                    "chunks",
                    **extras,
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

        path = self._dir_for(filename) / "chunks.parquet"
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
        path = self._dir_for(filename) / "chunks.parquet"
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
        path = self._dir_for(filename) / "chunks.parquet"
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
# Mixin: Embeddings (float16 .npy, zero metadata duplication)
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingsMixin(_PathResolverMixin):
    """
    Pure numerical embedding storage.
    * ONLY float16 vectors on disk.
    * NO text, NO metadata, NO chunk ids inside the .npy file.
    * Metadata lives in meta.json (single source of truth).
    * mmap_mode="r" keeps RAM overhead near zero until vectors are accessed.
    """

    def has_embeddings(self, filename: str) -> bool:
        return self._has_artifact(filename, "embeddings")

    def save_embeddings(self, filename: str, vectors: np.ndarray) -> None:
        if vectors is None or len(vectors) == 0:
            darwin.log(f"No vectors to save for {filename}", LogType.ERROR)
            return

        d = self._dir_for(filename)
        if d is None:
            return

        path = d / "embeddings.f16.npy"

        # C-contiguous float16: optimal for both disk layout and mmap paging
        vectors = np.ascontiguousarray(vectors.astype(np.float16))
        np.save(path, vectors)

        dim = vectors.shape[1]
        count = vectors.shape[0]
        model_name = _normalize_model_name(get_current_model_name())

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
                    record.last_seen_path or filename,
                    "embeddings",
                    embedding_model=model_name,
                    embedding_dimensions=dim,
                )
            except Exception as reg_err:
                darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)

        self.invalidate(filename)
        darwin.log(
            f"Embeddings saved: {path.name} ({count} vectors, {dim}d, float16)",
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
        if d is None:
            return None
        path = d / "embeddings.f16.npy"

        # ── Model compatibility check (file missing path) ─────────────────
        if not path.exists():
            meta = self.load_meta(filename)
            stored_model = meta.get("embedding_model", "unknown")
            current_model = get_current_model_name()
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

        # ── Model compatibility check (file exists path) ──────────────────
        meta = self.load_meta(filename)
        stored_model = meta.get("embedding_model", "unknown")
        current_model = get_current_model_name()
        if stored_model not in ("unknown", current_model):
            darwin.log(
                f"Embeddings for '{filename}' were built with '{stored_model}', "
                f"but current model is '{current_model}'. Using a different embedding "
                f"model breaks cosine-similarity search. Re-embedding required.",
                LogType.WARNING,
            )
            return None

        # ── Dimension sanity check ────────────────────────────────────────
        expected_dim = meta.get("embedding_dimensions")
        if expected_dim and path.stat().st_size > 0:
            # Quick header read: np.load with mmap to avoid full load for dim check
            header = np.load(path, mmap_mode="r")
            if header.shape[1] != expected_dim:
                darwin.log(
                    f"Embedding dimension mismatch on disk: expected {expected_dim}, got {header.shape[1]}.",
                    LogType.WARNING,
                )

        # mmap_mode='r' → read-only memory map.  No RAM consumed until touched.
        mode = "r" if mmap else None
        vectors = np.load(path, mmap_mode=mode)

        if hasattr(self, "_artifact_cache") and mmap:
            self._artifact_cache[cache_key] = vectors

        darwin.log(
            f"Loaded embeddings: {vectors.shape[0]} vectors, {vectors.shape[1]}d (mmap={mmap})",
            LogType.INFO,
        )
        return vectors

    def get_embedding_metadata(self, filename: str) -> dict | None:
        """Return embedding metadata from global meta.json (single source of truth)."""
        meta = self.load_meta(filename)
        return {
            "embedding_model": meta.get("embedding_model"),
            "dimensions": meta.get("embedding_dimensions"),
            "count": meta.get("embedding_count"),
            "updated_at": meta.get("updates", {}).get("embeddings"),
        }

    def is_embedding_compatible(self, filename: str) -> bool:
        try:
            v = self.load_embeddings(filename, mmap=True)
            return v is not None
        except (FileNotFoundError, OSError, ValueError):
            return False


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Vector (USearch) — row index == embedding row
# ═══════════════════════════════════════════════════════════════════════════

class VectorMixin(_PathResolverMixin):
    """
    ANN index via USearch.
    Keys are row indices (0..N-1) aligned with chunks.parquet and embeddings.f16.npy.
    No chunk_ids are stored inside the index or its metadata.
    """

    def has_vector(self, filename: str) -> bool:
        return self._has_artifact(filename, "vector")

    def save_vector(self, filename: str, vectors: np.ndarray) -> None:
        d = self._dir_for(filename)
        if d is None:
            return

        from usearch.index import Index

        n, dim = vectors.shape
        # USearch index built as f32 for broad compatibility; source vectors may be f16
        index = Index(ndim=dim, metric="cos", dtype="f32")
        keys = np.arange(n, dtype=np.int64)
        index.add(keys=keys, vectors=vectors.astype(np.float32))
        index.save(str(d / "vector.usearch"))

        # vector.meta.json: index-only metadata (no embedding model info — that's in meta.json)
        write_json(d / "vector.meta.json", {
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
                self.registry.update_pipeline(record.last_seen_path or filename, "vector")
            except Exception as reg_err:
                darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)

        self.invalidate(filename)
        darwin.log(f"Vector index saved: {n} vectors, {dim}d", LogType.SUCCESS)

    def load_vector(self, filename: str):
        cache_key = f"vector:{filename}"
        if hasattr(self, "_artifact_cache") and cache_key in self._artifact_cache:
            return self._artifact_cache[cache_key]

        d = self._dir_for(filename)
        if d is None:
            return None
        index_path = d / "vector.usearch"
        meta_path = d / "vector.meta.json"

        if not index_path.exists() or not meta_path.exists():
            return None

        # ── Model compatibility check ─────────────────────────────────────
        # The embedding model used to build the index is stored in global meta.json.
        # Searching with a different model produces meaningless cosine scores.
        global_meta = self.load_meta(filename)
        stored_model = global_meta.get("embedding_model", "unknown")
        current_model = get_current_model_name()
        if stored_model not in ("unknown", current_model):
            darwin.log(
                f"Vector index for '{filename}' was built with '{stored_model}' "
                f"but current model is '{current_model}'. Using a different embedding "
                f"model breaks ANN search. Re-embedding required.",
                LogType.WARNING,
            )
            return None

        # ── Dimension compatibility check ─────────────────────────────────
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
        Return (row_index, cosine_similarity) tuples.
        Row indices map 1:1 to chunks.parquet and embeddings.f16.npy rows.
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

        if hasattr(matches, "keys") and hasattr(matches, "distances"):
            for key, dist in zip(matches.keys, matches.distances):
                idx = int(key)
                score = 1.0 - float(dist)
                results.append((idx, score))
        else:
            for match in matches:
                idx = int(match.key)
                score = 1.0 - float(match.distance)
                results.append((idx, score))

        return results

    def load_vector_metadata(self, filename: str) -> Optional[dict]:
        path = self._dir_for(filename) / "vector.meta.json"
        if not path.exists():
            return None
        return read_json(path)

    def is_vector_compatible(self, filename: str) -> bool:
        try:
            index = self.load_vector(filename)
            return index is not None
        except FileNotFoundError:
            return False


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: BM25
# ═══════════════════════════════════════════════════════════════════════════

class BM25Mixin(_PathResolverMixin):
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
            # Numerical arrays → npz (no object arrays, no pickle)
            npz_path = d / "bm25.npz"
            np.savez_compressed(
                npz_path,
                doc_lengths=np.array(doc_lengths, dtype=np.int32),
            )

            # Tokens → parquet list<string>.  Avoids numpy object arrays entirely
            # and gives us dictionary compression on repeated tokens.
            tokens_path = d / "bm25_tokens.parquet"
            tokens_records = [{"tokens": list(tok_list)} for tok_list in tokens]
            tokens_table = records_to_table(tokens_records)
            pq.write_table(tokens_table, tokens_path, compression="zstd")

            # Scalar metadata → json
            meta_path = d / "bm25_meta.json"
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
                        "bm25",
                        chunk_count=chunk_count,
                        vocab_size=vocab_size,
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

        npz_path = d / "bm25.npz"
        meta_path = d / "bm25_meta.json"
        tokens_path = d / "bm25_tokens.parquet"

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

class StopwordsMixin(_PathResolverMixin):

    def has_stopwords(self, filename: str) -> bool:
        return self._has_artifact(filename, "stopwords")

    def save_stopwords(self, filename: str, stopwords: set[str]) -> None:
        if not stopwords:
            darwin.log(f"No stopwords to save for {filename}", LogType.ERROR)
            return
        d = self._dir_for(filename)
        if d is None:
            return
        write_json(d / "stopwords.json", {"stopwords": sorted(stopwords)})
        self._update_meta(filename, "stopwords")
        record = self._record_for(filename)
        if record:
            try:
                self.registry.update_pipeline(record.last_seen_path or filename, "stopwords")
            except Exception as reg_err:
                darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)
        self.invalidate(filename)

    def load_stopwords(self, filename: str) -> set[str]:
        path = self._dir_for(filename) / "stopwords.json"
        if not path.exists():
            return set()
        return set(read_json(path).get("stopwords", []))


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Graph
# ═══════════════════════════════════════════════════════════════════════════

class GraphMixin(_PathResolverMixin):

    def has_graph(self, filename: str) -> bool:
        return self._has_artifact(filename, "graph")

    def has_graph_html(self, filename: str) -> bool:
        return self._has_artifact(filename, "graph_vis")

    def save_graph(self, filename: str, nodes: list[Any], edges: list[Any]) -> None:
        if not nodes and not edges:
            darwin.log(f"No graph data to save for {filename}", LogType.ERROR)
            return

        d = self._dir_for(filename) / "graph"
        d.mkdir(parents=True, exist_ok=True)

        # Write nodes and edges as separate parquet files for modular access.
        # Callers can load only nodes if they don't need edge traversal.
        if nodes:
            nodes_table = records_to_table([normalize_record(n) for n in nodes])
            pq.write_table(nodes_table, d / "nodes.parquet", compression="zstd")
        if edges:
            edges_table = records_to_table([normalize_record(e) for e in edges])
            pq.write_table(edges_table, d / "edges.parquet", compression="zstd")

        self._update_meta(filename, "graph", extras={
            "graph": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "built_at": now(),
            }
        })

        record = self._record_for(filename)
        if record:
            try:
                self.registry.update_pipeline(
                    record.last_seen_path or filename,
                    "graph",
                    node_count=len(nodes),
                    edge_count=len(edges),
                )
            except Exception as reg_err:
                darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)

        self.invalidate(filename)
        darwin.log(f"Graph saved: {len(nodes)} nodes, {len(edges)} edges", LogType.SUCCESS)

    def save_graph_html(self, filename: str, html: str) -> None:
        if html is None:
            darwin.log(f"No graph HTML to save for {filename}", LogType.ERROR)
            return

        d = self._dir_for(filename) / "graph"
        d.mkdir(parents=True, exist_ok=True)
        (d / "graph.html").write_text(html, encoding="utf-8")
        self._update_meta(filename, "graph_vis")
        record = self._record_for(filename)
        if record:
            try:
                self.registry.update_pipeline(record.last_seen_path or filename, "graph_vis")
            except Exception as reg_err:
                darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)
        self.invalidate(filename)

    def load_graph(self, filename: str) -> dict:
        """Return graph as {nodes: [...], edges: [...]}."""
        cache_key = f"graph:{filename}"
        if hasattr(self, "_artifact_cache") and cache_key in self._artifact_cache:
            return self._artifact_cache[cache_key]

        d = self._dir_for(filename) / "graph"
        nodes_path = d / "nodes.parquet"
        edges_path = d / "edges.parquet"

        result: dict[str, list] = {"nodes": [], "edges": []}
        if nodes_path.exists():
            result["nodes"] = table_to_records(pq.read_table(nodes_path))
        if edges_path.exists():
            result["edges"] = table_to_records(pq.read_table(edges_path))

        if hasattr(self, "_artifact_cache"):
            self._artifact_cache[cache_key] = result
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Wiki
# ═══════════════════════════════════════════════════════════════════════════

class WikiMixin(_PathResolverMixin):

    def has_wiki(self, filename: str) -> bool:
        return self._has_artifact(filename, "wiki")

    def save_wiki(self, filename: str, pages: list[Any]) -> None:
        if not pages:
            darwin.log(f"No wiki pages to save for {filename}", LogType.ERROR)
            return

        d = self._dir_for(filename) / "wiki"
        d.mkdir(parents=True, exist_ok=True)

        dicts = [normalize_record(p) for p in pages]
        table = records_to_table(dicts)
        pq.write_table(table, d / "wiki.parquet", compression="zstd")

        record = self._record_for(filename)
        if record:
            try:
                self.registry.update_pipeline(
                    record.last_seen_path or filename,
                    "wiki",
                    page_num=len(pages),
                )
            except Exception as reg_err:
                darwin.log(f"Failed to update registry for {filename}: {reg_err}", LogType.WARNING)

        self.invalidate(filename)
        darwin.log(f"Wiki saved: {len(pages)} pages", LogType.SUCCESS)

    def load_wiki(self, filename: str) -> list[dict] | None:
        path = self._dir_for(filename) / "wiki" / "wiki.parquet"
        if not path.exists():
            return None
        return table_to_records(pq.read_table(path))


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Search — high-level semantic retrieval (USearch + numpy fallback)
# ═══════════════════════════════════════════════════════════════════════════

class SearchMixin(_PathResolverMixin):
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
        USearch ANN path: query vector → row indices → parquet fetch → reconstruction.
        """
        matches = self.search_vector(filename, query_vec, k=k)
        if not matches:
            return []

        indices, scores = zip(*matches)

        table = self.load_records(filename, columns=columns)
        if table.num_rows == 0:
            return []

        taken = table.take(pa.array(list(indices)))
        rows = taken.to_pylist()

        for row, score in zip(rows, scores):
            row["_score"] = score

        if cls is not None:
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

        # Cosine similarity via dot product (assumes L2-normalized embeddings)
        similarities = np.dot(vectors.astype(np.float32), query_vec.astype(np.float32))

        if k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        scores = similarities[top_indices].tolist()

        table = self.load_records(filename, columns=columns)
        if table.num_rows == 0:
            return []

        taken = table.take(pa.array(top_indices.tolist()))
        rows = taken.to_pylist()

        for row, score in zip(rows, scores):
            row["_score"] = float(score)

        if cls is not None:
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

        all_results: list[dict] = []
        per_doc_k = max(5, top_k // len(filenames) + 1)

        for filename in filenames:
            rows = self.search_single(
                query_vec, filename, k=per_doc_k,
                min_score=min_score, columns=columns,
                force_numpy=force_numpy,
            )
            all_results.extend(rows)

        all_results.sort(key=lambda r: r.get("_score", 0.0), reverse=True)
        return all_results[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Migration — automatic upgrade from legacy JSON/NPZ formats
# ═══════════════════════════════════════════════════════════════════════════

class MigrationMixin(_PathResolverMixin):
    """
    One-shot migration helpers.  Old files are backed up to *.bak so the
    operation is reversible.  Migration is idempotent — safe to run multiple
    times because we skip if the new artifact already exists.
    """

    def migrate_all(self, filename: str) -> dict[str, bool]:
        """Run every available migration. Returns {stage: succeeded}."""
        report = {
            "chunks": self._migrate_chunks_json(filename),
            "embeddings": self._migrate_embeddings_npz(filename),
            "graph": self._migrate_graph_json(filename),
            "wiki": self._migrate_wiki_json(filename),
        }
        migrated = [k for k, v in report.items() if v]
        if migrated:
            darwin.log(
                f"Migration complete for {filename}: {', '.join(migrated)}",
                LogType.SUCCESS,
            )
        return report

    def _migrate_chunks_json(self, filename: str) -> bool:
        d = self._dir_for(filename)
        old = d / "chunks.json"
        new = d / "chunks.parquet"
        if not old.exists() or new.exists():
            return False
        try:
            records = read_json(old) or []
            if not records:
                return False
            # save_records is provided by RecordsMixin in the final DarwinStore MRO
            self.save_records(filename, records)
            old.rename(d / "chunks.json.bak")
            return True
        except Exception as e:
            darwin.log(f"chunks.json migration failed: {e}", LogType.WARNING)
            return False

    def _migrate_embeddings_npz(self, filename: str) -> bool:
        d = self._dir_for(filename)
        new = d / "embeddings.f16.npy"
        if new.exists():
            return False

        migrated = False
        for npz_file in d.glob("*.npz"):
            if npz_file.name == "bm25.npz":
                continue
            try:
                # mmap_mode='r' avoids loading the entire old npz into RAM
                data = np.load(npz_file, allow_pickle=False, mmap_mode="r")
                if "vectors" not in data:
                    continue

                vectors = data["vectors"]
                f16 = np.ascontiguousarray(vectors.astype(np.float16))
                np.save(d / "embeddings.f16.npy", f16)

                self._update_meta(filename, "embeddings", extras={
                    "embedding_dimensions": int(f16.shape[1]),
                    "embedding_count": int(f16.shape[0]),
                    "embedding_model": get_current_model_name(),
                    "migrated_from": str(npz_file.name),
                })

                npz_file.rename(npz_file.with_suffix(".npz.bak"))
                migrated = True
                darwin.log(
                    f"Migrated embeddings {npz_file.name} → embeddings.f16.npy",
                    LogType.SUCCESS,
                )
                break
            except Exception as e:
                darwin.log(f"Skipping {npz_file.name} migration: {e}", LogType.WARNING)
        return migrated

    def _migrate_graph_json(self, filename: str) -> bool:
        d = self._dir_for(filename) / "graph"
        old = d / "graph.json"
        new_nodes = d / "nodes.parquet"
        if not old.exists() or new_nodes.exists():
            return False
        try:
            data = read_json(old) or {}
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            self.save_graph(filename, nodes, edges)
            old.rename(old.with_suffix(".json.bak"))
            return True
        except Exception as e:
            darwin.log(f"graph.json migration failed: {e}", LogType.WARNING)
            return False

    def _migrate_wiki_json(self, filename: str) -> bool:
        d = self._dir_for(filename) / "wiki"
        old = d / "wiki.json"
        new = d / "wiki.parquet"
        if not old.exists() or new.exists():
            return False
        try:
            data = read_json(old) or {}
            pages = data.get("pages", [])
            self.save_wiki(filename, pages)
            old.rename(old.with_suffix(".json.bak"))
            return True
        except Exception as e:
            darwin.log(f"wiki.json migration failed: {e}", LogType.WARNING)
            return False