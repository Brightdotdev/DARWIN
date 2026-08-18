from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from src.core.Storage.CacheManager import CacheManager
from src.core.Storage.Constants import ARTIFACT_MANIFEST, CORE_STAGES
from src.core.Storage.DataModels import Source
from src.core.Storage.LocalRegistry import LocalRegistry, SourceManager, ContentIndex
from src.core.Storage.MetaManager import MetaManager
from src.core.Storage.StorePaths import StorePaths
from src.core.Storage.StoreRegistry import StoreRegistry
from src.core.Storage.StoreSearch import StoreSearch
from src.core.Storage.StoreUtils import (
    get_current_embedding_model,
    get_current_model_dimensions,
    now,
    read_json,
    to_dict,
    write_json,
)
from src.core.Storage.Transanctions import write_text_atomic
from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.AppPaths import set_darwin_base_dir


class DarwinStore(StoreSearch):
    """
    Single unified storage class for Darwin RAG.
    Manages file registration, pipeline stages, search, integrity, and health.
    """

    def __init__(self, darwin_dir: Path | None = None, name: Optional[str] = None):
        self._darwin_dir = (darwin_dir or set_darwin_base_dir()).resolve()
        self.paths = StorePaths(self._darwin_dir)
        self.paths.indexes_dir.mkdir(parents=True, exist_ok=True)

        # Internal components - per-store local registry
        self._registry = LocalRegistry(self.paths.registry_db_path)
        self._sources = SourceManager(self._registry)
        self._contents = ContentIndex(self._registry)
        self._cache = CacheManager(maxsize=256)
        self._meta = MetaManager(self.paths)
        self._manifest = ARTIFACT_MANIFEST

        # Store registry (central tracking for multi-store management)
        self._store_registry = StoreRegistry()
        self._store_name = name or self._darwin_dir.name
        self._store_id = self._register_self()

    def bootstrap(self, *, quiet: bool = False) -> None:
        """Optional initialization: load models and print dashboard.

        Args:
            quiet: If True, suppress dashboard and model-init log output.
        """
        if not quiet:
            self.print_dashboard()
        try:
            from src.core.Models.ModelManager import modelManager

            modelManager.initialize()
        except Exception as e:
            if not quiet:
                darwin.warning(f"Model manager initialization skipped: {e}")

    def _register_self(self) -> str:
        """Register this store in the central registry and write .darwin_store_id."""
        store_id_file = self.paths.indexes_dir / ".darwin_store_id"
        try:
            store_id = store_id_file.read_text().strip()
        except (FileNotFoundError, OSError):
            import uuid

            store_id = str(uuid.uuid4())
            try:
                store_id_file.write_text(store_id)
            except OSError:
                # Another process may have created it — read theirs
                store_id = store_id_file.read_text().strip()
        # Register/update in central registry, passing the user-provided name
        self._store_registry.register(self._darwin_dir, store_id, name=self._store_name)
        return store_id

    def _resolve_hash(self, filename: str) -> str | None:
        source = self._sources.find_by_name(filename)
        if not source:
            darwin.warning(f"File not registered: {filename} :: ignore if parsing")
            return None
        return source.content_hash

    # --------------------------------------------------------------------
    # Source management (from StoreSources)
    # --------------------------------------------------------------------
    def register(
        self, file_path: str | Path, page_count: int = 0, file_size: int = 0
    ) -> Source:
        source, _ = self._sources.register(file_path, page_count, file_size)
        return source

    def register_url(
        self, slug: str, url: str, content_hash: str, page_count: int = 0
    ) -> Source:
        """Register a URL source. Returns the Source object (not a tuple)."""
        source, _ = self._sources.register_url(slug, url, content_hash, page_count)
        return source

    def get_or_register(self, file_path: str | Path) -> Source:
        path = Path(file_path)
        source = self._sources.find_by_name(path.name)
        if source:
            return source
        return self.register(file_path)

    def remove_file(self, filename: str) -> bool:
        source = self._sources.find_by_name(filename)
        if not source:
            return False
        self._sources.remove_by_name(filename)
        if self._sources.count_for_hash(source.content_hash) == 0:
            self.paths.delete_hash_dir(source.content_hash)
        # Invalidate all cached artifacts for this hash
        self._cache.invalidate(source.content_hash)
        return True

    def ensure_image_dir(self, filename: str) -> Path:
        h = self._resolve_hash(filename)
        if h is None:
            darwin.error(f"File not registered: {filename}")
            return None
        return self.paths.ensure_image_dir(h)

    def get_source(self, display_name: str) -> Optional[Source]:
        return self._sources.find_by_name(display_name)

    # --------------------------------------------------------------------
    # Metadata helpers
    # --------------------------------------------------------------------
    def update_meta(
        self,
        content_hash: str,
        chunk_count: Optional[int] = None,
        page_count: Optional[int] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        self._contents.update_meta(
            content_hash, chunk_count, page_count, embedding_model
        )

    # --------------------------------------------------------------------
    # Artifact I/O (from ArtifactManager)
    # --------------------------------------------------------------------
    def _write_artifact(self, h: str, stage: str, write_fn: Callable[[], None]) -> None:
        self.paths.ensure_dir(h)
        write_fn()
        self._meta.update(h, stage)
        self._contents.set_stage(h, stage)
        self._cache.invalidate(h)

    def _read_artifact(
        self, h: str, stage: str, read_fn: Callable[[], Any]
    ) -> Any | None:
        cache_key = f"{h}:{stage}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        result = read_fn()
        if result is not None:
            self._cache.put(cache_key, result)
        return result

    def _delete_artifact(self, h: str, stage: str, paths: list[Path]) -> bool:
        removed = False
        for p in paths:
            if p.exists():
                p.unlink()
                removed = True
        self._contents.set_stage(h, stage, complete=False)
        self._cache.invalidate(h)
        return removed

    # --- Markdown ---
    def has_markdown(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        return False if h is None else self.paths.has_artifact(h, "markdown")

    def save_markdown(self, filename: str, doc: Any) -> None:
        error = getattr(doc, "error", None) or (
            doc.get("error") if isinstance(doc, dict) else None
        )
        if error:
            darwin.info(f"Skipped markdown for {filename}: {error}")
            return
        h = self._resolve_hash(filename)
        if h is None:
            return

        def _write():
            embedding = getattr(doc, "embedding_text", "") or doc.get(
                "embedding_text", ""
            )
        
            write_text_atomic(
                self.paths.artifact_path(h, "markdown", "source"), embedding
            )

            dd = to_dict(doc)
            skip = {
                "text",
                "error",
                "filename",
                "images",
                "embedding_text",
            }
            extras = {k: v for k, v in dd.items() if k not in skip}
            self._meta.update(h, "markdown", extras=extras)

        self._write_artifact(h, "markdown", _write)
        darwin.success(f"Saved markdown for {filename}")

    def load_markdown(self, filename: str) -> str | None:
        h = self._resolve_hash(filename)
        if h is None:
            return None
        path = self.paths.artifact_path(h, "markdown", "source")
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            darwin.error(f"Error reading markdown for {filename}: {e}")
            
            return None

    def delete_markdown(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        if h is None:
            return False
        return self._delete_artifact(
            h,
            "markdown",
            [
                self.paths.artifact_path(h, "markdown", "source"),
            ],
        )

    # --- Chunks ---
    def has_records(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        return False if h is None else self.paths.has_artifact(h, "chunks")

    def save_records(self, filename: str, records: list[Any]) -> None:
        if not records:
            darwin.error(f"No records to save for {filename}")
            return
        h = self._resolve_hash(filename)
        if h is None:
            return

        def _write():
            dicts = [to_dict(r) for r in records]
            table = pa.Table.from_pylist(dicts)
            path = self.paths.artifact_path(h, "chunks", "file")
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                table,
                path,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
            self._meta.update(h, "chunks", extras={"chunk_count": len(records)})
            self._contents.update_meta(h, chunk_count=len(records))

        self._write_artifact(h, "chunks", _write)
        darwin.success(f"Records saved: {len(records)} rows")

    def load_records(self, filename: str, columns: list[str] | None = None) -> pa.Table:
        h = self._resolve_hash(filename)
        if h is None:
            return pa.table({})
        cache_key = f"{h}:chunks"
        cached = self._cache.get(cache_key)
        if cached is not None and columns is None:
            return cached

        path = self.paths.artifact_path(h, "chunks", "file")
        if not path.exists():
            return pa.table({})

        table = pq.read_table(path, columns=columns)
        if columns is None:
            self._cache.put(cache_key, table)
        return table

    def query_records(
        self, filename: str, columns: list[str] | None = None, **filters
    ) -> list[dict[str, Any]]:
        h = self._resolve_hash(filename)
        if h is None:
            return []
        path = self.paths.artifact_path(h, "chunks", "file")
        if not path.exists():
            return []

        lf = pl.scan_parquet(path)
        if columns:
            lf = lf.select(columns)

        # Simple filter: equality and "in"
        exprs = []
        for key, value in filters.items():
            if key.endswith("__in"):
                field = key.removesuffix("__in")
                exprs.append(pl.col(field).is_in(list(value)))
            else:
                exprs.append(pl.col(key) == value)
        if exprs:
            combined = exprs[0]
            for e in exprs[1:]:
                combined = combined & e
            lf = lf.filter(combined)

        return lf.collect().to_dicts()

    def iter_records(
        self, filename: str, columns: list[str] | None = None, batch_size: int = 1000
    ) -> Iterator[dict[str, Any]]:
        h = self._resolve_hash(filename)
        if h is None:
            return iter([])
        path = self.paths.artifact_path(h, "chunks", "file")
        if not path.exists():
            return iter([])

        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        scanner = dataset.scanner(columns=columns, batch_size=batch_size)
        for batch in scanner.to_batches():
            yield from batch.to_pylist()

    def delete_records(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        if h is None:
            return False
        removed = self._delete_artifact(
            h, "chunks", [self.paths.artifact_path(h, "chunks", "file")]
        )
        self._contents.update_meta(h, chunk_count=0)
        return removed

    # --- Schema (structured data) ---
    def save_schema(self, filename: str, schema: dict) -> None:
        """Save structured schema metadata (keys, types, record_count)."""
        h = self._resolve_hash(filename)
        if h is None:
            return

        def _write():
            schema_path = self.paths.artifact_path(h, "schema", "file")
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(schema_path, schema)

        self._write_artifact(h, "schema", _write)

    def load_schema(self, filename: str) -> dict | None:
        """Load structured schema metadata."""
        h = self._resolve_hash(filename)
        if h is None:
            return None
        cache_key = f"{h}:schema"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        schema_path = self.paths.artifact_path(h, "schema", "file")
        if not schema_path.exists():
            return None

        result = read_json(schema_path)
        if result is not None:
            self._cache.put(cache_key, result)
        return result

    def has_schema(self, filename: str) -> bool:
        """Check if a file has structured schema metadata."""
        h = self._resolve_hash(filename)
        return False if h is None else self.paths.has_artifact(h, "schema")

    def list_structured_files(self) -> List[str]:
        """Return filenames of all indexed files that have structured schemas."""
        structured = []
        for filename in self.get_search_ready_files():
            if self.has_schema(filename):
                structured.append(filename)
        return structured

    def get_all_schemas(self) -> Dict[str, dict]:
        """Return {filename: schema} for all structured files in the store.

        Useful for:
          - Discovering what structured data is available
          - Checking for overlapping keys across files
          - Providing schema context to AI agents
        """
        schemas = {}
        for filename in self.list_structured_files():
            schema = self.load_schema(filename)
            if schema is not None:
                schemas[filename] = schema
        return schemas

    def find_overlapping_keys(self) -> Dict[str, List[str]]:
        """Find keys that appear in multiple structured files' schemas.

        Returns {key_name: [list_of_files_that_have_it]} for keys
        appearing in more than one file. Useful for disambiguation
        when filtering across multiple files.
        """
        all_schemas = self.get_all_schemas()
        key_to_files: Dict[str, List[str]] = {}

        for filename, schema in all_schemas.items():
            for key in schema.get("keys", []):
                key_to_files.setdefault(key, []).append(filename)

        return {k: files for k, files in key_to_files.items() if len(files) > 1}

    def validate_filters(
        self,
        filters: Dict[str, Any],
        files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Validate filter keys/types against schemas. Returns a validation report.

        Report contains:
          - valid: dict of filters that are valid (key exists in schema)
          - warnings: list of warning strings (missing keys, type mismatches)
          - overlapping: dict of keys found in multiple files
          - files_checked: list of files that were checked
          - schemas_found: whether any schemas were found at all
        """
        if not filters:
            return {
                "valid": {},
                "warnings": ["No filters provided"],
                "overlapping": {},
                "files_checked": [],
                "schemas_found": False,
            }

        target_files = files or self.list_structured_files()
        if not target_files:
            return {
                "valid": {},
                "warnings": ["No structured files found in store"],
                "overlapping": {},
                "files_checked": [],
                "schemas_found": False,
            }

        # Merge all schemas from target files
        merged_keys: Dict[str, str] = {}  # key -> type
        file_schemas: Dict[str, dict] = {}
        for filename in target_files:
            schema = self.load_schema(filename)
            if schema is None:
                continue
            file_schemas[filename] = schema
            for key, typ in schema.get("types", {}).items():
                if key not in merged_keys:
                    merged_keys[key] = typ

        if not file_schemas:
            return {
                "valid": {},
                "warnings": [f"No schemas found for files: {target_files}"],
                "overlapping": {},
                "files_checked": target_files,
                "schemas_found": False,
            }

        # Check each filter
        valid_filters = {}
        warnings = []

        # Import operator parser from StructuredSearch
        from src.core.Retrival.StructuredSearch import _parse_filter_key, OPERATORS

        for filter_key, filter_value in filters.items():
            # Parse operator from filter key
            field_name, operator = _parse_filter_key(filter_key)

            # For nested fields, check the root key
            root_key = field_name.split(".")[0]

            # Check if the field (or root key for nested) exists in schema
            if root_key not in merged_keys and field_name not in merged_keys:
                available = sorted(merged_keys.keys())
                warnings.append(
                    f"Filter key '{field_name}' (from '{filter_key}') not found in any schema. "
                    f"Available keys: {available}"
                )
                continue

            # Validate operator
            if operator not in ("exact", *OPERATORS):
                warnings.append(
                    f"Unknown operator '{operator}' in '{filter_key}'. "
                    f"Supported: exact (default), {', '.join(sorted(OPERATORS))}"
                )
                continue

            # Type check (skip for operator-based filters that change semantics)
            if operator not in ("exists",):
                schema_type = merged_keys.get(root_key, merged_keys.get(field_name, "str"))

                if isinstance(filter_value, str):
                    if schema_type in ("int", "float") and operator in ("gt", "lt", "gte", "lte"):
                        try:
                            float(filter_value)
                        except ValueError:
                            warnings.append(
                                f"Filter '{filter_key}': numeric operator '{operator}' "
                                f"requires a number, got '{filter_value}'"
                            )
                            continue
                    elif schema_type in ("int", "float") and operator == "exact":
                        try:
                            if schema_type == "int":
                                int(filter_value)
                            else:
                                float(filter_value)
                        except ValueError:
                            warnings.append(
                                f"Filter '{filter_key}': schema type is {schema_type} "
                                f"but value '{filter_value}' cannot be converted"
                            )
                            continue
                    elif schema_type == "bool" and operator == "exact":
                        if filter_value.lower() not in ("true", "false", "1", "0", "yes", "no"):
                            warnings.append(
                                f"Filter '{filter_key}': schema type is bool "
                                f"but value '{filter_value}' is not a recognized boolean"
                            )
                            continue
                elif isinstance(filter_value, list) and operator == "in":
                    # In-list: each item should be checkable against schema type
                    pass  # allow any list
                elif operator in ("gt", "lt", "gte", "lte"):
                    if not isinstance(filter_value, (int, float)):
                        warnings.append(
                            f"Filter '{filter_key}': numeric operator requires a number"
                        )
                        continue

            valid_filters[filter_key] = filter_value

        # Check overlapping keys (using parsed field names)
        overlapping = self.find_overlapping_keys()
        overlap_warnings = []
        for key in valid_filters:
            field_name, _ = _parse_filter_key(key)
            root_key = field_name.split(".")[0]
            if root_key in overlapping:
                overlap_warnings.append(
                    f"Key '{root_key}' exists in multiple files: {overlapping[root_key]}"
                )

        return {
            "valid": valid_filters,
            "warnings": warnings + overlap_warnings,
            "overlapping": {k: v for k, v in overlapping.items() if k in [
                _parse_filter_key(key)[0].split(".")[0] for key in valid_filters
            ]},
            "files_checked": sorted(file_schemas.keys()),
            "schemas_found": True,
        }

    # --- Embeddings ---
    def has_embeddings(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        return False if h is None else self.paths.has_artifact(h, "embeddings")

    def save_embeddings(
        self, filename: str, vectors: np.ndarray, indices: list[int] | None = None
    ) -> None:
        if vectors is None or len(vectors) == 0:
            darwin.error(f"No vectors to save for {filename}")
            return
        h = self._resolve_hash(filename)
        if h is None:
            return

        def _write():
            vectors_f16 = np.ascontiguousarray(vectors.astype(np.float16))
            emb_path = self.paths.artifact_path(h, "embeddings", "file")
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(emb_path), vectors_f16)

            dim = vectors.shape[1]
            count = vectors.shape[0]
            model_name = get_current_embedding_model()

            if indices is not None:
                map_path = self.paths.artifact_path(h, "embeddings", "index_map")
                map_path.parent.mkdir(parents=True, exist_ok=True)
                write_json(map_path, indices)

            self._meta.update(
                h,
                "embeddings",
                extras={
                    "embedding_model": model_name,
                    "embedding_dimensions": dim,
                    "embedding_count": count,
                },
            )
            self._contents.update_meta(h, embedding_model=model_name)

        self._write_artifact(h, "embeddings", _write)
        darwin.success(
            f"Embeddings saved: {len(vectors)} vectors, {vectors.shape[1]}d"
        )

    def load_embeddings(self, filename: str, mmap: bool = True) -> np.ndarray | None:
        h = self._resolve_hash(filename)
        if h is None:
            return None

        cache_key = f"{h}:embeddings"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        emb_path = self.paths.artifact_path(h, "embeddings", "file")
        if not emb_path.exists():
            return None

        meta = self._meta.load(h)
        stored_model = meta.get("embedding_model", "unknown")
        current_model = get_current_embedding_model()
        if stored_model not in ("unknown", current_model):
            darwin.warning(
                f"Model mismatch for '{filename}': stored {stored_model}, current {current_model}"
            )
            return None

        mode = "r" if mmap else None
        vectors = np.load(str(emb_path), mmap_mode=mode)
        if mmap:
            self._cache.put(cache_key, vectors)
        return vectors

    def load_embedding_index_map(self, filename: str) -> list[int] | None:
        h = self._resolve_hash(filename)
        if h is None:
            return None
        path = self.paths.artifact_path(h, "embeddings", "index_map")
        if not path.exists():
            return None
        return read_json(path)

    def delete_embeddings(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        if h is None:
            return False
        removed = self._delete_artifact(
            h,
            "embeddings",
            [
                self.paths.artifact_path(h, "embeddings", "file"),
                self.paths.artifact_path(h, "embeddings", "index_map"),
            ],
        )
        self._contents.update_meta(h, embedding_model=None)
        return removed

    # --- Vector (USearch) ---
    def has_vector(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        return False if h is None else self.paths.has_artifact(h, "vector")

    def save_vector(self, filename: str, vectors: np.ndarray) -> None:
        h = self._resolve_hash(filename)
        if h is None:
            return

        def _write():
            from usearch.index import Index

            n, dim = vectors.shape
            index = Index(ndim=dim, metric="cos", dtype="f32")
            keys = np.arange(n, dtype=np.int64)
            index.add(keys=keys, vectors=vectors.astype(np.float32))

            idx_path = self.paths.artifact_path(h, "vector", "index")
            meta_path = self.paths.artifact_path(h, "vector", "meta")
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.parent.mkdir(parents=True, exist_ok=True)

            index.save(str(idx_path))
            write_json(
                meta_path,
                {
                    "dimension": dim,
                    "count": n,
                    "metric": "cos",
                    "dtype": "f32",
                    "created_at": now(),
                },
            )
            self._meta.update(
                h, "vector", extras={"vector_dimension": dim, "vector_count": n}
            )

        self._write_artifact(h, "vector", _write)
        darwin.success(
            f"Vector index saved: {len(vectors)} vectors, {vectors.shape[1]}d"
        )

    def load_vector(self, filename: str):
        h = self._resolve_hash(filename)
        if h is None:
            return None

        cache_key = f"{h}:vector"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        idx_path = self.paths.artifact_path(h, "vector", "index")
        meta_path = self.paths.artifact_path(h, "vector", "meta")
        if not idx_path.exists() or not meta_path.exists():
            return None

        meta = self._meta.load(h)
        stored_model = meta.get("embedding_model", "unknown")
        current_model = get_current_embedding_model()
        if stored_model not in ("unknown", current_model):
            darwin.warning(f"Vector model mismatch for '{filename}'")
            return None

        vector_meta = read_json(meta_path) or {}
        stored_dim = vector_meta.get("dimension", 0)
        expected_dim = get_current_model_dimensions()
        if expected_dim and stored_dim != expected_dim:
            darwin.warning(
                f"Vector dimension mismatch: stored {stored_dim}, expected {expected_dim}"
            )
            return None

        from usearch.index import Index

        index = Index.restore(str(idx_path))
        self._cache.put(cache_key, index)
        return index

    def search_vector(
        self, filename: str, query_vec: np.ndarray, k: int = 10
    ) -> list[tuple[int, float]]:
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

    def delete_vector(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        if h is None:
            return False
        return self._delete_artifact(
            h,
            "vector",
            [
                self.paths.artifact_path(h, "vector", "index"),
                self.paths.artifact_path(h, "vector", "meta"),
            ],
        )

    # --- BM25 ---
    def has_bm25(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        return False if h is None else self.paths.has_artifact(h, "bm25")

    def save_bm25(self, filename: str, data: Any) -> None:
        # Robust extraction: supports both dict and object with attributes
        tokens = (
            getattr(data, "tokens", None)
            if not isinstance(data, dict)
            else data.get("tokens")
        )
        doc_lengths = (
            getattr(data, "doc_lengths", None)
            if not isinstance(data, dict)
            else data.get("doc_lengths")
        )
        doc_freqs = (
            getattr(data, "doc_freqs", None)
            if not isinstance(data, dict)
            else data.get("doc_freqs")
        )
        avgdl = (
            getattr(data, "avgdl", None)
            if not isinstance(data, dict)
            else data.get("avgdl")
        )
        chunk_count = (
            getattr(data, "chunk_count", len(tokens) if tokens else 0)
            if not isinstance(data, dict)
            else data.get("chunk_count", len(tokens) if tokens else 0)
        )
        vocab_size = (
            getattr(data, "vocab_size", len(doc_freqs) if doc_freqs else 0)
            if not isinstance(data, dict)
            else data.get("vocab_size", len(doc_freqs) if doc_freqs else 0)
        )
        threshold = (
            getattr(data, "threshold", 0.0)
            if not isinstance(data, dict)
            else data.get("threshold", 0.0)
        )
        idf_smoothing = (
            getattr(data, "idf_smoothing", 0.0)
            if not isinstance(data, dict)
            else data.get("idf_smoothing", 0.0)
        )
        domain_stopwords = (
            getattr(data, "domain_stopwords", [])
            if not isinstance(data, dict)
            else data.get("domain_stopwords", [])
        )
        precomputed_idfs = (
            getattr(data, "precomputed_idfs", None)
            if not isinstance(data, dict)
            else data.get("precomputed_idfs")
        )
        # FIX: Extract chunk_indices
        chunk_indices = (
            getattr(data, "chunk_indices", None)
            if not isinstance(data, dict)
            else data.get("chunk_indices")
        )

        if not tokens or not doc_lengths:
            darwin.error(f"No BM25 data to save for {filename}")
            return

        h = self._resolve_hash(filename)
        if h is None:
            return

        def _write():
            npz_path = self.paths.artifact_path(h, "bm25", "index")
            meta_path = self.paths.artifact_path(h, "bm25", "meta")
            tokens_path = self.paths.artifact_path(h, "bm25", "tokens")

            npz_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            tokens_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                npz_path, doc_lengths=np.array(doc_lengths, dtype=np.int32)
            )

            tokens_records = [{"tokens": list(tok_list)} for tok_list in tokens]
            tokens_table = pa.Table.from_pylist(tokens_records)
            pq.write_table(tokens_table, tokens_path, compression="zstd")

            meta_data = {
                "avgdl": avgdl,
                "chunk_count": chunk_count,
                "vocab_size": vocab_size,
                "threshold": threshold,
                "idf_smoothing": idf_smoothing,
                "domain_stopwords": list(domain_stopwords),
                "doc_freqs": doc_freqs,
                "precomputed_idfs": precomputed_idfs or {},
            }

            # FIX: Save chunk_indices if present
            if chunk_indices is not None:
                meta_data["chunk_indices"] = chunk_indices

            write_json(meta_path, meta_data, compact=True)

        self._write_artifact(h, "bm25", _write)
        darwin.success(f"Saved BM25 index for {filename}")

    def load_bm25(self, filename: str) -> dict | None:
        h = self._resolve_hash(filename)
        if h is None:
            return None

        cache_key = f"{h}:bm25"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        npz_path = self.paths.artifact_path(h, "bm25", "index")
        meta_path = self.paths.artifact_path(h, "bm25", "meta")
        tokens_path = self.paths.artifact_path(h, "bm25", "tokens")

        if not npz_path.exists() or not meta_path.exists() or not tokens_path.exists():
            return None

        try:
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
                "chunk_indices": meta.get("chunk_indices"),
            }

            self._cache.put(cache_key, result)
            return result
        except Exception as e:
            darwin.error(f"Failed to load BM25 for {filename}: {e}")
            return None

    def delete_bm25(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        if h is None:
            return False
        return self._delete_artifact(
            h,
            "bm25",
            [
                self.paths.artifact_path(h, "bm25", "index"),
                self.paths.artifact_path(h, "bm25", "meta"),
                self.paths.artifact_path(h, "bm25", "tokens"),
            ],
        )

    # --- Stopwords ---
    def has_stopwords(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        return False if h is None else self.paths.has_artifact(h, "stopwords")

    def save_stopwords(self, filename: str, stopwords: set[str]) -> None:
        h = self._resolve_hash(filename)
        if h is None:
            return

        def _write():
            path = self.paths.artifact_path(h, "stopwords", "file")
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, {"stopwords": sorted(stopwords)})

        self._write_artifact(h, "stopwords", _write)

    def load_stopwords(self, filename: str) -> set[str]:
        h = self._resolve_hash(filename)
        if h is None:
            return set()
        path = self.paths.artifact_path(h, "stopwords", "file")
        if not path.exists():
            return set()
        data = read_json(path) or {}
        return set(data.get("stopwords", []))

    def delete_stopwords(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        if h is None:
            return False
        return self._delete_artifact(
            h, "stopwords", [self.paths.artifact_path(h, "stopwords", "file")]
        )

    # --------------------------------------------------------------------
    # Status & Health (merged from StoreStatus + StoreHealth)
    # --------------------------------------------------------------------
    def file_status(self, file_path: str | Path) -> dict:
        path = Path(file_path)
        name = path.name
        source = self._sources.find_by_name(name)
        if not source:
            source, _ = self._sources.register(path)

        status = self._get_content_status(source.content_hash)
        return {
            "source_uuid": source.uuid,
            "file_hash": source.content_hash,
            "original_name": source.display_name,
            "last_seen_path": source.current_path,
            "file_size_mb": round(source.file_size / (1024 * 1024), 1),
            "page_count": status.get("page_count", 0),
            "is_parsed": status.get("stages", {}).get("markdown", False),
            "is_chunked": status.get("stages", {}).get("chunks", False),
            "is_embedded": status.get("stages", {}).get("embeddings", False),
            "is_kw_searchable": status.get("is_kw_searchable", False),
            "is_embedding_searchable": status.get("is_embedding_searchable", False),
            "is_hybrid_searchable": status.get("is_hybrid_searchable", False),
            "has_graph": status.get("stages", {}).get("graph", False)
            and status.get("stages", {}).get("graph_vis", False),
            "has_wiki": status.get("stages", {}).get("wiki", False),
            "progress_percent": status.get("progress_percent", 0),
            "status_summary": status.get("status_summary", "registered"),
            "missing_stages": status.get("missing_stages", []),
            "embedding_model": status.get("embedding_model"),
            "chunk_count": status.get("chunk_count", 0),
            "created_at": source.created_at,
            "updated_at": source.updated_at,
        }

    def status(self) -> dict:
        all_sources = self._sources.list_all()
        files: dict[str, dict] = {}
        total_chunks = 0
        total_pages = 0
        kw_searchable = 0

        for source in all_sources:
            st = self._get_content_status(source.content_hash)
            total_chunks += st.get("chunk_count", 0)
            total_pages += st.get("page_count", 0)
            if st.get("is_kw_searchable", False):
                kw_searchable += 1
            files[source.display_name] = {
                "file_hash": source.content_hash,
                "original_name": source.display_name,
                "last_seen_path": source.current_path,
                "file_size_mb": round(source.file_size / (1024 * 1024), 1),
                "page_count": st.get("page_count", 0),
                "is_parsed": st.get("stages", {}).get("markdown", False),
                "is_chunked": st.get("stages", {}).get("chunks", False),
                "is_embedded": st.get("stages", {}).get("embeddings", False),
                "is_kw_searchable": st.get("is_kw_searchable", False),
                "is_embedding_searchable": st.get("is_embedding_searchable", False),
                "is_hybrid_searchable": st.get("is_hybrid_searchable", False),
                "has_graph": st.get("stages", {}).get("graph", False)
                and st.get("stages", {}).get("graph_vis", False),
                "has_wiki": st.get("stages", {}).get("wiki", False),
                "progress_percent": st.get("progress_percent", 0),
                "status_summary": st.get("status_summary", "registered"),
                "missing_stages": st.get("missing_stages", []),
                "embedding_model": st.get("embedding_model"),
                "chunk_count": st.get("chunk_count", 0),
                "created_at": source.created_at,
                "updated_at": source.updated_at,
            }

        stats = {
            "total_files": len(all_sources),
            "files_needing_work": sum(
                1
                for s in files
                if (src := self._sources.find_by_name(s))
                and not self._get_content_status(src.content_hash)["is_core_complete"]
            ),
            "keyword_searchable_files": kw_searchable,
            "total_chunks": total_chunks,
            "total_pages": total_pages,
        }

        return {
            "darwin_dir": str(self._darwin_dir),
            "stats": stats,
            "files": files,
        }

    def _get_content_status(self, content_hash: str) -> dict:
        content = self._contents.get(content_hash)
        if not content:
            return {"hash": content_hash}

        all_stages = [
            "markdown",
            "chunks",
            "embeddings",
            "vector",
            "bm25",
            "stopwords",
            "graph",
            "graph_vis",
            "wiki",
        ]
        stages = {s: s in content.stages for s in all_stages}

        weights = {
            "markdown": 10,
            "chunks": 15,
            "embeddings": 25,
            "bm25": 25,
            "vector": 25,
        }
        score = sum(w for s, w in weights.items() if stages.get(s))
        missing = [
            s
            for s in ["markdown", "chunks", "embeddings", "bm25", "vector"]
            if not stages.get(s)
        ]

        # Read extra metadata from meta.json
        meta = self._meta.load(content_hash)
        chunk_count = meta.get("chunk_count", 0)
        page_count = meta.get("page_count", 0)
        embedding_model = meta.get("embedding_model")

        is_core_complete = all(
            stages[s] for s in ["markdown", "chunks", "embeddings", "bm25", "vector"]
        )
        is_kw = stages.get("bm25", False) and stages.get("stopwords", False)
        is_sem = stages.get("embeddings", False) and stages.get("vector", False)

        status_summary = (
            "complete — {} chunks".format(chunk_count)
            if is_core_complete
            else ("parsed" if stages.get("markdown") else "registered")
        )

        return {
            "hash": content_hash,
            "stages": stages,
            "progress_percent": min(score, 100),
            "is_core_complete": is_core_complete,
            "missing_stages": missing,
            "is_kw_searchable": is_kw,
            "is_embedding_searchable": is_sem,
            "is_hybrid_searchable": is_kw and is_sem,
            "embedding_model": embedding_model,
            "chunk_count": chunk_count,
            "page_count": page_count,
            "status_summary": status_summary,
        }

    def get_searchables(self, filename: str) -> dict:
        source = self._sources.find_by_name(filename)
        if not source:
            return {
                "filename": filename,
                "kw_searchable": False,
                "semantic_searchable": False,
                "hybrid_searchable": False,
                "missing_stages": ["not_registered"],
            }
        status = self._get_content_status(source.content_hash)
        missing = status.get("missing_stages", [])
        kw = status.get("is_kw_searchable", False)
        sem = status.get("is_embedding_searchable", False)
        return {
            "filename": filename,
            "kw_searchable": kw,
            "semantic_searchable": sem,
            "hybrid_searchable": kw and sem,
            "missing_stages": missing,
        }

    def get_search_ready_files(
        self, filenames: str | list[str] | None = None, mode: str = "hybrid"
    ) -> list[str]:
        targets = []
        if filenames is None:
            targets = [s.display_name for s in self._sources.list_all()]
        elif isinstance(filenames, str):
            targets = [filenames]
        else:
            targets = filenames

        ready = []
        for name in targets:
            if self.is_search_ready(name, mode):
                ready.append(name)
        return ready

    def is_search_ready(self, filename: str, mode: str = "hybrid") -> bool:
        try:
            source = self._sources.find_by_name(filename)
            if not source:
                return False
            status = self._get_content_status(source.content_hash)
        except Exception:
            return False

        if mode == "kw":
            return status.get("is_kw_searchable", False)
        elif mode == "semantic":
            return status.get("is_embedding_searchable", False)
        elif mode == "hybrid":
            return status.get("is_hybrid_searchable", False)
        return False

    def get_pipeline_report(self, filename: str | None = None) -> dict | list[dict]:

        def _make(source) -> dict:
            st = self._get_content_status(source.content_hash)
            return {
                "filename": source.display_name,
                "hash": source.content_hash[:16] + "…",
                "last_seen": source.current_path,
                "stages": st.get("stages", {}),
                "completed": [
                    name for name, done in st.get("stages", {}).items() if done
                ],
                "missing": st.get("missing_stages", []),
                "core_complete": st.get("is_core_complete", False),
                "progress_percent": st.get("progress_percent", 0),
                "chunk_count": st.get("chunk_count", 0),
                "embedding_model": st.get("embedding_model"),
            }

        if filename:
            src = self._sources.find_by_name(filename)
            if not src:
                darwin.error(f"File not registered: {filename}")
                return None
            return _make(src)

        return [_make(s) for s in self._sources.list_all()]

    def print_pipeline_report(self, filename: str | None = None) -> None:
        data = self.get_pipeline_report(filename)
        files = [data] if filename else data

        darwin.info("Pipeline Report")

        for f in files:
            icon = "✓" if f["core_complete"] else "○"
            darwin.info(f"{icon} {f['filename']}")
            darwin.info(
                f"Hash {f['hash']} | "
                f"Progress {f['progress_percent']}% | "
                f"Chunks {f['chunk_count']}"
            )

            stage_str = " | ".join(
                f"{'✓' if done else '·'} {name}" for name, done in f["stages"].items()
            )
            darwin.info(f"{stage_str}")

            if f["missing"]:
                darwin.warning(f"Missing stages: {', '.join(f['missing'])}")

    # --- Integrity & Repair ---
    def check_integrity(self, content_hash: str) -> dict:
        content = self._contents.get(content_hash)
        if not content:
            return {
                "hash": content_hash,
                "error": "Not registered in contents table",
                "is_healthy": False,
            }

        registry_done = []
        actually_missing = []
        actually_present = []
        tracked_stages = [
            "markdown",
            "chunks",
            "embeddings",
            "vector",
            "bm25",
            "stopwords",
            "graph",
            "graph_vis",
            "wiki",
        ]

        for stage in ARTIFACT_MANIFEST:
            if stage not in tracked_stages:
                continue
            registry_says = stage in content.stages
            files_exist = self.paths.has_artifact(content_hash, stage)

            if registry_says and not files_exist:
                actually_missing.append(stage)
            elif not registry_says and files_exist:
                actually_present.append(stage)
            elif registry_says and files_exist:
                registry_done.append(stage)

        return {
            "hash": content_hash,
            "is_healthy": len(actually_missing) == 0 and len(actually_present) == 0,
            "registry_done": registry_done,
            "actually_missing": actually_missing,
            "actually_present": actually_present,
        }

    def repair(self, content_hash: Optional[str] = None, dry_run: bool = False) -> dict:
        hashes = (
            [content_hash]
            if content_hash
            else [c.hash for c in self._contents.list_all()]
        )
        result = {
            "hashes_checked": len(hashes),
            "healthy": 0,
            "repaired": 0,
            "removed": 0,
            "repairs": [],
        }

        for h in hashes:
            report = self.check_integrity(h)
            if "error" in report:
                continue

            content = self._contents.get(h)
            if not content:
                continue

            if report["is_healthy"]:
                result["healthy"] += 1
                continue

            missing = report["actually_missing"]
            present = report["actually_present"]

            core_missing = [s for s in CORE_STAGES if s in missing]
            if len(core_missing) == len(CORE_STAGES):
                if not dry_run:
                    # Remove all sources for this hash; _maybe_gc_content handles cleanup
                    sources = self._sources.get_by_hash(h)
                    if sources:
                        for src in sources:
                            self._sources.remove_by_name(src.display_name)
                    else:
                        # No sources — orphaned content. Clean up directly.
                        self._registry.execute(
                            "DELETE FROM contents WHERE hash = ?", (h,)
                        )
                        self._registry.commit()
                        self.paths.delete_hash_dir(h)
                result["removed"] += 1
                result["repairs"].append(
                    {
                        "hash": h,
                        "action": "removed",
                        "reason": "all core artifacts missing",
                    }
                )
                continue

            if missing and not dry_run:
                for stage in missing:
                    self._contents.set_stage(h, stage, complete=False)
                result["repaired"] += 1
                result["repairs"].append(
                    {"hash": h, "action": "cleared_flags", "stages": missing}
                )

            if present and not dry_run:
                for stage in present:
                    self._contents.set_stage(h, stage, complete=True)
                if not missing:
                    result["repaired"] += 1
                result["repairs"].append(
                    {"hash": h, "action": "set_flags", "stages": present}
                )

        return result

    def full_audit(self) -> dict:
        report = self.repair(dry_run=True)
        all_hashes = [c.hash for c in self._contents.list_all()]
        broken = []
        for h in all_hashes:
            integrity = self.check_integrity(h)
            if not integrity.get("is_healthy", True):
                broken.append(
                    {
                        "hash": h,
                        "missing": integrity.get("actually_missing", []),
                        "orphans": integrity.get("actually_present", []),
                    }
                )

        return {
            "summary": {
                "total_hashes": len(all_hashes),
                "healthy": report["healthy"],
                "broken": len(broken),
                "would_repair": len(report["repairs"]),
            },
            "broken_hashes": broken,
        }

    def repair_registry(
        self, filename: str | None = None, dry_run: bool = False
    ) -> dict:
        if filename:
            h = self._resolve_hash(filename)
            if h is None:
                darwin.error("File not registered")
                return {}
            return self.repair(h, dry_run=dry_run)
        return self.repair(dry_run=dry_run)

    def full_registry_health_check(self) -> dict:
        return self.full_audit()

    def get_embedding_health(self, model_id: str) -> dict:
        from src.core.Models.ModelManager import modelManager
        from src.core.Models.Lib.ModelRegistry import ModelCategory

        health = {
            "active_model": model_id,
            "up_to_date": [],
            "needs_reembed": [],
            "model_missing": [],
            "never_embedded": [],
        }

        for content in self._contents.list_all():
            h = content.hash
            names = [s.display_name for s in self._sources.get_by_hash(h)]
            name = names[0] if names else h[:8]

            meta = self._meta.load(h)
            stored_model = meta.get("embedding_model")

            if not stored_model:
                health["never_embedded"].append(name)
                continue

            if stored_model == model_id:
                health["up_to_date"].append(name)
            else:
                if modelManager.is_available(ModelCategory.EMBEDDING, stored_model):
                    health["needs_reembed"].append(name)
                else:
                    health["model_missing"].append(name)

        return health

    def get_files_needing_reembed(
        self, filenames: str | list[str] | None = None
    ) -> list[dict]:
        current_model = get_current_embedding_model()
        current_dim = get_current_model_dimensions()

        targets = []
        if filenames is None:
            targets = [s.display_name for s in self._sources.list_all()]
        elif isinstance(filenames, str):
            targets = [filenames]
        else:
            targets = filenames

        result = []
        for name in targets:
            source = self._sources.find_by_name(name)
            if not source:
                continue
            content = self._contents.get(source.content_hash)
            if not content:
                continue

            meta = self._meta.load(source.content_hash)
            stored_model = meta.get("embedding_model")
            stored_dim = meta.get("embedding_dimensions", 0)

            status = "ok"
            if not stored_model:
                status = "never_embedded"
            elif stored_model != current_model:
                status = "model_mismatch"
            elif current_dim and stored_dim and stored_dim != current_dim:
                status = "dimension_mismatch"

            result.append(
                {
                    "filename": name,
                    "hash": source.content_hash[:16] + "…",
                    "stored_model": stored_model,
                    "current_model": current_model,
                    "stored_dimensions": stored_dim,
                    "current_dimensions": current_dim,
                    "status": status,
                    "needs_reembed": status != "ok",
                }
            )

        return result

    # --------------------------------------------------------------------
    # Dashboard convenience (prints using DashboardService)
    # --------------------------------------------------------------------
    def print_dashboard(self) -> None:

        status = self.status()
        stats = status["stats"]
        health = self.full_registry_health_check()
        W = 54

        darwin.info(f"{'-- Store ' + '-' * (W - 8)}", style="bold cyan")
        darwin.info("")
        darwin.info(f"  Directory: {self._darwin_dir}", style="bold white")
        darwin.info(
            f"  Files: {stats['total_files']}  |  "
            f"Searchable: {stats['keyword_searchable_files']}  |  "
            f"Chunks: {stats['total_chunks']}",
        )
        darwin.info(
            f"  Pages: {stats['total_pages']}  |  "
            f"Needs work: {stats['files_needing_work']}",
        )

        if health["summary"]["broken"] > 0:
            darwin.warning(
                f"  Registry integrity issue: {health['summary']['broken']} hash(es) "
                f"have missing artifacts but are marked done"
            )
        darwin.info("")
        

    # --------------------------------------------------------------------
    # Registry management (static methods)
    # --------------------------------------------------------------------
    @staticmethod
    def list_stores() -> list[dict]:
        return StoreRegistry().list_stores()

    @staticmethod
    def load_store(store_id: str) -> DarwinStore:
        reg = StoreRegistry()
        info = reg.get_store(store_id)
        if not info:
            darwin.error(f"Store with ID {store_id} not found")
            darwin.warning(f"To prevent crash using default store")
            return DarwinStore()
        return DarwinStore(Path(info["darwin_dir"]))

    @staticmethod
    def get_default_store() -> Optional[DarwinStore]:
        reg = StoreRegistry()
        info = reg.get_default_store()
        if not info:
            return None
        return DarwinStore(Path(info["darwin_dir"]))

    @staticmethod
    def load_store_by_name(name: str) -> DarwinStore:
        """Load a store by its human-readable name (case-sensitive)."""
        reg = StoreRegistry()
        stores = reg.list_stores()
        for s in stores:
            if s.get("name") == name:
                return DarwinStore(Path(s["darwin_dir"]))
        darwin.error(f"Store with name '{name}' not found")
        return None

    # --------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------
    @staticmethod
    def _human_bytes(size: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"

    @property
    def darwin_dir(self) -> Path:
        return self._darwin_dir

    @property
    def store_id(self) -> str:
        return self._store_id

    @property
    def store_name(self) -> str:
        return self._store_name
