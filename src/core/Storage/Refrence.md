# DarwinStore API Reference

This document describes the complete public interface of the refactored `DarwinStore` module. The module is a content‑addressable storage system for RAG pipelines, handling file registration, pipeline artifacts (markdown, chunks, embeddings, vector indices, BM25, stopwords, graphs, wiki data), integrity checks, and search.

---

## Table of Contents

1. [Initialization and Store Management](#1-initialization-and-store-management)
2. [Source (File) Registration](#2-source-file-registration)
3. [Pipeline Artifacts – Markdown](#3-pipeline-artifacts--markdown)
4. [Pipeline Artifacts – Chunks](#4-pipeline-artifacts--chunks)
5. [Pipeline Artifacts – Embeddings](#5-pipeline-artifacts--embeddings)
6. [Pipeline Artifacts – Vector Index (USearch)](#6-pipeline-artifacts--vector-index-usearch)
7. [Pipeline Artifacts – BM25](#7-pipeline-artifacts--bm25)
8. [Pipeline Artifacts – Stopwords](#8-pipeline-artifacts--stopwords)
9. [Pipeline Artifacts – Graph and Graph Visualisation](#9-pipeline-artifacts--graph-and-graph-visualisation)
10. [Pipeline Artifacts – Wiki](#10-pipeline-artifacts--wiki)
11. [Search and Retrieval](#11-search-and-retrieval)
12. [Status and Health](#12-status-and-health)
13. [Integrity and Repair](#13-integrity-and-repair)
14. [Dashboard (Disk Usage and Treemap)](#14-dashboard-disk-usage-and-treemap)
15. [Store Registry (Multi‑Store Management)](#15-store-registry-multi‑store-management)
16. [Data Models](#16-data-models)
17. [Exceptions and Error Handling](#17-exceptions-and-error-handling)

---

## 1. Initialization and Store Management

### `DarwinStore(darwin_dir: Optional[Path] = None, name: Optional[str] = None)`

Creates or opens a DarwinStore at the given directory.

- **`darwin_dir`**: filesystem path where the store lives. Defaults to `~/.darwin` (global default).
- **`name`**: human‑readable name for the store. If omitted, the directory name is used. This name is registered in the central store registry.

**Behaviour:**
- Ensures the directory exists.
- Initialises the SQLite registry (`registry.db`) inside the store.
- Creates a `.darwin_store_id` file inside the store (if missing) to uniquely identify the store.
- Registers the store in the central registry (`~/.darwin/stores_registry.db`).
- Loads the active embedding model (calls `modelManager.initialize()`).
- Prints a dashboard summary to the logger.

**Returns:** an instance of `DarwinStore`.

**Example:**
```python
from src.core.Storage.DarwinStore import DarwinStore

# Default store
default_store = DarwinStore()

# Custom store with a friendly name
my_store = DarwinStore("/data/project_x", name="Project X")
```

---

### `DarwinStore.list_stores() -> list[dict]` *(static)*

Lists all stores registered in the central registry.

**Returns:** a list of dictionaries, each containing:
- `id` (UUID),
- `slug` (unique slug),
- `name` (human‑readable),
- `darwin_dir` (absolute path),
- `created_at`, `updated_at`, `last_opened`,
- `is_default` (1/0),
- `cache_json` (cached metadata).

**Example:**
```python
stores = DarwinStore.list_stores()
for s in stores:
    print(s['name'], s['darwin_dir'])
```

---

### `DarwinStore.load_store(store_id: str) -> DarwinStore` *(static)*

Loads a registered store by its unique ID.

- **`store_id`**: the UUID of the store.

**Raises:** `ValueError` if not found.

**Example:**
```python
store = DarwinStore.load_store("a1b2c3d4-...")
```

---

### `DarwinStore.load_store_by_name(name: str) -> DarwinStore` *(static)*

Loads a store by its human‑readable name (case‑sensitive).

- **`name`**: the registered name.

**Raises:** `ValueError` if not found.

**Example:**
```python
store = DarwinStore.load_store_by_name("Project X")
```

---

### `DarwinStore.get_default_store() -> Optional[DarwinStore]` *(static)*

Returns the store marked as default in the central registry, or `None` if none.

---

## 2. Source (File) Registration

Sources are user‑facing file entries. They have a `display_name` (usually the filename) and point to a `content_hash` (SHA‑256 of file content). Multiple sources can share the same hash (deduplication).

### `register(file_path: str | Path, page_count: int = 0, file_size: int = 0) -> Source`

Registers a file from the local filesystem.

- **`file_path`**: path to the file.
- **`page_count`**: optional page count (used for PDFs etc.).
- **`file_size`**: optional size in bytes; if 0, it will be stat()’ed.

**Returns:** a `Source` dataclass (see [Data Models](#16-data-models)) with the registration details.

**Side effects:**
- If the file is new (new hash), a `Content` row is created.
- If the file already exists with the same hash, only the `updated_at` is refreshed.
- If the file exists but the hash changed (file modified), the old hash is garbage‑collected if no other source references it.

**Example:**
```python
source = store.register("/home/user/docs/report.pdf")
```

---

### `register_url(slug: str, url: str, content_hash: str, page_count: int = 0) -> Source`

Registers a URL as a source with a pre‑computed content hash (e.g., after fetching and hashing the body).

- **`slug`**: a unique identifier (acts as `display_name`).
- **`url`**: the original URL (stored in the `current_path` field).
- **`content_hash`**: SHA‑256 hex digest of the fetched content.
- **`page_count`**: optional.

**Returns:** a `Source` object.

**Example:**
```python
hash = compute_sha256(fetched_bytes)
source = store.register_url("wikipedia_darwin", "https://en.wikipedia.org/wiki/Darwin", hash)
```

---

### `get_or_register(file_path: str | Path) -> Source`

Convenience: if a file with that name is already registered, return its `Source`; otherwise register it.

**Example:**
```python
source = store.get_or_register("report.pdf")
```

---

### `remove_file(filename: str) -> bool`

Removes a source entry by its `display_name`.

- **`filename`**: the exact name (case‑sensitive).

**Returns:** `True` if the source existed and was removed.

**Side effects:**
- If no other source references the same `content_hash`, the entire hash directory and its artifacts are deleted from disk.
- The cache is invalidated for that hash.

---

### `get_source(display_name: str) -> Optional[Source]`

Returns the `Source` record for the given `display_name`, or `None` if not found.

---

### `ensure_image_dir(filename: str) -> Path`

Ensures the `images/` subdirectory exists inside the hash folder for the given file, and returns its path.

- **Raises:** `ValueError` if the file is not registered.

**Example:**
```python
img_dir = store.ensure_image_dir("report.pdf")
# you can now save extracted images into img_dir
```

---

## 3. Pipeline Artifacts – Markdown

Each registered file can have a parsed markdown representation. Two files are stored:
- `markdown.md` – the **embedding** text (usually cleaned, without display‑only formatting).
- `display-markdown.md` – a version for human viewing (may include formatting).

### `has_markdown(filename: str) -> bool`

Checks if markdown artifacts exist for the given file.

---

### `save_markdown(filename: str, doc: Any) -> None`

Saves the markdown representation. `doc` can be a dataclass or dict with the following expected fields:
- `display_text` (str) – display version.
- `embedding_text` (str) – embedding version.
- Any other metadata (title, author, etc.) will be stored in `meta.json` under the `markdown` stage.

**Side effects:**
- Creates the markdown files.
- Updates `meta.json` with the provided metadata.
- Sets the `markdown` stage flag in the registry.

**Example:**
```python
doc = {
    "display_text": "# Darwin\n\nCharles Darwin...",
    "embedding_text": "Darwin Charles natural selection...",
    "title": "Darwin",
    "author": "Unknown"
}
store.save_markdown("report.pdf", doc)
```

---

### `load_markdown(filename: str) -> str | None`

Returns the raw embedding markdown content as a string, or `None` if absent.

---

### `delete_markdown(filename: str) -> bool`

Deletes both markdown files and clears the `markdown` stage flag.

---

## 4. Pipeline Artifacts – Chunks

Chunks are stored as a Parquet file (`chunks.parquet`) with one row per chunk. The table typically contains columns like `index`, `text`, `metadata`, etc.

### `has_records(filename: str) -> bool`

Checks if the chunks Parquet exists.

---

### `save_records(filename: str, records: list[Any]) -> None`

Saves a list of chunk records. Each record can be a dict or a dataclass with a `to_dict()` method (or simple dataclass). The list is converted to a PyArrow Table and written as Parquet with Zstandard compression.

**Side effects:**
- Writes the Parquet file.
- Updates `meta.json` with `chunk_count`.
- Updates the `chunk_count` column in the `contents` registry.
- Sets the `chunks` stage flag.

**Example:**
```python
records = [
    {"index": 0, "text": "First chunk...", "page": 1},
    {"index": 1, "text": "Second chunk...", "page": 2}
]
store.save_records("report.pdf", records)
```

---

### `load_records(filename: str, columns: Optional[list[str]] = None) -> pa.Table`

Reads the entire chunk table into a PyArrow Table. If `columns` is given, only those columns are loaded. The result is cached (if no column filter) to speed up repeated calls.

**Returns:** an empty table if the file is not registered or the Parquet does not exist.

---

### `query_records(filename: str, columns: Optional[list[str]] = None, **filters) -> list[dict]`

Queries the chunks using Polars lazy evaluation. Supports simple filters:
- Equality: `index=5` or `text="foo"`.
- `__in`: `index__in=[1,2,3]`.

**Returns:** a list of dicts representing the rows (materialised).

**Example:**
```python
results = store.query_records("report.pdf", columns=["text", "score"], index__in=[0, 2])
```

---

### `iter_records(filename: str, columns: Optional[list[str]] = None, batch_size: int = 1000) -> Iterator[dict]`

Returns an iterator over the records, yielding dictionaries in batches. Useful for memory‑efficient streaming.

---

### `delete_records(filename: str) -> bool`

Deletes the Parquet file and clears the `chunks` stage flag. Also resets the `chunk_count` to 0 in the registry.

---

## 5. Pipeline Artifacts – Embeddings

Embeddings are stored as a float16 NumPy array (`embeddings.f16.npy`) and optionally an `index_map.json` (mapping row positions to chunk indices).

### `has_embeddings(filename: str) -> bool`

Checks if the embeddings array exists.

---

### `save_embeddings(filename: str, vectors: np.ndarray, indices: Optional[list[int]] = None) -> None`

Saves the embeddings.

- **`vectors`**: a 2D NumPy array of shape `(n_chunks, dim)`.
- **`indices`**: optional list of chunk indices corresponding to each row. If provided, it is saved as `index_map.json`.

**Side effects:**
- Writes the `.npy` file (float16).
- Writes `index_map.json` if `indices` is given.
- Updates `meta.json` with embedding model name, dimensions, and count.
- Updates the registry's `embedding_model` field.
- Sets the `embeddings` stage flag.

**Example:**
```python
import numpy as np
vectors = np.random.randn(100, 384).astype(np.float32)
store.save_embeddings("report.pdf", vectors)
```

---

### `load_embeddings(filename: str, mmap: bool = True) -> np.ndarray | None`

Loads the embeddings array. If `mmap=True` (default), returns a memory‑mapped array (read‑only) to avoid loading into RAM. Also caches the result.

**Returns:** `None` if the file is missing or the model name stored in `meta.json` does not match the current active model (to prevent dimension mismatches).

**Warning:** If the stored model differs from the current active model, a warning is logged and `None` is returned.

---

### `load_embedding_index_map(filename: str) -> list[int] | None`

Loads the `index_map.json` file if present.

---

### `delete_embeddings(filename: str) -> bool`

Deletes both the `.npy` and the `index_map.json` files, clears the `embeddings` stage flag, and resets the `embedding_model` field in the registry.

---

## 6. Pipeline Artifacts – Vector Index (USearch)

The vector index is a USearch index built on the embeddings for fast cosine similarity search. It is stored as two files: `vectors/vector.usearch` and `vectors/vector.meta.json`.

### `has_vector(filename: str) -> bool`

Checks if the USearch index exists.

---

### `save_vector(filename: str, vectors: np.ndarray) -> None`

Builds a USearch index (cosine metric, float32) from the given vectors and saves it.

- **`vectors`**: a 2D NumPy array of shape `(n_vectors, dim)`.

**Side effects:**
- Writes the `.usearch` index file.
- Writes a metadata JSON with dimension, count, etc.
- Updates `meta.json` with vector dimension and count.
- Sets the `vector` stage flag.

**Example:**
```python
store.save_vector("report.pdf", embeddings)   # embeddings is a numpy array
```

---

### `load_vector(filename: str)` -> `usearch.Index | None`

Restores the USearch index. The index is memory‑mapped and cached.

**Returns:** a USearch `Index` object or `None` if:
- The index files are missing.
- The stored model name does not match the current active model.
- The stored dimensions do not match the current model's dimensions.

---

### `search_vector(filename: str, query_vec: np.ndarray, k: int = 10) -> list[tuple[int, float]]`

Performs a k‑NN search using the loaded USearch index.

- **`query_vec`**: a 1D or 2D numpy array (if 2D, only the first vector is used).
- **`k`**: number of neighbours to return.

**Returns:** a list of tuples `(chunk_index, similarity_score)` where similarity is `1 - cosine_distance`. Scores are in the range [0,1] (cosine similarity). If an `index_map` exists, the chunk indices are mapped accordingly.

**If the index is missing, returns an empty list.**

---

### `delete_vector(filename: str) -> bool`

Deletes the USearch index and its metadata file, and clears the `vector` stage flag.

---

## 7. Pipeline Artifacts – BM25

BM25 index data consists of:
- `bm25.npz` – compressed doc lengths.
- `bm25_meta.json` – metadata (avgdl, doc_freqs, etc.).
- `bm25_tokens.parquet` – token lists per chunk.

### `has_bm25(filename: str) -> bool`

Checks if all BM25 artifacts exist.

---

### `save_bm25(filename: str, data: Any) -> None`

Saves BM25 index data. `data` can be a dict or an object with the following attributes:
- `tokens`: list of lists of tokens (one per chunk).
- `doc_lengths`: list of ints (length per chunk).
- `doc_freqs`: dict mapping token → document frequency.
- `avgdl`: float (average doc length).
- `chunk_count`, `vocab_size`, `threshold`, `idf_smoothing`, `domain_stopwords`, `precomputed_idfs` (optional).

**Side effects:**
- Writes the three files.
- Updates `meta.json` with BM25‑related info.
- Sets the `bm25` stage flag.

**Example:**
```python
bm25_data = {
    "tokens": [["darwin", "evolution"], ["natural", "selection"]],
    "doc_lengths": [2, 2],
    "doc_freqs": {"darwin": 1, "evolution": 1, "natural": 1, "selection": 1},
    "avgdl": 2.0
}
store.save_bm25("report.pdf", bm25_data)
```

---

### `load_bm25(filename: str) -> dict | None`

Loads the BM25 data as a dictionary containing all fields (`tokens`, `doc_lengths`, `doc_freqs`, `avgdl`, etc.). The result is cached.

**Returns:** `None` if any file is missing or an error occurs.

---

### `delete_bm25(filename: str) -> bool`

Deletes all BM25 artifacts and clears the `bm25` stage flag.

---

## 8. Pipeline Artifacts – Stopwords

A set of domain‑specific stopwords stored as a JSON file.

### `has_stopwords(filename: str) -> bool`

Checks existence.

---

### `save_stopwords(filename: str, stopwords: set[str]) -> None`

Writes the set to `stopwords.json`.

---

### `load_stopwords(filename: str) -> set[str]`

Returns the set of stopwords, or an empty set if absent.

---

### `delete_stopwords(filename: str) -> bool`

Deletes the JSON file and clears the `stopwords` stage flag.

---

## 9. Pipeline Artifacts – Graph and Graph Visualisation

The graph data is stored as two Parquet files (`nodes.parquet` and `edges.parquet`). The visualisation is stored as an HTML file.

### `has_graph(filename: str) -> bool`

Returns `True` only if **both** `graph` and `graph_vis` stages exist (i.e., nodes/edges and HTML are present).

---

### `save_graph(filename: str, nodes: list[Any], edges: list[Any]) -> None`

Saves the graph nodes and edges as Parquet files.

- Each node/edge record can be a dict or dataclass (converted with `to_dict`).

**Side effects:**
- Writes `nodes.parquet` and `edges.parquet`.
- Updates `meta.json` with node/edge counts.
- Sets the `graph` stage flag.

---

### `save_graph_html(filename: str, html: str) -> None`

Saves the HTML visualisation string to `graph.html`. Sets the `graph_vis` stage flag.

---

### `load_graph(filename: str) -> dict`

Returns a dict with keys `"nodes"` and `"edges"`, each being a PyArrow `Table` converted to a dict of columns (using `to_pydict()`). Returns `{"nodes": [], "edges": []}` if files are missing.

---

### `delete_graph(filename: str) -> bool`

Deletes both `nodes.parquet` and `edges.parquet` and clears the `graph` stage flag.

---

### `delete_graph_html(filename: str) -> bool`

Deletes `graph.html` and clears the `graph_vis` stage flag.

---

## 10. Pipeline Artifacts – Wiki

A dictionary of wiki‑style pages, stored as a JSON file.

### `has_wiki(filename: str) -> bool`

Checks if the `wiki` stage exists and the JSON file is present.

---

### `save_wiki(filename: str, wiki: dict) -> None`

Writes the wiki dict to `wiki.json`. Sets the `wiki` stage flag.

---

### `load_wiki(filename: str) -> dict | None`

Returns the parsed JSON, or `None` if missing.

---

### `delete_wiki(filename: str) -> bool`

Deletes the JSON and clears the `wiki` stage flag.

---

## 11. Search and Retrieval

These methods provide hybrid search (vector + keyword) and fallback modes.

### `search_single(query_vec: np.ndarray, filename: str, k: int = 10, min_score: float = 0.0, columns: Optional[list[str]] = None, force_numpy: bool = False) -> list[dict]`

Performs vector search over a single file.

- **`query_vec`**: query embedding (1D numpy array).
- **`filename`**: name of the registered file.
- **`k`**: number of top results to return.
- **`min_score`**: minimum similarity score (0‑1) to include.
- **`columns`**: which chunk columns to return (if `None`, all columns are returned).
- **`force_numpy`**: if `True`, uses the NumPy dot‑product fallback even if a USearch index exists; if `False` (default), tries USearch first, then falls back to NumPy if USearch fails or is absent.

**Returns:** a list of dicts, each containing the chunk record plus an extra `"_score"` key (the similarity). The list is sorted by descending score.

**Example:**
```python
query = embed("What is natural selection?")
results = store.search_single(query, "report.pdf", k=5)
for r in results:
    print(r['text'], r['_score'])
```

---

### `search_multi(query_vec: np.ndarray, filenames: list[str], top_k: int = 10, min_score: float = 0.0, columns: Optional[list[str]] = None, force_numpy: bool = False) -> list[dict]`

Searches across multiple files, merging results.

- **`filenames`**: list of file names to search.
- **`top_k`**: final number of results to return.
- Other parameters as in `search_single`.

**Algorithm:** for each file, retrieves `max(5, top_k // len(filenames) + 1)` results, then takes the global top `top_k` by score.

**Returns:** list of dicts (each with `_score`) globally sorted.

---

### `retrieve_by_vector(filename: str, query_vec: np.ndarray, k: int = 10, columns: Optional[list[str]] = None, cls: Any = None) -> list[Any]`

Low‑level method that uses the USearch index directly (no fallback). Returns a list of chunk records (dicts) with `_score` added. The `cls` parameter is ignored (kept for compatibility; you can pass `None`).

---

### `retrieve_by_numpy(filename: str, query_vec: np.ndarray, k: int = 10, columns: Optional[list[str]] = None, cls: Any = None) -> list[Any]`

Low‑level method that uses NumPy dot product on the loaded embeddings (memory‑intensive). Returns results in the same format.

---

### `search_vector(filename: str, query_vec: np.ndarray, k: int = 10) -> list[tuple[int, float]]`

As described in the Vector section – returns raw `(chunk_index, score)` pairs.

---

## 12. Status and Health

These methods provide information about the pipeline state and search readiness.

### `file_status(file_path: str | Path) -> dict`

Returns a detailed status dictionary for a given file (by filename). If the file is not registered, it is registered first.

**Output keys:**
- `source_uuid`, `file_hash`, `original_name`, `last_seen_path`, `file_size_mb`, `page_count`
- `is_parsed`, `is_chunked`, `is_embedded`
- `is_kw_searchable`, `is_embedding_searchable`, `is_hybrid_searchable`
- `has_graph`, `has_wiki`
- `progress_percent`, `status_summary`, `missing_stages`
- `embedding_model`, `chunk_count`
- `created_at`, `updated_at`

---

### `status() -> dict`

Aggregate status of all registered files.

**Output:**
```python
{
    "darwin_dir": "/path/to/store",
    "stats": {
        "total_files": int,
        "files_needing_work": int,
        "keyword_searchable_files": int,
        "total_chunks": int,
        "total_pages": int
    },
    "files": {
        "filename.pdf": { ... same as file_status ... },
        ...
    }
}
```

---

### `get_searchables(filename: str) -> dict`

Returns a dict with booleans `kw_searchable`, `semantic_searchable`, `hybrid_searchable`, and `missing_stages`. If the file is not registered, `missing_stages` is `["not_registered"]`.

---

### `is_search_ready(filename: str, mode: str = "hybrid") -> bool`

Checks if the file is ready for search in the given mode: `"kw"` (requires BM25 + stopwords), `"semantic"` (requires embeddings + vector), or `"hybrid"` (both).

---

### `get_search_ready_files(filenames: Optional[Union[str, list[str]]] = None, mode: str = "hybrid") -> list[str]`

Returns a list of filenames (from the provided list or all files) that are search‑ready in the given mode.

---

### `get_pipeline_report(filename: Optional[str] = None) -> dict | list[dict]`

Returns a detailed pipeline report for one or all files. Each entry contains:
- `filename`, `hash` (truncated), `last_seen`
- `stages`: dict of stage names → boolean
- `completed`: list of stage names that are done
- `missing`: list of missing core stages
- `core_complete`, `progress_percent`, `chunk_count`, `embedding_model`

---

### `print_pipeline_report(filename: Optional[str] = None) -> None`

Pretty‑prints the pipeline report to the logger.

---

### `get_embedding_health(model_id: str) -> dict`

Analyses the embedding model status for all contents. Returns a dict:
```python
{
    "active_model": "model_id",
    "up_to_date": ["file1", ...],
    "needs_reembed": ["file2", ...],
    "model_missing": ["file3", ...],   # stored model no longer available
    "never_embedded": ["file4", ...]
}
```

---

### `get_files_needing_reembed(filenames: Optional[Union[str, list[str]]] = None) -> list[dict]`

Returns a list of per‑file statuses regarding re‑embedding. Each entry:
- `filename`, `hash`
- `stored_model`, `current_model`
- `stored_dimensions`, `current_dimensions`
- `status`: `"ok"`, `"never_embedded"`, `"model_mismatch"`, `"dimension_mismatch"`
- `needs_reembed`: boolean

---

## 13. Integrity and Repair

These methods compare the registry stage flags against the actual on‑disk artifacts and can repair inconsistencies.

### `check_integrity(content_hash: str) -> dict`

Examines a single content hash. Returns a dict with:
- `hash`, `is_healthy` (bool)
- `registry_done`: stages that are both flagged and present
- `actually_missing`: stages flagged but missing on disk
- `actually_present`: stages present but not flagged

---

### `repair(content_hash: Optional[str] = None, dry_run: bool = False) -> dict`

Performs repair operations on one or all content hashes.

- **`content_hash`**: if given, only that hash is repaired; otherwise all.
- **`dry_run`**: if `True`, no changes are made; only a report is generated.

**Returns:** a dict with:
- `hashes_checked`, `healthy`, `repaired`, `removed`
- `repairs`: list of actions taken (with hash, action, stages)

**Behaviour:**
- If all core stages are missing, the entire content entry and its sources are removed (after checking no sources remain? Actually it also deletes sources – careful).
- Otherwise, missing stages are cleared from the registry, and present stages are set.

---

### `full_audit() -> dict`

Calls `repair(dry_run=True)` and also lists broken hashes with their missing/orphan stages. Returns:
```python
{
    "summary": {
        "total_hashes": int,
        "healthy": int,
        "broken": int,
        "would_repair": int
    },
    "broken_hashes": [
        {"hash": "...", "missing": [...], "orphans": [...]}
    ]
}
```

---

### `repair_registry(filename: Optional[str] = None, dry_run: bool = False) -> dict`

Convenience wrapper around `repair` that accepts a filename (resolves to hash) or `None` for all files.

---

### `full_registry_health_check() -> dict`

Alias for `full_audit()`.

---

## 14. Dashboard (Disk Usage and Treemap)

The dashboard functionality is now a separate service class `DashboardService` to keep the core store lean. You create an instance by passing a store.

### `DashboardService(store: DarwinStore)`

**Methods:**

### `get_treemap(subpath: Optional[str] = None) -> dict`

Builds a hierarchical tree of the entire store directory and returns a dict suitable for React treemap. If `subpath` is given (e.g., `"indexes/abc123..."`), only that subtree is returned.

---

### `print_treemap(subpath: Optional[str] = None) -> None`

Prints the treemap as an ASCII tree to the console.

---

### `get_disk_breakdown(filenames: Optional[Union[str, list[str]]] = None) -> dict`

Returns a per‑file breakdown of disk usage by artifact type (markdown, chunks, embeddings, etc.). If `filenames` is `None`, includes all files.

Output:
```python
{
    "scope": "all" | "selected",
    "files": {
        "file.pdf": {
            "markdown": {"bytes": 1234, "human": "1.2 KB"},
            "chunks": {...},
            ...
            "total": {"bytes": ..., "human": ...}
        }
    },
    "grand_total": {"bytes": ..., "human": ...}   # only for scope="all"
}
```

---

## 15. Store Registry (Multi‑Store Management)

These static methods operate on the central registry and are available directly on the `DarwinStore` class.

- `list_stores()`
- `load_store(store_id)`
- `load_store_by_name(name)`
- `get_default_store()`

All described in [Section 1](#1-initialization-and-store-management).

---

## 16. Data Models

### `Source` (dataclass)
- `uuid`: str
- `display_name`: str
- `current_path`: Optional[str]
- `content_hash`: str
- `file_size`: int
- `mtime`: float
- `created_at`: str (ISO)
- `updated_at`: str (ISO)

### `Content` (dataclass)
- `hash`: str
- `stages`: set[str] (stage names that are complete)
- `created_at`: str
- `updated_at`: str
- *Properties*: `has_markdown`, `has_chunks`, etc. (derived from stages)

---

## 17. Exceptions and Error Handling

Most methods that fail due to missing files or mismatches return `None` or empty results. However, some operations raise explicit exceptions:

- `ValueError`: invalid content hash, invalid path, file not registered (for `ensure_image_dir`).
- `NotADirectoryError`: if a supplied path exists but is not a directory.
- `KeyError`: when `get_pipeline_report(filename)` is given a non‑registered file.
- `RuntimeError`: when `load_vector` encounters a model dimension mismatch? Actually it logs a warning and returns `None`; we might want to raise in future versions, but currently it returns `None`.

We recommend checking return values for `None` and handling missing data gracefully.

---

*End of API Reference*