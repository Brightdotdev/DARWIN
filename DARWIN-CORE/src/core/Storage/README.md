# Storage — DarwinStore

**What it is:** The central persistence layer. Manages file registration, artifact I/O, search indexes, store health/audit/repair, and multi-store registry. Everything in Darwin reads from or writes to a `DarwinStore`.

**Why it exists:** RAG needs a structured way to track source files, their processed artifacts (markdown, chunks, embeddings, vectors, BM25), and provide query interfaces for retrieval. This replaces ad-hoc file management with a consistent store abstraction.

## Files

| File | Lines | Purpose |
|---|---|---|
| `DarwinStore.py` | 1376 | Central store — register files, read/write artifacts, status, health, audit, repair |
| `StoreSearch.py` | 269 | Mixin — USearch vector search, NumPy fallback, single/multi-file retrieval |
| `StorePaths.py` | — | SHA256 hashing, artifact path resolution via `ARTIFACT_MANIFEST` |
| `LocalRegistry.py` | — | Per-store SQLite registry (sources table, content index) |
| `StoreRegistry.py` | — | Central multi-store registry (SQLite) |
| `MetaManager.py` | — | Per-hash meta.json management |
| `CacheManager.py` | — | LRU in-memory artifact cache (max 256 entries) |
| `Constants.py` | — | `ARTIFACT_MANIFEST`, `CORE_STAGES` definitions |
| `DataModels.py` | — | `Source` dataclass |
| `StoreUtils.py` | — | Utility functions |
| `DashboardService.py` | — | Dashboard rendering |
| `DiskUsage.py` | — | Disk space analysis |
| `Transanctions.py` | — | Atomic file writes |

## Usage

```python
from src.core.Storage.DarwinStore import DarwinStore

# Get or create default store
store = DarwinStore()
store.bootstrap()

# Or load specific store
store = DarwinStore.load_store("store-id")

# Register a file
store.register_file("path/to/doc.pdf")

# Write artifacts
store.write_artifact(file_hash, "markdown", "## Content...")
store.write_artifact(file_hash, "chunks", chunk_df)   # Polars DataFrame
store.write_artifact(file_hash, "embeddings", emb_array)  # NumPy array

# Read artifacts
md = store.read_artifact(file_hash, "markdown")
chunks = store.read_artifact(file_hash, "chunks")

# Search (via StoreSearch mixin)
results = store.retrieve_by_vector(query_emb, top_k=10)

# Status
status = store.file_status(filename)
health = store.get_embedding_health()
report = store.full_audit()

# Multi-store
stores = DarwinStore.list_stores()
default = DarwinStore.get_default_store()
```

## Artifact Manifest

Defined in `Constants.py`:

| Artifact | File Pattern | Format |
|---|---|---|
| `meta` | `meta.json` | JSON |
| `markdown` | `markdown.md` | Text |
| `markdown-display` | `display-markdown.md` | Text |
| `chunks` | `chunks.parquet` | Parquet |
| `embeddings` | `vectors/embeddings.f16.npy` | NumPy float16 |
| `vector-index` | `vectors/vector.usearch` | USearch |
| `vector-meta` | `vectors/vector.meta.json` | JSON |
| `bm25` | `bm25.npz` | SciPy NPZ |
| `bm25-tokens` | `bm25_tokens.parquet` | Parquet |
| `stopwords` | `stopwords.json` | JSON |
| `images` | `images/` | PNG files |
