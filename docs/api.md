# API — Darwin Facade

**What it is:** The public Python API for darwin-rag. A single `Darwin` class that wraps all orchestrators behind a flat, documented surface — no need to juggle `StoreOrchestrator`, `RetrievalOrchestrator`, `PipelineOrchestrator`, or `LoggerOrchestrator` directly.

**Why it exists:** Two audiences, two interfaces. The CLI (`darwin` / `darwin-admin`) is for humans. The MCP server is for AI agents. The `Darwin` class is for Python developers — anyone embedding darwin-rag as a library in their own app, script, or agent.

## Files

| File | Lines | Purpose |
|---|---|---|
| `api.py` | ~320 | `Darwin` class — all tiers, sync + async |
| `__init__.py` | 1 | Re-exports `Darwin` from `core` |

## Quick Start

```python
from core import Darwin

# Default store (registered during setup)
d = Darwin()

# Or point at a specific directory
d = Darwin("/data/my-project", name="project-alpha")

# Ingest
d.ingest("./papers", recursive=True)

# Search
results = d.search("what is this paper about")
for p in results["passages"]:
    print(f"[{p['score']:.3f}] {p['source']}: {p['text'][:80]}...")

# Structured search (CSVs, JSON)
records = d.search_records(filters={"status": "active", "year__gte": 2024})

# Async (FastAPI, agent loops)
hits = await d.search_async("query", method="semantic")
```

## Tiers

### Core (the 90% use case)

| Method | Description |
|---|---|
| `ingest(target, *, files, ocr, recursive, rerun, strict, stages)` | Parse + chunk + index files into the store |
| `search(query, *, method, top_k, max_passages, max_chars, ...)` | Hybrid / semantic / keyword search |
| `search_records(*, filters, query, mode, top_k, files, ...)` | Structured field search (CSVs, JSON) |
| `schema(*, file, files, list_all)` | Inspect schemas for structured files |

### Admin

| Method | Description |
|---|---|
| `status()` | Store identity and health overview |
| `list_files(filename)` | Pipeline report for all files (or one) |
| `file_status(filename)` | Deep pipeline status for a single file |
| `logs(*, session, level, limit)` | Query session logs |

### Advanced

| Method | Description |
|---|---|
| `validate_filters(filters, files)` | Pre-flight filter validation against schemas |
| `overlapping_keys()` | Find keys shared across structured files |
| `purge(stages, filenames, dry_run)` | Delete pipeline artifacts |

### Config

| Method | Description |
|---|---|
| `setup()` | Run the interactive setup wizard (static, CLI only) |

### Store Lifecycle

| Method | Description |
|---|---|
| `create_store(path, name)` | Create a new store at a given path (static) |
| `list_stores()` | List all registered stores (static) |

### Properties

| Property | Type | Description |
|---|---|---|
| `path` | `str` | Absolute path to the store directory |
| `store_id` | `str` | Unique store identifier |
| `store_name` | `str` | Human-readable store name |

## Async Variants

Every Core and Admin method has an `_async` counterpart:

```python
await d.search_async("query")
await d.search_records_async(filters={"status": "active"})
await d.ingest_async("./papers", recursive=True)
await d.schema_async(list_all=True)
await d.list_files_async()
await d.file_status_async("doc.pdf")
await d.logs_async(session="PIPELINE", level="quiet")
```

Async methods use `asyncio.to_thread()` internally — they offload blocking orchestrator calls to a thread pool.

## Constructor

```python
Darwin(
    store_path: str | Path | None = None,  # None = default store
    name: str | None = None,               # Optional human-readable name
    quiet: bool = True,                    # Suppress dashboard on bootstrap
)
```

- `store_path=None` resolves to the registered default store via `StoreOrchestrator`
- `quiet=True` suppresses dashboard/console output (default for library use; set `False` for CLI-style startup)
- Bootstrap is eager — models are initialized on construction

## Search Parameters

### `search()` — Text Search

| Param | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Search query |
| `method` | `str` | `"hybrid"` | `"hybrid"`, `"semantic"`, or `"keyword"` |
| `top_k` | `int` | `5` | Max results |
| `max_passages` | `int` | `5` | Max passages to return |
| `max_chars` | `int` | `6000` | Max characters in response |
| `min_passage_score` | `float` | `0.15` | Minimum score threshold |
| `merge_strategy` | `str` | `"adjacent"` | `"none"`, `"adjacent"`, or `"aggressive"` |
| `max_per_source` | `int` | `2` | Max passages per source file |
| `diversity_rerank` | `bool` | `True` | Penalize repeated sources |
| `prefer_sources` | `list[str]` | `None` | Boost specific files |
| `exclude_sources` | `list[str]` | `None` | Exclude specific files |
| `files` | `list[str]` | `None` | Restrict to specific files |
| `save` | `bool` | `False` | Persist results to disk |

### `search_records()` — Structured Search

| Param | Type | Default | Description |
|---|---|---|---|
| `filters` | `dict` | `None` | Django-style filter dict |
| `query` | `str` | `""` | Text query for hybrid fallback |
| `mode` | `str` | `"auto"` | `"auto"`, `"structured"`, or `"text"` |
| `top_k` | `int` | `5` | Max results |
| `files` | `list[str]` | `None` | Restrict to specific files |
| `rerank` | `bool` | `False` | Apply reranking |
| `save` | `bool` | `False` | Persist results to disk |

#### Filter Operators (Django-style double-underscore)

| Operator | Example | Description |
|---|---|---|
| `__exact` | `{"status__exact": "failed"}` | Exact match (default for bare keys) |
| `__contains` | `{"name__contains": "test"}` | Substring match (case-insensitive) |
| `__startswith` | `{"name__startswith": "api"}` | Prefix match |
| `__endswith` | `{"name__endswith": "test"}` | Suffix match |
| `__gt` / `__lt` | `{"latency__gt": 100}` | Numeric comparison |
| `__gte` / `__lte` | `{"year__gte": 2024}` | Numeric comparison (inclusive) |
| `__neq` | `{"status__neq": "active"}` | Not equal |
| `__in` | `{"year__in": [2023, 2024]}` | Value in list |
| `__exists` | `{"archived__exists": false}` | Field existence check |

Bare keys (e.g. `{"status": "failed"}`) default to exact match. Strings are case-insensitive.

### `ingest()` — Ingestion

| Param | Type | Default | Description |
|---|---|---|---|
| `target` | `str` | required | Path to file, directory, or URL |
| `files` | `list[str]` | `None` | Subset of filenames to ingest |
| `ocr` | `bool` | `False` | Run OCR on scanned documents |
| `recursive` | `bool` | `False` | Recurse into subdirectories |
| `rerun` | `bool` | `False` | Re-process existing artifacts |
| `strict` | `bool` | `True` | Halt on first failure |
| `stages` | `list[str]` | `["ingestion", "indexing"]` | Pipeline stages to run |

### `schema()` — Schema Inspection

| Param | Type | Default | Description |
|---|---|---|---|
| `file` | `str` | `None` | Single filename to inspect |
| `files` | `list[str]` | `None` | Multiple filenames + overlapping keys |
| `list_all` | `bool` | `False` | All indexed structured files |

## Return Shapes

### `search()` returns:
```python
{
    "passages": [{"text": "...", "score": 0.85, "source": "doc.pdf", "heading": "...", "page": 26}],
    "total": 5,
    "total_chars": 3200,
    "sources": ["doc.pdf"],
    "max_score": 0.855,
    "avg_score": 0.72,
    "elapsed_ms": 45,
    "fallback_used": False,
    "fallback_reason": None
}
```

### `search_records()` returns:
```python
# Structured mode
{
    "records": [{"fields": {"name": "test", "status": "active"}, "score": 0.9, "source": "data.csv", "page": 1}],
    "passages": [],
    "total": 3,
    "mode": "structured",
    "warning": None
}

# Text fallback mode
{
    "records": [],
    "passages": [{"text": "...", "score": 0.7, "source": "data.csv"}],
    "total": 2,
    "mode": "text",
    "warning": "No structured matches — fell back to hybrid search"
}
```

### `ingest()` returns:
```python
{
    "success": True,
    "stages_completed": ["ingestion", "indexing"],
    "stages_failed": [],
    "summary": {"files_processed": 5, "chunks_created": 120}
}
```

### `schema()` returns:
```python
{
    "schemas": {"data.csv": {"keys": ["name", "status", "year"], "total_records": 100}},
    "total_files": 1,
    "overlapping_keys": {"name": ["data.csv", "users.json"]}
}
```

## Error Handling

All methods return `dict` — no exceptions bubble up to the caller. Errors are surfaced as keys:

```python
{"error": "No schema for 'foo.csv' — not indexed as structured."}
{"success": False, "error": "Path is not writable: /readonly"}
```

Orchestrator warnings (clamped values, fallback triggers) are logged to the session log, not raised.

## Escape Hatch

Access the underlying `DarwinStore` directly:

```python
store = d._store  # DarwinStore instance
```

This is undocumented and unsupported — use at your own risk.

## Internal Wiring

```
Darwin.__init__
  └─ StoreOrchestrator(darwin_dir, name)
       └─ bootstrap(initialize_models=True, quiet=True)

d.ingest()
  └─ PipelineOrchestrator(store).build_pipeline(PipelineParameters)

d.search()
  └─ RetrievalOrchestrator(store).retrieve_and_resolve(RetrievalParameters)

d.search_records()
  └─ RetrievalOrchestrator(store).search_lists(StructuredSearchParams)

d.logs()
  └─ LoggerOrchestrator.get_logs(session, levels, limit)
```

Lazy initialization: `RetrievalOrchestrator` and `PipelineOrchestrator` are created on first use, not on `Darwin()` construction.
