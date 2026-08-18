# search_darwin

Search the DARWIN knowledge base for relevant passages.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | string | *(required)* | Search query (max 4096 chars) |
| `method` | `"hybrid"` \| `"semantic"` \| `"keyword"` | `"hybrid"` | Search strategy |
| `store` | string \| null | `null` | Store ID, name, or path. `null` = default store |
| `files` | string[] \| null | `null` | Filter to specific file(s) |
| `top_k` | int | `5` | Number of top results to return |
| `max_passages` | int | `5` | Maximum passages to retrieve |
| `max_chars` | int | `6000` | Character limit on returned text |
| `min_passage_score` | float | `0.15` | Minimum relevance score to include a passage |
| `merge_strategy` | `"none"` \| `"adjacent"` \| `"aggressive"` | `"adjacent"` | How to merge overlapping passages |
| `fill_gaps` | bool | `false` | Fill gaps between non-adjacent results |
| `max_per_source` | int | `2` | Max passages from any single file |
| `prefer_sources` | string[] \| null | `null` | Favor passages from these files |
| `exclude_sources` | string[] \| null | `null` | Exclude passages from these files |
| `filter_structural` | bool | `true` | Filter out boilerplate/structural content |
| `structural_penalty_factor` | float | `0.5` | Penalty multiplier for non-body content (0.0-1.0). Lower = harsher. |
| `diversity_rerank` | bool | `true` | Apply 15% compound penalty per extra result from same source |
| `include_audit` | bool | `false` | Include per-passage scoring breakdown in results |
| `fallback_strategy` | `"expand"` \| `"relax"` \| `"both"` | `"both"` | Fallback when hybrid search is too narrow |
| `rerank` | bool | `false` | Apply reranking to final results |
| `save` | bool | `false` | Save results as JSON for later retrieval |
| `results_dir` | string \| null | `null` | Custom directory for saved results. `null` = default (`{base}/search-results/`) |

## Usage guidance

- **Check `summary` first** for a quick overview before reading individual passages.
- **`method="hybrid"`** (default) combines keyword + semantic; best for most queries.
- **`method="semantic"`** is better for conceptual or fuzzy questions.
- **`method="keyword"`** is better for exact terms, names, or code references.
- If results are sparse, lower `min_passage_score` (e.g. `0.15`).
- `fallback_used=true` in the response means the search broadened automatically — check `fallback_reason` for why.
- Use `prefer_sources`/`exclude_sources` to target or skip specific files.
- Pass a `store` to search a non-default store.
- Use `save=true` to persist results. Retrieve them later via the `darwin://search/results` resource or `get_search_result_by_id`.

### Power-user tuning

- **`include_audit=true`** adds per-passage scoring breakdown: `raw_score`, `penalty_factor`, `structural_type`, `was_penalized`. Use this to debug why results rank the way they do.
- **`diversity_rerank=false`** disables the 15% compound penalty for multiple results from the same source. Use when you want pure relevance ranking, e.g. deep-diving into one file.
- **`structural_penalty_factor`** (0.0-1.0) controls how aggressively non-body content (tables, captions, footnotes) is penalized. Default `0.5` = 50% penalty. Set `1.0` to disable structural penalties entirely.
- **`max_per_source`** caps passages per file. Default `2`. Raise to `5-10` for deeper coverage of specific documents; lower to `1` for maximum diversity.

## Response

```json
{
  "success": true,
  "data": {
    "passages": [
      {
        "text": "Machine learning is a subset of AI...",
        "metadata": {
          "source": "ml-intro.pdf",
          "page_start": 12,
          "page_end": 12,
          "chunk_indices": [42],
          "chunk_count": 1,
          "is_merged": false,
          "gap_filled": false,
          "structural_type": "body",
          "penalty_factor": 1.0,
          "was_penalized": false,
          "avg_score": 0.85,
          "semantic_score": 0.9,
          "keyword_score": 0.7,
          "rrf_score": 0.85,
          "rerank_score": null,
          "matched_terms": ["machine learning"]
        },
        "audit": null
      }
    ],
    "total": 5,
    "total_chars": 3200,
    "sources": ["ml-intro.pdf", "ai-handbook.pdf"],
    "max_score": 0.92,
    "avg_score": 0.71,
    "elapsed_ms": 145,
    "fallback_used": false,
    "fallback_reason": null
  }
}
```

---

# get_search_results

List all saved search results. Consider using the `darwin://search/results` resource for quick context lookups without a tool call.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results_dir` | string \| null | `null` | Custom directory. `null` = default (`{base}/search-results/`) |

## Response

Returns a list of saved result files with summary metadata:

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "filename": "machine-learning_20260718_143022.json",
        "query": "machine learning",
        "method": "hybrid",
        "total_passages": 5,
        "max_score": 0.92,
        "elapsed_ms": 145,
        "fallback_used": false,
        "saved_at": "2026-07-18T14:30:22"
      }
    ],
    "total": 1
  }
}
```

---

# get_search_result_by_id

Load a saved search result by its filename.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | string | *(required)* | Filename from `get_search_results` (e.g. `"machine-learning_20260718_143022.json"`) |
| `results_dir` | string \| null | `null` | Custom directory. `null` = default (`{base}/search-results/`) |

## Response

Returns the full `RetrievalContext` (same structure as `search_darwin` response data). If not found:

```json
{
  "success": false,
  "error": "Result 'xyz.json' not found in {path}. Run search_darwin with save=true to create one."
}
```

---

# search_lists

Search structured data (JSON arrays, CSVs, spreadsheets) by field values with hybrid text fallback. Routes through `RetrievalOrchestrator.search_lists()` — same orchestration pattern as `search_darwin`.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | string \| null | `null` | Text query for hybrid/re-ranking (optional if filters provided) |
| `filters` | object \| null | `null` | Dict of field-value pairs to filter by (see filter syntax below) |
| `mode` | `"auto"` \| `"structured"` \| `"text"` | `"auto"` | `"auto"` = try structured, fallback to text. `"structured"` = filter only. `"text"` = hybrid only |
| `store` | string \| null | `null` | Store name/path/ID. `null` = default store |
| `files` | string[] \| null | `null` | Limit search to specific structured file(s) |
| `top_k` | int | `5` | Max results to return |
| `rerank` | bool | `false` | Apply semantic re-ranking to filtered results |
| `filter_boost_factor` | float | `0.2` | Score boost added to filter matches (0.0-1.0) |
| `save` | bool | `false` | Save results to disk |

### Filter syntax (Django-style double underscore operators)

| Operator | Example | Description |
|---|---|---|
| *(exact)* | `{"status": "open"}` | Exact match (case-insensitive for strings) |
| `__contains` | `{"title__contains": "login"}` | Substring match |
| `__startswith` | `{"title__startswith": "error"}` | Prefix match |
| `__endswith` | `{"title__endswith": "404"}` | Suffix match |
| `__gt` | `{"score__gt": 5}` | Greater than |
| `__lt` | `{"score__lt": 10}` | Less than |
| `__gte` | `{"score__gte": 5}` | Greater than or equal |
| `__lte` | `{"score__lte": 10}` | Less than or equal |
| `__neq` | `{"status__neq": "closed"}` | Not equal |
| `__in` | `{"status__in": ["open", "pending"]}` | Match any value in list |
| `__exists` | `{"email__exists": true}` | Field existence check |

Nested fields use dot-notation: `{"user.name": "Bright"}`, `{"user.role__contains": "admin"}`

## Usage guidance

- Use `get_schema` first to discover available fields and their types.
- **`mode="auto"`** (default) tries structured filtering first, falls back to hybrid text search if no filter matches.
- **`mode="structured"`** returns empty if no filters match — no text fallback.
- **`mode="text"`** skips structured search entirely, goes straight to hybrid text.
- Combine filters with a query for re-ranked results: `filters={"gender": "Female"}, query="nurse"` ranks filter matches by semantic similarity to "nurse".
- `filter_boost_factor` works like `structural_penalty_factor` — additive constant added to filter match scores.
- Use `rerank=true` to apply cross-encoder re-ranking when both `query` and `filters` are provided.
- Call `get_schema` with `validate_filters` to pre-validate filter keys before searching.

## Response (structured mode)

```json
{
  "success": true,
  "mode": "structured",
  "records": [
    {
      "fields": {"status": "open", "priority": "high", "assignee": "Alice"},
      "score": 1.0,
      "source": "tickets.csv",
      "chunk_index": 42,
      "heading": ""
    }
  ],
  "passages": [],
  "total": 5,
  "warning": null
}
```

## Response (text fallback)

```json
{
  "success": true,
  "mode": "text",
  "records": [],
  "passages": [
    {
      "text": "Matching passage text...",
      "metadata": { "source": "tickets.csv", "avg_score": 0.85, "..." : "..." }
    }
  ],
  "total": 3,
  "warning": "No structured matches found — fell back to hybrid text search"
}
```

---

# get_schema

Get schema information for structured files (JSON arrays, CSVs, spreadsheets). Four modes of operation.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | string \| null | `null` | Single file to inspect (e.g. `"tickets.csv"`) |
| `files` | string[] \| null | `null` | Multiple files — returns mapped schemas + overlapping key detection |
| `list_all` | bool \| null | `null` | `true` = return schemas for ALL indexed structured files |
| `validate_filters` | object \| null | `null` | Filter dict to validate against schemas before searching |
| `store` | string \| null | `null` | Store name/path/ID. `null` = default store |

**Exactly one mode** must be provided: `file`, `files`, `list_all=true`, or `validate_filters`.

## Usage guidance

- **Single file** (`file="tickets.csv"`) — discover available fields, types, and record count.
- **Multiple files** (`files=["a.csv", "b.json"]`) — compare schemas, detect overlapping keys that could cause ambiguous filters.
- **List all** (`list_all=true`) — overview of every structured file in the store.
- **Validate filters** (`validate_filters={"status": "open"}`) — pre-check filter keys/types before calling `search_lists`. Returns warnings for missing keys or type mismatches.

Always call this before `search_lists` to know what fields are available and how to spell them.

## Response (single file)

```json
{
  "success": true,
  "data": {
    "keys": ["status", "priority", "assignee", "created_at"],
    "types": {"status": "str", "priority": "str", "assignee": "str", "created_at": "str"},
    "record_count": 1200,
    "file_type": "csv",
    "index_mode": "structured"
  }
}
```

## Response (list all)

```json
{
  "success": true,
  "data": {
    "schemas": {
      "tickets.csv": {"keys": ["status", "priority"], "record_count": 1200},
      "orders.json": {"keys": ["order_id", "total"], "record_count": 500}
    },
    "total_files": 2,
    "overlapping_keys": {
      "status": ["tickets.csv", "orders.json"]
    }
  }
}
```

## Response (validate filters)

```json
{
  "success": true,
  "data": {
    "valid": {"status": "open"},
    "warnings": ["Filter key 'foo' not found in any schema. Available keys: ['status', 'priority', 'assignee']"],
    "files_checked": ["tickets.csv"],
    "schemas_found": true
  }
}
```
