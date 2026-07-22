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
- Use `save=true` to persist results. Retrieve them later with `get_search_results` and `get_search_result_by_id`.

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

List all saved search results.

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
