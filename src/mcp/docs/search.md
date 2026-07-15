# search_darwin

Search the DARWIN knowledge base for relevant passages.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | string | *(required)* | Search query (max 4096 chars) |
| `method` | `"hybrid"` \| `"semantic"` \| `"keyword"` | `"hybrid"` | Search strategy |
| `store` | string \| null | `null` | Store ID, name, or path. `null` = default store |
| `files` | string \| null | `null` | Filter to a specific file |
| `top_k` | int | `5` | Number of top results to return |
| `max_passages` | int | `5` | Maximum passages to retrieve |
| `max_chars` | int | `6000` | Character limit on returned text |
| `min_passage_score` | float | `0.3` | Minimum relevance score to include a passage |
| `merge_strategy` | `"none"` \| `"adjacent"` \| `"aggressive"` | `"adjacent"` | How to merge overlapping passages |
| `fill_gaps` | bool | `false` | Fill gaps between non-adjacent results |
| `max_per_source` | int | `2` | Max passages from any single file |
| `prefer_sources` | string \| null | `null` | Favor passages from this file |
| `exclude_sources` | string \| null | `null` | Exclude passages from this file |
| `filter_structural` | bool | `true` | Filter out boilerplate/structural content |
| `fallback_strategy` | `"expand"` \| `"relax"` \| `"both"` | `"both"` | Fallback when hybrid search is too narrow |
| `rerank` | bool | `false` | Apply reranking to final results |

## Usage guidance

- **Check `summary` first** for a quick overview before reading individual passages.
- **`method="hybrid"`** (default) combines keyword + semantic; best for most queries.
- **`method="semantic"`** is better for conceptual or fuzzy questions.
- **`method="keyword"`** is better for exact terms, names, or code references.
- If results are sparse, lower `min_passage_score` (e.g. `0.15`).
- `fallback_used=true` in the response means the search broadened automatically — check `fallback_reason` for why.
- Use `prefer_sources`/`exclude_sources` to target or skip specific files.
- Pass a `store` to search a non-default store.

## Response

```json
{
  "success": true,
  "data": {
    "passages": [{"text": "...", "score": 0.85, "source": "...", "position": 12}],
    "total": 5,
    "total_chars": 3200,
    "sources": ["file1.md", "file2.md"],
    "max_score": 0.92,
    "avg_score": 0.71,
    "elapsed_ms": 145,
    "fallback_used": false,
    "fallback_reason": null
  }
}
```
