# create_store

Create a new DarwinStore at a given filesystem path for isolating data.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | string | *(required)* | Filesystem path for the new store |
| `name` | string \| null | `null` | Optional human-readable name |

## Usage guidance

- Create a project-specific store before running pipeline to isolate data.
- Pass the returned store ID to other tools via the `store` parameter.
- `name` is optional; if omitted the store is identified by path only.

---

# list_files

List all files tracked by a store with their pipeline status.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | string \| null | `null` | Filter to a single file; omit for all files |
| `store` | string \| null | `null` | Store ID, name, or path. `null` = default store |

Returns the pipeline report — each file shows completed stages.
Use this to find exact filenames before running `purge_artifacts`.

---

# file_status

Get detailed status for a single file across all pipeline stages.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | string | *(required)* | Exact filename as used during ingestion |
| `store` | string \| null | `null` | Store ID, name, or path. `null` = default store |

Use this to check what artifacts exist for a specific file.

---

# Resources

MCP resources are read-only data sources the agent can fetch without tool calls.

| Resource | Description |
|---|---|
| `darwin://store/report` | Pipeline report — all files with completed stages and aggregate stats |
| `darwin://search/results` | Saved search results index — lists all previously saved searches |
