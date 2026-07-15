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

## list_files

List all files tracked by the default store with their pipeline status.

No parameters. Returns the pipeline report — each file shows completed stages.
Use this to find exact filenames before running `purge_artifacts`.

## file_status

Get detailed status for a single file across all pipeline stages.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | string | *(required)* | Exact filename as used during ingestion |

Use this to check what artifacts exist for a specific file.
