# run_pipeline

Run ingestion and/or indexing pipeline stages on files or directories.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stages` | string[] | *(required)* | Stages to run (see valid values below) |
| `target` | string | `""` | File path, directory path, or URL to ingest |
| `store` | string \| null | `null` | Store ID, name, or path. `null` = default store |
| `files` | string[] \| null | `null` | Filename(s) to process within target directory (omit for all) |
| `ocr` | bool | `false` | Enable OCR for images/PDFs during ingestion |
| `recursive` | bool | `false` | Recurse into subdirectories |
| `precise` | bool | `false` | Use precise parsing (slower, more accurate) |
| `rerun` | bool | `false` | Re-process files that already have artifacts |
| `threshold` | float | `0.5` | Embedding similarity threshold for indexing |
| `idf_smoothing` | float | `1.5` | IDF smoothing factor for BM25 indexing |
| `only_kw` | bool | `false` | Index keyword embeddings only |
| `only_semantic` | bool | `false` | Index semantic embeddings only |
| `strict` | bool | `true` | Halt pipeline on first stage failure |

## Valid stage values

- **Ingestion**: `ingestion`, `ingest`, `extract`, `parse`, `chunk`, `ocr`, `clean`
- **Indexing**: `indexing`, `index`, `embed`

## Target parameter

- **Directory**: ingests all files within it. Use `recursive=true` for subdirectories.
- **File**: ingests a single file.
- **URL**: fetches and ingests a remote file.
- Use `files` to target specific filenames within a directory (omit to process all).

## Usage guidance

- Use `rerun=true` to re-process files that already have artifacts.
- `strict=true` (default) halts on first failure; set `false` to continue through errors.
- Logs in the response are filtered to **WARNING/ERROR/SUCCESS only**.
- Pass `store` to run against a specific store.
- Use `list_files` to check which stages are already complete before running.

---

# purge_artifacts

Delete pipeline artifacts for specific files.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stages` | string[] | *(required)* | Stages to purge: `"ingestion"` or `"indexing"` (or aliases) |
| `filenames` | string[] | *(required)* | Exact filenames to purge |
| `dry_run` | bool | `false` | Preview what would be deleted without making changes |

**Always pass `dry_run=true` first.** Filenames must match the originals used during ingestion. Use `list_files` to see available filenames.

---

# list_files

List all files tracked by the default store with their pipeline status. This is the primary tool for checking pipeline status (replaces the deprecated `pipeline_report`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | string \| null | `null` | Filter to a single file; omit for all files |

Returns the pipeline report — each file shows completed stages.
Use this to find exact filenames before running `purge_artifacts`.
