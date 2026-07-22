# run_pipeline

Run ingestion and/or indexing pipeline stages on files or directories.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stages` | string[] | *(required)* | Stages to run (see valid values below) |
| `target` | string | `""` | Local path or static URL to ingest |
| `store` | string \| null | `null` | Store ID, name, or path. `null` = default store |
| `files` | string[] \| null | `null` | Filename(s) to process (omit for all) |
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

## Usage guidance

- Always pass a `target` path or URL for ingestion stages.
- Use `rerun=true` to re-process files that already have artifacts.
- `strict=true` (default) halts on first failure; set `false` to continue through errors.
- Logs in the response are filtered to **WARNING/ERROR/SUCCESS only** — formatted as `[session][LEVEL] message`.
- Pass `store` to run against a specific store.
- Run `pipeline_report` before to check which stages are already complete.

## purge_artifacts

Delete pipeline artifacts for specific files.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stages` | string[] | *(required)* | Stages to purge |
| `filenames` | string[] | *(required)* | Exact filenames to purge |
| `dry_run` | bool | `false` | Preview what would be deleted without making changes |

**Always pass `dry_run=true` first.** Filenames must match the originals used during ingestion.

## pipeline_report

Get pipeline report for files — shows which stages are complete per file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | string \| null | `null` | Check a specific file; omit for all files |

Use this before running pipeline to decide if `rerun` is needed.
