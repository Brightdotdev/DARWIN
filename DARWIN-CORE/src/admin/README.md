# Darwin Admin CLI

Full orchestrator control for power users. 30 subcommands across 7 groups.

## Usage

```bash
darwin-admin <group> <command> [options]
# or:   
python -m src.admin <group> <command> [options]
```

| Global Option | Description |
|---|---|
| `--verbose`, `-v` | Enable verbose error output (shows full tracebacks) |

## Commands

### `setup` — First-time configuration

| Command | Description |
|---------|-------------|
| `interactive` | Guided numbered menus — detect hardware, choose models, set API keys |
| `profile --profile recommended` | Non-interactive, install recommended model profile |
| `onboard --preferences embedding=... nlp=...` | Non-interactive with explicit model IDs |

### `models` — Model lifecycle

| Command | Description |
|---------|-------------|
| `status` | Show installed status per category |
| `list [--category] [--provider]` | List all registered models (with optional filters) |
| `install <model_id>... [--no-download] [--force]` | Install one or more models |
| `uninstall <model_id>... [--keep-files]` | Remove models |
| `switch <category> <model_id> [--no-download]` | Change active model for a category |
| `keys set/status/remove --provider <name> [--key]` | Manage API keys |

### `store` — Store introspection

| Command | Description |
|---------|-------------|
| `status` | Store overview (sources, pipeline stages, searchable files) |
| `files` | List all files with pipeline stage and search readiness |
| `file <filename>` | Detailed status for one file |
| `audit` | Full store audit |
| `health [--model <id>]` | Health check (embedding or registry) |
| `repair [--dry-run]` | Repair integrity issues |

### `pipeline` — Pipeline execution

The pipeline processes documents in two stages:

1. **Ingestion** — Parse files → extract text → chunk into passages → save as Parquet
2. **Indexing** — Build BM25 keyword index (NPZ) + generate embeddings (NumPy float16) + build vector index (USearch)

You can run both together (`run`), or each stage independently (`ingest`, `index`).

| Command | Description |
|---------|-------------|
| `run <target> [--ocr] [--recursive] [options]` | Full ingest + index |
| `ingest <target> [options]` | Parse + chunk only (no indexing) |
| `index [--files] [--store] [--threshold] [options]` | BM25 + embedding indexing only |
| `purge <filenames>... [--stages] [--store]` | Remove artifacts for files |

**Required parameters:**
- `run` and `ingest` require a `target` — a file path, directory, or static URL
- `purge` requires one or more `filenames` to clean up

**Optional parameters (common):**
- `--store <id|name|path>` — Target a specific store (default: auto-detect)
- `--files <file1,file2>` — Process only specific files (comma-separated)
- `--ocr` — Enable OCR for scanned PDFs (ingestion only)
- `--recursive` — Recurse into subdirectories (ingestion only)
- `--rerun` — Re-process files even if already done

**Indexing parameters:**
- `--threshold <float>` — Minimum relevance score to include a chunk (default: 0.5). Lower = more results, higher = more precise
- `--idf-smoothing <float>` — BM25 IDF smoothing factor (default: 1.5). Higher = more weight on rare terms
- `--only-kw` — Build keyword index only (skip embeddings)
- `--only-semantic` — Build vector index only (skip BM25)
- `--precise` — Precision parsing mode (slower but more accurate text extraction)

**What happens during a run:**
```
darwin-admin pipeline run ./docs --ocr
```
1. Store is located (or created if new)
2. Files are discovered in `./docs/` (recursively if `--recursive`)
3. Each file is parsed (PDF→markdown, HTML→text, DOCX→text, etc.)
4. OCR is applied to scanned pages (if `--ocr` and Tesseract available)
5. Parsed text is chunked into passages (~512 tokens each)
6. BM25 keyword index is built from chunks
7. Embeddings are generated and stored (NumPy + USearch)
8. Results are logged with success/failure per file

**Expected output:**
- Successful files show `OK` with stage completion
- Failed files show `ERR` with the specific error
- Summary shows total files processed, chunks created, time elapsed

### `logs` — Log management

| Command | Description |
|---------|-------------|
| `query [--session] [--levels] [--limit]` | Query log entries |
| `sessions` | List all log sessions |
| `session-info <session-id>` | Detailed info for one session |
| `stats` | Log statistics |
| `purge [--older-than]` | Purge old logs |
| `purge-session <session-id>` | Purge a specific session |
| `export [--format] [--output]` | Export logs |
| `compact` | Compact log storage |
| `set-retention <days>` | Set log retention policy |

### `system` — Diagnostics

| Command | Description |
|---------|-------------|
| `info` | Hardware detection + model recommendations |

### `status` — Overview

| Command | Description |
|---------|-------------|
| `status [--store] [--initialize]` | Dashboard: store files, installed models, preferences |

By default, `status` shows store info and file counts without initializing models (fast). Use `--initialize` to also run model initialization and show model readiness status (slower).

## Examples

```bash
# Interactive setup (guided menus, numbered selection)
darwin-admin setup interactive

# Non-interactive setup (Docker/CI)
darwin-admin setup profile --profile recommended --no-download
darwin-admin setup onboard --preferences embedding=intfloat/multilingual-e5-small-onnx

# Model management
darwin-admin models status
darwin-admin models list --category embedding
darwin-admin models install intfloat/multilingual-e5-small-onnx
darwin-admin models switch embedding intfloat/multilingual-e5-large-onnx
darwin-admin models keys set openai sk-...

# Store inspection
darwin-admin store status
darwin-admin store files
darwin-admin store audit
darwin-admin store health

# Pipeline operations
darwin-admin pipeline run /data/docs --ocr --recursive
darwin-admin pipeline ingest /data/papers/
darwin-admin pipeline index --threshold 0.6
darwin-admin pipeline index --files doc1.pdf --store my-store
darwin-admin pipeline purge file1.pdf file2.pdf --stages ingestion

# System info and status
darwin-admin system info
darwin-admin status                     # fast — no model init
darwin-admin status --initialize        # full — with model init

# Verbose error output (shows tracebacks)
darwin-admin --verbose pipeline run /data/docs --ocr
```
