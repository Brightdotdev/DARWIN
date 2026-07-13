# Darwin Admin CLI

Full orchestrator control for power users. 31 subcommands across 8 groups.

## Usage

```bash
darwin-admin <group> <command> [options]
# or:
python -m src.admin <group> <command> [options]
```

| Global Option | Description |
|---|---|
| `--verbose`, `-v` | Enable verbose error output (shows full tracebacks) |

## Module Structure

The admin module is split by concern across multiple files:

```
src/admin/
  cli.py                 — Entry point, argparse builder, lazy orchestrator singletons
  renderer.py            — AdminRenderer: ANSI output, tables, panels, trees
  setup.py               — Interactive onboarding + config helpers
  commands_models.py     — Model lifecycle: status, list, install, uninstall, switch, keys
  commands_store.py      — Store introspection: status, files, file, audit, health, repair
  commands_pipeline.py   — Pipeline execution: run, ingest, index, purge
  commands_logs.py       — Log management: query, sessions, stats, purge, export, compact
  commands_system.py     — System info + top-level status dashboard
  uninstall.py           — Full darwin uninstall (discover stores, clean artifacts, remove data)
```

## Commands

### `setup` — First-time configuration

| Command | Description |
|---------|-------------|
| `interactive` | Guided per-category setup — detect hardware, pick models, configure logging |
| `profile --profile recommended` | Non-interactive, install recommended model profile |
| `onboard --preferences embedding=... nlp=...` | Non-interactive with explicit model IDs |

#### Interactive setup flow

The interactive setup walks through each step sequentially:

1. **Detect system** — RAM, disk, CPU, GPU, hardware tier (fast, no model init)
2. **Per-category model selection** — one category at a time:
   - **Embedding (required):** Lists all models, user picks one (recommended is default)
   - **Optional categories** (reranker, nlp, ocr, vision): Shows recommended model + requirements, user can accept, browse all, or skip
3. **API keys** — prompts only for keys needed by selected API models
4. **Install** — downloads local models only (API models just need keys)
5. **Logging** — mode (terminal/sse/both/none), persistence, retention, SSE port

#### Config detection

When run, setup checks all three config files:
- `darwin.json` — setup config (models, keys, environment)
- `logging.json` — logging mode, SSE port, retention
- `models.json` — active model per category

If all three exist, offers: change specific setting / full re-run / exit.
If partial: offers to complete missing pieces.
If none: runs full setup.

### `models` — Model lifecycle

| Command | Description |
|---------|-------------|
| `status` | Show installed status per category |
| `list [--category] [--provider]` | List all registered models (with optional filters) |
| `install <model_id>... [--no-download] [--force]` | Install one or more models |
| `discard --category <cat> [--all] [--active] [--inactive] [--delete]` | Discard models per category |
| `switch <category> <model_id> [--no-download]` | Change active model for a category |
| `keys set/status/remove --provider <name> [--key]` | Manage API keys |

#### Discard (category-based)

Discard models from one or more categories. Supports multiple `--category` flags.

```bash
# Interactive: shows installed models, user picks which to discard
darwin-admin models discard --category embedding

# Discard all installed models in a category
darwin-admin models discard --category embedding --all --delete

# Discard only the active model
darwin-admin models discard --category embedding nlp --active --delete

# Discard only inactive models (keep the active one)
darwin-admin models discard --category ocr --inactive --delete
```

| Flag | Description |
|------|-------------|
| `--category`, `-c` | Category name(s) to discard from (required, can repeat) |
| `--all` | Discard all installed models in each category |
| `--active` | Discard only the active model |
| `--inactive` | Discard only inactive models (keep the active one) |
| `--delete` | Delete model files from disk (default: keep files) |

**Behavior when no scope flag is given:**
- Single installed model → discarded directly
- Multiple installed models → interactive selector (pick one, all, or skip)

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

### `uninstall` — Remove darwin

Safely removes all darwin data from the system. Discovers all registered stores (main + extra), removes only darwin artifacts from extra stores (keeping user files), cleans the main data directory, and optionally uninstalls the pip package.

```bash
darwin-admin uninstall              # full cleanup + pip uninstall
darwin-admin uninstall --dry-run   # preview what would be removed
darwin-admin uninstall --keep-package  # clean data only, keep the package
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview what would be removed without making changes |
| `--keep-package` | Only clean data, skip `pip uninstall darwin-core` |

**What it does:**

1. **Discovers stores** — queries `~/.darwin/stores_registry.db` for all registered stores (main + extra)
2. **Cleans extra stores** — for each store outside the main base dir, removes only `indexes/`, `registry.db`, `.darwin_store_id`, `checkpoints/`, `conversations/` (leaves your original files untouched)
3. **Removes global registry** — deletes `~/.darwin/`
4. **Removes main data directory** — deletes `%LOCALAPPDATA%/Darwin` (Windows), `~/Library/Application Support/Darwin` (macOS), or `~/.local/share/darwin` (Linux)
5. **Uninstalls package** — runs `pip uninstall darwin-core --yes` (skipped with `--keep-package`)

## Logging modes

The logging configuration supports four modes:

| Mode | Description |
|------|-------------|
| `terminal` | Logs to console only (default) |
| `sse` | Stream logs via SSE (for MCP / remote viewing) |
| `both` | Console + SSE simultaneously |
| `none` | Silent (no log output) |

Additional settings:
- **persist** — Save logs to SQLite history (separate from mode)
- **retention_days** — How long to keep logs (3, 7, 14, 30 days, or never)
- **sse_port** — Port for SSE streaming (shared with MCP server, auto-resolved if busy)

These are saved to `logging.json` in the config directory. MCP reads the port from this file at startup.

## Examples

```bash
# Interactive setup (per-category guided flow)
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

# Discard models (category-based)
darwin-admin models discard --category embedding --all --delete
darwin-admin models discard --category embedding nlp --active --delete
darwin-admin models discard --category ocr --inactive --delete

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

# Uninstall (clean all darwin data)
darwin-admin uninstall --dry-run        # preview first
darwin-admin uninstall --keep-package   # remove data only
darwin-admin uninstall                  # full cleanup + pip uninstall

# Verbose error output (shows tracebacks)
darwin-admin --verbose pipeline run /data/docs --ocr
```
