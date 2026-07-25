# Darwin Admin CLI

Full orchestrator control for power users. 36 subcommands across 9 groups.

## Usage

```bash
darwin-admin <group> <command> [options]
# or:
python -m src.admin <group> <command> [options]
```

| Global Option | Description |
|---|---|
| `--verbose`, `-v` | Enable verbose error output (shows full tracebacks) |

Ctrl+C at any point exits cleanly (no partial state saved).

---

## Module Structure

```
src/admin/
  lib.py                 — Types, enums, constants, dataclasses (no magic strings)
  cli.py                 — Entry point, argparse builder, lazy orchestrator singletons
  renderer.py            — AdminRenderer: ANSI output, tables, panels, trees, interactive prompts
  setup.py               — Interactive onboarding + config helpers + logging reconfigure
  commands_models.py     — Model lifecycle: status, list, install, discard, switch, keys
  commands_store.py      — Store introspection: status, files, file, audit, health, repair
  commands_pipeline.py   — Pipeline execution: run, ingest, index, purge
  commands_search.py     — Knowledge base search: search
  commands_logs.py       — Log management: query, sessions, stats, purge, export, compact
  commands_system.py     — System info + top-level status dashboard
  uninstall.py           — Full uninstall (discover stores, clean artifacts, remove data)
```

### Architecture

```
cli.py  main()
  │
  ├── _r.show_banner()              ← prints ASCII banner (never saved to logs)
  ├── argparse parses command
  └── args.func(args)               ← dispatches to the correct handler
        │
        ├── setup.py                ← _setup_dispatch() → subcommand or interactive
        ├── commands_models.py      ← cmd_models_status/list/install/discard/switch/keys
        ├── commands_store.py       ← cmd_store_status/files/file/audit/health/repair
        ├── commands_pipeline.py    ← cmd_pipeline_run/ingest/index/purge
        ├── commands_search.py      ← cmd_search
        ├── commands_logs.py        ← cmd_logs_query/sessions/stats/purge/export/compact
        ├── commands_system.py      ← cmd_system_info / cmd_status
        └── uninstall.py            ← cmd_uninstall
```

**Lazy singletons** — Orchestrators (`_store`, `_pipeline`, `_models`) are created on first use, not at import time. Each command module receives a `_get_*` factory function, not an instance.

---

## Types (`lib.py`)

All shared types live in `lib.py`. No magic strings or raw dicts are passed between functions.

### Enums

| Enum | Values | Used by |
|------|--------|---------|
| `ModelCategory` | `embedding`, `reranker`, `nlp`, `ocr`, `vision` | Setup, models |
| `OutputMode` | `terminal`, `sse`, `both`, `none` | Setup logging |
| `LogFilter` | `all`, `quiet`, `silent` | Setup logging |
| `SetupAction` | `interactive`, `profile`, `onboard`, `validate`, `doctor`, `reset`, `logging` | Setup dispatch |
| `SetupPreset` | `required`, `recommended` | One-shot setup |
| `KeyAction` | `set`, `status`, `remove` | Models keys |
| `StoreAction` | `status`, `files`, `file`, `audit`, `health`, `repair` | Store commands |
| `PipelineAction` | `run`, `ingest`, `index`, `purge` | Pipeline commands |
| `LogAction` | `query`, `sessions`, `session-info`, `stats`, `purge`, `purge-session`, `export`, `compact`, `set-retention` | Log commands |
| `ConfigFile` | `darwin`, `logging`, `models` | Setup validation |

### Dataclasses

| Class | Fields | Used by |
|-------|--------|---------|
| `SetupStatus` | `darwin_json: bool`, `logging_json: bool`, `models_json: bool` | Setup detection |
| `DiskImpact` | `required_mb`, `available_mb`, `after_mb`, `warning` | Setup disk check |
| `LoggingConfig` | `mode`, `persist`, `retention_days`, `max_memory`, `sse_port`, `sse_host`, `log_filter` | Logging writes |
| `DoctorCheck` | `name`, `passed`, `fix` | Doctor diagnostics |
| `CategoryLabel` | `category`, `description`, `required` | Setup category order |

### Constants

| Constant | Default | Used by |
|----------|---------|---------|
| `DEFAULT_SSE_PORT` | `8765` | Setup logging, MCP server |
| `DEFAULT_SSE_HOST` | `127.0.0.1` | Setup logging |
| `DEFAULT_RETENTION_DAYS` | `14` | Setup logging |
| `DEFAULT_MAX_MEMORY` | `10_000` | Setup logging |

### Utilities

| Function | Purpose |
|----------|---------|
| `atomic_write_json(path, data)` | Write JSON atomically (temp file → os.replace). Interrupt-safe. |
| `is_ci()` | True in CI/CD environments |
| `is_ssh()` | True over SSH |
| `is_kubernetes()` | True in Kubernetes |
| `is_headless()` | True when no interactive terminal |

---

## Commands

### `setup` — First-time configuration

```bash
darwin-admin setup                          # Interactive setup (guided)
darwin-admin setup --preset recommended     # One-shot: models + logging (no prompts)
darwin-admin setup logging                  # Reconfigure logging only
darwin-admin setup validate                 # Validate current config
darwin-admin setup doctor                   # Diagnose common issues
darwin-admin setup reset                    # Reset config (keeps models with --keep-models)
```

| Subcommand | Description |
|------------|-------------|
| *(no args)* | Interactive guided setup (default) |
| `--preset required\|recommended` | One-shot: resolve models, download, configure logging |
| `logging` | Reconfigure logging interactively |
| `validate` | Validate config files and embedding model |
| `doctor` | Diagnose config issues, disk space, model status |
| `reset` | Reset configuration files |
| `profile --profile recommended` | Non-interactive, install recommended model profile |
| `onboard --preferences embedding=... nlp=...` | Non-interactive with explicit model IDs |

| Flag | Description |
|------|-------------|
| `--preset <required\|recommended>` | One-shot setup (no prompts) |
| `--yes`, `-y` | Skip confirmation prompts (use defaults for logging) |
| `--no-download` | Skip model downloads (assumes models are cached locally) |
| `--keep-models` | On reset, keep downloaded models and config files |

#### Interactive setup flow

```
Step 1: Detect system
  └── RAM, disk, CPU, GPU, hardware tier (fast, no model init)

Step 2: Configure models per category
  ├── embedding (required) — lists all models, user picks one
  ├── reranker (optional)  — recommended model + accept/browse/skip
  ├── ocr (optional)
  └── vision (optional)

Step 3: API keys
  └── Prompts only for keys needed by selected API models

Step 4: Install
  └── Downloads local models (API models just need keys)

Step 5: Logging
  └── Mode, persistence, retention, verbosity
```

#### Config detection

When run, setup checks three config files via `SetupStatus`:

- `darwin.json` — setup config (models, keys, environment)
- `logging.json` — logging mode, SSE port, retention
- `models.json` — active model per category

| State | Behavior |
|-------|----------|
| All three exist | Offers: change specific / full re-run / exit |
| Partial | Offers to complete missing pieces / full re-run / exit |
| None | Runs full setup |

#### One-shot preset (no prompts)

```bash
darwin-admin setup --preset recommended
```

Resolves models for your hardware tier, downloads them, writes `logging.json` with defaults (terminal, persist, 14d retention). Does **not** prompt for API keys — run `models keys set` afterward.

#### Atomic config writes

All config writes use `atomic_write_json()` — writes to a temp file, then `os.replace()`. If interrupted mid-write, only the `.tmp` file exists; the original config is never half-written.

---

### `models` — Model lifecycle

| Command | Description |
|---------|-------------|
| `status` | Show installed status per category |
| `list [--category] [--provider]` | List all registered models |
| `install <model_id>... [--no-download] [--force]` | Install one or more models |
| `discard --category <cat> [--all\|--active\|--inactive] [--delete]` | Discard models per category |
| `switch [category] [model_id] [--no-download]` | Change active model for a category |
| `keys set\|status\|remove --provider <name> [--key]` | Manage API keys |

#### Switch (fully interactive)

`switch` works with or without arguments:
- No args → prompts for category, then prompts for model
- `switch embedding` → prompts for model within embedding category
- `switch embedding model-id` → direct switch

When switching between providers, you'll see a comparison and can optionally discard the old key.

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

#### Keys (API key management)

```bash
darwin-admin models keys set openai sk-...    # Set a key
darwin-admin models keys status               # Show all key statuses
darwin-admin models keys remove openai        # Remove a key
```

`--provider` is required for `set` and `remove`. `--key` is required for `set`.

---

### `store` — Store introspection

| Command | Description |
|---------|-------------|
| `status` | Store overview (sources, pipeline stages, searchable files) |
| `files` | List all files with pipeline stage and search readiness |
| `file <filename>` | Detailed status for one file (formatted as tree) |
| `audit` | Full store audit (formatted as tree) |
| `health [--model <id>]` | Health check (embedding or registry) |
| `repair [--dry-run]` | Repair integrity issues |

---

### `pipeline` — Pipeline execution

The pipeline processes documents in two stages:

1. **Ingestion** — Parse files → extract text → chunk into passages → save as Parquet
2. **Indexing** — Build BM25 keyword index (NPZ) + generate embeddings (NumPy float16) + build vector index (USearch)

You can run both together (`run`), or each stage independently (`ingest`, `index`).

| Command | Description |
|---------|-------------|
| `run [target] [--files f1 f2] [--ocr] [--recursive] [options]` | Full ingest + index |
| `ingest [target] [--files f1 f2] [options]` | Parse + chunk only (no indexing) |
| `index [--files f1 f2] [--store] [--threshold] [options]` | BM25 + embedding indexing only |
| `purge <filenames>... [--stages] [--store]` | Remove artifacts for files |

**Target:**
- `run` and `ingest` accept an optional `target` — a directory path or static URL (default: `.`)
- If `--files` is provided, `target` is ignored for file discovery

**Optional parameters (common):**
- `--store <id|name|path>` — Target a specific store (default: auto-detect)
- `--files <file1> [file2 ...]` — Process only specific files (space-separated, repeatable)
- `--ocr` — Enable OCR for scanned PDFs (ingestion only)
- `--recursive` — Recurse into subdirectories (ingestion only)
- `--rerun` — Re-process files even if already done

**Indexing parameters:**
- `--threshold <float>` — Minimum relevance score (default: 0.5)
- `--idf-smoothing <float>` — BM25 IDF smoothing (default: 1.5)
- `--only-kw` — Build keyword index only (skip embeddings)
- `--only-semantic` — Build vector index only (skip BM25)
- `--precise` — Precision parsing mode (slower but more accurate)

---

### `search` — Knowledge base search

Search the DARWIN knowledge base and optionally save results.

```bash
darwin-admin search "machine learning"
darwin-admin search "AI safety" --method semantic --top-k 20
darwin-admin search "neural nets" --save
darwin-admin search "deep learning" --save --results-dir ./my-results
```

| Flag | Default | Description |
|------|---------|-------------|
| `query` | *(required)* | Search query string |
| `--method` | `hybrid` | Search strategy: `hybrid`, `semantic`, `keyword` |
| `--top-k` | `5` | Number of top results to return |
| `--max-passages` | `5` | Maximum passages to retrieve |
| `--max-chars` | `6000` | Character limit on returned text |
| `--min-score` | `0.15` | Minimum relevance score threshold |
| `--merge-strategy` | `adjacent` | Merging: `none`, `adjacent`, `aggressive` |
| `--files`, `-f` | *(all)* | Restrict to specific files (space-separated) |
| `--store` | *(default)* | Store ID, name, or path |
| `--save` | `false` | Save results as JSON for later retrieval |
| `--results-dir` | `{base}/search-results/` | Custom directory for saved results |
| `--rerank` | `false` | Enable reranking |
| `--structural-penalty-factor` | `0.5` | Penalty for non-body content (0.0-1.0). Lower = harsher. |
| `--diversity-rerank` | `true` | Source diversity penalty. Set `false` for deep dives into one file. |
| `--include-audit` | `false` | Per-passage scoring breakdown (raw_score, penalty_factor, etc.) |

#### Power-user examples

```bash
# Debug why results rank the way they do
darwin-admin search "neural nets" --include-audit

# Deep dive into one file — disable diversity penalty
darwin-admin search "attention mechanism" --files transformer.pdf --diversity-rerank false

# Get 10 passages from a single large document
darwin-admin search "chapter 3" --max-passages 10

# Disable structural penalties (tables, captions count equally)
darwin-admin search "data table" --structural-penalty-factor 1.0
```

---

### `logs` — Log management

| Command | Description |
|---------|-------------|
| `query [--session] [--levels] [--limit]` | Query log entries |
| `sessions` | List all sessions |
| `session-info <session-id>` | Detailed info for one session |
| `stats` | Log statistics |
| `purge [--days]` | Purge old logs |
| `purge-session <session-id>` | Purge a specific session |
| `export --output <path> [--session]` | Export logs to JSONL |
| `compact` | VACUUM SQLite database |
| `set-retention <days>` | Set log retention policy |

---

### `system` — Diagnostics

| Command | Description |
|---------|-------------|
| `info` | Hardware detection + model recommendations |

---

### `status` — Overview

| Command | Description |
|---------|-------------|
| `status [--store] [--initialize]` | Dashboard: store files, installed models, preferences |

By default, `status` shows store info and file counts without initializing models (fast). Use `--initialize` to also run model initialization (slower).

---

### `uninstall` — Remove darwin

Safely removes all darwin data. Discovers all registered stores, removes only darwin artifacts from extra stores (keeping user files), validates directories before deletion.

```bash
darwin-admin uninstall              # full cleanup + pip uninstall
darwin-admin uninstall --dry-run   # preview what would be removed
darwin-admin uninstall --keep-package  # clean data only, keep the package
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview what would be removed without making changes |
| `--keep-package` | Only clean data, skip `pip uninstall darwin-rag` |

**What it does:**

1. **Discovers stores** — queries `~/.darwin/stores_registry.db` for all registered stores
2. **Cleans extra stores** — removes only `indexes/`, `registry.db`, `.darwin_store_id`, `checkpoints/`, `conversations/` (leaves your original files untouched)
3. **Removes global registry** — validates `~/.darwin/` before deletion
4. **Removes main data directory** — with content validation
5. **Uninstalls package** — runs `pip uninstall darwin-rag --yes` (skipped with `--keep-package`)

---

## Logging modes

| Mode | Description |
|------|-------------|
| `terminal` | Logs to console only (default). ASCII banner shows on startup. |
| `sse` | Stream logs via SSE (for MCP / remote viewing) |
| `both` | Console + SSE simultaneously |
| `none` | Silent (no log output) |

Additional settings:
- **persist** — Save logs to SQLite history (separate from mode)
- **log_filter** — `all` (default), `quiet` (errors + warnings + success), `silent` (errors only)
- **retention_days** — 3, 7, 14, 30 days, or never
- **sse_port** — Port for SSE streaming (shared with MCP server, auto-resolved if busy)

These are saved to `logging.json`. MCP reads the port at startup.

---

## HTTP Endpoints

The MCP server exposes these HTTP endpoints (default: `http://127.0.0.1:8765`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stream` | GET | SSE stream of real-time log events |
| `/history` | GET | Query historical logs. Params: `?session=&levels=&limit=&before=` |
| `/banner` | GET | Darwin ASCII banner as JSON. Response: `{"banner": ["line1", ...]}` |
| `/health` | GET | Server health check (store, models) |
| `/dashboard` | GET | Web-based log dashboard (HTML) |

### Frontend integration example

```javascript
// Fetch banner once on load
const { banner } = await fetch("http://127.0.0.1:8765/banner").then(r => r.json());

// Connect to live log stream
const sse = new EventSource("http://127.0.0.1:8765/stream");
sse.onmessage = (e) => {
  const { message, log_type, session } = JSON.parse(e.data);
  // log_type: "info" | "warning" | "error" | "success"
};
```
