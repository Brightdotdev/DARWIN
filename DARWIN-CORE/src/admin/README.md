# Darwin Admin CLI

Full orchestrator control for power users. 17 subcommands across 5 groups.

## Usage

```bash
python -m src.admin <group> <command> [options]
# or via unified CLI:
darwin admin <group> <command> [options]
```

## Groups

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

| Command | Description |
|---------|-------------|
| `run <target> [--ocr] [--recursive] [options]` | Full ingest + index |
| `ingest <target> [options]` | Parse + chunk only |
| `index [--files] [--threshold] [options]` | BM25 + embedding indexing only |
| `purge <filenames>... [--stages]` | Remove artifacts for files |

### `system` — Diagnostics

| Command | Description |
|---------|-------------|
| `info` | Hardware detection + model recommendations |

## Interactive mode

```bash
python -m src.admin setup interactive
```

Shows numbered menus for:
1. System detection (RAM, CPU, GPU, tier)
2. Model recommendations per category
3. Model selection (or `all`)
4. API key prompts for provider-based models
5. Installation progress

## Non-interactive (Docker/automation)

```bash
# Profile-based
python -m src.admin setup profile --profile recommended --no-download

# Explicit preferences
python -m src.admin setup onboard \
  --preferences embedding=intfloat/multilingual-e5-small-onnx \
  --no-download

# Pipeline automation
python -m src.admin pipeline run /data/docs --ocr --recursive

# Health monitoring
python -m src.admin store health
```
