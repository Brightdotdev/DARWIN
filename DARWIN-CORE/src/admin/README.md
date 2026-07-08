# Admin — Full Orchestrator Control

**What it is:** A comprehensive CLI for power users to control every aspect of Darwin — models, store, pipeline, logs, system, and setup. Built with Rich for beautiful terminal output.

**Why it exists:** While the root `darwin` CLI covers common operations, power users and Docker deployments need fine-grained control over every subsystem. The admin CLI exposes the full orchestrator API.

## Entry Points

```bash
darwin admin <command> [ARGS]...
```

Or directly:

```bash
python -m src.admin <command> [ARGS]...
```

## Files

| File | Lines | Purpose |
|---|---|---|
| `cli.py` | 919 | Main admin CLI — 6 command groups |
| `__main__.py` | — | `python -m src.admin` entry |
| `__init__.py` | — | Package init |

## Command Groups

### `setup`

Onboarding and configuration.

```bash
darwin admin setup interactive          # Guided onboarding
darwin admin setup profile --profile recommended
darwin admin setup onboard --preferences embedding=intfloat/...
```

### `models`

Model lifecycle management.

```bash
darwin admin models status              # Show all model statuses
darwin admin models list                # List registered models
darwin admin models install <model_id>  # Install a model
darwin admin models uninstall <model_id># Uninstall a model
darwin admin models switch <cat> <id>   # Switch active model
darwin admin models keys set <provider> <key>  # Set API key
```

### `store`

Store inspection and maintenance.

```bash
darwin admin store status               # Overall store health
darwin admin store files                # List all files with status
darwin admin store file <filename>      # Detailed file status
darwin admin store audit                # Full store audit
darwin admin store health               # Embedding health check
darwin admin store repair [--dry-run]   # Repair issues
```

### `pipeline`

Document processing pipeline.

```bash
darwin admin pipeline run <target> [--ocr] [--recursive]  # Full ingest+index
darwin admin pipeline ingest <target>   # Parse + chunk only
darwin admin pipeline index <target>    # BM25 + embedding only
darwin admin pipeline purge <file> --stages ingestion  # Delete artifacts
```

### `logs`

Structured log management.

```bash
darwin admin logs query                 # Query logs with filters
darwin admin logs sessions              # List sessions
darwin admin logs session-info <id>     # Session details
darwin admin logs stats                 # Log statistics
darwin admin logs purge                 # Purge old logs
darwin admin logs purge-session <id>    # Purge a session
darwin admin logs export <file>         # Export logs to JSON
darwin admin logs compact               # Compact SQLite DB
darwin admin logs set-retention <days>  # Set retention policy
```

### `system`

System information.

```bash
darwin admin system info                # Hardware, models, store, config
```

## Architecture

Each command group lazily loads its orchestrator:

```python
@cli.group()
def models():
    pass

@models.command()
def status():
    orch = _get_models()  # Lazy-loaded ModelOrchestrator
    ...
```

The CLI uses Rich for all output: Table, Panel, Tree, console markup.
