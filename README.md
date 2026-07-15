# Darwin Core

**Local-first RAG engine** — ingest documents, index them (BM25 + embeddings), and search via hybrid/semantic/keyword retrieval. Runs as an MCP server for AI agent integration or via CLI. All local, all offline-capable.

Built by [BrightDotDev](https://github.com/brightdotdev).

[![License: MIT with Attribution](https://img.shields.io/badge/License-MIT%20with%20Attribution-blue.svg)](LICENSE)

---

## Quick Start

> **Not on PyPI yet.** Install directly from the repo.

```bash
# 1. Clone
git clone <repo-url>
cd darwin-core

# 2. Create venv & install
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e .
# or with uv:
uv pip install -e .

# 3. Setup (interactive — detect hardware, pick models, set API keys)
darwin-admin setup interactive

# 4. Start MCP server for AI agents
darwin mcp

# 5. Or use SSE/HTTP for remote access
darwin mcp --http --port 8765
```

> **Note:** `darwin` and `darwin-admin` are available after activating the virtual environment.
> If you prefer not to activate the venv each time, use `uv run darwin ...` or add
> `.venv/Scripts` (Windows) / `.venv/bin` (macOS/Linux) to your PATH.

---

## Architecture

```
darwin (user CLI)        darwin-admin (power-user CLI)
  │                         │
  ├── mcp [--sse|--http]    ├── setup interactive|profile|onboard
  │   [--port] [--temp-port]│   models status|list|install|switch|keys
  │   [--host]              │   store files|audit|health|repair
  └── config [target]       │   pipeline run|ingest|index|purge
                            │   logs query|sessions|stats|export
                            │   system info
                            │   uninstall [--dry-run] [--keep-package]
                            └── status [--initialize]
        │
MCP Server (FastMCP) ──> 7 tools + 3 resources: search, pipeline, store
        │
Orchestrators ──> RetrievalOrch | PipelineOrch | StoreOrch
        │
Core Engine:
  ┌─────────────────────────────────────────────┐
  │  DarwinStore — file registry, artifacts,    │
  │  search indexes (Parquet, NumPy, USearch)   │
  ├─────────────────┬──────────────────────────┤
  │  Pipeline       │  Retrieval               │
  │  (ingest,       │  (hybrid, sem, kw)       │
  │   index)        │                          │
  └─────────────────┴──────────────────────────┘
```

### Layers

| Layer | What | Entry point |
|---|---|---|
| **User CLI** | `darwin` — mcp, config | `src/cli.py` |
| **Admin CLI** | `darwin-admin` — setup, models, store, pipeline, logs, system, uninstall, status | `src/admin/` |
| **MCP** | 7 tools + 3 resources for AI agents | `src/mcp/` |
| **Orchestrators** | Retrieval, Pipeline, Store | `src/core/Orchestrators/` |
| **lib.py** | Typed params with Literal enums | `src/core/Orchestrators/lib.py` |
| **Store** | Parquet, NumPy, USearch, BM25 persistence | `src/core/Storage/` |

---

## Modules

| Module | Path | Purpose |
|---|---|---|
| **CLI** | `src/cli.py` | User entry point (`darwin`) — mcp, config |
| **Admin** | `src/admin/` | Admin entry point (`darwin-admin`) — 31 subcommands |
| **Admin modules** | `src/admin/renderer.py` | Terminal output (ANSI, tables, panels) |
| | `src/admin/setup.py` | Interactive onboarding + config helpers |
| | `src/admin/commands_*.py` | Command handlers per group (models, store, pipeline, logs, system) |
| | `src/admin/uninstall.py` | Full darwin uninstall (discover stores, clean artifacts, remove data) |
| **MCP** | `src/mcp/` | FastMCP server + 7 tools + 3 resources |
| **Orchestrators** | `src/core/Orchestrators/` | Retrieval, Pipeline, Store |
| **Orchestrators/lib.py** | `src/core/Orchestrators/lib.py` | Parameter library with Literal types |
| **Storage** | `src/core/Storage/` | DarwinStore — file registry, artifacts, indices |
| **Pipeline** | `src/core/Pipeline/` | Ingestion (parse + chunk), Indexing (BM25 + embeddings) |
| **Retrieval** | `src/core/Retrieval/` | Hybrid, semantic, keyword search with reranking |
| **Models** | `src/core/Models/` | Model registry (40+), inference factories, downloaders |
| **Logger** | `src/core/Logger/` | Structured session logging, SSE streaming |
| **Lib** | `src/lib/` | Path resolution, key management, network utils |

---

## CLI Reference — `darwin` (User)

After activating the virtual environment and running `pip install -e .`, the `darwin` command is available in that session.

### `darwin mcp`

Start the MCP server for AI agent integration.

```bash
darwin mcp                        # stdio (default — for agent subprocess)
darwin mcp --sse --port 8765      # SSE (legacy network transport)
darwin mcp --http --port 8765     # Streamable HTTP (recommended)
darwin mcp --temp-port 9000       # Override port for this session only
```

| Option | Default | Description |
|---|---|---|
| `--sse` | false | Legacy SSE transport |
| `--http` | false | Streamable HTTP transport (recommended) |
| `--port` | from logging.json or 8765 | Port for network mode |
| `--temp-port` | none | Override port for this session only (does not save) |
| `--host` | 127.0.0.1 | Host for network mode |

**Stdio mode:** Logs are silenced on stdout to keep the JSON-RPC channel clean. Logs still go to SQLite history (if enabled) and SSE stream (if configured).

**SSE/HTTP mode:** Logs visible on terminal. Port is shared with the logger SSE server (same port from `logging.json`).

See [src/mcp/README.md](src/mcp/README.md) for full MCP reference (tools, resources, endpoints).

### `darwin config`

Generate client config for MCP integration.

```bash
darwin config                 # generate config for all clients
darwin config claude          # Claude Desktop config only
darwin config cursor          # Cursor config only
darwin config vscode          # VS Code (Continue) config only
darwin config sse             # Network SSE config
darwin config http            # Network Streamable HTTP config
darwin config all --copy      # generate all + copy to clipboard
```

| Option | Default | Description |
|---|---|---|
| `--host` | 127.0.0.1 | Host for network config |
| `--port` | 8765 | Port for network config |
| `--copy` | false | Copy generated config to clipboard |

---

## CLI Reference — `darwin-admin` (Power User)

Full orchestrator control. 31 subcommands across 8 groups.

```bash
darwin-admin <group> <command> [options]
```

See [src/admin/README.md](src/admin/README.md) for complete reference.

### Groups

| Group | Commands | Description |
|---|---|---|
| `setup` | `interactive`, `profile`, `onboard` | First-time configuration |
| `models` | `status`, `list`, `install`, `discard`, `switch`, `keys` | Model lifecycle |
| `store` | `status`, `files`, `file`, `audit`, `health`, `repair` | Store introspection |
| `pipeline` | `run`, `ingest`, `index`, `purge` | Pipeline execution |
| `logs` | `query`, `sessions`, `session-info`, `stats`, `purge`, `export`, `compact`, `set-retention` | Log management |
| `system` | `info` | Hardware detection + recommendations |
| `uninstall` | — | Remove all darwin data and optionally the package |
| `status` | — | Overview dashboard |

---

## Orchestrator Parameters (`lib.py`)

All orchestrator parameters are defined in `src/core/Orchestrators/lib.py` — the single source of truth. Option-based string values use `Literal` types for IDE intellisense and MCP JSON Schema enum generation.

```python
from src.core.Orchestrators.lib import RetrievalParameters, PipelineParameters

# IDE shows valid options for Literal fields
params = RetrievalParameters(
    query="what is Darwin?",
    method="hybrid",              # Literal["hybrid", "semantic", "keyword"]
    merge_strategy="adjacent",    # Literal["none", "adjacent", "aggressive"]
    fallback_strategy="both",     # Literal["expand", "relax", "both"]
)
cfg = params.build()
ctx = orchestrator.retrieve_and_resolve(cfg)
```

---

## Docker

The entrypoint auto-runs `darwin setup preset server --yes` on first start. No manual init needed.

When running with `--sse` or `--http`, the server also exposes:
- `GET /health` — returns `{"status": "ok", "checks": {...}}` (503 when degraded)
- `GET /stream` — SSE log stream for dashboards
- `GET /history` — query historical logs (`?session=&levels=&limit=`)

```bash
# Build & start (auto-setup on first run, MCP SSE on port 8080)
docker compose up -d

# Or manually:
docker build -t darwin .
docker run -d --name darwin -p 8080:8080 -v darwin_data:/data darwin

# Check logs
docker compose logs -f

# Stop
docker compose down

# Wipe data (volume)
docker compose down -v
```

| Variable | Default | Description |
|---|---|---|
| `DARWIN_BASE_DIR` | `/data` | Container path for persistent data (mount a volume here) |

### docker-compose.yml

```yaml
services:
  darwin:
    build: .
    environment:
      - DARWIN_BASE_DIR=/data
    volumes:
      - darwin_data:/data
    ports:
      - "8080:8080"
    command: ["darwin", "mcp", "--sse", "--port", "8080"]

volumes:
  darwin_data:
```

---

## Data Flow

1. **Ingest**: File → parse (PDF→markdown, HTML→text, etc.) → chunk (Parquet records) → store
2. **Index**: Chunks → BM25 index (NPZ) + embeddings (NumPy float16) + vector index (USearch) → store
3. **Search**: Query → hybrid fusion (RRF of BM25 + semantic scores) / pure semantic / pure keyword → filter, penalize, merge, resolve passages
4. **MCP**: AI agent → FastMCP tool call → orchestrator → JSON result

---

## Storage Layout

```
%LOCALAPPDATA%/Darwin/                     # Windows
~/Library/Application Support/Darwin/      # macOS
~/.local/share/darwin/                     # Linux
  ├── config/
  │   ├── darwin.json                      # Setup config (models, keys, environment)
  │   └── logging.json                     # Logging mode, SSE port, retention
  ├── indexes/
  │   ├── registry.db
  │   └── <sha256-hash>/
  │       ├── meta.json
  │       ├── markdown.md
  │       ├── chunks.parquet
  │       ├── bm25.npz
  │       └── vectors/
  ├── logs/
  └── models/
      ├── config/
      │   └── models.json                  # Active model per category
      └── <model-id>/                      # Downloaded model files
```

---

## Security

Darwin includes input validation and safety hardening across all MCP tools and CLI commands:

| Protection | Scope | Detail |
|---|---|---|
| **Query length limit** | Search | Max 4096 characters |
| **Path traversal block** | Store, Pipeline | `..` in paths rejected |
| **Filename validation** | Store, Pipeline | Invalid characters, empty names, length limits |
| **Stage whitelist** | Pipeline | Only valid stage names accepted |
| **Model ID validation** | Models | Length limits, invalid characters blocked |
| **Category validation** | Models | Must match registered `ModelCategory` values |
| **Port availability** | MCP Server | Auto-scans for free port, saves to config |
| **Health checks** | `/health` endpoint | Checks store and models, returns 503 when degraded |
| **Stdio log silencing** | MCP stdio | Logger silenced on stdout to protect JSON-RPC channel |
| **Error propagation** | All tools | Try/except with structured error returns instead of silent swallowing |
| **Thread safety** | TerminalUi, SQLiteHistoryStore | Lock-protected console output and DB access |
| **History store lifecycle** | DarwinLogger | Old history closed on reconfigure, SSE server shutdown on exit |

---

## Running Tests

```bash
# Full suite
python -m pytest tests/ -v

# Quick smoke test (15 tests)
python -m pytest tests/smoke_test.py -v
```

---

## License

MIT with Attribution — see [LICENSE](LICENSE).

Core architecture and implementation by [BrightDotDev](https://github.com/brightdotdev).
