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
cd DARWIN-CORE

# 2. Install (works with pip and uv)
pip install -e .
# or: uv pip install -e .
# or: uv sync                        # if using uv.lock

# 3. Bootstrap (creates store, downloads required models)
darwin init

# 4. Ingest documents
darwin ingest ./my-docs/

# 5. Search
darwin search "what is graph rag"

# 6. Start MCP server for AI agents
darwin mcp
```

---

## Architecture

```
darwin (user CLI)        darwin-admin (power-user CLI)
  │                         │
  ├── init                  ├── models status|install|switch|...
  ├── setup                 ├── store files|audit|health|repair
  ├── mcp [--sse|--http]    ├── pipeline run|ingest|index|purge
  ├── ingest <target>       ├── logs query|sessions|stats|export|...
  ├── search <query>        ├── system info
  ├── config [target]       └── status
  └── status
        │
MCP Server (FastMCP) ──> 11 tools + 5 resources: search, pipeline, models, store
        │
Orchestrators ──> RetrievalOrch | PipelineOrch | ModelOrch | StoreOrch
        │
Core Engine:
  ┌─────────────────────────────────────────────┐
  │  DarwinStore — file registry, artifacts,    │
  │  search indexes (Parquet, NumPy, USearch)   │
  ├─────────────┬────────────┬──────────────────┤
  │  Pipeline   │  Retrival  │  Models          │
  │  (ingest,   │  (hybrid,  │  (registry,      │
  │   index)    │   sem, kw) │   inference)     │
  └─────────────┴────────────┴──────────────────┘
```

### Layers

| Layer | What | Entry point |
|---|---|---|
| **User CLI** | `darwin` — init, setup, mcp, ingest, search, config, status | `src/cli.py` |
| **Admin CLI** | `darwin-admin` — models, store, pipeline, logs, system, status | `src/admin/cli.py` |
| **MCP** | 11 tools + 5 resources for AI agents | `src/mcp/` |
| **Orchestrators** | Retrieval, Pipeline, Models, Store | `src/core/Orchestrators/` |
| **lib.py** | Typed params with Literal enums | `src/core/Orchestrators/lib.py` |
| **Store** | Parquet, NumPy, USearch, BM25 persistence | `src/core/Storage/` |

---

## Modules

| Module | Path | Purpose |
|---|---|---|
| **CLI** | `src/cli.py` | User entry point (`darwin`) |
| **Admin** | `src/admin/cli.py` | Admin entry point (`darwin-admin`) |
| **MCP** | `src/mcp/` | FastMCP server + 11 tools + 5 resources |
| **Orchestrators** | `src/core/Orchestrators/` | Retrieval, Pipeline, Models, Store |
| **Orchestrators/lib.py** | `src/core/Orchestrators/lib.py` | Parameter library with Literal types |
| **Storage** | `src/core/Storage/` | DarwinStore — file registry, artifacts, indices |
| **Pipeline** | `src/core/Pipeline/` | Ingestion (parse + chunk), Indexing (BM25 + embeddings) |
| **Retrival** | `src/core/Retrival/` | Hybrid, semantic, keyword search with reranking |
| **Models** | `src/core/Models/` | Model registry (40+), inference factories, downloaders |
| **Logger** | `src/core/Logger/` | Structured session logging |
| **Features** | `src/features/` | Knowledge graphs, summarization, image analysis |
| **Agent** | `src/agent/LLM/` | LLM client, config, manager |
| **Lib** | `src/lib/` | Path resolution, key management, network utils |

---

## CLI Reference — `darwin` (User)

After `pip install -e .`, the `darwin` command is available globally.

### `darwin init`

Bootstrap store + download required models.

```bash
darwin init                          # default: required models only
darwin init --profile recommended    # also install OCR, reranker, vision
darwin init --no-download            # register models without downloading
```

| Option | Default | Description |
|---|---|---|
| `--profile` | `required` | Model profile: `required` or `recommended` |
| `--no-download` | false | Register models without downloading files |

### `darwin setup`

One-shot setup (directories, models, workspace).

```bash
darwin setup                     # interactive guided setup
darwin setup preset server       # non-interactive server preset
darwin setup models              # configure models
darwin setup embeddings          # configure embedding model only
darwin setup logs                # configure logging
darwin setup storage             # configure storage directory
darwin setup paths               # show all derived paths
darwin setup validate            # validate current configuration
darwin setup status              # show current state, disk usage
darwin setup reset               # clear all config and start fresh
darwin setup doctor              # diagnose common issues
```

See [src/Setup.py](src/Setup.py) for full setup command reference.

### `darwin mcp`

Start the MCP server for AI agent integration.

```bash
darwin mcp                        # stdio (default — for agent subprocess)
darwin mcp --sse --port 8765      # SSE (legacy network transport)
darwin mcp --http --port 8765     # Streamable HTTP (recommended)
```

| Option | Default | Description |
|---|---|---|
| `--sse` | false | Legacy SSE transport |
| `--http` | false | Streamable HTTP transport (recommended) |
| `--port` | 8765 | Port for network mode |
| `--host` | 127.0.0.1 | Host for network mode |

See [src/mcp/README.md](src/mcp/README.md) for full MCP reference (tools, resources, endpoints).

### `darwin ingest`

Quick one-off ingest.

```bash
darwin ingest ./my-docs/
darwin ingest ./papers/ --ocr --recursive
darwin ingest https://example.com/page --store my-store
```

| Option | Description |
|---|---|
| `--store` | Store ID, name, or path (default: auto) |
| `--ocr` | Enable OCR for scanned documents |
| `--recursive` | Recurse into subdirectories |

### `darwin search`

Quick one-off search.

```bash
darwin search "what is graph rag"
darwin search "NVIDIA Nemotron" --method semantic --top-k 10
darwin search "DNA" --store my-store
```

| Option | Default | Description |
|---|---|---|
| `--method` | hybrid | Search method: `hybrid`, `semantic`, `keyword` |
| `--top-k` | 5 | Number of results |
| `--store` | auto | Store ID, name, or path |

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

### `darwin status`

Overview: store files, installed models, preferences.

```bash
darwin status
darwin status --store my-store
```

---

## CLI Reference — `darwin-admin` (Power User)

Full orchestrator control. 30 subcommands across 7 groups.

```bash
darwin-admin <group> <command> [options]
```

See [src/admin/README.md](src/admin/README.md) for complete reference.

### Groups

| Group | Commands | Description |
|---|---|---|
| `setup` | `interactive`, `profile`, `onboard` | First-time configuration |
| `models` | `status`, `list`, `install`, `uninstall`, `switch`, `keys` | Model lifecycle |
| `store` | `status`, `files`, `file`, `audit`, `health`, `repair` | Store introspection |
| `pipeline` | `run`, `ingest`, `index`, `purge` | Pipeline execution |
| `logs` | `query`, `sessions`, `session-info`, `stats`, `purge`, `export`, `compact`, `set-retention` | Log management |
| `system` | `info` | Hardware detection + recommendations |
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
```

---

## Security

Darwin includes input validation and safety hardening across all MCP tools and CLI commands:

| Protection | Scope | Detail |
|---|---|---|
| **Query length limit** | Search | Max 4096 characters |
| **Path traversal block** | Store, Pipeline | `..` in paths rejected |
| **Filename validation** | Store, Pipeline | Invalid characters, empty names, length limits |
| **Stage whitelist** | Pipeline | Only valid stage names accepted (`ingest`, `extract`, `parse`, `chunk`, `ocr`, `clean`, `index`, `embed`, `ingestion`, `indexing`) |
| **Model ID validation** | Models | Length limits, invalid characters blocked |
| **Category validation** | Models | Must match registered `ModelCategory` values |
| **Port availability** | MCP Server | Auto-scans for free port with `SO_REUSEADDR` |
| **Health checks** | `/health` endpoint | Checks store (file count) and models (all_ready), returns 503 when degraded |
| **Error propagation** | All tools | Try/except with structured error returns instead of silent swallowing |
| **Thread safety** | TerminalUi, SQLiteHistoryStore | Lock-protected console output and DB access |
| **History store lifecycle** | DarwinLogger | Old history closed on reconfigure, SSE server shutdown on exit |

---

## Running Tests

```bash
# Full suite
python -m pytest tests/ -v

# Quick smoke test
python -m pytest tests/smoke_test.py -v
```

---

## License

MIT with Attribution — see [LICENSE](LICENSE).

Core architecture and implementation by [BrightDotDev](https://github.com/brightdotdev).
