# Darwin Core

**Local-first RAG engine** — ingest documents, index them (BM25 + embeddings), and search via hybrid/semantic/keyword retrieval. Runs as an MCP server for AI agent integration or via CLI. All local, all offline-capable.

Built by [BrightDotDev](https://github.com/brightdotdev).

[![License: MIT with Attribution](https://img.shields.io/badge/License-MIT%20with%20Attribution-blue.svg)](LICENSE)

---

## Dev Quick Start

> **Not a published package yet.** Clone and run directly for development/testing.

```bash
# 1. Clone
git clone <repo-url>
cd DARWIN-CORE

# 2. Create virtual environment (Python 3.11+)
python -m venv .venv

# 3. Activate
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 4. Install deps
pip install -r requirements.txt   # or just install as you go

# 5. Bootstrap
python -m src.cli init

# 6. Ingest documents
python -m src.cli ingest ./my-docs/

# 7. Search
python -m src.cli search "what is graph rag"

# 8. Start MCP server for AI agents
python -m src.cli mcp
```

---

## Architecture

```
CLI (src/cli.py) ──> init | mcp | admin | ingest | search
                        │
MCP Server (FastMCP) ──> 10 tools: search, pipeline, models, store
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

| Layer | What | For |
|---|---|---|
| **CLI** | `python -m src.cli` | Users |
| **MCP** | 10 agent tools | AI agents |
| **Admin** | 17 orchestrator commands | Power users |
| **Orchestrators** | Retrieval, Pipeline, Models, Store | Programmatic API |
| **lib.py** | Typed params with Literal enums | Single source of truth for defaults |
| **Store** | Parquet, NumPy, USearch, BM25 persistence | Local disk |

---

## Modules

| Module | Path | Purpose |
|---|---|---|
| **CLI** | `src/cli.py` | Main entry point |
| **Admin** | `src/admin/cli.py` | Full orchestrator control (17 commands) |
| **MCP** | `src/mcp/` | FastMCP server + 10 tools |
| **Orchestrators** | `src/core/Orchestrators/` | Retrieval, Pipeline, Models, Store |
| **Orchestrators/lib.py** | `src/core/Orchestrators/lib.py` | Parameter library with Literal types |
| **Storage** | `src/core/Storage/` | DarwinStore — file registry, artifacts, indices |
| **Pipeline** | `src/core/Pipeline/` | Ingestion (parse + chunk), Indexing (BM25 + embeddings) |
| **Retrival** | `src/core/Retrival/` | Hybrid, semantic, keyword search with reranking |
| **Models** | `src/core/Models/` | Model registry (40+), inference factories, downloaders |
| **Logger** | `src/core/Logger/` | Structured session logging |
| **Features** | `src/features/` | Knowledge graphs, summarization, image OCR/vision |
| **Agent** | `src/agent/LLM/` | LLM client, config, manager |
| **Lib** | `src/lib/` | Path resolution, key management, utilities |

---

## CLI Reference

### `python -m src.cli init`

Bootstrap store + download required models.

```bash
python -m src.cli init                          # default: required models only
python -m src.cli init --profile recommended    # also install OCR, reranker, vision
python -m src.cli init --no-download            # register models without downloading
```

### `python -m src.cli mcp`

Start the MCP server for AI agent integration.

```bash
python -m src.cli mcp                        # stdio (for agent subprocess)
python -m src.cli mcp --sse --port 8080      # SSE (for network/Docker)
```

| Option | Default | Description |
|---|---|---|
| `--sse` | false | Use SSE transport instead of stdio |
| `--port` | 8080 | SSE server port |

### `python -m src.cli admin`

Full orchestrator control — 17 commands across 5 groups.

```bash
python -m src.cli admin setup interactive
python -m src.cli admin models status
python -m src.cli admin store files
python -m src.cli admin pipeline run ./docs --ocr --recursive
python -m src.cli admin system info
```

See [src/admin/README.md](src/admin/README.md) for full reference.

### `python -m src.cli ingest`

Quick one-off ingest shortcut.

```bash
python -m src.cli ingest <target> [--ocr] [--recursive]
```

Delegates to admin pipeline ingest under the hood.

### `python -m src.cli search`

Quick one-off search.

```bash
python -m src.cli search "what is Darwin?"
python -m src.cli search "NVIDIA Nemotron" --method semantic --top-k 10
```

| Option | Default | Description |
|---|---|---|
| `--method` | hybrid | Search method: hybrid, semantic, keyword |
| `--top-k` | 5 | Number of results |

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

```bash
# Build
docker build -t darwin .

# Run MCP server on port 8080
docker run -p 8080:8080 -v darwin-data:/data/darwin darwin

# Or run init first, then MCP
docker run -v darwin-data:/data/darwin darwin init
docker run -p 8080:8080 -v darwin-data:/data/darwin darwin mcp --sse
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

## Running Tests

```bash
python -m pytest tests/smoke_test.py -v
```

---

## License

MIT with Attribution — see [LICENSE](LICENSE).

Core architecture and implementation by [BrightDotDev](https://github.com/brightdotdev).
