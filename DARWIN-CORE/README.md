# Darwin Core

**Local-first RAG engine** — ingest documents, index them (BM25 + embeddings), and search via hybrid/semantic/keyword retrieval. Runs as an MCP server for AI agent integration or via CLI. All local, all offline-capable.

Built by [BrightDotDev](https://github.com/brightdotdev).

[![License: MIT with Attribution](https://img.shields.io/badge/License-MIT%20with%20Attribution-blue.svg)](LICENSE)

---

## Quick Start

```bash
# Install
pip install -e .

# One-time setup
darwin init

# Ingest documents
darwin ingest ./my-docs/

# Search
darwin search "what is graph rag"

# Start MCP server for AI agents
darwin mcp
```

---

## Architecture

```
CLI (src/cli.py) ──> darwin init | setup | mcp | admin | ingest | search
                        │
MCP Server (FastMCP) ──> 10 tools: search, pipeline, models, store
                        │
Orchestrators ──> RetrievalOrch | PipelineOrch | ModelOrch | StoreOrch | LoggerOrch
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
| **CLI** | `darwin` command | Users |
| **MCP** | 10 agent tools | AI agents |
| **Admin** | 17 orchestrator commands | Power users |
| **Orchestrators** | Retrieval, Pipeline, Models, Store, Logger | Programmatic API |
| **Store** | Parquet, NumPy, USearch, BM25 persistence | Local disk |

---

## Modules

| Module | Path | Lines | Purpose |
|---|---|---|---|
| **CLI** | `src/cli.py` | 156 | Main entry point (`darwin` command) |
| **Setup** | `src/Setup.py` | ~400 | Interactive setup wizard |
| **Admin** | `src/admin/cli.py` | 919 | Full orchestrator control |
| **MCP** | `src/mcp/` | — | FastMCP server + tools |
| **Logger** | `src/core/Logger/` | 8 files | Structured logging (SQLite, SSE, terminal) |
| **Storage** | `src/core/Storage/` | 17 files | DarwinStore — file registry, artifacts, indices |
| **Orchestrators** | `src/core/Orchestrators/` | 5 files | High-level orchestration |
| **Pipeline** | `src/core/Pipeline/` | ~25 files | Ingestion (parse + chunk), Indexing (BM25 + embeddings) |
| **Retrival** | `src/core/Retrival/` | ~15 files | Hybrid, semantic, keyword search with reranking |
| **Models** | `src/core/Models/` | ~25 files | Model registry (40+), 5 inference factories, downloaders |
| **Features** | `src/features/` | 4 files | Knowledge graphs, summarization, image OCR/vision |
| **Agent** | `src/agent/LLM/` | 9 files | LLM client, config, manager |
| **Lib** | `src/lib/` | 6 files | Path resolution, key management, utilities |

---

## CLI Reference

### `darwin init`

Bootstrap store + download required models.

```bash
darwin init                     # default: required models only
darwin init --profile recommended  # also install OCR, reranker, vision
darwin init --no-download          # register models without downloading
```

Delegates to `src.core.init.run_init()`.

### `darwin setup`

One-shot setup via Click CLI. Delegates to `src/Setup.py`.

```
darwin setup [OPTIONS] COMMAND [ARGS]...
```

| Command | Description |
|---|---|
| `run` | Guided interactive onboarding |
| `preset <name>` | One-shot preset (minimal, recommended, complete, local-only, api-only, server) |
| `models` | Install/update specific models |
| `embeddings` | Set embedding model only |
| `logs` | Configure logging |
| `storage` | Set storage directory |
| `paths` | Show all derived paths |
| `validate` | Validate current configuration |

### `darwin mcp`

Start the MCP server so AI agents can search, ingest, and manage models.

```bash
darwin mcp                       # stdio mode (for agent subprocess)
darwin mcp --sse --port 8080     # SSE mode (for network/Docker)
```

| Option | Default | Description |
|---|---|---|
| `--sse` | `false` | Use SSE transport instead of stdio |
| `--port` | `8080` | SSE server port |

**MCP Tools:**

| Tool | File | Description |
|---|---|---|
| `search_darwin` | `mcp/tools/Search.py` | Full-featured search with configurable method, merging, fallback, reranking |
| `run_pipeline` | `mcp/tools/Pipeline.py` | Run ingestion/indexing pipeline stages |
| `purge_artifacts` | `mcp/tools/Pipeline.py` | Delete pipeline artifacts for specific files |
| `pipeline_report` | `mcp/tools/Pipeline.py` | Get pipeline report for files |
| `model_status` | `mcp/tools/Models.py` | Show model status across categories |
| `install_model` | `mcp/tools/Models.py` | Install a model |
| `switch_model` | `mcp/tools/Models.py` | Switch active model for a category |
| `recommend_models` | `mcp/tools/Models.py` | Get model recommendations based on hardware |
| `list_files` | `mcp/tools/Store.py` | List all files with pipeline status |
| `file_status` | `mcp/tools/Store.py` | Detailed status for one file |

### `darwin admin`

Full orchestrator control for power users.

```
darwin admin <command> [ARGS]...
```

| Group | Commands | Description |
|---|---|---|
| `setup` | `interactive`, `profile`, `onboard` | Onboarding and configuration |
| `models` | `status`, `list`, `install`, `uninstall`, `switch`, `keys` | Model lifecycle |
| `store` | `status`, `files`, `file`, `audit`, `health`, `repair` | Store management |
| `pipeline` | `run`, `ingest`, `index`, `purge` | Pipeline operations |
| `logs` | `query`, `sessions`, `stats`, `purge`, `export`, `compact`, `set-retention` | Log management |
| `system` | `info` | System information |

```bash
darwin admin models status
darwin admin store files
darwin admin pipeline run ./docs --ocr --recursive
darwin admin pipeline purge myfile.pdf --stages ingestion
darwin admin logs query --levels error --limit 10
darwin admin system info
```

### `darwin ingest`

Quick one-off ingest shortcut.

```bash
darwin ingest <target> [--ocr] [--recursive]
```

| Argument | Description |
|---|---|
| `target` | File path, directory, or URL to ingest |
| `--ocr` | Enable OCR for images in documents |
| `--recursive` | Recursively ingest directories |

Delegates to admin pipeline ingest under the hood.

### `darwin search`

Quick one-off search.

```bash
darwin search <query> [--method <method>] [--top-k <n>]
darwin search "book of enoch"
darwin search "NVIDIA Nemotron" --method semantic --top-k 10
```

| Option | Default | Description |
|---|---|---|
| `--method` | `hybrid` | Search method: hybrid, semantic, keyword |
| `--top-k` | `10` | Number of results |
| `--store` | default | Store ID to search |

---

## Development

```bash
git clone <repo-url>
cd darwin-core
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -e .

# Run tests
python -m pytest tests/smoke_test.py -v
```

### Docker

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
%LOCALAPPDATA%/Darwin/           # Windows
~/Library/Application Support/Darwin/   # macOS
~/.local/share/darwin/           # Linux
  ├── config/
  │   ├── logging.json
  │   └── stores_registry.db     # Central multi-store registry
  ├── indexes/
  │   ├── registry.db            # Per-store SQLite (sources, content)
  │   └── <sha256-hash>/         # Artifacts per unique file content
  │       ├── meta.json          # Source metadata
  │       ├── markdown.md        # Parsed markdown
  │       ├── chunks.parquet     # Chunk records (Polars)
  │       ├── bm25.npz           # BM25 sparse matrix
  │       ├── bm25_tokens.parquet
  │       ├── vectors/           # USearch index + NumPy embeddings
  │       └── images/            # Extracted images
  ├── logs/
  │   └── logs.db                # SQLite log history
  └── models/
      ├── config/models.json     # Active model configuration
      ├── embedding/             # ONNX embedding models
      ├── nlp/                   # spaCy NLP models
      ├── reranker/              # ONNX cross-encoder models
      ├── vision/                # ONNX vision models
      └── ocr/                   # ONNX OCR models
```

---

## Key Dependencies

| Category | Packages |
|---|---|
| **ONNX Inference** | `onnxruntime` |
| **MCP Protocol** | `fastmcp`, `uvicorn`, `starlette`, `sse-starlette` |
| **Vector Search** | `usearch` |
| **NLP** | `spacy`, `pytextrank` |
| **PDF** | `pymupdf`, `pymupdf4llm` |
| **Data** | `polars`, `pyarrow`, `numpy` |
| **Terminal UI** | `rich`, `click` |
| **LLM API** | `litellm` |
| **OCR** | `rapidocr-onnxruntime` |
| **Security** | `keyring` |

---

## License

MIT with Attribution — see [LICENSE](LICENSE).

Core architecture and implementation by [BrightDotDev](https://github.com/brightdotdev).
