# Architecture — For Developers and Contributors

This document explains what each part of Darwin does and why it exists. It's logic-heavy, not code-heavy — the code changes, the ideas don't.

---

## How Darwin works

Darwin is a RAG (Retrieval-Augmented Generation) engine. The core loop is simple:

1. **Ingest** — take documents in any format, parse them into markdown, split them into chunks
2. **Index** — build a keyword index (BM25) and a vector index (embeddings) from those chunks
3. **Search** — when a query comes in, run it against both indexes, merge the results, and return the best passages

The MCP server sits on top of this and exposes search, pipeline, and store operations as tools that AI agents can call.

---

## The two CLIs

Darwin has two command-line interfaces because there are two audiences:

**`darwin`** is for end users. It does two things: start the MCP server (`darwin mcp`) and generate client configs (`darwin config`). That's it.

**`darwin-admin`** is for power users and operators. It has 36 subcommands across 9 groups — setup, models, store, pipeline, search, logs, system, uninstall, status. This is where all the real work happens.

The split exists because a user connecting their AI agent shouldn't need to know about model management or store repair. And an operator managing a Darwin deployment shouldn't need the MCP server cluttering their terminal.

---

## Core modules

### Storage (`src/core/Storage/`)

The persistence layer. Everything in Darwin reads from or writes to a `DarwinStore`.

A store is a directory with a SQLite registry and a set of artifact files per document: markdown, chunks (Parquet), BM25 index (NPZ), embeddings (NumPy float16), and a vector index (USearch). The store handles file registration, artifact I/O, health checks, audit, and repair.

There's also a multi-store registry — you can have multiple stores for different projects, and Darwin tracks them all in a central SQLite database.

**Why it exists:** RAG needs structured persistence. Without a store, you'd be managing raw files and hoping the indexes stay consistent. The store guarantees that if a document is registered, all its artifacts are either all present or all missing.

### Pipeline (`src/core/Pipeline/`)

The document processing pipeline. Two independent phases:

**Ingestion** takes raw documents and turns them into chunks. Parsing converts PDFs, HTML, DOCX, CSV, EPUB, and URLs into markdown. Chunking splits that markdown into sentence-aware segments with structural classification (header, body, table, list) and directional overlap.

**Indexing** takes chunks and builds search indexes. BM25 indexing creates a sparse keyword matrix. Embedding indexing generates dense vectors and builds a USearch vector index.

The two phases are separate because you might want to re-index without re-parsing (say, after switching embedding models). Or you might want to parse new documents without re-indexing everything. Keeping them independent saves time and compute.

### Retrieval (`src/core/Retrival/`)

The search engine. Three search strategies:

**Hybrid** fuses BM25 keyword scores with semantic embedding scores using Reciprocal Rank Fusion (RRF). This is the default because it combines the precision of keyword matching with the conceptual understanding of embeddings.

**Semantic** uses USearch vector similarity — good for fuzzy, conceptual queries.

**Keyword** uses BM25 Okapi scoring — good for exact terms, names, and code references.

On top of raw search, the retrieval layer adds: structural penalties (downweighting headers, tables, footnotes), source diversity penalties (preventing one file from dominating results), passage merging (combining adjacent chunks from the same page), and fallback detection (automatically relaxing thresholds when results are sparse).

Optional reranking uses an ONNX cross-encoder to re-score the top results.

### Orchestrators (`src/core/Orchestrators/`)

The glue layer between the CLI/MCP and the core engine. Each orchestrator encapsulates a domain:

- **RetrievalOrchestrator** — full retrieval pipeline: fetch, filter, penalize, merge, fallback, resolve
- **PipelineOrchestrator** — validate parameters, run stages, purge artifacts
- **StoreOrchestrator** — status, files, audit, repair, registry operations
- **ModelOrchestrator** — install, switch, uninstall, upgrade, detect hardware
- **LoggerOrchestrator** — query, purge, export, retention

The orchestrators exist because the core modules (DarwinStore, Pipeline, Retrieval) are low-level. Orchestrators provide the high-level operations that CLI commands and MCP tools need, with proper error handling and session logging.

All orchestrator parameters are defined in `src/core/Orchestrators/lib.py` — the single source of truth. Option-based values use `Literal` types for IDE autocomplete and MCP JSON Schema enum generation.

### Models (`src/core/Models/`)

Model management for 40+ AI models across 5 categories: embedding, reranker, NLP, vision, and OCR.

Handles the full lifecycle: discovery (what models can I run on my hardware?), installation (download from HuggingFace), switching (change the active model for a category), and inference (ONNX runtime, spaCy, API fallbacks).

Models are split into tiers (DOLPHIN for light hardware, ORCA for medium, WHALE for heavy) so Darwin can automatically pick the best model for your system during setup.

The model registry is a master list of all available models with their metadata — hardware requirements, loader type, provider, and tier. This is what makes the `--preset` and interactive setup possible.

### Logger (`src/core/Logger/`)

A session-based structured logging framework. Not standard Python `logging` — Darwin needs progress bars, spinners, SSE streaming for real-time dashboards, styled terminal output, and structured log queries.

The logger outputs to three places: terminal (Rich-based), SQLite history (for queries and export), and SSE stream (for the web dashboard and remote monitoring). The MCP server shares the SSE port with the logger.

All Darwin components log through a global `darwin` singleton, so there's one consistent logging interface everywhere.

### MCP (`src/mcp/`)

The MCP (Model Context Protocol) server. Exposes Darwin as 8 tools + 2 resources that AI agents can call.

Three transports: stdio (for subprocess by Claude Desktop, Cursor, etc.), SSE (legacy network), and Streamable HTTP (recommended for production). All transports share the same tools and validation logic.

The MCP server also serves a live log dashboard at `/dashboard` and a health check at `/health`.

### Lib (`src/lib/`)

Shared utilities that don't belong to any specific module: platform-aware path resolution, API key management via OS keyring, terminal helpers, internet connectivity checks.

### Features (`src/features/`)

Optional modules that extend core RAG: extractive summarization (spaCy + pytextrank) and image processing (OCR text extraction + vision captioning). These are not required for basic search and pipeline operations.

---

## Data flow

```
Documents
    │
    ▼
Pipeline: Ingestion
    │  Parse (PDF→markdown, HTML→text, etc.)
    │  Chunk (sentence-aware, structural classification)
    ▼
Pipeline: Indexing
    │  BM25 keyword index (NPZ)
    │  Embedding vectors (NumPy float16 + USearch)
    ▼
DarwinStore
    │  Per-document artifacts in SHA256-hashed directories
    │  SQLite registry tracking what's been processed
    │
    ▼
Search Query
    │
    ▼
Retrieval
    │  Hybrid/semantic/keyword search
    │  Structural penalties, diversity, merging
    │  Optional ONNX reranking
    ▼
Resolved Passages
    │
    ▼
MCP / CLI output
```

---

## MCP tools

| Tool | What it does |
|---|---|
| `search_darwin` | Query the knowledge base (hybrid/semantic/keyword) |
| `get_search_results` | List saved search results |
| `get_search_result_by_id` | Load a saved result by filename |
| `run_pipeline` | Ingest + index documents from a path or URL |
| `purge_artifacts` | Delete pipeline artifacts for specific files |
| `create_store` | Create a new DarwinStore |
| `list_files` | All files with pipeline status |
| `file_status` | Detailed status for one file |

Resources: `darwin://store/report` (pipeline report), `darwin://search/results` (saved search index).

---

## Configuration files

Darwin uses three config files in your platform's data directory:

- **`darwin.json`** — models, API keys, environment settings. Written by `darwin-admin setup`.
- **`logging.json`** — log mode, SSE port, retention. Written by setup, read by the MCP server at startup.
- **`models.json`** — active model per category. Updated when you switch models.

---

## Docker

The Docker setup uses a `python:3.11-slim` base image. The entrypoint validates that a volume is mounted, runs first-time setup if needed, and starts the MCP server.

See [docs/setup.md](docs/setup.md) for Docker instructions.
