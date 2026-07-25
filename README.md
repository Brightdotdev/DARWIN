# Darwin RAG

Local-first RAG engine. Ingest documents, index them with BM25 and embeddings, and search via hybrid/semantic/keyword retrieval. Runs as an MCP server for AI agent integration or via CLI. All local, all offline-capable.

Built by [BrightDotDev](https://github.com/brightdotdev).

[![License: MIT with Attribution](https://img.shields.io/badge/License-MIT%20with%20Attribution-blue.svg)](LICENSE)

---

## Setup

Internet is required for first-time setup to download and validate models. After that, everything runs offline.

### Interactive (guided)

Detects your hardware, shows available models, lets you pick. Takes about 2 minutes.

```bash
pip install darwin-rag
darwin-admin setup interactive
```

### One-shot: required (embedding only)

Downloads and configures only the embedding model. No prompts, no choices. Fastest path to a working install.

```bash
darwin-admin setup --preset required
```

### One-shot: recommended (embedding + reranker + OCR)

Also downloads a reranker and OCR model. Better search quality out of the box.

```bash
darwin-admin setup --preset recommended
```

After setup, start the MCP server:

```bash
darwin mcp --http --port 8765
```

For a full setup walkthrough including Docker, from source, and API key configuration, see [docs/setup.md](docs/setup.md).

---

## Connecting your agent

```bash
# Generate config for Claude Desktop, Cursor, etc.
darwin config claude

# Generate config for all clients
darwin config all --copy
```

Or point any MCP client directly:
- **SSE:** `http://localhost:8765/sse`
- **HTTP:** `http://localhost:8765/mcp`

---

## Commands

**`darwin`** — user CLI (mcp, config)

**`darwin-admin`** — power-user CLI (setup, models, store, pipeline, search, logs, system, uninstall, status)

See [docs/admin.md](docs/admin.md) for the full command reference.

---

## Documentation

| Doc | What |
|---|---|
| [setup.md](docs/setup.md) | Full setup walkthrough |
| [architecture.md](docs/architecture.md) | For developers and contributors |
| [mcp.md](docs/mcp.md) | MCP server, tools, resources |
| [admin.md](docs/admin.md) | Admin CLI reference |
| [storage.md](docs/storage.md) | DarwinStore |
| [pipeline.md](docs/pipeline.md) | Ingestion & indexing |
| [retrieval.md](docs/retrieval.md) | Search engine |
| [models.md](docs/models.md) | Model registry & inference |
| [logger.md](docs/logger.md) | Structured logging |
| [orchestrators.md](docs/orchestrators.md) | High-level business logic |

---

## License

MIT with Attribution — see [LICENSE](LICENSE).

Core architecture and implementation by [BrightDotDev](https://github.com/brightdotdev).
