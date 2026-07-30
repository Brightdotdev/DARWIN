<!-- mcp-name: io.github.Brightdotdev/darwin-rag -->

# Darwin RAG

A local-first RAG engine that ingests documents, indexes them with BM25 + dense embeddings, and exposes search via an MCP server for AI agent integration.

- **Ingestion** — PDF, Markdown, HTML, images (OCR), CSV, Excel, ODS, URLs
- **Indexing** — BM25 keyword + dense embedding hybrid index with configurable chunking strategies
- **Search** — Hybrid, semantic, or keyword retrieval with reranking, diversity rerank, and structural penalties
- **Generation** — LLM-backed answer synthesis via LiteLLM (OpenAI, Anthropic, Gemini, etc.)
- **Observability** — Structured logging with per-session history, queryable via MCP
- **Isolation** — Multiple independent stores for tenant/project separation
- **Deployment** — stdio (AI agent subprocess), SSE, or Streamable HTTP; Docker-ready
- **Local-first** — Everything runs locally, fully offline-capable after setup

Built by [BrightDotDev](https://github.com/brightdotdev).  
[![License: MIT with Attribution](https://img.shields.io/badge/License-MIT%20with%20Attribution-blue.svg)](LICENSE)

---

## Quick Start

### 1. Install

```bash
pip install darwin-rag
```

### 2. Set up models

```bash
# Interactive — detects hardware, pick your models
darwin-admin setup interactive

# Or one-shot (embedding-only, no prompts)
darwin-admin setup --preset required
```

### 3. Start the MCP server & connect

```bash
# stdio mode — for AI agent subprocess (Claude Desktop, Cursor, etc.)
darwin mcp

# Or HTTP mode — for remote clients
darwin mcp --http --port 8765
```

Configure your MCP client:

```json
{
  "mcpServers": {
    "darwin": {
      "command": "darwin",
      "args": ["mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-..."  // At least one LLM provider key
      }
    }
  }
}
```

Or generate config automatically:

```bash
darwin config claude          # Claude Desktop config
darwin config cursor          # Cursor config
darwin config all --copy      # All clients + copy to clipboard
```

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_darwin` | Query the knowledge base with hybrid/semantic/keyword search |
| `get_search_results` | List saved search results |
| `get_search_result_by_id` | Load a saved search result by filename |
| `run_pipeline` | Ingest + index documents from a path or URL |
| `purge_artifacts` | Delete pipeline artifacts for specific files |
| `create_store` | Create a new isolated data store |
| `list_files` | List all tracked files with pipeline status |
| `file_status` | Detailed status for a single file across all stages |
| `get_logs` | Query session logs (oldest first, INFO excluded) |

> Full documentation: [docs/mcp.md](https://github.com/BrightDotDev/DARWIN/blob/main/docs/mcp.md)

---

## Remote / HTTP Mode

Start the server on a network-accessible endpoint:

```bash
# SSE transport (legacy)
darwin mcp --sse --host 0.0.0.0 --port 8765

# Streamable HTTP transport (recommended for production)
darwin mcp --http --host 0.0.0.0 --port 8765
```

Configure your MCP client with the URL:

```json
{
  "mcpServers": {
    "darwin": {
      "url": "http://your-host:8765/mcp"  // or /sse for SSE mode
    }
  }
}
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No* | OpenAI provider key |
| `ANTHROPIC_API_KEY` | No* | Anthropic provider key |
| `GEMINI_API_KEY` | No* | Google Gemini provider key |
| `MISTRAL_API_KEY` | No* | Mistral AI provider key |
| `GROQ_API_KEY` | No* | Groq provider key |
| `COHERE_API_KEY` | No* | Cohere provider key |
| `TOGETHER_API_KEY` | No* | Together AI provider key |
| `OPENROUTER_API_KEY` | No* | OpenRouter provider key |
| `DEEPSEEK_API_KEY` | No* | DeepSeek provider key |
| `DARWIN_BASE_DIR` | No | Override the base data directory |
| `NO_COLOR` | No | Set to any value to disable ANSI color output |

*\* At least one LLM provider key is required for answer generation. Search/indexing works without any.*

---

## Setup Details

| Command | What it does |
|---------|-------------|
| `darwin-admin setup interactive` | Guided setup — detect hardware, choose models |
| `darwin-admin setup --preset required` | Download embedding model only (fastest) |
| `darwin-admin setup --preset recommended` | Embedding + reranker + OCR models |
| `darwin-admin setup logging` | Reconfigure logging only |
| `darwin-admin setup validate` | Validate current setup |

See [docs/setup.md](https://github.com/BrightDotDev/DARWIN/blob/main/docs/setup.md) for the full walkthrough including Docker, from-source install, and API key configuration.

---

## CLI Reference

**`darwin`** — User CLI

| Command | Description |
|---------|-------------|
| `darwin mcp` | Start MCP server (stdio, `--sse` or `--http` for network) |
| `darwin config [client]` | Generate MCP client config |

**`darwin-admin`** — Power-user CLI

| Command | Description |
|---------|-------------|
| `darwin-admin setup` | Setup models, logging, and configuration |
| `darwin-admin status` | System status overview |
| `darwin-admin models` | Model registry: list, install, switch, keys |
| `darwin-admin store` | Data store: status, files, audit, health, repair |
| `darwin-admin pipeline` | Ingestion pipeline: run, ingest, index, purge |
| `darwin-admin search` | Interactive search |
| `darwin-admin logs` | Structured log viewer and management |
| `darwin-admin system` | System information |
| `darwin-admin uninstall` | Remove Darwin data and configuration |

See [docs/admin.md](https://github.com/BrightDotDev/DARWIN/blob/main/docs/admin.md) for the full command reference.

---

## Documentation

| Doc | What |
|-----|------|
| [setup.md](https://github.com/BrightDotDev/DARWIN/blob/main/docs/setup.md) | Full setup walkthrough |
| [mcp.md](https://github.com/BrightDotDev/DARWIN/blob/main/docs/mcp.md) | MCP server, tools, resources, transports |
| [admin.md](https://github.com/BrightDotDev/DARWIN/blob/main/docs/admin.md) | Admin CLI reference |
| [architecture.md](https://github.com/BrightDotDev/DARWIN/blob/main/docs/architecture.md) | For developers and contributors |
| [pipeline.md](https://github.com/BrightDotDev/DARWIN/blob/main/docs/pipeline.md) | Ingestion & indexing |
| [retrieval.md](https://github.com/BrightDotDev/DARWIN/blob/main/docs/retrieval.md) | Search engine |
| [storage.md](https://github.com/BrightDotDev/DARWIN/blob/main/docs/storage.md) | DarwinStore |
| [models.md](https://github.com/BrightDotDev/DARWIN/blob/main/docs/models.md) | Model registry & inference |
| [logger.md](https://github.com/BrightDotDev/DARWIN/blob/main/docs/logger.md) | Structured logging |
| [orchestrators.md](https://github.com/BrightDotDev/DARWIN/blob/main/docs/orchestrators.md) | High-level business logic |

---

## Contributing

Found a bug? Want to add something? You're welcome here.

- **Issues** — open one at [github.com/BrightDotDev/DARWIN/issues](https://github.com/BrightDotDev/DARWIN/issues)
- **Code** — fork, branch, PR. Keep it minimal.
- **AI-generated code is fine** — but you own what you ship. Test it before submitting.

Read [CONTRIBUTING.md](https://github.com/BrightDotDev/DARWIN/blob/main/CONTRIBUTING.md) for the full guidelines.

---

## License

MIT with Attribution — see [LICENSE](LICENSE).

Core architecture and implementation by [BrightDotDev](https://github.com/brightdotdev).
