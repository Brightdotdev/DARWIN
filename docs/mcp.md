# Darwin MCP Server

Model Context Protocol server exposing Darwin as AI-agent tools and resources,
with built-in log streaming for real-time dashboards.

## Quick start

```bash
# stdio (default — for subprocess by AI assistants)
darwin mcp

# SSE (legacy network transport)
darwin mcp --sse --port 8765

# Streamable HTTP (recommended for production)
darwin mcp --http --port 8765

# Override port for this session only (does not save to config)
darwin mcp --temp-port 9000
```

All transports share the same 9 tools, 2 resources, and validation logic.
The unified server (`UnifiedServer.py`) handles SSE, HTTP, log streaming,
history, and health in a single process.

## Transports

| Transport | Use case | Command | Logs on stdout? |
|-----------|----------|---------|-----------------|
| **stdio** | AI agent subprocess (Claude, Cursor) | `darwin mcp` | No — silenced for clean JSON-RPC |
| **SSE** | Legacy network / Docker | `darwin mcp --sse` | Yes — visible on terminal |
| **Streamable HTTP** | Production / remote clients | `darwin mcp --http` | Yes — visible on terminal |

## Port sharing

MCP and the logger SSE server share the same port. The port is read from
`logging.json` at startup (field: `sse_port`). If the port is busy,
`UnifiedServer._resolve_port()` scans for a free one and saves it back
to `logging.json`.

```bash
# The port in logging.json is used by both MCP and logger
darwin mcp --http          # reads port from logging.json
darwin mcp --temp-port 9000  # overrides for this session only
```

## Endpoints

When running with `--sse` or `--http`, the unified server provides:

| Endpoint | Type | Description |
|----------|------|-------------|
| `/sse` | MCP SSE | AI agent tool calls (SSE mode) |
| `/messages/` | MCP POST | AI agent message handling (SSE mode) |
| `/mcp` | MCP Streamable HTTP | AI agent tool calls (HTTP mode) |
| `/stream` | SSE | Real-time darwin log events (UI dashboard) |
| `/history` | GET | Query historical logs (`?session=&levels=&limit=&before=`) |
| `/health` | GET | Health check (checks store + models) |
| `/dashboard` | GET | Live log viewer web UI |
| `/banner` | GET | ASCII banner as JSON |

**Logs never flow to the AI agent.** The `/stream` endpoint is for UI
consumers (web dashboards, IDE plugins). The agent uses `/sse` or `/mcp`
for tool calls only.

## Available tools (9)

### Search
- **`search_darwin`** — Query the knowledge base with hybrid/semantic/keyword search.
- **`get_search_results`** — List saved search results.
- **`get_search_result_by_id`** — Load a saved search result by filename.

### Pipeline
- **`run_pipeline`** — Ingest + index documents from a path or URL.
- **`purge_artifacts`** — Delete pipeline artifacts for specific files.

### Store
- **`create_store`** — Create a new DarwinStore at a given path.
- **`list_files`** — All files with pipeline status (supports custom store).
- **`file_status`** — Detailed status for one file (supports custom store).

### Logs
- **`get_logs`** — Query session logs (oldest first). Filters: session name, verbosity level. INFO is never returned.

## Available resources (2)

| URI | Description |
|-----|-------------|
| `darwin://store/report` | Pipeline report — all files with completed stages and aggregate stats |
| `darwin://search/results` | Saved search results index — lists all previously saved searches |

## Tool annotations

All tools include MCP-compliant annotations for client-side safety:

| Tool | readOnly | destructive | idempotent | openWorld |
|------|----------|-------------|------------|-----------|
| `search_darwin` | yes | no | yes | no |
| `get_search_results` | yes | no | yes | no |
| `get_search_result_by_id` | yes | no | yes | no |
| `run_pipeline` | no | no | no | no |
| `purge_artifacts` | no | **yes** | yes | no |
| `create_store` | no | no | yes | no |
| `list_files` | yes | no | yes | no |
| `file_status` | yes | no | yes | no |
| `get_logs` | yes | no | yes | no |

## Input validation

All tool parameters are validated before reaching orchestrators:

- **Search**: query max 4096 chars, non-empty
- **Pipeline**: stage whitelist, target path non-empty + max 2048 chars, invalid characters blocked, filenames validated
- **Store**: path max 2048 chars, `..` traversal blocked, invalid characters blocked, filename validation on `file_status`, name max 256 chars
- **Logs**: session must be one of `PIPELINE`, `RETRIEVAL`, `Ingestion`, `Indexing` (or omitted for all), limit clamped to 1-1000

All tools return structured `{"success": bool, "data": ..., "error": ...}` responses. Exceptions are caught and returned as error objects (traceback logged server-side only).

## Health endpoint

`GET /health` checks store and models:

```json
{
  "status": "ok",
  "checks": {
    "store": { "status": "ok", "files": 12 },
    "models": { "status": "ok", "all_ready": true, "summary": "5/5 ready" }
  }
}
```

Returns HTTP 503 when any check is degraded or errored.

## Client config generation

```bash
# Generate config for Claude Desktop
darwin config claude

# Generate config for all clients + copy to clipboard
darwin config all --copy

# Network config (SSE or HTTP)
darwin config sse --host 127.0.0.1 --port 8765
darwin config http --host 127.0.0.1 --port 8765
```

## Adding a new tool

1. Create or edit a file in `src/mcp/tools/`
2. Add a `register(mcp, ...)` function with `@mcp.tool(annotations=ToolAnnotations(...))` decorators
3. Wire it in `src/mcp/app_factory.py`:
   ```python
   import src.mcp.tools.MyTool as MyTool
   MyTool.register(mcp, dependency)
   ```

## Adding a new resource

1. Add a function in `src/mcp/tools/Resources.py` with `@mcp.resource("darwin://...")`
2. Wire it in the `register()` function (already called from Server.py and UnifiedServer.py)

## Architecture

```
Logger/                   mcp/
  DarwinLogger.py           UnifiedServer.py  ← subscribes to darwin events, SSE + HTTP + health
  TerminalUi.py             Server.py         ← stdio entry point
  SQLiteHistoryStore.py     app_factory.py    ← single source of truth for tool registration
  lib.py                    tools/*.py        ← tool implementations
                            tools/Resources.py ← resource definitions
```

- **Logger** is pure — it emits events and stores them. Zero imports from Orchestrators / Storage / Pipeline.
- **mcp/UnifiedServer** uses FastMCP with `custom_route()` to mount log SSE, history, and health endpoints alongside the MCP transports. Health checks store, returns 503 when degraded. Saves resolved port back to `logging.json`.
- **mcp/Server** builds a FastMCP via `app_factory.create_mcp_app()` and runs stdio transport.
- **mcp/app_factory** is the single source of truth — both Server.py and UnifiedServer.py call `create_mcp_app()` so adding a tool only requires one edit.
- **DarwinLogger.configure(mode=SSE/BOTH)** starts `UnifiedServer` in a background thread. `shutdown()` closes SSE server + history store.
- **Stdio mode** — logger is silenced (`mode=none`) to keep stdout clean for JSON-RPC. Logs still go to SQLite history and SSE stream if configured.
- **Tool annotations** — all 9 tools have `ToolAnnotations` (readOnlyHint, destructiveHint, idempotentHint, openWorldHint).
- **Input validation** — all tools validate parameters before reaching orchestrators. Exceptions returned as structured error objects, never silently swallowed.
- **Async tools** — all tool functions are async with blocking orchestrator calls dispatched via `run_in_executor`.
