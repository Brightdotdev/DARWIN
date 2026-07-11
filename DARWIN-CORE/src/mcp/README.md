# Darwin MCP Server

Model Context Protocol server exposing Darwin as AI-agent tools,
with built-in log streaming for real-time dashboards.

## Quick start

```bash
# stdio (default — for subprocess by AI assistants)
python -m src.mcp

# SSE (network — for Docker / remote clients + log streaming)
python -m src.mcp --sse --port 8765
```

## Transports

| Transport | Use case | Command |
|-----------|----------|---------|
| stdio | AI agent subprocess (Claude, Cursor) | `python -m src.mcp` |
| SSE | Network / Docker / remote clients | `python -m src.mcp --sse --port 8765` |

## Endpoints (SSE mode)

When running with `--sse`, the unified server provides these endpoints:

| Endpoint | Type | Description |
|----------|------|-------------|
| `/sse` | MCP SSE | AI agent tool calls |
| `/messages/` | MCP POST | AI agent message handling |
| `/stream` | SSE | Real-time darwin log events (UI dashboard) |
| `/history` | GET | Query historical logs (`?session=&levels=&limit=`) |
| `/health` | GET | Health check (returns `ok`) |

**Logs never flow to the AI agent.**  The `/stream` endpoint is for UI
consumers (web dashboards, IDE plugins).  The agent uses `/sse` for
tool calls only.

## Available tools (10)

### Search
- **`search_darwin`** — Query the knowledge base with hybrid/semantic/keyword search.

### Pipeline
- **`run_pipeline`** — Ingest + index documents from a path or URL.
- **`purge_artifacts`** — Delete pipeline artifacts for specific files.
- **`pipeline_report`** — Get processing status for files.

### Models
- **`model_status`** — Check which models are installed per category.
- **`install_model`** — Download and register a model.
- **`switch_model`** — Change the active model for a category.
- **`recommend_models`** — Get model recommendations for your hardware.

### Store
- **`list_files`** — All files with pipeline status.
- **`file_status`** — Detailed status for one file.

## Adding a new tool

1. Create or edit a file in `src/mcp/tools/`
2. Add a `register(mcp, ...)` function with `@mcp.tool()` decorators
3. Wire it in `src/mcp/Server.py` (stdio) and `src/mcp/UnifiedServer.py` (SSE):
   ```python
   import src.mcp.tools.MyTool as MyTool
   MyTool.register(mcp, dependency)
   ```

## Architecture

```
Logger/                   mcp/
  DarwinLogger.py           UnifiedServer.py  ← subscribes to darwin events
  TerminalUi.py             Server.py         ← CLI entry point
  SQLiteHistoryStore.py     tools/*.py        ← tool implementations
  DarwinMP.py
```

- **Logger** is pure — it emits events and stores them.  Zero imports
  from Orchestrators / Storage / Pipeline.
- **mcp/UnifiedServer** uses FastMCP with `custom_route()` to mount
  log SSE, history, and health endpoints alongside the MCP SSE transport.
- **mcp/Server** is a thin CLI that delegates to `UnifiedServer` (SSE) or
  creates a stdio FastMCP (stdio).
- **DarwinLogger.configure(mode=SSE/BOTH)** starts `UnifiedServer` in a
  background thread, same as before.
