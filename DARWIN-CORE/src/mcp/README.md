# Darwin MCP Server

Model Context Protocol server exposing Darwin as AI-agent tools.

## Quick start

```bash
# stdio (default — for subprocess by AI assistants)
python -m src.mcp

# SSE (network — for Docker / remote clients)
python -m src.mcp --sse --port 8080
```

## Available tools (10)

### Search
- **`search_darwin`** — Query the knowledge base with hybrid/semantic/keyword search. Returns merged passages with source metadata. All option fields use `Literal` types — your IDE shows valid choices.

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

## Transports

| Transport | Use case | Command |
|-----------|----------|---------|
| stdio | AI agent subprocess (Claude, Cursor) | `python -m src.mcp` |
| SSE | Network / Docker / remote clients | `python -m src.mcp --sse --port 8080` |

## Adding a new tool

1. Create or edit a file in `src/mcp/tools/`
2. Add a `register(mcp, ...)` function with `@mcp.tool()` decorators
3. Wire it in `src/mcp/Server.py`:
   ```python
   import src.mcp.tools.MyTool as MyTool
   MyTool.register(mcp, dependency)
   ```

## Parameter types

Tools that accept option-based string values (method, merge strategy, fallback strategy, log mode) use `Literal` type hints. This means:
- Your IDE shows valid choices via autocomplete
- MCP client sees `enum` in the JSON Schema
- Invalid values are caught at the type-checker level
