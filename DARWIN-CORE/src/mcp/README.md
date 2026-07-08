# MCP Server — AI Agent Integration

**What it is:** A FastMCP server that exposes Darwin's full functionality as 10 MCP tools. AI agents (Claude, etc.) can search, ingest, index, manage models, and inspect the store through the Model Context Protocol.

**Why it exists:** Darwin is designed as an MCP server first. AI agents need a standard protocol (not a CLI) to interact with the RAG engine. FastMCP implements the MCP spec with stdio and SSE transports.

## Entry Points

```bash
# Via CLI
darwin mcp                    # stdio (stdin/stdout)
darwin mcp --sse --port 8080  # HTTP SSE on port 8080

# Directly
python -m src.mcp             # stdio mode (via __main__.py)
```

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Package init |
| `__main__.py` | Entry: `python -m src.mcp` |
| `Server.py` | FastMCP server bootstrap — creates store, orchestrators, registers 10 tools |
| `clibase.py` | MCP logging configuration helper |
| `tools/Search.py` | `search_darwin` tool |
| `tools/Pipeline.py` | `run_pipeline`, `purge_artifacts`, `pipeline_report` tools |
| `tools/Models.py` | `model_status`, `install_model`, `switch_model`, `recommend_models` tools |
| `tools/Store.py` | `list_files`, `file_status` tools |

## Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `search_darwin` | Retrieve passages for a query | query, method, top_k, max_passages, rerank, files, merge_strategy |
| `run_pipeline` | Run ingestion/indexing pipeline | parsing, chunking, indexing, stages |
| `purge_artifacts` | Delete artifacts for files | stages, filenames |
| `pipeline_report` | Get pipeline report for a file | filename |
| `model_status` | Show model status across categories | — |
| `install_model` | Install a model | model_id |
| `switch_model` | Switch active model | category, model_id |
| `recommend_models` | Hardware-aware recommendations | — |
| `list_files` | List all files with pipeline status | — |
| `file_status` | Detailed status for one file | filename |

## Usage

### From Python (FastMCP client)

```python
import asyncio
from mcp import Client
from mcp.transport import StdioClientTransport

async def main():
    transport = StdioClientTransport(["darwin", "mcp"])
    client = Client(transport)
    await client.connect()

    result = await client.call_tool("search_darwin", {
        "query": "machine learning",
        "method": "hybrid",
        "top_k": 5,
    })
    print(result)

asyncio.run(main())
```

### From Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "darwin": {
      "command": "darwin",
      "args": ["mcp"]
    }
  }
}
```

### From CLI (for testing)

```bash
# Start SSE server
darwin mcp --sse --port 8080

# Test with curl
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search_darwin","arguments":{"query":"test"}}}'
```

## Internal MCP Server

There's also a separate logger MCP server at `src/core/Logger/McpServer.py` that exposes 11 tools for pipeline, retrieval, store, and model operations. This is used internally and by the old MCP stack.

## Architecture

```
AI Agent ──> FastMCP transport ──> Server.py
  (stdio/SSE)                       ├── DarwinStore
                                    ├── RetrievalOrchestrator
                                    ├── PipelineOrchestrator
                                    └── ModelOrchestrator
```
