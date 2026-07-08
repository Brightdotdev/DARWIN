import sys
import argparse

from mcp.server.fastmcp import FastMCP

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
from src.core.Orchestrators.PipelineOrchestrator import PipelineOrchestrator

import src.mcp.tools.Search as Search
import src.mcp.tools.Pipeline as Pipeline
import src.mcp.tools.Models as Models
import src.mcp.tools.Store as Store

mcp = FastMCP("darwin")

try:
    store = DarwinStore()
    store.bootstrap()

    retrieval = RetrievalOrchestrator(store=store)
    pipeline = PipelineOrchestrator(store=store)

    Search.register(mcp, retrieval)
    Pipeline.register(mcp, pipeline)
    Models.register(mcp)
    Store.register(mcp, store)

    _init_ok = True
except Exception as e:
    print(f"[darwin-mcp] Init failed: {e}", file=sys.stderr)
    _init_ok = False


def _serve_stdio():
    mcp.run(transport="stdio")


def _serve_sse(port: int):
    mcp.run(transport="sse", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Darwin MCP Server")
    parser.add_argument(
        "--sse", action="store_true", help="Run in SSE mode (default: stdio)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for SSE mode (default: 8080)"
    )
    args = parser.parse_args()

    if args.sse:
        _serve_sse(args.port)
    else:
        _serve_stdio()
