"""
Darwin MCP server entry point.

Run with:  python -m src.mcp
"""

import asyncio
from mcp.server.fastmcp import FastMCP

from src.core.Orchestrators.StoreOrchestrator import StoreOrchestrator
from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
from src.core.Orchestrators.PipelineOrchestrator import PipelineOrchestrator
import src.mcp.tools.Search as Search
import src.mcp.tools.Pipeline as Pipeline


def create_mcp_app(
    host: str = "127.0.0.1",
    port: int = 8765,
) -> FastMCP:
    """Build and return a FastMCP app with all Darwin tools registered."""
    from src.core.Orchestrators.StoreOrchestrator import StoreOrchestrator
    from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
    from src.core.Orchestrators.PipelineOrchestrator import PipelineOrchestrator
    from src.core.Orchestrators.ModelOrchestrator import modelsOrchestrator
    import src.mcp.tools.Search as Search
    import src.mcp.tools.Pipeline as Pipeline
    import src.mcp.tools.Models as Models
    import src.mcp.tools.Store as Store
    import src.mcp.tools.Resources as Resources

    mcp = FastMCP("darwin-mcp", host=host, port=port)

    orch = StoreOrchestrator()
    orch.bootstrap(quiet=True)
    retrieval = RetrievalOrchestrator(store=orch.store)
    pipeline = PipelineOrchestrator(store=orch.store)

    Search.register(mcp, retrieval)
    Pipeline.register(mcp, pipeline)
    Models.register(mcp)
    Store.register(mcp, orch.store)
    Resources.register(mcp, orch.store, models_orchestrator=modelsOrchestrator)

    return mcp


def run_stdio(host: str = "127.0.0.1", port: int = 8765):
    """Start the Darwin MCP server using stdio transport."""
    mcp = create_mcp_app(host=host, port=port)
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Darwin MCP Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_stdio(args.host, args.port)
