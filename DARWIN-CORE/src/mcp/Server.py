"""
MCP server — CLI entry point.

Transports:
  * stdio (default) — AI agent subprocess (Claude, Cursor, etc.)
  * SSE            — Network / Docker / remote clients (includes log streaming)
"""

import argparse

from src.mcp.UnifiedServer import UnifiedServer


def build_mcp_app():
    """Return a FastMCP instance wired with all tools (no HTTP)."""
    from mcp.server.fastmcp import FastMCP
    from src.core.Orchestrators.StoreOrchestrator import StoreOrchestrator
    from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
    from src.core.Orchestrators.PipelineOrchestrator import PipelineOrchestrator
    import src.mcp.tools.Search as Search
    import src.mcp.tools.Pipeline as Pipeline
    import src.mcp.tools.Models as Models
    import src.mcp.tools.Store as Store

    mcp = FastMCP("darwin")
    orch = StoreOrchestrator()
    orch.bootstrap()
    retrieval = RetrievalOrchestrator(store=orch.store)
    pipeline = PipelineOrchestrator(store=orch.store)
    Search.register(mcp, retrieval)
    Pipeline.register(mcp, pipeline)
    Models.register(mcp)
    Store.register(mcp, orch.store)
    return mcp


def run_stdio():
    """Serve MCP tools over stdio (no log streaming)."""
    mcp = build_mcp_app()
    mcp.run(transport="stdio")


def run_sse(host: str = "127.0.0.1", port: int = 8765):
    """Serve MCP tools + log streaming + history over SSE."""
    server = UnifiedServer(host=host, port=port)
    server.run_foreground()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Darwin MCP Server")
    p.add_argument("--sse", action="store_true", help="Run in SSE mode (default: stdio)")
    p.add_argument("--port", type=int, default=8765, help="Port for SSE mode")
    p.add_argument("--host", default="127.0.0.1", help="Host for SSE mode")
    return p
