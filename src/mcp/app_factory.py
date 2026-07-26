"""
Shared MCP app factory — single source of truth for tool registration.

Both Server.py (stdio) and UnifiedServer.py (SSE/HTTP) call
``create_mcp_app()`` so adding a tool only requires one edit.
"""

import sys
from typing import Tuple

from mcp.server.fastmcp import FastMCP

from src.core.Orchestrators.StoreOrchestrator import StoreOrchestrator
from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
from src.core.Orchestrators.PipelineOrchestrator import PipelineOrchestrator
import src.mcp.tools.Search as Search
import src.mcp.tools.Pipeline as Pipeline
import src.mcp.tools.Store as Store
import src.mcp.tools.Resources as Resources
import src.mcp.tools.Logs as Logs


def create_mcp_app(
    host: str = "127.0.0.1",
    port: int = 8765,
) -> Tuple[FastMCP, StoreOrchestrator]:
    """Build and return a FastMCP app with all Darwin tools registered.

    Returns (mcp, orch) — the orchestrator is needed by UnifiedServer
    for health checks and custom routes.
    """
    mcp = FastMCP("darwin-mcp", host=host, port=port)

    orch = StoreOrchestrator()
    orch.bootstrap(quiet=True)

    try:
        retrieval = RetrievalOrchestrator(store=orch.store)
    except Exception as e:
        print(f"[MCP] WARNING: RetrievalOrchestrator init failed: {e}", file=sys.stderr)
        retrieval = None

    try:
        pipeline = PipelineOrchestrator(store=orch.store)
    except Exception as e:
        print(f"[MCP] WARNING: PipelineOrchestrator init failed: {e}", file=sys.stderr)
        pipeline = None

    Search.register(mcp, retrieval)
    Pipeline.register(mcp, pipeline)
    Store.register(mcp, orch.store)
    Resources.register(mcp, orch.store)
    Logs.register(mcp)

    return mcp, orch
