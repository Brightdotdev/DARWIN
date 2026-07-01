# src/mcp/server.py
"""
Unified Darwin MCP Server
Exposes: ModelsOrchestrator + PipelineOrchestrator tools
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("darwin")

# Register tool modules
from src.mcp.tools.Model import register_model_tools
from src.mcp.tools.Pipleine import register_pipeline_tools
from src.mcp.tools.Store import register_store_tools
register_model_tools(mcp)
register_pipeline_tools(mcp)
register_store_tools(mcp)

if __name__ == "__main__":
    mcp.run()