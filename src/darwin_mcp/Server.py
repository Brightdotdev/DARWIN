"""
Darwin MCP server entry point (stdio transport).

Run with:  python -m darwin_mcp
"""

import asyncio

from darwin_mcp.app_factory import create_mcp_app


def run_stdio(host: str = "127.0.0.1", port: int = 8765):
    """Start the Darwin MCP server using stdio transport."""
    mcp, _ = create_mcp_app(host=host, port=port)
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Darwin MCP Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_stdio(args.host, args.port)
