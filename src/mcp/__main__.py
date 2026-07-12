"""
Entry point for running the Darwin MCP server.

Supports three transport modes:
  SSE  (legacy)     : python -m src.mcp --sse
  HTTP (modern)     : python -m src.mcp --http
  stdio (default)   : python -m src.mcp

The default port is read from logging.json (the same port used by the
logging SSE server). If not found, falls back to 8765.
"""

import argparse

from src.mcp.UnifiedServer import UnifiedServer
from src.mcp.Server import run_stdio


def _read_logging_port() -> int:
    """Read sse_port from logging.json. Returns 8765 if unavailable."""
    import json
    from pathlib import Path

    try:
        from src.lib.AppPaths import default_config_dir
        path = default_config_dir() / "logging.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("sse_port", 8765)
    except Exception:
        pass
    return 8765


def main():
    default_port = _read_logging_port()

    parser = argparse.ArgumentParser(description="Darwin MCP server")
    transport_group = parser.add_mutually_exclusive_group()
    transport_group.add_argument(
        "--sse",
        action="store_const",
        const="sse",
        dest="transport",
        default="stdio",
        help="Run with SSE transport (legacy, default for backward compat)",
    )
    transport_group.add_argument(
        "--http",
        action="store_const",
        const="http",
        dest="transport",
        default=None,
        help="Run with Streamable HTTP transport (modern, recommended)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args()

    if args.transport == "stdio":
        run_stdio(args.host, args.port)
    else:
        server = UnifiedServer(
            host=args.host, port=args.port, transport=args.transport
        )
        server.run_foreground(transport=args.transport)


if __name__ == "__main__":
    main()
