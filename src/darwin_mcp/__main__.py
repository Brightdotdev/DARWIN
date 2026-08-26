"""
Entry point for running the Darwin MCP server.

Supports three transport modes:
  SSE  (legacy)     : python -m darwin_mcp --sse
  HTTP (modern)     : python -m darwin_mcp --http
  stdio (default)   : python -m darwin_mcp

The default port is read from logging.json (the same port used by the
logging SSE server). If not found, falls back to 8765.
"""

import argparse

from darwin_mcp.UnifiedServer import UnifiedServer
from darwin_mcp.Server import run_stdio
from cli import _read_logging_port


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
        help="Run with SSE transport (legacy)",
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
