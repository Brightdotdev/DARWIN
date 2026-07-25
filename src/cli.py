"""
darwin — Local-first RAG engine.

User CLI. Starts the MCP server and generates client configs.
For admin tasks (setup, models, pipeline, status) use darwin-admin.
"""

import argparse
import sys


DEFAULT_PORT = 8765


def _read_logging_port() -> int:
    """Read sse_port from logging.json. Returns DEFAULT_PORT if unavailable."""
    import json
    try:
        from src.lib.AppPaths import default_logging_config
        path = default_logging_config()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return int(data.get("sse_port", DEFAULT_PORT))
    except (json.JSONDecodeError, ValueError, OSError) as e:
        import warnings
        warnings.warn(f"Failed to read logging.json: {e}")
    return DEFAULT_PORT


def _mcp(args):
    from src.core.Logger.DarwinLogger import darwin
    from src.mcp.Server import run_stdio
    from src.mcp.UnifiedServer import UnifiedServer

    port = args.temp_port or args.port or _read_logging_port()

    if args.http:
        server = UnifiedServer(host=args.host, port=port, transport="http")
        server.run_foreground(transport="http")
    elif args.sse:
        server = UnifiedServer(host=args.host, port=port, transport="sse")
        server.run_foreground(transport="sse")
    else:
        darwin.configure()
        run_stdio(host=args.host, port=port)


def _config(args):
    import json
    import os

    target = args.target or "all"
    python = sys.executable
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    configs = {}

    if target in ("claude", "claude-desktop", "all"):
        configs["Claude Desktop"] = {
            "mcpServers": {
                "darwin": {
                    "command": python,
                    "args": ["-m", "src.mcp"],
                    "cwd": workspace,
                }
            }
        }

    if target in ("cursor", "all"):
        configs["Cursor"] = {
            "mcpServers": {
                "darwin": {
                    "command": python,
                    "args": ["-m", "src.mcp"],
                    "cwd": workspace,
                }
            }
        }

    if target in ("vscode", "all"):
        configs["VS Code (Continue)"] = {
            "mcpServers": {
                "darwin": {
                    "command": python,
                    "args": ["-m", "src.mcp"],
                    "cwd": workspace,
                }
            }
        }

    if target in ("sse", "http", "all"):
        host = args.host or "127.0.0.1"
        port = args.port or _read_logging_port()
        transport = "http" if target == "http" else "sse"
        endpoint = f"http://{host}:{port}"
        if transport == "sse":
            endpoint += "/sse"
        configs[f"Network ({transport.upper()})"] = {
            "mcpServers": {
                "darwin": {
                    "url": endpoint,
                }
            }
        }

    if not configs:
        print(f"Unknown target: {target}")
        print("Valid targets: claude, cursor, vscode, sse, http, all")
        return

    output = json.dumps(configs, indent=2)
    print(output)

    if args.copy:
        try:
            import subprocess
            if sys.platform == "win32":
                subprocess.run(["clip"], input=output.encode(), check=True)
            else:
                subprocess.run(["pbcopy"], input=output.encode(), check=True)
            print("\nCopied to clipboard!")
        except Exception:
            print("\nCould not copy to clipboard")


def main():
    from src.core.Logger.DarwinLogger import darwin
    darwin.configure()
    parser = argparse.ArgumentParser(
        prog="darwin",
        description=(
            "Local-first RAG engine — start MCP server for AI agents.\n"
            "For admin tasks (setup, models, pipeline) use: darwin-admin"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── mcp ──
    m = sub.add_parser("mcp", help="Start MCP server")
    group = m.add_mutually_exclusive_group()
    group.add_argument(
        "--sse", action="store_true", help="Legacy SSE transport"
    )
    group.add_argument(
        "--http", action="store_true", help="Streamable HTTP transport (recommended)"
    )
    m.add_argument(
        "--port", type=int, default=8765, help="Port (default: from logging.json or 8765)"
    )
    m.add_argument(
        "--temp-port", type=int, default=None,
        help="Override port for this session only (does not save to config)"
    )
    m.add_argument(
        "--host", default="127.0.0.1", help="Host (default: 127.0.0.1)"
    )
    m.set_defaults(func=_mcp)

    # ── config ──
    cfg = sub.add_parser("config", help="Generate client config for MCP integration")
    cfg.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["claude", "cursor", "vscode", "sse", "http", "all"],
        help="Client to generate config for (default: all)",
    )
    cfg.add_argument("--host", default="127.0.0.1", help="Host for network config")
    cfg.add_argument("--port", type=int, default=8765, help="Port for network config")
    cfg.add_argument("--copy", action="store_true", help="Copy to clipboard")
    cfg.set_defaults(func=_config)

    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    if args.command == "mcp":
        from src.admin.setup import _setup_status
        _status = _setup_status()
        if not _status.any_configured:
            print("No config found — run  darwin-admin setup  first\n")
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
