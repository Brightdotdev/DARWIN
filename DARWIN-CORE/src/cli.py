"""
darwin — Local-first RAG engine.

Usage:
    darwin init              Bootstrap store + download required models
    darwin setup [options]   One-shot setup (directories, models, workspace)
    darwin mcp [--sse]       Start MCP server (stdio or SSE)
    darwin admin <command>   Full admin CLI (models, store, pipeline, logs)
    darwin status            Overview: store files, installed models, preferences
    darwin ingest <target>   Quick one-off ingest
    darwin search <query>    Quick one-off search
"""

import argparse
import sys


def _init(args):
    from src.core.init import run_init

    run_init(profile=args.profile, no_download=args.no_download)


def _setup(args):
    from src.Setup import _build_cli

    setup_cli = _build_cli()
    rest = args.args
    if not rest or rest[0] not in setup_cli.commands:
        rest = ["run"] + rest
    sys.argv = [sys.argv[0]] + rest
    setup_cli(standalone_mode=False, prog_name=f"{sys.argv[0]} setup")


def _mcp(args):
    from src.mcp.Server import mcp

    if args.sse:
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")


def _admin(args):
    from src.admin.cli import main as admin_main

    sys.argv = [sys.argv[0]] + args.args
    admin_main()


def _ingest(args):
    from src.admin.cli import main as admin_main

    admin_args = ["pipeline", "ingest", args.target]
    if args.store:
        admin_args.extend(["--store", args.store])
    if args.ocr:
        admin_args.append("--ocr")
    if args.recursive:
        admin_args.append("--recursive")
    sys.argv = [sys.argv[0]] + admin_args
    admin_main()


def _search(args):
    from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
    from src.core.Orchestrators.lib import RetrievalParameters, resolve_store

    store = resolve_store(args.store)
    store.bootstrap()
    orch = RetrievalOrchestrator(store=store)
    params = RetrievalParameters(
        query=args.query, method=args.method, top_k=args.top_k, store=args.store
    )
    ctx = orch.retrieve_and_resolve(params)
    if ctx and not ctx.is_empty():
        from rich.console import Console

        c = Console()
        for i, p in enumerate(ctx.passages, 1):
            c.print(
                f"\n[bold]#{i}[/bold] [cyan]{p.metadata.source}[/cyan] p.{p.metadata.page_start}"
            )
            c.print(p.text[:300] + ("..." if len(p.text) > 300 else ""))
        c.print(
            f"\n[green]{ctx.total_passages} passages from {len(ctx.sources_used)} sources[/green]"
        )
    else:
        print("No results.")


def _status(args):
    from src.admin.cli import cmd_status

    cmd_status(args)


def main():
    from src.core.Logger.DarwinLogger import darwin
    darwin.configure(show_banner=True, override=True)
    parser = argparse.ArgumentParser(
        prog="darwin",
        description="Local-first RAG engine — ingest, index, search, and serve via MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── init ──
    i = sub.add_parser("init", help="Bootstrap store + download required models")
    i.add_argument("--profile", default="required", choices=["required", "recommended"])
    i.add_argument(
        "--no-download", action="store_true", help="Register models without downloading"
    )
    i.set_defaults(func=_init)

    # ── setup ──
    s = sub.add_parser("setup", help="One-shot setup: directories, models, workspace")
    s.add_argument(
        "args", nargs=argparse.REMAINDER, help="Setup options (see --help after this)"
    )
    s.set_defaults(func=_setup)

    # ── mcp ──
    m = sub.add_parser("mcp", help="Start MCP server")
    m.add_argument(
        "--sse", action="store_true", help="Run in SSE mode (default: stdio)"
    )
    m.add_argument(
        "--port", type=int, default=8080, help="Port for SSE (default: 8080)"
    )
    m.set_defaults(func=_mcp)

    # ── admin ──
    a = sub.add_parser("admin", help="Full admin CLI (all orchestrator controls)")
    a.add_argument(
        "args", nargs=argparse.REMAINDER, help="Admin subcommand and arguments"
    )
    a.set_defaults(func=_admin)

    # ── ingest ──
    ing = sub.add_parser("ingest", help="Quick one-off ingest")
    ing.add_argument("target", help="Path or static URL to ingest")
    ing.add_argument("--store", help="Store ID, name, or path (default: auto)")
    ing.add_argument("--ocr", action="store_true", help="Enable OCR")
    ing.add_argument("--recursive", action="store_true")
    ing.set_defaults(func=_ingest)

    # ── search ──
    se = sub.add_parser("search", help="Quick one-off search")
    se.add_argument("query", help="Search query")
    se.add_argument(
        "--method", default="hybrid", choices=["hybrid", "semantic", "keyword"]
    )
    se.add_argument("--top-k", type=int, default=5)
    se.add_argument("--store", help="Store ID, name, or path (default: auto)")
    se.set_defaults(func=_search)

    # ── status ──
    st = sub.add_parser("status", help="Overview: store files, installed models, preferences")
    st.add_argument("--store", help="Store ID, name, or path")
    st.set_defaults(func=_status)

    args, unknown = parser.parse_known_args()
    if args.command in ("setup", "admin") and unknown:
        args.args = unknown
    elif unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    args.func(args)


if __name__ == "__main__":
    main()
