"""
ingestion/main.py - Unified ingestion pipeline: parsing + chunking
"""

import sys
import asyncio
import argparse
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.features.Ingestion.Parsing.main import handle_parsing
from src.features.Ingestion.Chunking.main import handle_chunking
from src.features.Ingestion.Parsing.lib import ParsingArgs
from src.features.Ingestion.Chunking.lib import ChunkingArgs
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket


# ── Unified handler ────────────────────────────────────────────────────────────

def handle_ingestion(
    parsing_args: ParsingArgs,
    chunking_args: ChunkingArgs,
) -> None:
    from src.features.Storage.DarwinStore import DarwinStore

    darwin.log("")
    darwin.log("━" * 50, LogType.INFO)
    darwin.log("  INGESTION PIPELINE", LogType.INFO)
    darwin.log("━" * 50, LogType.INFO)
    darwin.log("")

    store = DarwinStore()

    # ── Resolve target filenames ──
    if parsing_args.cluster:
        candidates = [Path(f).name for f in parsing_args.cluster]
    elif parsing_args.file:
        candidates = [Path(parsing_args.file).name]
    else:
        candidates = None  # directory mode — let handle_parsing resolve

    # ── Markdown-aware parse gate ──
    if parsing_args.skip_if_markdown and candidates:
        needs_parse  = [f for f in candidates if not store.has_markdown(f)]
        has_markdown = [f for f in candidates if store.has_markdown(f)]

        if has_markdown:
            darwin.log(f"[ 1 / 2 ]  Parsing — skipped for {len(has_markdown)} file(s) (markdown exists)", LogType.INFO)
            for f in has_markdown:
                darwin.log(f"  ✓ {f}", LogType.INFO)
            darwin.log("")

        if needs_parse:
            darwin.log(f"[ 1 / 2 ]  Parsing — {len(needs_parse)} file(s) have no markdown", LogType.INFO)
            darwin.log("")
            parsing_args.cluster = needs_parse if len(needs_parse) > 1 else None
            parsing_args.file    = needs_parse[0] if len(needs_parse) == 1 else parsing_args.file

            docs = handle_parsing(parsing_args)
            freshly_parsed = [doc.filename for doc in docs if not doc.error]
        else:
            freshly_parsed = []

        parsed_names = has_markdown + freshly_parsed

    else:
        # ── Normal parse path ──
        darwin.log("[ 1 / 2 ]  Parsing", LogType.INFO)
        darwin.log("")

        docs = handle_parsing(parsing_args)

        if not docs:
            darwin.log("Parsing produced no documents — aborting ingestion.", LogType.ERROR)
            return

        parsed_names  = [doc.filename for doc in docs if not doc.error]
        failed_parse  = [doc.filename for doc in docs if doc.error]

        if failed_parse:
            darwin.log(f"Skipping {len(failed_parse)} failed parse(s) from chunking queue", LogType.WARNING)

    if not parsed_names:
        darwin.log("No documents available to chunk — aborting.", LogType.ERROR)
        return

    darwin.log("")
    darwin.log("━" * 50, LogType.INFO)
    darwin.log("")

    # ── Phase 2: Chunking ──
    darwin.log("[ 2 / 2 ]  Chunking", LogType.INFO)
    darwin.log("")

    chunking_args.cluster = parsed_names
    handle_chunking(chunking_args)

    darwin.log("")
    darwin.log("━" * 50, LogType.INFO)
    darwin.log("  INGESTION COMPLETE", LogType.SUCCESS)
    darwin.log("━" * 50, LogType.INFO)
    darwin.log("")
# ── CLI entry point ────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified ingestion pipeline — parse and chunk documents"
    )

    # ── Target ──
    parser.add_argument("path",          nargs="?", default=".",  help="File or directory to ingest (default: current directory)")
    parser.add_argument("--file",  "-f", default=None,            help="Ingest a single file")
    parser.add_argument("--cluster", "-cl", nargs="+", default=None, metavar="FILE", help="Ingest a specific set of files (space-separated paths)")

    # ── Parsing flags ──
    parser.add_argument("--recursive", "-r",   action="store_true", help="Scan directories recursively")
    parser.add_argument("--images",    "-img",  action="store_true", help="Extract images from documents")
    parser.add_argument("--caption",   "-c",    action="store_true", help="Generate captions for images (requires --images)")
    parser.add_argument("--precise",            action="store_true", help="Use pymupdf4llm for layout-accurate PDF extraction")
    parser.add_argument("--rerun",     "-R",    action="store_true", help="Reprocess already-parsed files")
    parser.add_argument("--no-save",            action="store_true", help="Skip saving markdown to store (preview only)")

    # ── Chunking flags ──
    parser.add_argument("--inspect",   "-i",    action="store_true", help="Print chunk inspection output after chunking")

    # ── WebSocket ──
    parser.add_argument("--ws-port",   type=int, default=8765,       help="WebSocket server port (default: 8765)")
    parser.add_argument("--no-ws",     action="store_true",          help="Disable WebSocket server")

    args = parser.parse_args()

    # ── Mutual exclusion ──
    if args.cluster and args.file:
        darwin.log("--cluster and --file are mutually exclusive", LogType.ERROR)
        return 1

    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        parsing_args = ParsingArgs(
            target=args.path if not args.file else str(Path(args.file).parent),
            file=args.file,
            cluster=args.cluster,
            images=args.images,
            caption=args.caption,
            recursive=args.recursive,
            rerun=args.rerun,
            precise=args.precise,
            save=not args.no_save,
        )

        chunking_args = ChunkingArgs(
            target=args.path,
            project=args.path,
            inspect=args.inspect,
            rerun=args.rerun,
        )

        handle_ingestion(parsing_args, chunking_args)
        return 0

    except KeyboardInterrupt:
        darwin.log("")
        darwin.log("Goodbye!", LogType.INFO)
        return 130
    except Exception as e:
        import traceback
        darwin.log(f"Fatal error: {e}", LogType.ERROR)
        darwin.log(''.join(traceback.format_exception(e)), LogType.ERROR)
        return 1
    finally:
        shutdown_websocket(ws_server)


if __name__ == "__main__":
    asyncio.run(main())