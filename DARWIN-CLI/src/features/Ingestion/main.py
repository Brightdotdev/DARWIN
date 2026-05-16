"""
ingestion/main.py - Unified ingestion pipeline: parsing + chunking (refactored)
"""

import sys
import asyncio
import argparse
from pathlib import Path
from typing import List, Optional

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.features.Ingestion.Parsing.main import handle_parsing
from src.features.Ingestion.Chunking.main import handle_chunking
from src.features.Ingestion.Parsing.lib import ParsingArgs
from src.features.Ingestion.Chunking.lib import ChunkingArgs
from src.features.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket
from src.lib.Utils import ensure_writable_directory

# ── helpers ───────────────────────────────────────────────────────────────

def resolve_target(path: str, files: Optional[List[str]]) -> str:
    if not files:
        return path

    parents = {str(Path(f).parent) for f in files}
    if len(parents) == 1:
        return parents.pop()

    darwin.log("Files span multiple directories. Using provided path.", LogType.WARNING)
    return path


def normalize_filenames(files: Optional[List[str]]) -> Optional[List[str]]:
    if not files:
        return None
    return [Path(f).name for f in files]


def split_markdown_state(files: List[str], store: DarwinStore):
    has_md = [f for f in files if store.has_markdown(f)]
    needs_md = [f for f in files if not store.has_markdown(f)]
    return has_md, needs_md


# ── pipeline ──────────────────────────────────────────────────────────────

def run_parsing(parsing_args: ParsingArgs, store: DarwinStore) -> List[str]:
    files = normalize_filenames(parsing_args.files)

    if parsing_args.skip_if_markdown and files:
        has_md, needs_md = split_markdown_state(files, store)

        if has_md:
            darwin.log(f"[ 1 / 2 ] Parsing skipped for {len(has_md)} file(s)", LogType.INFO)
            for f in has_md:
                darwin.log(f"  ✓ {f}", LogType.INFO)
            darwin.log("")

        if not needs_md:
            return has_md

        darwin.log(f"[ 1 / 2 ] Parsing {len(needs_md)} file(s)", LogType.INFO)
        darwin.log("")

        args = ParsingArgs(**{**vars(parsing_args), "files": needs_md})
        docs = handle_parsing(args,store=store)

        parsed = [d.filename for d in docs if not d.error]
        return has_md + parsed

    # normal path
    darwin.log("[ 1 / 2 ] Parsing", LogType.INFO)
    darwin.log("")

    docs = handle_parsing(parsing_args,store=store)
    if not docs:
        return []

    parsed = [d.filename for d in docs if not d.error]
    failed = [d.filename for d in docs if d.error]

    if failed:
        darwin.log(f"Skipping {len(failed)} failed parse(s)", LogType.WARNING)
        for f in failed:
            darwin.log(f"  ✗ {f}", LogType.WARNING)
        darwin.log("")

    return parsed


def run_chunking(filenames: List[str], base_args: ChunkingArgs, store: DarwinStore) -> None:
    darwin.log("")
    darwin.log("━" * 50, LogType.INFO)
    darwin.log("")
    darwin.log("[ 2 / 2 ] Chunking", LogType.INFO)
    darwin.log("")

    args = ChunkingArgs(**{**vars(base_args), "files": filenames})
    handle_chunking(args, store=store)


def handle_ingestion(parsing_args: ParsingArgs, chunking_args: ChunkingArgs, store_dir: str = None) -> None:
    darwin.log("")
    darwin.log("━" * 50, LogType.INFO)
    darwin.log("  INGESTION PIPELINE", LogType.INFO)
    darwin.log("━" * 50, LogType.INFO)
    darwin.log("")
    if store_dir and ensure_writable_directory(store_dir):
        store = DarwinStore(store_dir)
    else:
        store = DarwinStore()
    

    parsed_files = run_parsing(parsing_args, store)

    if not parsed_files:
        darwin.log("No documents available to chunk. Aborting.", LogType.ERROR)
        return

    run_chunking(parsed_files, chunking_args, store)

    darwin.log("")
    darwin.log("━" * 50, LogType.INFO)
    darwin.log("  INGESTION COMPLETE", LogType.SUCCESS)
    darwin.log("━" * 50, LogType.INFO)
    darwin.log("")


# ── CLI ───────────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified ingestion pipeline - parse and chunk documents"
    )

    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--file", "-f", nargs="+", default=None)

    # parsing
    parser.add_argument("--recursive", "-r", action="store_true")
    parser.add_argument("--images", "-img", action="store_true")
    parser.add_argument("--caption", "-c", action="store_true")
    parser.add_argument("--precise", action="store_true")
    parser.add_argument("--rerun", "-R", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--skip-if-markdown", action="store_true")

    # chunking
    parser.add_argument("--inspect", "-i", action="store_true")

    # websocket
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--no-ws", action="store_true")

    parser.add_argument("--store", "-s", default=None, help="Store directory")

    args = parser.parse_args()

    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        target = resolve_target(args.path, args.file)

        parsing_args = ParsingArgs(
            target=target,
            files=args.file,
            images=args.images,
            caption=args.caption,
            recursive=args.recursive,
            rerun=args.rerun,
            precise=args.precise,
            save=not args.no_save,
            skip_if_markdown=args.skip_if_markdown,
        )

        chunking_args = ChunkingArgs(
            inspect=args.inspect,
            rerun=args.rerun,
        )

        handle_ingestion(parsing_args, chunking_args,store_dir=args.store_dir)
        return 0

    except KeyboardInterrupt:
        darwin.log("")
        darwin.log("Goodbye!", LogType.INFO)
        return 130

    except Exception as e:
        import traceback
        darwin.log(f"Fatal error: {e}", LogType.ERROR)
        darwin.log("".join(traceback.format_exception(e)), LogType.ERROR)
        return 1

    finally:
        shutdown_websocket(ws_server)


if __name__ == "__main__":
    asyncio.run(main())