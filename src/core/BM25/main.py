"""
src/core/BM25/main.py - BM25 indexing pipeline entry point

Reads chunks from .darwin, builds BM25 sparse index, saves bundle.
Run after chunking, before or after embedding.

Usage examples
--------------
# index everything chunked in current directory
uv run src/core/BM25/main.py

# index a specific file
uv run src/core/BM25/main.py --file biology.pdf

# adjust domain stopword threshold (lower = more aggressive)
uv run src/core/BM25/main.py --threshold 0.25

# use aggressive IDF smoothing (boost rare terms more)
uv run src/core/BM25/main.py --idf-smoothing 1.0

# inspect output — show vocabulary stats and sample tokens
uv run src/core/BM25/main.py --inspect

# force re-index even if fresh index exists
uv run src/core/BM25/main.py --rerun
"""

import sys
import argparse
import asyncio
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.BM25.BM25Indexer import (
    run_bm25_indexing_pipeline,
    inspect_bm25_index,
)
from src.core.BM25.lib import (
    BM25IndexingArgs,
    DOMAIN_STOP_THRESHOLD,
    DEFAULT_IDF_SMOOTHING,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket
from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.Utils import scan_folder


# ── Inspection helper ──────────────────────────────────────────────────────────

def _print_inspection(filename: str, store) -> None:
    try:
        data = store.load_bm25(filename)
        inspect_bm25_index(data)
    except FileNotFoundError:
        darwin.log("Could not load BM25 index for inspection", LogType.ERROR)




def _is_bm25_indexable(filename: str | None, status: dict) -> bool:
    """Determine which files to search based on args and registry status."""
    if filename is None:
        return False
        
    key = Path(filename).name
    file_data = status.get("files", {}).get(key)
    print("*************")
    print(file_data)
    print("*************")
    if file_data is None:
        return False
    return file_data.get("is_chunked", False)


# ── Core handler ───────────────────────────────────────────────────────────────

def handle_bm25_indexing(args: BM25IndexingArgs) -> None:
    store = DarwinStore()

    darwin.log("")
    darwin.log("BM25 indexing", LogType.INFO)
    darwin.log(
        f"Threshold: {args.threshold} | IDF smoothing: {args.idf_smoothing} | "
        f"Rerun: {args.rerun} | Inspect: {args.inspect}",
        LogType.INFO
    )
    darwin.log("")

    status = store.status()
    filenames = []

    if args.file:
        if not _is_bm25_indexable(args.file, status):
            darwin.log(f"Cannot index {args.file} — no chunks found", LogType.WARNING)
            return        
        filenames = [args.file]
    else:    
        # Get all files with chunks from store
        filenames = [
            fname for fname, info in status.get("files", {}).items()
             if info.get("is_chunked", False)
        ]
    
    if not filenames:
        darwin.log("No chunked files found. Run chunking first.", LogType.ERROR)
        return

    darwin.log(f"Indexing {len(filenames)} file(s)", LogType.INFO)
    

    succeeded = failed = skipped = 0

    for filename in filenames:
        if not args.rerun and store.has_bm25(filename):
            darwin.log(f"Skipping {filename} (fresh index exists)", LogType.INFO)
            skipped += 1
            continue

        try:
            chunks = store.load_chunks(filename)
        except FileNotFoundError:
            darwin.log(f"{filename} — no chunks found, skipping", LogType.ERROR)
            failed += 1
            continue

        try:
            data = run_bm25_indexing_pipeline(
                filename=filename,
                chunks=chunks,
                store=store,
                threshold=args.threshold,
                idf_smoothing=args.idf_smoothing,
            )

            if data is not None:
                succeeded += 1
                if args.inspect:
                    _print_inspection(filename, store)
            else:
                skipped += 1

        except Exception as e:
            darwin.log(f"{filename} — {type(e).__name__}: {e}", LogType.ERROR)
            failed += 1

    darwin.log("")
    darwin.log(f"done — {succeeded} indexed · {skipped} skipped · {failed} failed", LogType.SUCCESS)


# ── CLI ────────────────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build BM25 sparse index from chunked documents"
    )
    parser.add_argument("path",              nargs="?", default=".",             help="Project directory")
    parser.add_argument("--file",     "-f",  default=None,                       help="Index a single file")
    parser.add_argument("--project",  "-p",  default=None,                       help="Project directory (default: cwd)")
    parser.add_argument("--threshold","-t",  type=float, default=DOMAIN_STOP_THRESHOLD,
                        help=f"Domain stopword threshold (default: {DOMAIN_STOP_THRESHOLD})")
    parser.add_argument("--idf-smoothing", "-s", type=float, default=DEFAULT_IDF_SMOOTHING,
                        help=f"IDF smoothing parameter (default: {DEFAULT_IDF_SMOOTHING})")
    parser.add_argument("--inspect",  "-i",  action="store_true",                help="Print index stats and samples")
    parser.add_argument("--rerun",    "-R",  action="store_true",                help="Re-index even if fresh index exists")
    parser.add_argument("--ws-port", type=int, default=8765,                     help="WebSocket server port (default: 8765)")
    parser.add_argument("--no-ws",   action="store_true",                        help="Disable WebSocket server")

    args = parser.parse_args()

    # Validate target path
    target_path = Path(args.path).resolve()
    if not target_path.exists():
        darwin.log(f"Path does not exist: {args.path}", LogType.ERROR)
        return 1

    # Setup WebSocket
    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        bm25_args = BM25IndexingArgs(
            target=args.path,
            file=args.file,
            project=args.project or args.path,
            threshold=args.threshold,
            idf_smoothing=args.idf_smoothing,
            inspect=args.inspect,
            rerun=args.rerun,
        )
        
        handle_bm25_indexing(bm25_args)
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