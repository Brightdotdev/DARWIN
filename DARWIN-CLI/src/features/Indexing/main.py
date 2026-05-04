"""
main.py - Document parsing pipeline entry point
"""

import sys
import multiprocessing
import time
import asyncio
from pathlib import Path

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.Indexing.lib import IndexingArgs
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Logger.WebsocketHelpers import setup_websocket,shutdown_websocket


from src.features.Indexing.Embedding.lib import EmbeddingArgs

from src.features.Indexing.BM25.lib import (
    BM25IndexingArgs,
    DOMAIN_STOP_THRESHOLD,
    DEFAULT_IDF_SMOOTHING,
)



def _validate_inputs(args: IndexingArgs):
    target_path = Path(args.path).resolve()

    if not target_path.exists():
        darwin.log(f"Path does not exist: {args.path}", LogType.ERROR)
        return None

    if args.file:
        file_path = Path(args.file).resolve()

        if not file_path.exists():
            darwin.log(f"File not found: {args.file}", LogType.ERROR)
            return None

        if not file_path.is_file():
            darwin.log(f"Not a file: {args.file}", LogType.ERROR)
            return None

        return file_path

    return target_path

# ── handle_indexing ─────────────────────────────────────────────────────────────

def handle_indexing(args: IndexingArgs) -> None:
    from src.features.Indexing.Embedding.main import handle_embedding
    from src.features.Indexing.BM25.main import handle_bm25_indexing

    target = _validate_inputs(args)
    if target is None:
        return

    if args.cluster:
        files = args.cluster
    elif args.file:
        files = [args.file]
    else:
        files = None  # both handlers resolve from store internally

    darwin.log("Indexing", LogType.INFO)
    darwin.log(
        f"Threshold: {args.threshold} | IDF smoothing: {args.idf_smoothing} | "
        f"Inspect: {args.inspect} | Rerun: {args.rerun}",
        LogType.INFO
    )

    if files:
        darwin.log(f"Cluster: {len(files)} file(s)", LogType.INFO)

    darwin.log("")

    if files:
        succeeded = failed = 0

        for fname in files:
            try:
                darwin.log(f"[1/2] BM25 — {fname}", LogType.INFO)
                handle_bm25_indexing(BM25IndexingArgs(
                    target=args.path,
                    file=fname,
                    threshold=args.threshold,
                    idf_smoothing=args.idf_smoothing,
                    inspect=args.inspect,
                    rerun=args.rerun,
                ))

                darwin.log(f"[2/2] Embedding — {fname}", LogType.INFO)
                handle_embedding(EmbeddingArgs(
                    target=args.path,
                    file=fname,
                    inspect=args.inspect,
                    rerun=args.rerun,
                ))

                succeeded += 1

            except Exception as e:
                darwin.log(f"{fname} — {type(e).__name__}: {e}", LogType.ERROR)
                failed += 1

        darwin.log("")
        darwin.log(f"Indexing complete — {succeeded} indexed · {failed} failed", LogType.SUCCESS)

    else:
        darwin.log("[1/2] BM25 indexing", LogType.INFO)
        handle_bm25_indexing(BM25IndexingArgs(
            target=args.path,
            file=args.file,
            threshold=args.threshold,
            idf_smoothing=args.idf_smoothing,
            inspect=args.inspect,
            rerun=args.rerun,
        ))

        darwin.log("[2/2] Embedding indexing", LogType.INFO)
        handle_embedding(EmbeddingArgs(
            target=args.path,
            file=args.file,
            inspect=args.inspect,
            rerun=args.rerun,
        ))

        darwin.log("Indexing complete", LogType.SUCCESS)
# ── CLI entry point ────────────────────────────────────────────────────────────

async def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run indexing (BM25 + embeddings)")
    parser.add_argument("path", nargs="?", default=".", help="File or directory to parse (default: current directory)")
    parser.add_argument("--rerun", "-R", action="store_true", help="Reprocess cached files")
    parser.add_argument("--file", "-f", default=None, help="Parse a single file instead of the store")
    
    parser.add_argument("--threshold","-t",  type=float, default=DOMAIN_STOP_THRESHOLD,
                        help=f"Domain stopword threshold (default: {DOMAIN_STOP_THRESHOLD})")
    
    parser.add_argument("--idf-smoothing", "-s", type=float, default=DEFAULT_IDF_SMOOTHING,
                        help=f"IDF smoothing parameter (default: {DEFAULT_IDF_SMOOTHING})")
    parser.add_argument("--inspect",  "-i",  action="store_true", help="Print index stats and samples")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port (default: 8765)")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")
    parser.add_argument(
    "--cluster", "-cl",
    nargs="+",
    default=None,
    metavar="FILE",
    help="Index a specific set of files (space-separated filenames)",
)

    args = parser.parse_args()
    
    if args.cluster and args.file:
        darwin.log("--cluster and --file are mutually exclusive", LogType.ERROR)
        return 1

    # Use shared WebSocket setup — only ONE server starts
    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:

        indexing_args = IndexingArgs(
    path=args.path,
    rerun=args.rerun,
    file=args.file,
    cluster=args.cluster,        # ← add this
    inspect=args.inspect,
    threshold=args.threshold,
    idf_smoothing=args.idf_smoothing,
)

        handle_indexing(indexing_args)
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