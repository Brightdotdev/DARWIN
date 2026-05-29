"""
main.py - Document indexing pipeline entry point

Usage:
  uv run main.py                                        # index all stored files
  uv run main.py -f biology.pdf chemistry.pdf           # index specific files
  uv run main.py -f biology.pdf --only-kw               # keyword index only
  uv run main.py -f biology.pdf --only-semantic         # embedding only
  uv run main.py --rerun                                # force reindex everything
"""

import sys
import os
import warnings
import multiprocessing
import asyncio
from pathlib import Path

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.path.insert(0, str(Path(__file__).parent.parent))



from src.features.Indexing.lib import IndexingArgs
from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity
from src.lib.Utils import ensure_writable_directory
from src.core.Storage.DarwinStore import DarwinStore

from src.features.Indexing.Embedding.lib import EmbeddingArgs

from src.features.Indexing.BM25.lib import (
    BM25IndexingArgs,
    DOMAIN_STOP_THRESHOLD,
    DEFAULT_IDF_SMOOTHING,
)


# ── input validation ───────────────────────────────────────────────────────────

def _validate_inputs(args: IndexingArgs) -> bool:
    """Validate path and any explicitly provided files. Returns False on error."""
  
    if args.files:
        for f in args.files:
            file_path = Path(f).resolve()
            if not file_path.exists():
                darwin.log(f"File not found: {f}", LogType.ERROR)
                return False
            if not file_path.is_file():
                darwin.log(f"Not a file: {f}", LogType.ERROR)
                return False

    return True


# ── main handler ──────────────────────
def handle_indexing(args: IndexingArgs, store: DarwinStore) -> None:
    from src.features.Indexing.Embedding.main import handle_embedding
    from src.features.Indexing.BM25.main import handle_bm25_indexing

  
    if not _validate_inputs(args):
        return


    files = args.files or None

    run_kw       = not args.only_semantic
    run_semantic = not args.only_kw


    steps      = ([" KW"] if run_kw else []) + (["Semantic"] if run_semantic else [])
    step_count = len(steps)

    def step_label(name: str) -> str:
        idx = steps.index(name) + 1
        return f"[{idx}/{step_count}]"

    darwin.log("Indexing", LogType.INFO)
    darwin.log(
        f"Threshold: {args.threshold} | IDF smoothing: {args.idf_smoothing} | "
        f"Inspect: {args.inspect} | Rerun: {args.rerun}",
        LogType.INFO,
    )
    if files:
        darwin.log(f"Files: {files}", LogType.INFO)
    darwin.log("")


    if run_kw:
        darwin.log(f"{step_label(' KW')} Keyword (BM25) indexing", LogType.INFO)
        try:
            handle_bm25_indexing(BM25IndexingArgs(
                
                files=files,     
                threshold=args.threshold,
                idf_smoothing=args.idf_smoothing,
                inspect=args.inspect,
                rerun=args.rerun,
            ),store=store)
        except Exception as e:
            import traceback
            darwin.log(f"BM25 indexing failed — {type(e).__name__}: {e}", LogType.ERROR)
            darwin.log(traceback.format_exc())
        
        darwin.log("")


    if run_semantic:
        darwin.log(f"{step_label('Semantic')} Semantic embedding", LogType.INFO)
        try:
            handle_embedding(EmbeddingArgs(
                files=files,          # full batch — model loads once, embeds all
                inspect=args.inspect,
                rerun=args.rerun,
            ),store=store)
        except Exception as e:
            import traceback
            darwin.log(f"Embedding failed — {type(e).__name__}: {e}", LogType.ERROR)
            darwin.log(traceback.format_exc())
        darwin.log("")

    darwin.log("Indexing complete", LogType.SUCCESS)


# ── CLI entry point ────────────────────────────────────────────────────────────

async def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run indexing (BM25 + embeddings)")

    parser.add_argument(
        "path", nargs="?", default=".",
        help="Project directory (default: current directory)",
    )
    parser.add_argument(
        "--file", "-f", nargs="+", default=None, dest="files",
        metavar="FILE",
        help="One or more files to index. Omit to index all stored files.",
    )
    parser.add_argument("--rerun", "-R", action="store_true", help="Reprocess already-indexed files")
    parser.add_argument(
        "--threshold", "-t", type=float, default=DOMAIN_STOP_THRESHOLD,
        help=f"Domain stopword threshold (default: {DOMAIN_STOP_THRESHOLD})",
    )
    parser.add_argument(
        "--idf-smoothing", "-s", type=float, default=DEFAULT_IDF_SMOOTHING,
        help=f"IDF smoothing parameter (default: {DEFAULT_IDF_SMOOTHING})",
    )
    parser.add_argument("--inspect", "-i", action="store_true", help="Print index stats and samples")

    # pass selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--only-kw",       action="store_true", help="Run keyword (BM25) indexing only")
    mode.add_argument("--only-semantic", action="store_true", help="Run semantic embedding only")

    # websocket
    parser.add_argument("--verbosity", choices=["normal", "quiet"], default="normal", help="Log verbosity")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port (default: 8765)")
    parser.add_argument("--no-ws",   action="store_true",    help="Disable WebSocket server")
    parser.add_argument("--store-dir",  default=None,   help="Darwin store directory")

    args = parser.parse_args()


    darwin.configure(
        enable_websocket=not args.no_ws,
        ws_port=args.ws_port,
        enable_history=True,
        verbosity_default=Verbosity.QUIET if args.verbosity == "quiet" else Verbosity.NORMAL,
        show_banner=True,
    )

    try:
        indexing_args = IndexingArgs(
            path=args.path,
            rerun=args.rerun,
            files=args.files,           # list[str] | None
            inspect=args.inspect,
            threshold=args.threshold,
            idf_smoothing=args.idf_smoothing,
            only_kw=args.only_kw,
            only_semantic=args.only_semantic,
        )
        
        if args.store_dir and ensure_writable_directory(args.store_dir):
            store = DarwinStore(args.store_dir)
        else:
            store = DarwinStore()


        

        with darwin.session("indexing"):
            handle_indexing(indexing_args, store)
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
        darwin.shutdown()


if __name__ == "__main__":
    asyncio.run(main())