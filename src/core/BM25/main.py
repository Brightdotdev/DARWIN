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
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.BM25.BM25Indexer import (
    run_bm25_indexing_pipeline,
    inspect_bm25_index,
)
from src.core.BM25.BM25Utils import (
    BM25IndexingArgs,
    DOMAIN_STOP_THRESHOLD,
    DEFAULT_IDF_SMOOTHING,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


# ── Inspection helper ──────────────────────────────────────────────────────────

def _print_inspection(filename: str, store) -> None:
    try:
        data = store.load_bm25(filename)
        inspect_bm25_index(data)
    except FileNotFoundError:
        TerminalUI.error("Could not load BM25 index for inspection")


# ── Core handler ───────────────────────────────────────────────────────────────

def handle_bm25_indexing(args: BM25IndexingArgs) -> None:
    store = DarwinStore()

    TerminalUI.print_darwin()
    TerminalUI.blank()
    TerminalUI.bold("BM25 indexing")
    TerminalUI.dim(
        f"Threshold: {args.threshold} | "
        f"IDF smoothing: {args.idf_smoothing} | "
        f"Rerun: {args.rerun} | "
        f"Inspect: {args.inspect}"
    )
    TerminalUI.blank()

    if args.file:
        filenames = [args.file]
    else:
        status    = store.status()
        filenames = [
            fname for fname, info in status.get("files", {}).items()
            if info["components"]["chunks"]
        ]

    if not filenames:
        TerminalUI.error("No chunked files found. Run chunking first.")
        return

    TerminalUI.bold(f"Indexing {len(filenames)} file(s)")

    succeeded = failed = skipped = 0

    for filename in filenames:
        if not args.rerun and store.is_fresh(filename) and store.has_bm25(filename):
            TerminalUI.dim(f"Skipping {filename} (fresh index exists)")
            skipped += 1
            continue

        try:
            chunks = store.load_chunks(filename)
        except FileNotFoundError:
            TerminalUI.error(f"{filename} — no chunks found, skipping")
            failed += 1
            continue

        try:
            data = run_bm25_indexing_pipeline(
                filename      = filename,
                chunks        = chunks,
                store         = store,
                threshold     = args.threshold,
                idf_smoothing = args.idf_smoothing,
                rerun         = args.rerun,
            )

            if data is not None:
                succeeded += 1
                if args.inspect:
                    _print_inspection(filename, store)
            else:
                skipped += 1

        except Exception as e:
            TerminalUI.error(f"{filename} — {type(e).__name__}: {e}")
            failed += 1

    TerminalUI.blank()
    TerminalUI.success(f"done — {succeeded} indexed · {skipped} skipped · {failed} failed")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
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

    args = parser.parse_args()

    handle_bm25_indexing(BM25IndexingArgs(
        target        = args.path,
        file          = args.file,
        project       = args.project or args.path,
        threshold     = args.threshold,
        idf_smoothing = args.idf_smoothing,
        inspect       = args.inspect,
        rerun         = args.rerun,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())