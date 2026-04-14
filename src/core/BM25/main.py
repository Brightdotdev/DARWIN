"""
src/core/BM25/main.py - BM25 indexing pipeline entry point

Reads chunks from .darwin, builds BM25 sparse index, saves .bm25 bundle.
Run after chunking, before or after embedding.

Usage examples
--------------
# index everything chunked in current directory
uv run src/core/BM25/main.py

# index a specific file
uv run src/core/BM25/main.py --file biology.pdf

# adjust domain stopword threshold (lower = more aggressive)
uv run src/core/BM25/main.py --threshold 0.25

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

from src.core.BM25.BM25Indexer import run_bm25_indexing_pipeline, index_chunks_for_bm25
from src.core.BM25.BM25Utils import BM25IndexingArgs, DOMAIN_STOP_THRESHOLD
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


# ============================================================================
# Inspect — show what's in the .bm25 after saving
# ============================================================================

def _print_inspection(filename: str, store: DarwinStore) -> None:
    """Load the saved .bm25 and print stats."""
    try:
        data = store.load_bm25(filename)
    except FileNotFoundError:
        TerminalUI.error("Could not load BM25 index for inspection")
        return
    
    TerminalUI.blank()
    TerminalUI.rule(f"BM25 index inspection — {filename}")
    TerminalUI.blank()
    
    TerminalUI.table(
        headers=["metric", "value"],
        rows=[
            ["chunks indexed", str(data.chunk_count)],
            ["vocabulary size", str(data.vocab_size)],
            ["avg chunk length", f"{data.avgdl:.1f} tokens"],
            ["domain stopwords", str(len(data.domain_stopwords))],
            ["threshold used", f"{data.threshold:.2f}"],
        ],
    )
    
    TerminalUI.blank()
    TerminalUI.dim("── domain stopword samples ──")
    TerminalUI.blank()
    
    if data.domain_stopwords:
        # Show top 15 domain stopwords
        sample = data.domain_stopwords[:15]
        TerminalUI.panel(", ".join(sample), title="domain stops", border_style="dim")
    else:
        TerminalUI.dim("No domain stopwords detected")
    
    TerminalUI.blank()
    TerminalUI.dim("── token samples (first 3 chunks) ──")
    TerminalUI.blank()
    
    for i in range(min(3, len(data.tokens))):
        tokens = data.tokens[i]
        preview = " ".join(tokens[:20])
        if len(tokens) > 20:
            preview += "..."
        TerminalUI.info(f"chunk {i}", f"{len(tokens)} tokens")
        TerminalUI.dim(f"  {preview}")
        TerminalUI.blank()
    
    TerminalUI.rule()


# ============================================================================
# Core handler
# ============================================================================

def handle_bm25_indexing(args: BM25IndexingArgs) -> None:
    store = DarwinStore(Path(args.project) if args.project else Path.cwd())
    

    TerminalUI.print_darwin()
    TerminalUI.blank()
    TerminalUI.bold("BM25 indexing")
    TerminalUI.dim(f"Threshold: {args.threshold} | Rerun: {args.rerun} | Inspect: {args.inspect}")
    TerminalUI.blank()
    # Resolve which files to index
    if args.file:
        filenames = [args.file]
    else:
        status = store.status()
        filenames = [
            fname for fname, info in status.get("files", {}).items()
            if info["components"]["chunks"]
        ]
    
    if not filenames:
        TerminalUI.error("No chunked files found. Run chunking first.")
        return
    
    TerminalUI.bold(f"BM25 indexing {len(filenames)} file(s) — threshold={args.threshold}")
    
    succeeded = failed = skipped = 0
    
    for filename in filenames:
        if not args.rerun and store.is_fresh(filename) and store.has_bm25(filename):
            TerminalUI.dim(f"skipping {filename} (fresh BM25 index exists)")
            skipped += 1
            continue
        
        chunks = store.load_chunks(filename)
        if not chunks:
            TerminalUI.error(f"{filename} — no chunks found, skipping")
            failed += 1
            continue
        
        try:
            data = run_bm25_indexing_pipeline(
                filename=filename,
                chunks=chunks,
                store=store,
                threshold=args.threshold,
                rerun=args.rerun,
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


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build BM25 sparse index from chunked documents"
    )
    parser.add_argument("path", nargs="?", default=".", help="Project directory")
    parser.add_argument("--file", "-f", default=None, help="Index a single file")
    parser.add_argument("--project", "-p", default=None, help="Project directory (default: cwd)")
    parser.add_argument("--threshold", "-t", type=float, default=DOMAIN_STOP_THRESHOLD,
                       help=f"Domain stopword threshold (default: {DOMAIN_STOP_THRESHOLD})")
    parser.add_argument("--inspect", "-i", action="store_true", help="Print index stats and samples")
    parser.add_argument("--rerun", "-R", action="store_true", help="Re-index even if fresh index exists")
    
    args = parser.parse_args()
    
    handle_bm25_indexing(BM25IndexingArgs(
        target=args.path,
        file=args.file,
        project=args.project or args.path,
        threshold=args.threshold,
        inspect=args.inspect,
        rerun=args.rerun,
    ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())