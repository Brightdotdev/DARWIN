

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


from src.lib.Utils import ensure_writable_directory

from src.core.Pipeline.Indexing.BM25.BM25Indexer import (
    run_bm25_indexing_pipeline,
)
from src.core.Pipeline.Indexing.BM25.lib import (
    BM25IndexingArgs,
    DOMAIN_STOP_THRESHOLD,
    DEFAULT_IDF_SMOOTHING,
    
)

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin




# ── Core handler ───────────────────────────────────────────────────────────────

def handle_bm25_indexing(args: BM25IndexingArgs, store:DarwinStore) -> None:

    darwin.info("")
    darwin.info("BM25 keyword indexing started")
    darwin.info(
        f"Threshold: {args.threshold} | IDF smoothing: {args.idf_smoothing} | "
        f"Rerun: {args.rerun}"
    )
    darwin.info("")

    status = store.status()
    filenames = []

    if args.files:
        filenames = [f for f in args.files if store.has_records(f)]
    else:    
        filenames = [
            fname for fname, info in status.get("files", {}).items()
             if info.get("is_chunked", False)
        ]
    
    if not filenames:
        darwin.error("No chunked files found — run chunking first")
        return

    darwin.info(f"Indexing {len(filenames)} file(s)")
    

    succeeded = failed = skipped = 0

    for filename in filenames:
        if not args.rerun and store.has_bm25(filename):
            darwin.info(f"Skipping {filename} — fresh index already exists")
            skipped += 1
            continue

        try:
            chunks_rows = store.query_records(filename=filename, columns=["clean"])
            
        except FileNotFoundError:
            darwin.error(f"{filename} has no chunks, skipping")
            failed += 1
            continue

        try:
            data = run_bm25_indexing_pipeline(
                filename=filename,
                chunks=chunks_rows,
                store=store,
                threshold=args.threshold,
                idf_smoothing=args.idf_smoothing,
            )

            if data is not None:
                succeeded += 1
            else:
                skipped += 1

        except Exception as e:
            darwin.error(f"{filename} — {type(e).__name__}: {e}")
            failed += 1

    darwin.info("")
    darwin.success(f"Done — {succeeded} indexed / {skipped} skipped / {failed} failed")


# ── CLI ────────────────────────────────────────────────────────────────────────

async def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build BM25 sparse index from chunked documents"
    )
    
    parser.add_argument("--file", "-f", nargs="+", default=None,
                        help="Files to process (paths or bare filenames). Omit for all.")

    parser.add_argument("--threshold","-t",  type=float, default=DOMAIN_STOP_THRESHOLD,
                        help=f"Domain stopword threshold (default: {DOMAIN_STOP_THRESHOLD})")
    parser.add_argument("--idf-smoothing", "-s", type=float, default=DEFAULT_IDF_SMOOTHING,
                        help=f"IDF smoothing parameter (default: {DEFAULT_IDF_SMOOTHING})")
    parser.add_argument("--rerun",    "-R",  action="store_true",                help="Re-index even if fresh index exists")
    parser.add_argument("--store-dir", default=None, help="Darwin store directory")

    args = parser.parse_args()




    darwin.configure(
        
    )

    
    if args.store_dir and ensure_writable_directory(args.store_dir):
        store = DarwinStore(args.store_dir)
    else:
        store = DarwinStore()

    try:
        bm25_args = BM25IndexingArgs(
            files=args.file,
            threshold=args.threshold,
            idf_smoothing=args.idf_smoothing,
            rerun=args.rerun,
        )
        
        with darwin.session("bm25_indexing"):
            handle_bm25_indexing(bm25_args, store)
        return 0

    except KeyboardInterrupt:
        darwin.info("")
        darwin.info("Goodbye!")
        return 130
    except Exception as e:
        import traceback
        darwin.error(f"Fatal error: {e}")
        darwin.error(''.join(traceback.format_exception(e)))
        return 1
    finally:
        darwin.shutdown()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())