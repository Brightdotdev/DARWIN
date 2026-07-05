import sys
import multiprocessing
import asyncio
from pathlib import Path

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.Pipeline.Indexing.lib import IndexingArgs
from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.Utils import ensure_writable_directory
from src.core.Storage.DarwinStore import DarwinStore

from src.core.Pipeline.Indexing.Embedding.lib import EmbeddingArgs
from src.core.Pipeline.Indexing.BM25.lib import (
    BM25IndexingArgs,
    DOMAIN_STOP_THRESHOLD,
    DEFAULT_IDF_SMOOTHING,
)

def _resolve_input_files(
    raw: list[str | Path] | None,
    store: "DarwinStore",
) -> list[str] | None:
    """
    Convert user input (paths, Path objects, or bare filenames) into a list of
    registered filenames that actually have chunks (records). Returns:
        - None if raw is None (meaning "all files")
        - empty list if raw is not None but none of the inputs correspond to a valid file
        - list of canonical filenames (basenames) that have chunks.
    """
    if raw is None:
        return None

    resolved: list[str] = []
    for r in raw:
        # Convert Path to string if needed
        r_str = str(r)
        candidate = Path(r_str).name
        if store.has_records(candidate):
            resolved.append(candidate)
        else:
            from src.core.Logger.DarwinLogger import darwin, LogType
            darwin.log(f"'{r_str}' → '{candidate}' has no chunks — skipping", LogType.WARNING)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = []
    for f in resolved:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique

# ── main handler ──────────────────────

def handle_indexing(args: IndexingArgs, store: DarwinStore) -> None:
    from src.core.Pipeline.Indexing.Embedding.main import handle_embedding
    from src.core.Pipeline.Indexing.BM25.main import handle_bm25_indexing

    # Resolve user‑provided input (paths or basenames) to registered filenames with chunks
    files = _resolve_input_files(args.files, store)
    if args.files is not None and files == []:
        darwin.log("No valid input files have chunks. Aborting indexing.", LogType.ERROR)
        return

    run_kw = not args.only_semantic
    run_semantic = not args.only_kw

    steps = ([" KW"] if run_kw else []) + (["Semantic"] if run_semantic else [])
    step_count = len(steps)

    def step_label(name: str) -> str:
        idx = steps.index(name) + 1
        return f"[{idx}/{step_count}]"

    darwin.log("Indexing", LogType.INFO)
    darwin.log(
        f"Threshold: {args.threshold} | IDF smoothing: {args.idf_smoothing} | "
        f"Rerun: {args.rerun}",
        LogType.INFO,
    )
    if files:
        darwin.log(f"Files: {files}", LogType.INFO)
    else:
        darwin.log("Indexing all files with chunks", LogType.INFO)
    darwin.log("")

    if run_kw:
        darwin.log(f"{step_label(' KW')} Keyword (BM25) indexing", LogType.INFO)
        try:
            handle_bm25_indexing(BM25IndexingArgs(
                files=files,          # None means all stored files
                threshold=args.threshold,
                idf_smoothing=args.idf_smoothing,
                rerun=args.rerun,
            ), store=store)
        except Exception as e:
            import traceback
            darwin.log(f"BM25 indexing failed — {type(e).__name__}: {e}", LogType.ERROR)
            darwin.log(traceback.format_exc())
        darwin.log("")

    if run_semantic:
        darwin.log(f"{step_label('Semantic')} Semantic embedding", LogType.INFO)
        try:
            handle_embedding(EmbeddingArgs(
                files=files,
                rerun=args.rerun,
            ), store=store)
        except Exception as e:
            import traceback
            darwin.log(f"Embedding failed — {type(e).__name__}: {e}", LogType.ERROR)
            darwin.log(traceback.format_exc(), LogType.ERROR)
        darwin.log("")

    darwin.log("Indexing complete", LogType.SUCCESS)


# ── CLI entry point ────────────────────────────────────────────────────────────

async def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run indexing (BM25 + embeddings)")
    parser.add_argument(
        "path", nargs="?", default=".",
        help="Project directory (unused – only for CLI compatibility)",
    )
    parser.add_argument(
        "--file", "-f", nargs="+", default=None, dest="files",
        metavar="FILE",
        help="One or more files to index (full path or bare filename). Omit to index all.",
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

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--only-kw", action="store_true", help="Run keyword (BM25) indexing only")
    mode.add_argument("--only-semantic", action="store_true", help="Run semantic embedding only")

    
    parser.add_argument("--sse-port", type=int, default=8765, help="WebSocket server port")
    
    parser.add_argument("--store-dir", default=None, help="Darwin store directory")

    args = parser.parse_args()

    darwin.configure(
        
        sse_port=args.sse_port,
        
        
        show_banner=True,
    )

    try:
        indexing_args = IndexingArgs(
            path=args.path,
            rerun=args.rerun,
            files=args.files,
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