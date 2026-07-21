import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Pipeline.Indexing.Embedding.MemoryMonitor import MemoryMonitor
from src.core.Pipeline.Indexing.Embedding.BatchingStrategy import BatchingStrategy
from src.core.Pipeline.Indexing.Embedding.Embedder import EmbeddingSession
from src.core.Pipeline.Indexing.Embedding.lib import EmbeddingArgs
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Models.ModelConfig import get_active_model
from src.core.Models.Lib.ModelRegistry import ModelCategory
from src.core.Models.EmbeddingConfig import get_preset
from src.core.Logger.DarwinLogger import darwin
from src.lib.Utils import ensure_writable_directory


# ─────────────────────────────────────────────────────────────────────────────
# Handler — SYNC to match caller (handle_indexing is sync)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_one_file(
    session: EmbeddingSession,
    filename: str,
    store: DarwinStore,
    args: EmbeddingArgs,
    _lock: threading.Lock,
) -> tuple[str, str]:
    """Embed a single file. Returns (filename, 'ok'|'skip'|'fail')."""
    if not args.rerun and store.has_embeddings(filename):
        with _lock:
            darwin.warning(f"Skipping {filename} — embeddings already exist")
        return filename, "skip"

    chunks = store.query_records(filename=filename, columns=["clean", "index"])
    if not chunks:
        with _lock:
            darwin.error(f"{filename} has no chunks")
        return filename, "fail"

    try:
        vectors = session.embed_file(
            filename=filename,
            chunks=chunks,
            store=store,
            rerun=args.rerun,
            batch_size=args.batch_size,
            use_length_batching=args.use_length_batching,
        )
        if vectors is not None and len(vectors) > 0:
            return filename, "ok"
        return filename, "skip"
    except Exception as e:
        with _lock:
            darwin.error(f"{filename} — {type(e).__name__}: {e}")
        return filename, "fail"


def handle_embedding(args: EmbeddingArgs, store: DarwinStore) -> None:
    monitor = MemoryMonitor()
    monitor.log_summary()

    darwin.info("")
    darwin.info("Embedding pipeline started")
    darwin.info(f"Rerun={args.rerun} | "
                f"Batch={args.batch_size or 'auto'} | "
                f"LengthBatching={args.use_length_batching}")
    darwin.info("")

    if args.files:
        filenames = [f for f in args.files if store.has_records(f)]
    else:
        filenames = [
            fname for fname, info in store.status().get("files", {}).items()
            if info.get("is_chunked", False)
        ]

    if not filenames:
        darwin.error("No chunked files found — run chunking first")
        return

    model_key = get_active_model(ModelCategory.EMBEDDING.value.lower())
    if not model_key:
        darwin.error("No embedding model configured — run setup first")
        return

    preset = get_preset(model_key)
    is_api = preset.api_batch_limit is not None if preset else False

    darwin.info(f"Embedding {len(filenames)} file(s) [backend={'api' if is_api else 'local'}]")

    succeeded = skipped = failed = 0
    lock = threading.Lock()

    with EmbeddingSession(model_key=model_key) as session:
        if not session.embedder:
            darwin.error("Failed to acquire embedding model")
            return

        if is_api:
            for filename in filenames:
                _, status = _embed_one_file(session, filename, store, args, lock)
                if status == "ok":
                    succeeded += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1
        else:
            workers = min(2, len(filenames))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_embed_one_file, session, fn, store, args, lock): fn
                    for fn in filenames
                }
                for future in as_completed(futures):
                    _, status = future.result()
                    if status == "ok":
                        succeeded += 1
                    elif status == "skip":
                        skipped += 1
                    else:
                        failed += 1

    strategy = BatchingStrategy()
    history = strategy.get_history()
    if history:
        darwin.info(f"Batching history: {history}")

    profiles = []
    for mk in history:
        prof = strategy.get_profile(mk)
        if prof:
            profiles.append(f"{mk}: {prof}")
    if profiles:
        darwin.info(f"Throughput profiles: {' | '.join(profiles)}")

    darwin.info("")
    darwin.success(
        f"Done — {succeeded} embedded / {skipped} skipped / {failed} failed"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI — kept async for standalone usage, delegates to sync handle_embedding
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Embed chunked documents")

    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--file", "-f", nargs="+", default=None)

    parser.add_argument("--rerun", "-R", action="store_true")

    # ── FIXED: batching overrides now wired through (Audit #1) ──────
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (default: hardware-aware auto)",
    )
    parser.add_argument(
        "--min-batch-cpu",
        type=int,
        default=None,
        help="Minimum batch size on CPU (default: 16)",
    )
    parser.add_argument(
        "--min-batch-gpu",
        type=int,
        default=None,
        help="Minimum batch size on GPU (default: 8)",
    )

    # NEW: length-aware batching controls
    parser.add_argument(
        "--length-batching",
        action="store_true",
        default=True,
        help="Group batches by token budget instead of item count (default: True)",
    )
    parser.add_argument(
        "--no-length-batching",
        dest="length_batching",
        action="store_false",
        help="Disable length-aware batching",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=2048,
        help="Max tokens per batch when length-batching (default: 2048)",
    )

    parser.add_argument("--store-dir", default=None, help="Darwin store directory")

    args = parser.parse_args()

    darwin.configure(
        
    )

    if args.store_dir and ensure_writable_directory(args.store_dir):
        store = DarwinStore(args.store_dir)
    else:
        store = DarwinStore()

    try:
        # FIXED: all CLI args now passed through (Audit #1)
        embedding_args = EmbeddingArgs(
            files=args.file,
            rerun=args.rerun,
            batch_size=args.batch_size,
            min_batch_cpu=args.min_batch_cpu,
            min_batch_gpu=args.min_batch_gpu,
            use_length_batching=args.length_batching,
            token_budget=args.token_budget,
        )

        with darwin.session("embedding"):
            handle_embedding(embedding_args, store)  # sync call
        return 0

    except KeyboardInterrupt:
        darwin.info("")
        darwin.info("Goodbye!")
        return 130

    except Exception as e:
        import traceback
        darwin.error(f"Fatal error: {e}")
        darwin.error("".join(traceback.format_exception(e)))
        return 1

    finally:
        darwin.shutdown()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())