import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Pipeline.Indexing.Embedding.MemoryMonitor import MemoryMonitor
from src.core.Pipeline.Indexing.Embedding.BatchingStrategy import BatchingStrategy
from src.core.Pipeline.Indexing.Embedding.Embedder import EmbeddingSession
from src.core.Pipeline.Indexing.Embedding.lib import EmbeddingArgs
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Models.ModelConfig import get_active_model
from src.core.Models.Lib.ModelRegistry import ModelCategory
from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.Utils import ensure_writable_directory


# ─────────────────────────────────────────────────────────────────────────────
# Handler — SYNC to match caller (handle_indexing is sync)
# ─────────────────────────────────────────────────────────────────────────────

def handle_embedding(args: EmbeddingArgs, store: DarwinStore) -> None:
    # ── Hardware summary ──────────────────────────────────────────────
    monitor = MemoryMonitor()
    monitor.log_summary()

    darwin.log("")
    darwin.log("Embedding pipeline", LogType.INFO)
    darwin.log(f"Rerun={args.rerun} | "
                f"Batch={args.batch_size or 'auto'} | "
                f"LengthBatching={args.use_length_batching}", LogType.INFO)
    darwin.log("")

    # ── Resolve files ──
    if args.files:
        filenames = [f for f in args.files if store.has_records(f)]
    else:
        filenames = [
            fname for fname, info in store.status().get("files", {}).items()
            if info.get("is_chunked", False)
        ]

    if not filenames:
        darwin.log("No chunked files found. Run chunking first.", LogType.ERROR)
        return

    model_key = get_active_model(ModelCategory.EMBEDDING.value.lower())
    if not model_key:
        darwin.log("No embedding model configured. Run setup first.", LogType.ERROR)
        return

    darwin.log(f"Embedding {len(filenames)} file(s)", LogType.INFO)

    # ── FIXED: model loaded ONCE via session, not per-file (Audit #4) ──
    succeeded = skipped = failed = 0

    with EmbeddingSession(model_key=model_key) as session:
        if not session.embedder:
            darwin.log("Failed to acquire embedding model", LogType.ERROR)
            return

        for i, filename in enumerate(filenames):
            # Skip logic
            if not args.rerun and store.has_embeddings(filename):
                darwin.log(f"Skipping {filename} (fresh embeddings exist)", LogType.WARNING)
                skipped += 1
                continue

            chunks = store.query_records(filename=filename, columns=["clean", "index"])

            if not chunks:
                darwin.log(f"{filename} — no chunks found", LogType.ERROR)
                failed += 1
                continue

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
                    succeeded += 1
                    
                else:
                    skipped += 1

            except Exception as e:
                darwin.log(f"{filename} — {type(e).__name__}: {e}", LogType.ERROR)
                failed += 1

    # ── Batching history from shared singleton (Audit #3) ─────────────
    strategy = BatchingStrategy()  # returns same singleton instance
    history = strategy.get_history()
    if history:
        darwin.log(f"Batching history: {history}", LogType.INFO)

    profiles = []
    for mk in history:
        prof = strategy.get_profile(mk)
        if prof:
            profiles.append(f"{mk}: {prof}")
    if profiles:
        darwin.log(f"Throughput profiles: {' | '.join(profiles)}", LogType.INFO)

    darwin.log("")
    darwin.log(
        f"Done — {succeeded} embedded · {skipped} skipped · {failed} failed",
        LogType.SUCCESS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI — kept async for standalone usage, delegates to sync handle_embedding
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> int:
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
    
    parser.add_argument("--sse-port", type=int, default=8765)

    args = parser.parse_args()

    darwin.configure(
        
        sse_port=args.sse_port,
        
        
        show_banner=True,
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
    import asyncio
    asyncio.run(main())