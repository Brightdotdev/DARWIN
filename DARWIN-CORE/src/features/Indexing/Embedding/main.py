"""
src/core/Embedding/main.py - Clean embedding pipeline (unified architecture)

Refactored to use the new layered architecture:
  MemoryMonitor → BatchingStrategy → EmbeddingEngine → Embedder
"""

import sys
import argparse
import asyncio
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


from src.features.Indexing.Embedding.MemoryMonitor import MemoryMonitor
from src.features.Indexing.Embedding.BatchingStrategy import BatchingStrategy

from src.features.Indexing.Embedding.Embedder import run_embedding_pipeline
from src.features.Indexing.Embedding.lib import EmbeddingArgs
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Models.ModelManager import modelManager
from src.core.Models.Lib.ModelRegistry import ModelCategory
from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity
from src.lib.Utils import ensure_writable_directory


# ─────────────────────────────────────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────────────────────────────────────

def handle_embedding(args: EmbeddingArgs, store: DarwinStore) -> None:
    
    # ── Hardware summary (new) ──────────────────────────────────────
    monitor = MemoryMonitor()
    monitor.log_summary()

    darwin.log("")
    darwin.log("Embedding pipeline", LogType.INFO)
    darwin.log(f"Rerun={args.rerun} | Inspect={args.inspect}", LogType.INFO)
    darwin.log("")

    # ── resolve files using store identity (hash-safe) ──
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

    darwin.log(f"Embedding {len(filenames)} file(s)", LogType.INFO)


    succeeded = skipped = failed = 0
    total = len(filenames)

    for i, filename in enumerate(filenames):

        # ── skip logic ──
        if not args.rerun and store.has_embeddings(filename):
            darwin.log(f"Skipping {filename} (fresh embeddings exist)", LogType.INFO)
            skipped += 1
            continue

        chunks = store.query_records(filename=filename, columns=["clean", "index"])

        if not chunks:
            darwin.log(f"{filename} — no chunks found", LogType.ERROR)
            failed += 1
            continue

        is_last = (i == total - 1)

        try:
            
            vectors = run_embedding_pipeline(
                filename=filename,
                chunks=chunks,
                store=store,
                rerun=args.rerun,
                unload_after=is_last,
            )

            if vectors is not None and len(vectors) > 0:
                succeeded += 1

                if args.inspect:
                    darwin.log(f"{filename} — shape={vectors.shape}, dtype={vectors.dtype}", LogType.INFO)
            else:
                
                skipped += 1

        except Exception as e:
            darwin.log(f"{filename} — {type(e).__name__}: {e}", LogType.ERROR)
            failed += 1

    # ── ensure model cleanup ──
    try:
        modelManager.unload(ModelCategory.EMBEDDING)
        darwin.log("Embedding model unloaded", LogType.INFO)
    except Exception as e:
        darwin.log(f"Unload warning: {e}", LogType.WARNING)

    # ── Batching history (new — for observability) ──────────────────
    strategy = BatchingStrategy()
    history = strategy.get_history()
    if history:
        darwin.log(f"Batching history: {history}", LogType.INFO)

    darwin.log("")
    darwin.log(
        f"Done — {succeeded} embedded · {skipped} skipped · {failed} failed",
        LogType.SUCCESS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(description="Embed chunked documents")

    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--file", "-f", nargs="+", default=None)

    parser.add_argument("--inspect", "-i", action="store_true")
    parser.add_argument("--rerun", "-R", action="store_true")

    # ── New: batching overrides (optional) ──────────────────────────
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

    parser.add_argument("--store-dir", default=None, help="Darwin store directory")
    parser.add_argument("--verbosity", choices=["normal", "quiet"], default="normal", help="Log verbosity")
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--no-ws", action="store_true")

    args = parser.parse_args()

    darwin.configure(
        enable_websocket=not args.no_ws,
        ws_port=args.ws_port,
        enable_history=True,
        verbosity_default=Verbosity.QUIET if args.verbosity == "quiet" else Verbosity.NORMAL,
        show_banner=True,
    )

    if args.store_dir and ensure_writable_directory(args.store_dir):
        store = DarwinStore(args.store_dir)
    else:
        store = DarwinStore()

    try:
        embedding_args = EmbeddingArgs(
            files=args.file,
            inspect=args.inspect,
            rerun=args.rerun,
        )

        with darwin.session("embedding"):
            handle_embedding(embedding_args, store)
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