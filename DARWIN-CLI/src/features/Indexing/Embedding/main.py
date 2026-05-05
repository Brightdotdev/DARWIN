"""
src/core/Embedding/main.py - Clean embedding pipeline (refactored, no architecture changes)
"""

import sys
import argparse
import asyncio
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.features.Indexing.Embedding.Embedder import run_embedding_pipeline
from src.features.Indexing.Embedding.lib import EmbeddingArgs
from src.features.Storage.DarwinStore import DarwinStore
from src.Models import ModelManager, ModelCategory
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket
from src.core.Logger.DarwinLogger import darwin, LogType


# ─────────────────────────────────────────────────────────────────────────────
# Inspection
# ─────────────────────────────────────────────────────────────────────────────

def _print_inspection(filename: str, store: DarwinStore, vectors, chunks) -> None:
    """
    Inspect embedding output for a file.
    """

    if not store.has_embeddings(filename):
        darwin.log("Could not load embeddings for inspection", LogType.ERROR)
        return

    if not chunks or vectors is None or len(vectors) == 0:
        darwin.log("Missing chunks or vectors for inspection", LogType.ERROR)
        return

    texts = [c.text for c in chunks]
    headings = [c.heading for c in chunks]
    pages = [c.page for c in chunks]

    darwin.log("")
    darwin.log(f"── embedding inspection — {filename} ──", LogType.INFO)
    darwin.log("")

    norms = np.linalg.norm(vectors, axis=1) if len(vectors) else []

    dot_self = float(np.dot(vectors[0], vectors[0])) if len(vectors) else 0.0

    darwin.log("  metric                          value", LogType.INFO)
    darwin.log("  ──────────────────────────────  ─────", LogType.INFO)
    darwin.log(f"  vectors shape                   {vectors.shape}", LogType.INFO)
    darwin.log(f"  dtype                           {vectors.dtype}", LogType.INFO)

    if vectors.ndim > 1:
        darwin.log(f"  embedding dim                   {vectors.shape[1]}", LogType.INFO)

    darwin.log(
        f"  L2 norms                        min={np.min(norms):.4f} max={np.max(norms):.4f}",
        LogType.INFO,
    )

    darwin.log(f"  v[0] · v[0]                    {dot_self:.4f}", LogType.INFO)

    darwin.log(f"  unique pages                   {len(set(pages))}", LogType.INFO)

    heading_count = sum(1 for h in headings if h)
    darwin.log(f"  chunks with heading            {heading_count}/{len(headings)}", LogType.INFO)

    darwin.log("")
    darwin.log("── sample chunks ──", LogType.INFO)
    darwin.log("")

    indices = [0, len(texts)//2, len(texts)-1] if len(texts) >= 3 else range(len(texts))

    for idx in indices:
        darwin.log(f"  index   {idx}", LogType.INFO)
        darwin.log(f"  heading {headings[idx] or '(none)'}", LogType.INFO)
        darwin.log(f"  page    {pages[idx]}", LogType.INFO)

        preview = texts[idx][:300].replace("\n", " ").strip()
        if len(texts[idx]) > 300:
            preview += "..."

        darwin.log(f"  text    {preview}", LogType.INFO)

        vec_preview = ", ".join(f"{v:.4f}" for v in vectors[idx][:3])
        darwin.log(f"  vector  [{vec_preview}, ...]", LogType.INFO)

        darwin.log("─" * 40, LogType.INFO)

    darwin.log("─" * 60, LogType.INFO)
    darwin.log("")


# ─────────────────────────────────────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────────────────────────────────────

def handle_embedding(args: EmbeddingArgs) -> None:
    store = DarwinStore()
    loader = ModelManager.get_loader()

    darwin.log("")
    darwin.log("Embedding pipeline", LogType.INFO)
    darwin.log(f"Rerun={args.rerun} | Inspect={args.inspect}", LogType.INFO)
    darwin.log("")

    # ── resolve files using store identity (hash-safe) ──
    if args.files:
        filenames = [f for f in args.files if store.has_chunks(f)]
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

        chunks = store.load_chunks(filename)
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

            if vectors is not None:
                succeeded += 1

                if args.inspect:
                    _print_inspection(filename, store, vectors, chunks)
            else:
                skipped += 1

        except Exception as e:
            darwin.log(f"{filename} — {type(e).__name__}: {e}", LogType.ERROR)
            failed += 1

    # ── ensure model cleanup ──
    try:
        loader.unload(ModelCategory.EMBEDDING)
        darwin.log("Embedding model unloaded", LogType.INFO)
    except Exception as e:
        darwin.log(f"Unload warning: {e}", LogType.WARNING)

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

    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--no-ws", action="store_true")

    args = parser.parse_args()

    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        embedding_args = EmbeddingArgs(
            files=args.file,
            inspect=args.inspect,
            rerun=args.rerun,
        )

        handle_embedding(embedding_args)
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
        shutdown_websocket(ws_server)


if __name__ == "__main__":
    asyncio.run(main())