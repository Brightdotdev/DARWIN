"""
src/core/Embedding/main.py - Embedding pipeline entry point

Reads chunks from .darwin, embeds them with the active model, saves .npz bundle.
Model is loaded once for the entire batch, unloaded when complete.

Usage examples
--------------
# embed everything chunked in current directory
uv run src/core/Embedding/main.py

# embed a specific file
uv run src/core/Embedding/main.py --file biology.pdf

# inspect output — print vector stats and sample chunk text
uv run src/core/Embedding/main.py --inspect

# force re-embed even if fresh embeddings exist
uv run src/core/Embedding/main.py --rerun
"""

import sys
import argparse
import asyncio
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Embedding.Embedder import run_embedding_pipeline
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Embedding.EmbedderUtils import EmbeddingArgs
from src.core.Models import get_loader, ModelCategory, setup
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket
from src.core.Logger.DarwinLogger import darwin, LogType


# ═══════════════════════════════════════════════════════════════════════════
# Inspect
# ═══════════════════════════════════════════════════════════════════════════

def _print_inspection(filename: str, store: DarwinStore) -> None:
    """Load the saved .npz and print stats."""
    try:
        data = store.load_embeddings(filename)
    except FileNotFoundError:
        darwin.log("Could not load embeddings for inspection", LogType.ERROR)
        return

    vectors  = data["vectors"]
    texts    = data["text"]
    headings = data["headings"]
    pages    = data["pages"]

    darwin.log("")
    darwin.log(f"── embedding inspection — {filename} ──", LogType.INFO)
    darwin.log("")

    norms    = np.linalg.norm(vectors, axis=1)
    dot_self = float(np.dot(vectors[0], vectors[0])) if len(vectors) else 0

    # Summary table
    darwin.log("  metric                          value", LogType.INFO)
    darwin.log("  ──────────────────────────────  ─────", LogType.INFO)
    darwin.log(f"  vectors shape                   {vectors.shape}", LogType.INFO)
    darwin.log(f"  dtype                           {vectors.dtype}", LogType.INFO)
    if vectors.ndim > 1:
        darwin.log(f"  embedding dim                   {vectors.shape[1]}", LogType.INFO)
    darwin.log(f"  L2 norms (should be ~1.0)       min={norms.min():.4f} max={norms.max():.4f}", LogType.INFO)
    darwin.log(f"  v[0] · v[0] (should be ~1.0)    {dot_self:.4f}", LogType.INFO)
    darwin.log(f"  unique pages                    {len(set(pages))}", LogType.INFO)
    heading_count = sum(1 for h in headings if h)
    darwin.log(f"  chunks with heading             {heading_count} / {len(headings)}", LogType.INFO)
    darwin.log("")

    darwin.log("── sample chunks from .npz ──", LogType.INFO)
    darwin.log("")

    indices = [0, len(texts) // 2, len(texts) - 1] if len(texts) >= 3 else list(range(len(texts)))
    for idx in indices:
        darwin.log(f"  index:     {idx}", LogType.INFO)
        darwin.log(f"  heading:   {headings[idx] or '(none)'}", LogType.INFO)
        darwin.log(f"  page:      {pages[idx]}", LogType.INFO)
        darwin.log("")
        preview = texts[idx][:300].replace("\n", " ").strip()
        if len(texts[idx]) > 300:
            preview += "..."
        darwin.log(f"  text:   {preview}", LogType.INFO)
        darwin.log("")
        darwin.log(f"  vector: [{vectors[idx][0]:.4f}, {vectors[idx][1]:.4f}, ... {vectors[idx][-1]:.4f}]", LogType.INFO)
        darwin.log("")
        darwin.log("─" * 40, LogType.INFO)
        darwin.log("")

    darwin.log("─" * 60, LogType.INFO)
    darwin.log("")


# ═══════════════════════════════════════════════════════════════════════════
# Handler
# ═══════════════════════════════════════════════════════════════════════════

def handle_embedding(args: EmbeddingArgs) -> None:
    """Main embedding handler — loads chunks, embeds them, saves vectors."""
    store = DarwinStore()
    
    # Bug fix: setup() may fail silently if models dir doesn't exist
    try:
        loader = get_loader()
    except RuntimeError:
        darwin.log("Initializing models...", LogType.INFO)
        try:
            loader = setup()
        except Exception as e:
            darwin.log(f"Failed to initialize models: {e}", LogType.ERROR)
            return

    darwin.log("")
    darwin.log("Embedding", LogType.INFO)
    darwin.log(f"Rerun: {args.rerun} | Inspect: {args.inspect}", LogType.INFO)
    darwin.log("")

    # Resolve which files to embed
    if args.file:
        filenames = [args.file]
    else:
        try:
            status = store.status()
            filenames = [
                fname for fname, info in status.get("files", {}).items()
                if info.get("components", {}).get("chunks", False)
            ]
        except Exception as e:
            darwin.log(f"Failed to read store status: {e}", LogType.ERROR)
            return

    if not filenames:
        darwin.log("No chunked files found. Run chunking first.", LogType.ERROR)
        return

    darwin.log(f"Embedding {len(filenames)} file(s)", LogType.INFO)

    succeeded = failed = skipped = 0
    total = len(filenames)

    for i, filename in enumerate(filenames):
        if not args.rerun and store.has_embeddings(filename):
            darwin.log(f"skipping {filename} (fresh embeddings exist)", LogType.INFO)
            skipped += 1
            continue

        chunks = store.load_chunks(filename)
        if not chunks:
            darwin.log(f"{filename} — no chunks found, skipping", LogType.ERROR)
            failed += 1
            continue

        # Only unload after the LAST file
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
                    _print_inspection(filename, store)
            else:
                skipped += 1

        except Exception as e:
            darwin.log(f"{filename} — {type(e).__name__}: {e}", LogType.ERROR)
            failed += 1

    # Bug fix: unload safety — e was undefined in original except block
    try:
        loader.unload(ModelCategory.EMBEDDING)
        darwin.log("Embedding model unloaded from memory", LogType.INFO)
    except Exception as unload_err:
        darwin.log(f"Failed to unload embedding model: {unload_err}", LogType.WARNING)

    darwin.log("")
    darwin.log(f"done — {succeeded} embedded · {skipped} skipped · {failed} failed", LogType.SUCCESS)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Embed chunked documents into vector index"
    )
    parser.add_argument("path",           nargs="?", default=".",  help="Project directory")
    parser.add_argument("--file",    "-f", default=None,           help="Embed a single file")
    parser.add_argument("--project", "-p", default=None,           help="Project directory (default: cwd)")
    parser.add_argument("--inspect", "-i", action="store_true",    help="Print vector stats and sample chunks")
    parser.add_argument("--rerun",   "-R", action="store_true",    help="Re-embed even if fresh embeddings exist")
    parser.add_argument("--ws-port", type=int, default=8765,       help="WebSocket server port (default: 8765)")
    parser.add_argument("--no-ws",   action="store_true",          help="Disable WebSocket server")

    args = parser.parse_args()

    # Validate target path
    target_path = Path(args.path).resolve()
    if not target_path.exists():
        darwin.log(f"Path does not exist: {args.path}", LogType.ERROR)
        return 1

    # Setup WebSocket
    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        embedding_args = EmbeddingArgs(
            target=args.path,
            file=args.file,
            project=args.project or args.path,
            inspect=args.inspect,
            rerun=args.rerun,
        )
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, handle_embedding, embedding_args)
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