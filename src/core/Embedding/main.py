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
from pathlib import Path

import numpy as np

from src.lib.AppPaths import default_models_dir

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Embedding.Embedder import run_embedding_pipeline
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI
from src.core.Embedding.EmbedderUtils import EmbeddingArgs
from src.core.Models import get_loader, ModelCategory, setup


# ═══════════════════════════════════════════════════════════════════════════
# Inspect
# ═══════════════════════════════════════════════════════════════════════════

def _print_inspection(filename: str, store: DarwinStore) -> None:
    """Load the saved .npz and print stats."""
    try:
        data = store.load_embeddings(filename)
    except FileNotFoundError:
        TerminalUI.error("Could not load embeddings for inspection")
        return

    vectors  = data["vectors"]
    texts    = data["text"]
    headings = data["headings"]
    pages    = data["pages"]

    TerminalUI.blank()
    TerminalUI.rule(f"embedding inspection — {filename}")
    TerminalUI.blank()

    norms    = np.linalg.norm(vectors, axis=1)
    dot_self = float(np.dot(vectors[0], vectors[0])) if len(vectors) else 0

    TerminalUI.table(
        headers=["metric", "value"],
        rows=[
            ["vectors shape",     str(vectors.shape)],
            ["dtype",             str(vectors.dtype)],
            ["embedding dim",     str(vectors.shape[1]) if vectors.ndim > 1 else "—"],
            ["L2 norms (should be ~1.0)", f"min={norms.min():.4f} max={norms.max():.4f}"],
            ["v[0] · v[0] (should be ~1.0)", f"{dot_self:.4f}"],
            ["unique pages",      str(len(set(pages)))],
            ["chunks with heading", f"{sum(1 for h in headings if h)} / {len(headings)}"],
        ],
    )

    TerminalUI.blank()
    TerminalUI.dim("── sample chunks from .npz ──")
    TerminalUI.blank()

    indices = [0, len(texts) // 2, len(texts) - 1] if len(texts) >= 3 else list(range(len(texts)))
    for idx in indices:
        TerminalUI.info("index",   str(idx))
        TerminalUI.info("heading", headings[idx] or "(none)")
        TerminalUI.info("page",    str(pages[idx]))
        TerminalUI.blank()
        preview = texts[idx][:300].replace("\n", " ").strip()
        if len(texts[idx]) > 300:
            preview += "..."
        TerminalUI.dim(f"  text: {preview}")
        TerminalUI.blank()
        TerminalUI.dim(f"  vector: [{vectors[idx][0]:.4f}, {vectors[idx][1]:.4f}, ... {vectors[idx][-1]:.4f}]")
        TerminalUI.blank()
        TerminalUI.divider()
        TerminalUI.blank()

    TerminalUI.rule()
    TerminalUI.blank()


# ═══════════════════════════════════════════════════════════════════════════
# Handler
# ═══════════════════════════════════════════════════════════════════════════

def handle_embedding(args: EmbeddingArgs) -> None:
    store = DarwinStore()
    
    try:
        loader = get_loader()
    except RuntimeError:
        TerminalUI.dim("Initializing models...")
        loader, system = setup()

    TerminalUI.print_darwin()
    TerminalUI.blank()
    TerminalUI.bold("Embedding")
    TerminalUI.dim(f" Rerun: {args.rerun} | Inspect: {args.inspect}")
    TerminalUI.blank()

    # Resolve which files to embed
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

    TerminalUI.bold(f"Embedding {len(filenames)} file(s)")

    succeeded = failed = skipped = 0
    total = len(filenames)

    for i, filename in enumerate(filenames):
        if not args.rerun  and store.has_embeddings(filename):
            TerminalUI.dim(f"skipping {filename} (fresh embeddings exist)")
            skipped += 1
            continue

        chunks = store.load_chunks(filename)
        if not chunks:
            TerminalUI.error(f"{filename} — no chunks found, skipping")
            failed += 1
            continue

        # Only unload after the LAST file
        is_last = (i == total - 1)

        try:
            vectors = run_embedding_pipeline(
                filename     = filename,
                chunks       = chunks,
                store        = store,
                rerun        = args.rerun,
                unload_after = is_last,  # ← free RAM after final file
            )

            if vectors is not None:
                succeeded += 1
                if args.inspect:
                    _print_inspection(filename, store)
            else:
                skipped += 1

        except Exception as e:
            TerminalUI.error(f"{filename} — {type(e).__name__}: {e}")
            failed += 1

    # ── Safety: ensure model is unloaded even if something went wrong ──
    try:
        loader.unload(ModelCategory.EMBEDDING)
        TerminalUI.dim("Embedding model unloaded from memory")
    except Exception:
        import traceback
        tb_str = ''.join(traceback.format_exception(e))
        TerminalUI.warn(tb_str)
        pass

    TerminalUI.blank()
    TerminalUI.success(f"done — {succeeded} embedded · {skipped} skipped · {failed} failed")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Embed chunked documents into vector index"
    )
    parser.add_argument("path",           nargs="?", default=".",  help="Project directory")
    parser.add_argument("--file",    "-f", default=None,           help="Embed a single file")
    parser.add_argument("--project", "-p", default=None,           help="Project directory (default: cwd)")
    parser.add_argument("--inspect", "-i", action="store_true",    help="Print vector stats and sample chunks")
    parser.add_argument("--rerun",   "-R", action="store_true",    help="Re-embed even if fresh embeddings exist")

    args = parser.parse_args()

    handle_embedding(EmbeddingArgs(
        target  = args.path,
        file    = args.file,
        project = args.project or args.path,
        inspect = args.inspect,
        rerun   = args.rerun,
    ))

    return 0


if __name__ == "__main__":
    sys.exit(main())