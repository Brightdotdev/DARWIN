"""
src/core/Chunking/main.py - Chunking pipeline entry point

Reads parsed markdown from .darwin, chunks it, saves chunks.json.
Can be run standalone to inspect chunking output before embedding.

Usage examples
--------------
# chunk everything already parsed in current directory
uv run src/core/Chunking/main.py

# chunk a specific file
uv run src/core/Chunking/main.py --file biology.pdf

# chunk with custom token budget (default 350 for e5-small)
uv run src/core/Chunking/main.py --max-tokens 512

# chunk and print a sample of chunks to terminal
uv run src/core/Chunking/main.py --inspect

# force rechunk even if chunks already exist
uv run src/core/Chunking/main.py --rerun

# chunk for a specific embedding model (sets token budget automatically)
uv run src/core/Chunking/main.py --model bge-m3
"""

import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Chunking.Chunker import chunk_document 
from src.core.Chunking.ChunkerUtils import Chunk, DEFAULT_MAX_TOKENS, DEFAULT_MIN_TOKENS, DEFAULT_OVERLAP,ChunkingArgs

from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI



# ============================================================================
# Inspect — print sample chunks to terminal so you can see what's coming out
# ============================================================================

def _print_inspection(chunks: list[Chunk], filename: str) -> None:
    """Print a readable sample of chunks to terminal."""
    TerminalUI.blank()
    TerminalUI.rule(f"chunk inspection — {filename}")
    TerminalUI.blank()

    total       = len(chunks)
    avg_tokens  = sum(c.token_count for c in chunks) / total if total else 0
    min_tok     = min(c.token_count for c in chunks) if chunks else 0
    max_tok     = max(c.token_count for c in chunks) if chunks else 0
    with_heading = sum(1 for c in chunks if c.heading)
    tables      = sum(1 for c in chunks if "[Table]" in c.text)
    captions    = sum(1 for c in chunks if "[Caption:" in c.text)
    overlaps    = sum(1 for c in chunks if c.is_overlap)

    TerminalUI.table(
        headers=["metric", "value"],
        rows=[
            ["total chunks",       str(total)],
            ["avg tokens",         f"{avg_tokens:.0f}"],
            ["min tokens",         str(min_tok)],
            ["max tokens",         str(max_tok)],
            ["with heading",       f"{with_heading} ({with_heading/total*100:.0f}%)"],
            ["table chunks",       str(tables)],
            ["caption chunks",     str(captions)],
            ["overlap chunks",     str(overlaps)],
        ],
    )

    TerminalUI.blank()
    TerminalUI.dim("── first 3 chunks ──")
    TerminalUI.blank()

    for chunk in chunks[:3]:
        _print_chunk(chunk)

    if total > 6:
        TerminalUI.dim(f"── middle chunk ({total // 2}) ──")
        TerminalUI.blank()
        _print_chunk(chunks[total // 2])

    TerminalUI.dim("── last chunk ──")
    TerminalUI.blank()
    _print_chunk(chunks[-1])

    TerminalUI.rule()
    TerminalUI.blank()


def _print_chunk(chunk: Chunk) -> None:
    TerminalUI.info("index",       str(chunk.index))
    TerminalUI.info("heading",     chunk.heading or "(none)")
    TerminalUI.info("page",        str(chunk.page))
    TerminalUI.info("tokens",      str(chunk.token_count))
    TerminalUI.info("overlap",     "yes" if chunk.is_overlap else "no")
    TerminalUI.blank()

    # show text preview — first 300 chars
    preview = chunk.text[:300].replace("\n", " ").strip()
    if len(chunk.text) > 300:
        preview += "..."
    TerminalUI.dim(f"  text:  {preview}")
    TerminalUI.blank()

    # show clean preview — first 300 chars
    clean_preview = chunk.clean[:300].replace("\n", " ").strip()
    if len(chunk.clean) > 300:
        clean_preview += "..."
    TerminalUI.dim(f"  clean: {clean_preview}")
    TerminalUI.blank()
    TerminalUI.divider()
    TerminalUI.blank()


# ============================================================================
# Core handler
# ============================================================================

def handle_chunking(args: ChunkingArgs) -> None:
    store = DarwinStore(Path(args.project) if args.project else Path.cwd())

    # resolve which files to chunk
    if args.file:
        filenames = [args.file]
    else:
        # all files that have been parsed (have markdown) in the store
        status    = store.status()
        filenames = [
            fname for fname, info in status.get("files", {}).items()
            if info["components"]["markdown"]
        ]

    if not filenames:
        TerminalUI.error("No parsed files found. Run parsing first.")
        TerminalUI.error("Read src/core/Parsing/README.md for more info.")
        return


    model_presets = {
    "e5-small": DEFAULT_MAX_TOKENS,
    "bge-m3": 800,
    "gritlm": 512,
    "gritlm-7b": 512,
    }

    max_tokens = model_presets.get(args.model.lower(), DEFAULT_MAX_TOKENS)
    TerminalUI.bold(f"Chunking {len(filenames)} file(s) — max_tokens={max_tokens}")

    if args.min_tokens > max_tokens:
        TerminalUI.dim(f"The minimuum number of tokens cant be higher than the max_tokens ({args.min_tokens} < {max_tokens}) defaulting to default min_tokens ({DEFAULT_MIN_TOKENS})")
        args.min_tokens = DEFAULT_MIN_TOKENS
    
    succeeded = failed = skipped = 0

    for filename in filenames:
        # skip if fresh chunks exist and not rerunning
        if not args.rerun and store.is_fresh(filename) and store.has_chunks(filename):
            TerminalUI.dim(f"skipping {filename} (fresh chunks exist)")
            skipped += 1
            continue

        # load embedding_text from parsed markdown
        # note: we chunk embedding_text not display_text
        tobe_chunked = args.embedding_text if args.embedding_text else store.load_markdown(filename)

        if tobe_chunked is None:
            TerminalUI.error(f"{filename} — no markdown found, skipping")
            failed += 1
            continue

        try:
            t      = TerminalUI.timer(f"chunk:{filename}")
            chunks = chunk_document(
                text       = tobe_chunked,
                source     = filename,
                max_tokens = max_tokens,
                min_tokens = args.min_tokens,
                overlap    = args.overlap,
            )
            t.done()

            if not chunks:
                TerminalUI.error(f"{filename} — produced 0 chunks")
                failed += 1
                continue

            store.save_chunks(
                filename   = filename,
                chunks     = chunks,
                max_tokens = max_tokens,
                chunk_model = args.model,
            )

            TerminalUI.success(
                f"{filename} — {len(chunks)} chunks · "
                f"avg {sum(c.token_count for c in chunks) // len(chunks)} tokens"
            )
            succeeded += 1

            if args.inspect:
                _print_inspection(chunks, filename)

        except Exception as e:
            TerminalUI.error(f"{filename} — {type(e).__name__}: {e}")
            failed += 1

    TerminalUI.blank()
    TerminalUI.success(f"done — {succeeded} chunked · {skipped} skipped · {failed} failed")


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk parsed documents into embedding-ready pieces"
    )
    parser.add_argument("path",          nargs="?", default=".",   help="Project directory")
    parser.add_argument("--file",   "-f", default=None,            help="Chunk a single file")
    parser.add_argument("--project","-p", default=None,            help="Project directory (default: cwd)")
    parser.add_argument("--model",  "-m", default="e5-small",      help="Embedding model preset (sets token budget)")
    parser.add_argument("--min-tokens",   type=int, default=DEFAULT_MIN_TOKENS, help="Min tokens before merging")
    parser.add_argument("--overlap",      type=int, default=DEFAULT_OVERLAP,    help="Sentences of overlap")
    parser.add_argument("--inspect","-i", action="store_true",     help="Print sample chunks to terminal")
    parser.add_argument("--rerun",  "-R", action="store_true",     help="Rechunk even if fresh chunks exist")

    args = parser.parse_args()

    handle_chunking(ChunkingArgs(
        target     = args.path,
        file       = args.file,
        project    = args.project or args.path,
        min_tokens = args.min_tokens,
        overlap    = args.overlap,
        model      = args.model,
        inspect    = args.inspect,
        rerun      = args.rerun,
    ))

    return 0


if __name__ == "__main__":
    sys.exit(main())