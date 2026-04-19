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

# chunk and print a sample of chunks to terminal
uv run src/core/Chunking/main.py --inspect

# force rechunk even if chunks already exist
uv run src/core/Chunking/main.py --rerun

# chunk for a specific embedding model (sets token budget automatically)
uv run src/core/Chunking/main.py --model bge-m3
"""

import sys
import argparse
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Chunking.Chunker import chunk_for_model
from src.core.Chunking.ChunkerUtils import Chunk, ChunkingArgs
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


# ── Inspection helpers ─────────────────────────────────────────────────────────

def _print_inspection(chunks: list[Chunk], filename: str) -> None:
    TerminalUI.blank()
    TerminalUI.rule(f"chunk inspection — {filename}")
    TerminalUI.blank()

    total        = len(chunks)
    avg_tokens   = sum(c.token_count for c in chunks) / total if total else 0
    min_tok      = min(c.token_count for c in chunks) if chunks else 0
    max_tok      = max(c.token_count for c in chunks) if chunks else 0
    with_heading = sum(1 for c in chunks if c.heading)
    tables       = sum(1 for c in chunks if "[Table]" in c.text)
    captions     = sum(1 for c in chunks if "[Caption:" in c.text)
    overlaps     = sum(1 for c in chunks if c.is_overlap)
    with_links   = sum(1 for c in chunks if c.links)
    total_links  = sum(len(c.links) for c in chunks)

    TerminalUI.table(
        headers=["metric", "value"],
        rows=[
            ["total chunks",   str(total)],
            ["avg tokens",     f"{avg_tokens:.0f}"],
            ["min tokens",     str(min_tok)],
            ["max tokens",     str(max_tok)],
            ["with heading",   f"{with_heading} ({with_heading/total*100:.0f}%)"],
            ["table chunks",   str(tables)],
            ["caption chunks", str(captions)],
            ["overlap chunks", str(overlaps)],
            ["chunks w/ links", str(with_links)],
            ["total links",    str(total_links)],
        ],
    )

    TerminalUI.blank()
    TerminalUI.dim("── first 3 chunks ──")
    for chunk in chunks[:3]:
        _print_chunk(chunk)

    if total > 6:
        TerminalUI.dim(f"── middle chunk ({total // 2}) ──")
        _print_chunk(chunks[total // 2])

    TerminalUI.dim("── last chunk ──")
    _print_chunk(chunks[-1])
    TerminalUI.rule()
    TerminalUI.blank()


def _print_chunk(chunk: Chunk) -> None:
    TerminalUI.info("index",   str(chunk.index))
    TerminalUI.info("heading", chunk.heading or "(none)")
    TerminalUI.info("page",    f"{chunk.page}–{chunk.page_end}")
    TerminalUI.info("tokens",  str(chunk.token_count))
    TerminalUI.info("overlap", "yes" if chunk.is_overlap else "no")
    TerminalUI.info("links",   str(len(chunk.links)))
    TerminalUI.blank()

    preview = chunk.text[:300].replace("\n", " ").strip()
    if len(chunk.text) > 300:
        preview += "..."
    TerminalUI.dim(f"  text:  {preview}")

    if chunk.links:
        for lk in chunk.links[:3]:
            label = f" [{lk.text}]" if lk.text else ""
            TerminalUI.dim(f"  link:  {lk.url}{label}")
        if len(chunk.links) > 3:
            TerminalUI.dim(f"  ... and {len(chunk.links) - 3} more links")

    TerminalUI.blank()
    TerminalUI.divider()
    TerminalUI.blank()


# ── Core handler ───────────────────────────────────────────────────────────────

def handle_chunking(args: ChunkingArgs) -> None:
    store = DarwinStore(Path(args.project) if args.project else Path.cwd())

    TerminalUI.print_darwin()
    TerminalUI.blank()
    TerminalUI.bold("Chunking")
    TerminalUI.dim(f"Rerun: {args.rerun} | Inspect: {args.inspect}")
    TerminalUI.blank()

    if args.file:
        filenames = [args.file]
    else:
        status    = store.status()
        filenames = [
            fname for fname, info in status.get("files", {}).items()
            if info["components"]["markdown"]
        ]

    if not filenames:
        TerminalUI.error("No parsed files found. Run parsing first.")
        return

    succeeded = failed = skipped = 0

    for filename in filenames:
        if not args.rerun and store.is_fresh(filename) and store.has_chunks(filename):
            TerminalUI.dim(f"skipping {filename} (fresh chunks exist)")
            skipped += 1
            continue

        tobe_chunked =  args.embedding_text if args.embedding_text else store.load_markdown(filename)

        if tobe_chunked is None:
            TerminalUI.error(f"{filename} — no markdown found, skipping")
            failed += 1
            continue

        # Load link metadata if the store has it — zero-cost when absent

        try:
            t = TerminalUI.timer(f"chunk:{filename}")
            chunks, chunk_metadata = chunk_for_model(
                text      = tobe_chunked,
                source    = filename,
            )
            t.done()

            if not chunks:
                TerminalUI.error(f"{filename} — produced 0 chunks")
                failed += 1
                continue

            total_links = sum(len(c.links) for c in chunks) if chunks else 0
          
            store.save_chunks(
                filename   = filename,
                chunks     = chunks,
                chunk_meta = {**chunk_metadata, "total_links": total_links},
                rerun      = args.rerun,
            )

            TerminalUI.success(
                f"{filename} — {len(chunks)} chunks · "
                f"avg {sum(c.token_count for c in chunks) // len(chunks)} tokens · "
                f"{total_links} links distributed"
            )
          
            succeeded += 1

            if args.inspect:
                _print_inspection(chunks, filename)

        except Exception as e:
            TerminalUI.error(f"{filename} — {type(e).__name__}: {e}")
            failed += 1

    TerminalUI.blank()
    TerminalUI.success(f"done — {succeeded} chunked · {skipped} skipped · {failed} failed")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk parsed documents into embedding-ready pieces"
    )
    parser.add_argument("path",           nargs="?", default=".",  help="Project directory")
    parser.add_argument("--file",    "-f", default=None,           help="Chunk a single file")
    parser.add_argument("--project", "-p", default=None,           help="Project directory (default: cwd)")
    parser.add_argument("--inspect", "-i", action="store_true",    help="Print sample chunks to terminal")
    parser.add_argument("--rerun",   "-R", action="store_true",    help="Rechunk even if fresh chunks exist")

    args = parser.parse_args()

    handle_chunking(ChunkingArgs(
        target  = args.path,
        file    = args.file,
        project = args.project or args.path,
        inspect = args.inspect,
        rerun   = args.rerun,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())