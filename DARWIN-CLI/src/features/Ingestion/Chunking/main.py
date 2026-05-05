"""
main.py - Document chunking pipeline entry point

Usage:
  uv run src/core/Ingestion/Chunking/main.py                        # chunk all parsed files
  uv run src/core/Ingestion/Chunking/main.py -f biology.pdf         # single file
  uv run src/core/Ingestion/Chunking/main.py -f docs/biology.pdf    # path — normalized to basename
  uv run src/core/Ingestion/Chunking/main.py -f biology.pdf chemistry.pdf --rerun
"""

import sys
import argparse
import time
import asyncio
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.features.Storage.DarwinStore import DarwinStore
from src.features.Ingestion.Chunking.Chunker import chunk_for_model
from src.features.Ingestion.Chunking.lib import Chunk, ChunkingArgs
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket
from src.core.Logger.DarwinLogger import darwin, LogType


# ── file resolution ────────────────────────────────────────────────────────────

def _resolve_files(raw_inputs: list[str], status: dict) -> list[str]:
    """
    Accept paths or bare filenames, strip directory components,
    validate against store status, and return canonical filenames.

    Examples that all resolve to "biology.pdf":
      biology.pdf
      docs/biology.pdf
      /home/user/documents/biology.pdf
      ../../data/biology.pdf
    """
    parsed_files = status.get("files", {})
    resolved: list[str] = []

    for raw in raw_inputs:
        candidate = Path(raw).name      # strip any leading path components

        if candidate in parsed_files:
            if parsed_files[candidate].get("is_parsed", False):
                resolved.append(candidate)
            else:
                darwin.log(
                    f"'{raw}' → '{candidate}' exists but is not parsed yet — skipping",
                    LogType.WARNING,
                )
        else:
            darwin.log(
                f"'{raw}' → '{candidate}' not found in store — skipping",
                LogType.WARNING,
            )

    # deduplicate, preserve order
    seen: set[str] = set()
    unique: list[str] = []
    for f in resolved:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return unique


# ── inspection helpers ─────────────────────────────────────────────────────────

def _print_inspection(chunks: list[Chunk], filename: str) -> None:
    """Print detailed chunk inspection output."""
    darwin.log("")
    darwin.log(f"── chunk inspection — {filename} ──", LogType.INFO)
    darwin.log("")

    total = len(chunks)
    if total == 0:
        darwin.log("  No chunks to inspect", LogType.WARNING)
        return

    avg_tokens   = sum(c.token_count for c in chunks) / total
    min_tok      = min(c.token_count for c in chunks)
    max_tok      = max(c.token_count for c in chunks)
    with_heading = sum(1 for c in chunks if c.heading)
    tables       = sum(1 for c in chunks if "[Table]" in c.text)
    captions     = sum(1 for c in chunks if "[Caption:" in c.text)
    overlaps     = sum(1 for c in chunks if c.is_overlap)
    with_links   = sum(1 for c in chunks if c.links)
    total_links  = sum(len(c.links) for c in chunks)

    darwin.log("  metric              value", LogType.INFO)
    darwin.log("  ──────────────────  ─────", LogType.INFO)
    darwin.log(f"  total chunks        {total}", LogType.INFO)
    darwin.log(f"  avg tokens          {avg_tokens:.0f}", LogType.INFO)
    darwin.log(f"  min tokens          {min_tok}", LogType.INFO)
    darwin.log(f"  max tokens          {max_tok}", LogType.INFO)
    pct_heading = (with_heading / total * 100) if total > 0 else 0
    darwin.log(f"  with heading        {with_heading} ({pct_heading:.0f}%)", LogType.INFO)
    darwin.log(f"  table chunks        {tables}", LogType.INFO)
    darwin.log(f"  caption chunks      {captions}", LogType.INFO)
    darwin.log(f"  overlap chunks      {overlaps}", LogType.INFO)
    darwin.log(f"  chunks w/ links     {with_links}", LogType.INFO)
    darwin.log(f"  total links         {total_links}", LogType.INFO)
    darwin.log("")

    preview_count = min(3, total)
    darwin.log(f"── first {preview_count} chunk(s) ──", LogType.INFO)
    for chunk in chunks[:preview_count]:
        _print_chunk(chunk)

    if total > 6:
        mid_idx = total // 2
        darwin.log(f"── middle chunk ({mid_idx}) ──", LogType.INFO)
        _print_chunk(chunks[mid_idx])

    darwin.log("── last chunk ──", LogType.INFO)
    _print_chunk(chunks[-1])
    darwin.log("─" * 60, LogType.INFO)
    darwin.log("")


def _print_chunk(chunk: Chunk) -> None:
    """Print a single chunk's details."""
    overlap_str = getattr(chunk, "overlap_direction", None)
    overlap_val = overlap_str.value if overlap_str else "none"

    darwin.log(f"  index:     {chunk.index}", LogType.INFO)
    darwin.log(f"  heading:   {chunk.heading or '(none)'}", LogType.INFO)
    darwin.log(f"  page:      {chunk.page}–{chunk.page_end}", LogType.INFO)
    darwin.log(f"  tokens:    {chunk.token_count}", LogType.INFO)
    darwin.log(f"  overlap:   {overlap_val}", LogType.INFO)
    darwin.log(f"  links:     {len(chunk.links) if chunk.links else 0}", LogType.INFO)
    darwin.log("")

    text = chunk.text or ""
    preview = text[:300].replace("\n", " ").strip()
    if len(text) > 300:
        preview += "..."
    darwin.log(f"  text:  {preview}", LogType.INFO)

    if chunk.links:
        max_links = min(3, len(chunk.links))
        for lk in chunk.links[:max_links]:
            label = f" [{lk.text}]" if lk.text else ""
            darwin.log(f"  link:  {lk.url}{label}", LogType.INFO)
        if len(chunk.links) > 3:
            darwin.log(f"  ... and {len(chunk.links) - 3} more links", LogType.INFO)

    darwin.log("")
    darwin.log("─" * 40, LogType.INFO)
    darwin.log("")


# ── core handler ───────────────────────────────────────────────────────────────

def handle_chunking(args: ChunkingArgs) -> None:
    store  = DarwinStore()
    status = store.status()

    darwin.log("")
    darwin.log("Chunking", LogType.INFO)
    darwin.log(f"Rerun: {args.rerun} | Inspect: {args.inspect}", LogType.INFO)
    darwin.log("")

    if args.files:
        filenames = _resolve_files(args.files, status)
    else:
        try:
            filenames = [
                fname
                for fname, info in status.get("files", {}).items()
                if info.get("is_parsed", False)
            ]
        except Exception as e:
            darwin.log(f"Failed to read store status: {e}", LogType.ERROR)
            return

    if not filenames:
        darwin.log("No parsed files found. Run parsing first.", LogType.ERROR)
        return

    succeeded = failed = 0

    for filename in filenames:
        try:
            record = store.registry._find_by_name(filename)
        except AttributeError:
            darwin.log(f"{filename} — registry method unavailable", LogType.ERROR)
            failed += 1
            continue

        if record is None:
            darwin.log(f"{filename} — not registered, run parsing first", LogType.ERROR)
            failed += 1
            continue

        if not args.rerun and store.has_chunks(filename):
            darwin.log(f"{filename} — already chunked (use --rerun to rechunk)", LogType.INFO)
            succeeded += 1
            continue

        tobe_chunked = store.load_markdown(filename)

        if tobe_chunked is None:
            darwin.log(f"{filename} — no markdown found, skipping", LogType.ERROR)
            failed += 1
            continue

        if hasattr(tobe_chunked, "embedding_text"):
            chunk_text = tobe_chunked.embedding_text
        elif isinstance(tobe_chunked, str):
            chunk_text = tobe_chunked
        else:
            darwin.log(
                f"{filename} — unexpected markdown type: {type(tobe_chunked)}",
                LogType.ERROR,
            )
            failed += 1
            continue

        try:
            t0 = time.perf_counter()

            chunks, chunk_metadata = chunk_for_model(
                text=chunk_text,
                source=filename,
            )

            elapsed = time.perf_counter() - t0
            darwin.log(f"  chunk:{filename} — {elapsed:.2f}s", LogType.INFO)

            if not chunks:
                darwin.log(f"{filename} — produced 0 chunks", LogType.ERROR)
                failed += 1
                continue

            total_links = sum(len(c.links) for c in chunks)

            store.save_chunks(
                filename=filename,
                chunks=chunks,
                chunk_meta={**chunk_metadata, "total_links": total_links},
            )

            avg_tokens = sum(c.token_count for c in chunks) // len(chunks)
            darwin.log(
                f"{filename} — {len(chunks)} chunks · "
                f"avg {avg_tokens} tokens · "
                f"{total_links} links distributed",
                LogType.SUCCESS,
            )

            succeeded += 1

            if args.inspect:
                _print_inspection(chunks, filename)

        except Exception as e:
            import traceback
            darwin.log(f"{filename} — {type(e).__name__}: {e}", LogType.ERROR)
            darwin.log(traceback.format_exc())
            failed += 1

    darwin.log("")
    darwin.log(f"done — {succeeded} chunked · {failed} failed", LogType.SUCCESS)


# ── CLI entry point ────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk parsed documents into embedding-ready pieces"
    )


    parser.add_argument(
        "--file", "-f", nargs="+", default=None, dest="files",
        metavar="FILE",
        help="Files to chunk (paths or bare filenames). Omit to chunk all parsed files.",
    )
    parser.add_argument("--inspect", "-i", action="store_true", help="Print sample chunks to terminal")
    parser.add_argument("--rerun",   "-R", action="store_true", help="Rechunk even if fresh chunks exist")
    parser.add_argument("--ws-port", type=int, default=8765,    help="WebSocket server port (default: 8765)")
    parser.add_argument("--no-ws",   action="store_true",       help="Disable WebSocket server")

    args = parser.parse_args()


    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        chunking_args = ChunkingArgs(
            files=args.files,   # list[str] | None
            inspect=args.inspect,
            rerun=args.rerun,
        )

        handle_chunking(chunking_args)
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