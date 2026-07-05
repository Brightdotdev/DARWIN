

from __future__ import annotations

import sys
import argparse
import time
import asyncio
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


from src.core.Storage.DarwinStore import DarwinStore
from src.core.Pipeline.Ingestion.Chunking.Chunker import chunk_document
from src.core.Pipeline.Ingestion.Chunking.lib import Chunk, ChunkingArgs
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


# ── core handler ───────────────────────────────────────────────────────────────

def handle_chunking(args: ChunkingArgs, store: DarwinStore) -> None:
    status = store.status()

    darwin.log("")
    darwin.log("Chunking", LogType.INFO)
    darwin.log(f"Rerun: {args.rerun}", LogType.INFO)
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
        # ── v2 API: resolve name → source → content hash ─────────────────
        source = store._sources.find_by_name(filename)
        if source is None:
            darwin.log(f"{filename} — not registered, run parsing first", LogType.ERROR)
            failed += 1
            continue

        content_hash = source.content_hash

        if not args.rerun and store.has_records(filename):
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

            chunks = chunk_document(
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


            # Save ALL chunks for citation / page resolution
            store.save_records(
                filename=filename,
                records=chunks,
            )
            

            avg_tokens = sum(c.token_count for c in chunks) // len(chunks)
            darwin.log(
                f"{filename} — {len(chunks)} chunks · "
                f"avg {avg_tokens} tokens · "
                f"{total_links} links distributed",
                LogType.SUCCESS,
            )

    
    

            succeeded += 1


        except Exception as e:
            import traceback
            darwin.log(f"{filename} — {type(e).__name__}: {e}", LogType.ERROR)
            darwin.log(traceback.format_exc(), LogType.ERROR)
            failed += 1

    darwin.log("")
    darwin.log(f"done — {succeeded} chunked · {failed} failed", LogType.SUCCESS)


# ── CLI entry point ────────────────────────────────────────────────────────────

async def main() -> int:
    from src.lib.Utils import ensure_writable_directory

    parser = argparse.ArgumentParser(
        description="Chunk parsed documents into embedding-ready pieces"
    )

    parser.add_argument(
        "--file", "-f", nargs="+", default=None, dest="files",
        metavar="FILE",
        help="Files to chunk (paths or bare filenames). Omit to chunk all parsed files.",
    )
    parser.add_argument("--rerun",   "-R", action="store_true", help="Rechunk even if fresh chunks exist")
    parser.add_argument("--sse-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--store-dir", default=None, help="Darwin store directory")

    args = parser.parse_args()


    darwin.configure(
        
        sse_port=args.sse_port,
        
        
        show_banner=True,
    )

    try:
        with darwin.session("chunking"):
            chunking_args = ChunkingArgs(
                files=args.files,
                rerun=args.rerun,
            )

            if args.store_dir and ensure_writable_directory(args.store_dir):
                store = DarwinStore(args.store_dir)
            else:
                store = DarwinStore()

            handle_chunking(chunking_args, store)

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