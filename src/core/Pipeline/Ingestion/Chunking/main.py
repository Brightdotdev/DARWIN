

from __future__ import annotations

import sys
import time
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


from core.Storage.DarwinStore import DarwinStore
from core.Pipeline.Ingestion.Chunking.Chunker import chunk_document
from core.Pipeline.Ingestion.Chunking.StructuredChunker import chunk_structured
from core.Pipeline.Ingestion.Chunking.lib import  ChunkingArgs
from core.Logger.DarwinLogger import darwin


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
                darwin.warning(
                    f"'{raw}' resolves to '{candidate}' but it is not parsed yet, skipping"
                )
        else:
            darwin.warning(
                f"'{raw}' resolves to '{candidate}' which was not found in the store, skipping"
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

    darwin.info("")
    darwin.info("Chunking")
    darwin.info(f"Rerun: {args.rerun}")
    darwin.info("")

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
            darwin.error(f"Failed to read store status: {e}")
            return

    if not filenames:
        darwin.error("No parsed files found. Run parsing first.")
        return

    succeeded = failed = 0

    for filename in filenames:
        # ── v2 API: resolve name → source → content hash ─────────────────
        source = store._sources.find_by_name(filename)
        if source is None:
            darwin.error(f"{filename} is not registered, run parsing first")
            failed += 1
            continue


        if not args.rerun and store.has_records(filename):
            darwin.warning(f"{filename} 'already chunked' (use --rerun to re-chunk)")
            succeeded += 1
            continue

        # ── Check if this is a structured file ──────────────────────────
        schema = store.load_schema(filename)
        is_structured = schema is not None

        if is_structured:
            # Structured path: use StructuredChunker
            try:
                import json as _json

                t0 = time.perf_counter()

                # Re-parse the original file to get records
                from core.Pipeline.Ingestion.Parsing.StructuredParser import (
                    parse_structured,
                )
                source_obj = store.get_source(filename)
                if not source_obj or not source_obj.current_path:
                    darwin.error(f"{filename}: source path not found, cannot re-parse")
                    failed += 1
                    continue

                from pathlib import Path as _Path
                doc = parse_structured(_Path(source_obj.current_path), filename)
                if doc.error:
                    darwin.error(f"{filename}: parse error: {doc.error}")
                    failed += 1
                    continue

                chunks = chunk_structured(doc)

                elapsed = time.perf_counter() - t0
                darwin.info(f"  Chunked {filename} (structured) in {elapsed:.2f}s")

                if not chunks:
                    darwin.error(f"{filename} produced 0 chunks")
                    failed += 1
                    continue

                store.save_records(filename=filename, records=chunks)

                avg_tokens = sum(c.token_count for c in chunks) // len(chunks)
                darwin.success(
                    f"{filename}: {len(chunks)} records, "
                    f"avg {avg_tokens} tokens"
                )

                succeeded += 1

            except Exception as e:
                import traceback
                darwin.error(f"{filename}: {type(e).__name__}: {e}")
                darwin.error(traceback.format_exc())
                failed += 1

            continue

        # ── Text path (existing chunk_document) ─────────────────────────

        chunk_text = store.load_markdown(filename)
        
        if chunk_text is None:
            darwin.error(f"{filename} has no markdown, skipping")
            failed += 1
            continue

        try:
            t0 = time.perf_counter()

            chunks = chunk_document(
                text=chunk_text,
                source=filename,
            )

            elapsed = time.perf_counter() - t0
            darwin.info(f"  Chunked {filename} in {elapsed:.2f}s")

            if not chunks:
                darwin.error(f"{filename} produced 0 chunks")
                failed += 1
                continue

            total_links = sum(len(c.links) for c in chunks)


            # Save ALL chunks for citation / page resolution
            store.save_records(
                filename=filename,
                records=chunks,
            )
            

            avg_tokens = sum(c.token_count for c in chunks) // len(chunks)
            darwin.success(
                f"{filename}: {len(chunks)} chunks, "
                f"avg {avg_tokens} tokens, "
                f"{total_links} links"
            )

    
    

            succeeded += 1


        except Exception as e:
            import traceback
            darwin.error(f"{filename}: {type(e).__name__}: {e}")
            darwin.error(traceback.format_exc())
            failed += 1

    darwin.info("")
    darwin.success(f"Done: {succeeded} chunked, {failed} failed")


# ── CLI entry point ────────────────────────────────────────────────────────────

async def main() -> int:
    from lib.Utils import ensure_writable_directory
    import argparse

    parser = argparse.ArgumentParser(
        description="Chunk parsed documents into embedding-ready pieces"
    )

    parser.add_argument(
        "--file", "-f", nargs="+", default=None, dest="files",
        metavar="FILE",
        help="Files to chunk (paths or bare filenames). Omit to chunk all parsed files.",
    )
    parser.add_argument("--rerun",   "-R", action="store_true", help="Rechunk even if fresh chunks exist")
    parser.add_argument("--store-dir", default=None, help="Darwin store directory")

    args = parser.parse_args()


    darwin.configure(
        
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