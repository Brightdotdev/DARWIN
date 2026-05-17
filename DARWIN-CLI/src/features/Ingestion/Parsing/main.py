# main.py — Document parsing pipeline with session, spinner, and throttled progress
from __future__ import annotations

import sys
import multiprocessing
import time
import asyncio
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.Ingestion.Parsing.Parser import parse as _parse, scan_directory
from src.features.Ingestion.Parsing.lib import (
    ParsedDocument,
    FileWithImages,
    CPU_COUNT,
    ParsingArgs,
    detect_type,
)
from src.features.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket
from src.core.Logger.History import setup_history


# ---------------------------------------------------------------------------
# Worker (unchanged)
# ---------------------------------------------------------------------------
def parse_document(
    path_with_images: FileWithImages,
    caption_images: bool,
    precise: bool,
) -> ParsedDocument:
    file_path = Path(path_with_images.file)
    try:
        return _parse(
            file_path,
            images_dir=path_with_images.image_dir,
            caption=caption_images,
            precise=precise,
        )
    except Exception as e:
        return ParsedDocument(
            display_text="", embedding_text="", filename=file_path.name,
            page_count=0, file_type=detect_type(file_path) or "unknown",
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Orchestrator — throttled progress updates
# ---------------------------------------------------------------------------
def parse_documents(
    files: list[FileWithImages],
    caption_images: bool = False,
    precise: bool = False,
) -> list[ParsedDocument]:
    if not files:
        return []

    total = len(files)
    mode_label = "precise" if precise else "fast"
    darwin.log(f"Queuing {total} files ({mode_label})", LogType.INFO)

    indexed_files = sorted(enumerate(files), key=lambda x: Path(x[1].file).stat().st_size)
    n_workers = min(total, max(1, CPU_COUNT // 2) if caption_images else CPU_COUNT)

    results: list[Optional[ParsedDocument]] = [None] * total
    completed_files: set[str] = set()
    file_errors: dict[str, str] = {}

    darwin.progress_start("parsing", total, f"Parsing {total} files")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_info = {}
        for orig_idx, fw in indexed_files:
            future = executor.submit(parse_document, fw, caption_images, precise)
            future_to_info[future] = (orig_idx, Path(fw.file).name)

        processed = 0
        last_update = 0
        UPDATE_INTERVAL = 5  # update progress bar every N files

        for future in as_completed(future_to_info):
            orig_idx, filename = future_to_info[future]
            processed += 1
            try:
                doc = future.result()
                results[orig_idx] = doc
                if doc.error:
                    file_errors[filename] = doc.error
                else:
                    completed_files.add(filename)
            except Exception as e:
                results[orig_idx] = ParsedDocument(
                    display_text="", embedding_text="", filename=filename,
                    page_count=0, file_type="unknown", error=str(e),
                )
                file_errors[filename] = str(e)

            # Throttled progress update
            if processed - last_update >= UPDATE_INTERVAL or processed == total:
                darwin.progress_update("parsing", processed)
                last_update = processed

    darwin.progress_end("parsing")

    # Summary logs
    if completed_files:
        darwin.log(f"Parsed: {len(completed_files)} files", LogType.SUCCESS)
    if file_errors:
        darwin.log(f"Failed: {len(file_errors)} files", LogType.ERROR)
        for name, err in list(file_errors.items())[:5]:
            darwin.log(f"  • {name}: {err}", LogType.ERROR)
        if len(file_errors) > 5:
            darwin.log(f"  … and {len(file_errors) - 5} more", LogType.ERROR)

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# handle_parsing (unchanged core, but now logged inside a session)
# ---------------------------------------------------------------------------
def handle_parsing(args: ParsingArgs, store: DarwinStore) -> list[ParsedDocument]:
    target_paths: list[Path] = []

    if args.files:
        for raw in args.files:
            p = Path(raw).resolve()
            if not p.exists():
                darwin.log(f"file not found: {raw}", LogType.ERROR)
                continue
            if not p.is_file():
                darwin.log(f"file provided is not a file: {raw}", LogType.ERROR)
                continue
            if detect_type(p) is None:
                darwin.log(f"Unsupported file type: {p.suffix} ({p.name})", LogType.ERROR)
                continue
            target_paths.append(p)
        if not target_paths:
            darwin.log("No valid supported files found.", LogType.ERROR)
            return []
    else:
        target_dir = Path(args.target).resolve()
        if not target_dir.exists():
            darwin.log(f"Directory not found: {args.target}", LogType.ERROR)
            return []
        target_paths = scan_directory(target_dir, recursive=args.recursive)

    if not target_paths:
        darwin.log(f"No supported files found in {args.target}", LogType.ERROR)
        darwin.log("Supported types: .pdf, .docx, .pptx, .txt, .md", LogType.INFO)
        return []

    darwin.log(f"Settings: Images={args.images} | Caption={args.caption} | Recursive={args.recursive} | Mode={'Precise' if args.precise else 'Fast'}", LogType.INFO)
    darwin.log("")

    for path in target_paths:
        store.registry.register(path)

    to_process = []
    skipped_count = 0
    for path in target_paths:
        if store.has_markdown(path.name) and not args.rerun:
            skipped_count += 1
            continue
        to_process.append(path)

    if skipped_count > 0:
        darwin.log(f"Skipped {skipped_count} cached files (use --rerun to reprocess)", LogType.INFO)
    if not to_process:
        darwin.log("All files already parsed!", LogType.SUCCESS)
        darwin.log("Use --rerun to reprocess with different settings", LogType.INFO)
        return []

    darwin.log(f"Processing {len(to_process)} file(s)...", LogType.INFO)

    files = [FileWithImages(file=path, image_dir=store.ensure_image_dir(path.name) if args.images else None) for path in to_process]

    t0 = time.perf_counter()
    docs = parse_documents(files, caption_images=args.caption, precise=args.precise)
    elapsed = time.perf_counter() - t0

    succeeded = 0
    failed = 0
    for doc in docs:
        if doc.error:
            failed += 1
            darwin.log(f"{doc.filename} - {doc.error}", LogType.ERROR)
        else:
            if args.save:
                matching_path = next((p for p in to_process if p.name == doc.filename), None)
                if matching_path:
                    store.registry.register(matching_path, page_count=doc.page_count)
                store.save_markdown(doc.filename, doc)
            succeeded += 1

    darwin.log("")
    darwin.log(f"Total time: {elapsed:.1f}s", LogType.INFO)
    if succeeded > 0:
        darwin.log(f"Successfully parsed {succeeded} file(s)", LogType.SUCCESS)
    if failed > 0:
        darwin.log(f"Failed to parse {failed} file(s)", LogType.WARNING)

    return docs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
async def main() -> int:
    parser = argparse.ArgumentParser(description="Parse documents into .darwin format")
    parser.add_argument("path", nargs="?", default=".", help="File or directory to parse")
    parser.add_argument("--rerun", "-R", action="store_true", help="Reprocess cached files")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan directories recursively")
    parser.add_argument("--images", "-img", action="store_true", help="Extract images from documents")
    parser.add_argument("--caption", "-c", action="store_true", help="Generate captions for images (requires --images)")
    parser.add_argument("--file", "-f", nargs="+", default=None, help="Parse specific file(s)")
    parser.add_argument("--precise", action="store_true", help="Use pymupdf4llm for layout-accurate extraction")
    parser.add_argument("--no-save", action="store_true", help="Don't save markdown files (preview only)")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")
    parser.add_argument("--store-dir", default=None, help="Darwin store directory")
    args = parser.parse_args()

    # Start WebSocket server (if not disabled)
    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    # Always record history
    history = setup_history()

    try:
        # Wrap the entire run in a session
        with darwin.session("parsing"):
            darwin.spinner_start("init", "Initializing parsing pipeline...")
            # Optional: a brief pause for spinner visibility (remove in production)
            time.sleep(0.3)

            parsing_args = ParsingArgs(
                target=args.path if not args.file else str(Path(args.file).parent),
                images=args.images,
                caption=args.caption,
                recursive=args.recursive,
                rerun=args.rerun,
                file=args.file,
                precise=args.precise,
                save=not args.no_save,
            )
            store = DarwinStore(args.store_dir or None)
            handle_parsing(parsing_args, store)
            darwin.spinner_stop("init", success=True)

        return 0

    except KeyboardInterrupt:
        darwin.log("Interrupted", LogType.WARNING)
        return 130
    except Exception as e:
        import traceback
        darwin.log(f"Fatal error: {e}", LogType.ERROR)
        darwin.log(''.join(traceback.format_exception(e)), LogType.ERROR)
        return 1
    finally:
        shutdown_websocket(ws_server)
        darwin.shutdown()


if __name__ == "__main__":
    asyncio.run(main())