from __future__ import annotations

import sys
import hashlib
import multiprocessing
import time
import asyncio
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple, Union
from urllib.parse import urlparse

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.Ingestion.Parsing.Parser import (
    parse as _parse,          # renamed to avoid conflict with our async function
    scan_directory,
    is_url,
)
from src.features.Ingestion.Parsing.lib import (
    ParsedDocument,
    FileWithImages,
    CPU_COUNT,
    ParsingArgs,
    detect_type,
    ExtractedImage,
)
from src.features.Ingestion.Parsing.PdfExtractors import (
    replace_image_slots,
    normalize_image_path,
)
from src.features.Images.ImageHandler import ImageHandler
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity


# ── URL utilities ────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """Normalize URL before slugging to ensure cache hits on minor variations."""
    return url.strip().rstrip("/")


def url_to_slug(url: str) -> str:
    """
    Deterministic, filesystem-safe identifier from a URL.
    Example: 'https://docs.python.org/3/library/asyncio.html'
          → 'url_docs_python_org_a3f7b2d1'
    """
    url = normalize_url(url)
    parsed = urlparse(url)
    netloc = parsed.netloc.replace(".", "_")
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
    return f"url_{netloc}_{url_hash}"


# ── Worker initializer (shared logging queue) ────────────────────────────────

def _worker_init(mp_queue):
    """Initialize worker process with the shared logging queue."""
    from src.core.Logger.DarwinLogger import darwin
    darwin.set_mp_queue(mp_queue)


# ── Worker function (NO CAPTIONING / NO OCR) ─────────────────────────────────

def parse_document_worker(
    source: Union[Path, str],
    image_dir: Optional[Path],
    display_name: str,
    precise: bool,
) -> ParsedDocument:
    """
    Parse a single source (file path or URL) without image enrichment.
    The parser handles both local files and URLs internally.
    """
    try:
        # The parser now accepts an optional display_name to override the filename
        doc = _parse(
            source,
            images_dir=image_dir,
            caption=False,          # we do captioning later in the main process
            precise=precise,
            display_name=display_name,
        )
        return doc
    except Exception as e:
        # Fallback error document
        return ParsedDocument(
            display_text="", embedding_text="", filename=display_name,
            page_count=0, file_type="unknown", error=f"{type(e).__name__}: {e}",
        )


# ── Parallel document parser ─────────────────────────────────────────────────

def parse_documents(
    tasks: List[Tuple[Union[Path, str], Optional[Path], str]],  # (source, image_dir, display_name)
    precise: bool = False,
) -> Tuple[List[ParsedDocument], Dict[str, List[ExtractedImage]]]:
    """
    Extract text and images from multiple sources in parallel.
    Sources can be local file paths or URL strings.

    Returns:
        - List of ParsedDocument (without enriched image text)
        - Dict mapping display_name -> list of ExtractedImage objects
    """
    if not tasks:
        return [], {}

    total = len(tasks)
    mode_label = "precise" if precise else "fast"
    darwin.log(f"Queuing {total} items ({mode_label})", LogType.INFO)

    # Sort by file size if local, else by display_name
    def _sort_key(item):
        source, _, name = item
        if isinstance(source, Path) and source.exists():
            return source.stat().st_size
        return 0  # URLs treated as smallest

    indexed_tasks = sorted(enumerate(tasks), key=lambda x: _sort_key(x[1]))
    n_workers = min(total, CPU_COUNT)

    results: List[Optional[ParsedDocument]] = [None] * total
    images_by_doc: Dict[str, List[ExtractedImage]] = {}

    darwin.progress_start("parsing", total, f"Parsing {total} items")
    darwin.start_mp_listener()
    mp_queue = darwin.get_mp_queue()

    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(mp_queue,),
        ) as executor:
            future_to_info = {}
            for orig_idx, (source, image_dir, display_name) in indexed_tasks:
                future = executor.submit(
                    parse_document_worker,
                    source,
                    image_dir,
                    display_name,
                    precise,
                )
                future_to_info[future] = (orig_idx, display_name)

            processed = 0
            last_update = 0
            UPDATE_INTERVAL = 5

            for future in as_completed(future_to_info):
                orig_idx, display_name = future_to_info[future]
                processed += 1
                try:
                    doc = future.result()
                    results[orig_idx] = doc
                    if doc.images:
                        images_by_doc[display_name] = doc.images
                except Exception as e:
                    results[orig_idx] = ParsedDocument(
                        display_text="", embedding_text="", filename=display_name,
                        page_count=0, file_type="unknown", error=str(e),
                    )

                if processed - last_update >= UPDATE_INTERVAL or processed == total:
                    darwin.progress_update("parsing", processed)
                    last_update = processed
    finally:
        darwin.progress_end("parsing")
        darwin.stop_mp_listener()

    succeeded = sum(1 for d in results if d is not None and not d.error)
    failed = total - succeeded
    if succeeded:
        darwin.log(f"Parsed: {succeeded} items", LogType.SUCCESS)
    if failed:
        darwin.log(f"Failed: {failed} items", LogType.ERROR)

    return [r for r in results if r is not None], images_by_doc


# ── Main pipeline handler ────────────────────────────────────────────────────

async def handle_parsing(args: ParsingArgs, store: DarwinStore) -> List[ParsedDocument]:
    target_paths: List[Path] = []
    target_urls: List[str] = []

    # ── Separate URLs from local files ──────────────────────────────────────
    if args.files:
        for raw in args.files:
            if is_url(raw):
                target_urls.append(raw)
            else:
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
        if not target_paths and not target_urls:
            darwin.log("No valid supported files or URLs found.", LogType.ERROR)
            return []
    else:
        target_dir = Path(args.target).resolve()
        if not target_dir.exists():
            darwin.log(f"Directory not found: {args.target}", LogType.ERROR)
            return []
        target_paths = scan_directory(target_dir, recursive=args.recursive)

    if not target_paths and not target_urls:
        darwin.log(f"No supported items found", LogType.ERROR)
        darwin.log("Supported: .pdf, .docx, .pptx, .txt, .md, URLs (http/https)", LogType.INFO)
        return []

    darwin.log(
        f"Settings: Images={args.images} | Caption={args.caption} | OCR={args.ocr} | "
        f"Recursive={args.recursive} | Mode={'Precise' if args.precise else 'Fast'}",
        LogType.INFO,
    )
    darwin.log("")

    # ── Build task list and check cache ─────────────────────────────────────
    # Each task: (source, image_dir, display_name)
    tasks_to_process = []
    skipped_count = 0

    # Local files
    for path in target_paths:
        display = path.name
        if store.has_markdown(display) and not args.rerun:
            skipped_count += 1
            continue
        image_dir = store.ensure_image_dir(display) if args.images else None
        tasks_to_process.append((path, image_dir, display))

    # URLs
    for url in target_urls:
        slug = url_to_slug(url)
        if store.has_markdown(slug) and not args.rerun:
            skipped_count += 1
            continue
        image_dir = store.ensure_image_dir(slug) if args.images else None
        tasks_to_process.append((url, image_dir, slug))  # source is URL string

    if skipped_count > 0:
        darwin.log(f"Skipped {skipped_count} cached item(s) (use --rerun to reprocess)", LogType.INFO)
    if not tasks_to_process:
        darwin.log("All items already parsed!", LogType.SUCCESS)
        darwin.log("Use --rerun to reprocess with different settings", LogType.INFO)
        return []

    total_items = len(tasks_to_process)
    darwin.log(f"Processing {total_items} item(s)...", LogType.INFO)

    t0 = time.perf_counter()

    # ── Pass 1: parallel extraction (no image enrichment) ───────────────────
    parsed_docs, images_by_display = parse_documents(tasks_to_process, precise=args.precise)

    # ── Pass 2: image enrichment (caption / OCR via ImageHandler) ──────────
    if args.images and (args.caption or args.ocr):
        all_images: List[ExtractedImage] = []
        for imgs in images_by_display.values():
            all_images.extend(imgs)

        if all_images:
            if args.caption and args.ocr:
                stages = "both"
            elif args.caption:
                stages = "vision"
            else:
                stages = "ocr"

            darwin.log(f"Enriching {len(all_images)} image(s) [{stages}]...", LogType.INFO)

            handler = ImageHandler()
            image_paths = [img.path for img in all_images]

            results = handler.process_batch(
                image_paths,
                stages=stages,
                max_workers=4,
                auto_ensure=True,
                unload_after=True,
                use_cache=True,
            )

            for img, result in zip(all_images, results):
                if result.error:
                    darwin.log(f"Image enrichment failed for {img.path.name}: {result.error}", LogType.WARNING)
                    img.caption = ""
                    continue

                parts = []
                if args.caption and result.description:
                    parts.append(result.description)
                if args.ocr and result.text:
                    parts.append(f"OCR: {result.text}")

                img.caption = " | ".join(parts) if parts else ""

            darwin.log("Image enrichment complete", LogType.SUCCESS)

    # ── Pass 3: apply enriched image text to document text ─────────────────
    for display_name, imgs in images_by_display.items():
        slot_map = {}
        for img in imgs:
            idx = img.index
            # Determine the image directory that was used (stored in img.path)
            # We need the relative path; we can use the store's image dir for this display_name
            img_dir = store.ensure_image_dir(display_name) if args.images else None
            rel_path = normalize_image_path(img.path, img_dir)
            slot_map[idx] = {
                "caption": img.caption,
                "rel_path": rel_path,
            }
        for doc in parsed_docs:
            if doc.filename == display_name:
                doc.display_text, doc.embedding_text = replace_image_slots(
                    doc.display_text, doc.embedding_text, slot_map
                )
                break

    elapsed = (time.perf_counter() - t0) / 60
    darwin.log(f"Total time: {elapsed:.1f}min", LogType.INFO)

    succeeded = sum(1 for d in parsed_docs if not d.error)
    failed = sum(1 for d in parsed_docs if d.error)
    if succeeded:
        darwin.log(f"Successfully parsed {succeeded} item(s)", LogType.SUCCESS)
    if failed:
        darwin.log(f"Failed: {failed} item(s)", LogType.WARNING)

    # ── Save markdown ───────────────────────────────────────────────────────
    if args.save:
        # Map of display_name -> original URL (if any) for metadata
        url_map = {url_to_slug(u): u for u in target_urls}

        for doc in parsed_docs:
            if doc.error:
                continue

            display = doc.filename
            # If this was a URL, add source_url to metadata
            if display in url_map:
                doc.metadata["source_url"] = url_map[display]

            # For local files, re-register to capture page_count
            if not display in url_map:
                # Find matching local path (if any) to register page_count
                matching_path = next((p for p in target_paths if p.name == display), None)
                if matching_path:
                    store.register(matching_path, page_count=doc.page_count)

            store.save_markdown(display, doc)

    return parsed_docs


# ── CLI entry point ──────────────────────────────────────────────────────────

async def main() -> int:
    from src.lib.Utils import ensure_writable_directory
    parser = argparse.ArgumentParser(description="Parse documents into .darwin format")
    parser.add_argument("path", nargs="?", default=".", help="File or directory to parse")
    parser.add_argument("--rerun", "-R", action="store_true", help="Reprocess cached files")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan directories recursively")
    parser.add_argument("--images", "-img", action="store_true", help="Extract images from documents")
    parser.add_argument("--caption", "-c", action="store_true", help="Generate captions for images (requires --images)")
    parser.add_argument("--ocr", action="store_true", help="Run OCR on extracted images (requires --images)")
    parser.add_argument("--file", "-f", nargs="+", default=None, help="Parse specific file(s) or URL(s)")
    parser.add_argument("--precise", action="store_true", help="Use precise extraction (layout-aware for PDFs, markdown output for HTML)")
    parser.add_argument("--no-save", action="store_true", help="Don't save markdown files (preview only)")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")
    parser.add_argument("--store-dir", default=None, help="Darwin store directory")
    parser.add_argument("--verbosity", choices=["normal", "quiet"], default="normal", help="Log verbosity level")
    args = parser.parse_args()

    if (args.caption or args.ocr) and not args.images:
        parser.error("--caption and --ocr require --images to be set")

    verbosity = Verbosity.NORMAL if args.verbosity == "normal" else Verbosity.QUIET
    darwin.configure(
        enable_websocket=not args.no_ws,
        ws_port=args.ws_port,
        enable_history=True,
        verbosity_default=verbosity,
        show_banner=True,
    )

    try:
        with darwin.session("parsing"):
            darwin.spinner_start("init", "Parsing...")
            time.sleep(0.3)

            parsing_args = ParsingArgs(
                target=args.path if not args.file else str(Path(args.file[0]).parent) if args.file and not is_url(args.file[0]) else ".",
                images=args.images,
                caption=args.caption,
                ocr=args.ocr,
                recursive=args.recursive,
                rerun=args.rerun,
                files=args.file,
                precise=args.precise,
                save=not args.no_save,
            )

            if args.store_dir and ensure_writable_directory(args.store_dir):
                store = DarwinStore(args.store_dir)
            else:
                store = DarwinStore()

            await handle_parsing(parsing_args, store)
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
        darwin.shutdown()


if __name__ == "__main__":
    asyncio.run(main())