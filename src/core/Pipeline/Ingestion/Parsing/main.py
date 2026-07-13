from __future__ import annotations

import sys
import multiprocessing
import time
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple, Union

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.Pipeline.Ingestion.Parsing.Parser import (
    parse as _parse,
    scan_directory,
    is_url,
)
from src.core.Pipeline.Ingestion.Parsing.lib import (
    ParsedDocument,
    url_to_slug,
    ParsingArgs,
    detect_type,
    ExtractedImage,
    MAX_PARSING_WORKERS,
)
from src.core.Pipeline.Ingestion.Parsing.PdfExtractors import (
    replace_image_slots,
    normalize_image_path,
)
from src.features.Images.ImageHandler import ImageHandler
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Logger.DarwinMP import DarwinMPBridge, worker_log

# ── Worker initializer (shared logging queue) ────────────────────────────────

def _worker_init(mp_queue):
    """Initialize worker process with the shared logging queue."""
    DarwinMPBridge.worker_init(mp_queue)  


# ── Worker function (NO CAPTIONING / NO OCR) ─────────────────────────────────

def parse_document_worker(
    source: Union[Path, str, Tuple[str, bytes, str]],
    image_dir: Optional[Path],
    display_name: str,
    precise: bool,
) -> ParsedDocument:
    """
    Parse a single source without image enrichment.
    Includes per-file timing for performance diagnostics.
    """
    t0 = time.perf_counter()
    try:
        doc = _parse(
            source,
            images_dir=image_dir,
            caption=False,
            precise=precise,
            display_name=display_name,
        )
        elapsed = time.perf_counter() - t0
        # Log timing from worker (goes through mp_queue)
        worker_log(f"  [{elapsed:.1f}s] {display_name}", LogType.INFO)
        
        return doc
    except Exception as e:
        elapsed = time.perf_counter() - t0
        worker_log(f"  [{elapsed:.1f}s] {display_name} FAILED", LogType.ERROR)
        return ParsedDocument(
            display_text="", embedding_text="", filename=display_name,
            page_count=0, file_type="unknown", error=f"{type(e).__name__}: {e}",
        )


# ── Parallel document parser ─────────────────────────────────────────────────

def parse_documents(
    tasks: List[Tuple[Union[Path, str, Tuple[str, bytes, str]], Optional[Path], str]],
    precise: bool = False,
) -> Tuple[List[ParsedDocument], Dict[str, List[ExtractedImage]]]:
    """
    Extract text and images from multiple sources in parallel.

    OPTIMIZATION: Caps workers at MAX_PARSING_WORKERS to reduce Windows spawn overhead.
    """
    if not tasks:
        return [], {}

    total = len(tasks)
    mode_label = "precise" if precise else "fast"
    darwin.info(f"Queuing {total} items ({mode_label})")

    def _sort_key(item):
        source, _, name = item
        if isinstance(source, Path) and source.exists():
            return source.stat().st_size
        return 0

    indexed_tasks = sorted(enumerate(tasks), key=lambda x: _sort_key(x[1]))

    # FIX #1: Cap workers to reduce Windows spawn overhead
    n_workers = min(total, MAX_PARSING_WORKERS)
    darwin.info(f"Using {n_workers} worker(s) (capped at {MAX_PARSING_WORKERS})")

    results: List[Optional[ParsedDocument]] = [None] * total
    images_by_doc: Dict[str, List[ExtractedImage]] = {}

    darwin.progress_start("parsing", total, f"Parsing {total} items")
    try:
        with DarwinMPBridge() as bridge:
            mp_queue = bridge.get_queue()
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

                    # FIX #2: Update progress on EVERY completion for better UX
                    darwin.progress_update("parsing", processed)

    finally:
        darwin.progress_end("parsing")

    succeeded = sum(1 for d in results if d is not None and not d.error)
    failed = total - succeeded
    if succeeded:
        darwin.success(f"Parsed: {succeeded} items")
    if failed:
        darwin.error(f"Failed: {failed} items")

    return [r for r in results if r is not None], images_by_doc


# ── Main pipeline handler ────────────────────────────────────────────────────

def handle_parsing(args: ParsingArgs, store: DarwinStore) -> List[ParsedDocument]:
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
                    darwin.error(f"File not found: {raw}")
                    continue
                if not p.is_file():
                    darwin.error(f"Path is not a file: {raw}")
                    continue
                if detect_type(p) is None:
                    darwin.error(f"Unsupported file type: {p.suffix} ({p.name})")
                    continue
                target_paths.append(p)
        if not target_paths and not target_urls:
            darwin.error("No valid supported files or URLs found.")
            return []
    else:
        target_dir = Path(args.target).resolve()
        if not target_dir.exists():
            darwin.error(f"Directory not found: {args.target}")
            return []
        target_paths = scan_directory(target_dir, recursive=args.recursive)

    if not target_paths and not target_urls:
        darwin.error("No supported items found")
        darwin.info("Supported: .pdf, .docx, .pptx, .txt, .md, URLs (http/https)")
        return []

    darwin.info(
        f"Settings: Images={args.images} | Caption={args.caption} | OCR={args.ocr} | "
        f"Recursive={args.recursive} | Mode={'Precise' if args.precise else 'Fast'}"
    )
    darwin.info("")

    # ── Register all local sources BEFORE cache check ─────────────────────
    for path in target_paths:
        store.register(path)

    # ── Fetch and register URLs with content-hash BEFORE cache check ──────
    from src.core.Pipeline.Ingestion.Parsing.UrlFetcher import fetch_url
    url_registered_ok = []
    for url in target_urls:
        slug = f"{url_to_slug(url)}"
        try:
            data, content_type, fetch_err = fetch_url(url, timeout=30.0)
            if fetch_err:
                darwin.error(f"Failed to fetch {url}: {fetch_err}")
                continue
            content_hash = hashlib.sha256(data).hexdigest()

            store.register_url(slug, url, content_hash=content_hash, page_count=0)
            url_registered_ok.append((slug, url, content_hash, data, content_type))
        except Exception as e:
            darwin.error(f"Failed to register URL {url}: {e}")

    # ── Build task list and check cache ─────────────────────────────────────
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

    # URLs — pass pre-fetched data tuple to avoid double-fetch
    for slug, url, content_hash, data, content_type in url_registered_ok:
        if store.has_markdown(slug) and not args.rerun:
            skipped_count += 1
            continue
        image_dir = store.ensure_image_dir(slug) if args.images else None
        tasks_to_process.append(((url, data, content_type), image_dir, slug))

    if skipped_count > 0:
        darwin.info(f"Skipped {skipped_count} cached item(s), use --rerun to reprocess")
    if not tasks_to_process:
        darwin.success("All items already parsed!")
        darwin.info("Use --rerun to reprocess with different settings")
        return []

    total_items = len(tasks_to_process)
    darwin.info(f"Processing {total_items} item(s)...")

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

            darwin.info(f"Enriching {len(all_images)} image(s) [{stages}]...")

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
                    darwin.warning(f"Image enrichment failed for {img.path.name}: {result.error}")
                    img.caption = ""
                    continue

                parts = []
                if args.caption and result.description:
                    parts.append(result.description)
                if args.ocr and result.text:
                    parts.append(f"OCR: {result.text}")

                img.caption = " | ".join(parts) if parts else ""

            darwin.success("Image enrichment complete")

    # ── Pass 3: apply enriched image text to document text ─────────────────
    for display_name, imgs in images_by_display.items():
        slot_map = {}
        for img in imgs:
            idx = img.index
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
    darwin.info(f"Total time: {elapsed:.1f}min")

    succeeded = sum(1 for d in parsed_docs if not d.error)
    failed = sum(1 for d in parsed_docs if d.error)
    if succeeded:
        darwin.success(f"Successfully parsed {succeeded} item(s)")
    if failed:
        darwin.warning(f"Failed: {failed} item(s)")

    # ── Save markdown and update registry ──────────────────────────────────
    if args.save:
        for doc in parsed_docs:
            if doc.error:
                continue

            display = doc.filename
            source = store.get_source(display)
            if not source:
                darwin.warning(f"No source found for {display}")
                continue

            content_hash = source.content_hash
            store.update_meta(content_hash, page_count=doc.page_count)

            if doc.metadata is None:
                doc.metadata = {}

            store.save_markdown(display, doc)

    return parsed_docs


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> int:
    from src.lib.Utils import ensure_writable_directory
    import argparse
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
    parser.add_argument("--sse-port", type=int, default=8765, help="WebSocket server port")
    
    parser.add_argument("--store-dir", default=None, help="Darwin store directory")
   
    args = parser.parse_args()

    if (args.caption or args.ocr) and not args.images:
        parser.error("--caption and --ocr require --images to be set")


    darwin.configure(
        
        sse_port=args.sse_port,
        
        
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

            handle_parsing(parsing_args, store)
            darwin.spinner_stop("init", success=True)

        return 0

    except KeyboardInterrupt:
        darwin.warning("Interrupted")
        return 130
    except Exception as e:
        import traceback
        darwin.error(f"Fatal error: {e}")
        darwin.error(''.join(traceback.format_exception(e)))
        return 1
    finally:
        darwin.shutdown()


if __name__ == "__main__":
    main()