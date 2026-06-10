# main.py — Document parsing pipeline
# Extraction is parallel; captioning runs once in the main process afterward.

from __future__ import annotations

import sys
import multiprocessing
import time
import asyncio
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.Ingestion.Parsing.Parser import (
    parse as _parse,
    scan_directory,
    is_url,
)
from src.features.Ingestion.Parsing.BlobHandler import BlobHandler
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
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity


# ── Worker initializer (shared logging queue) ────────────────────────────────

def _worker_init(mp_queue):
    """Initialize worker process with the shared logging queue."""
    from src.core.Logger.DarwinLogger import darwin
    darwin.set_mp_queue(mp_queue)


# ── Worker function (NO CAPTIONING) ─────────────────────────────────────────

def parse_document_worker(
    path_with_images: FileWithImages,
    precise: bool,
) -> ParsedDocument:
    """
    Parse a single file or URL. Captioning is disabled.
    Returns a ParsedDocument that may contain a list of extracted images.
    """
    file_path = path_with_images.file
    try:
        # Force caption=False inside the parser
        # file_path can now be a Path (local) or str (URL)
        doc = _parse(
            file_path,
            images_dir=path_with_images.image_dir,
            caption=False,
            precise=precise,
        )
        return doc
    except Exception as e:
        # For URLs, file_path is already a string; for local files, convert
        filename = file_path if isinstance(file_path, str) else Path(file_path).name
        file_type = "url" if isinstance(file_path, str) else (detect_type(Path(file_path)) or "unknown")
        return ParsedDocument(
            display_text="", embedding_text="", filename=filename,
            page_count=0, file_type=file_type, error=f"{type(e).__name__}: {e}",
        )


# ── Parallel document parser ─────────────────────────────────────────────────

def parse_documents(
    files: List[FileWithImages],
    precise: bool = False,
) -> Tuple[List[ParsedDocument], Dict[str, List[ExtractedImage]]]:
    """
    Extract text and images from multiple files/URLs in parallel.
    Returns:
        - List of ParsedDocument (without captions)
        - Dict mapping original filename/URL -> list of ExtractedImage objects
    """
    if not files:
        return [], {}

    total = len(files)
    mode_label = "precise" if precise else "fast"
    darwin.log(f"Queuing {total} items ({mode_label})", LogType.INFO)

    # Sort by file size (smallest first) for better progress visibility.
    # For URLs we can't get size cheaply, so they sort to the end by default.
    def _sort_key(item):
        fw = item[1]
        if isinstance(fw.file, str) and is_url(fw.file):
            return float("inf")  # URLs last
        return Path(fw.file).stat().st_size

    indexed_files = sorted(enumerate(files), key=_sort_key)
    n_workers = min(total, CPU_COUNT)

    results: List[Optional[ParsedDocument]] = [None] * total
    completed_files: set[str] = set()
    file_errors: Dict[str, str] = {}
    images_by_file: Dict[str, List[ExtractedImage]] = {}

    darwin.progress_start("parsing", total, f"Parsing {total} items")

    # Start the multiprocessing log listener once for the whole app
    darwin.start_mp_listener()
    mp_queue = darwin.get_mp_queue()

    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(mp_queue,),
        ) as executor:
            future_to_info = {}
            for orig_idx, fw in indexed_files:
                future = executor.submit(parse_document_worker, fw, precise)
                # Use URL string or filename as the key
                label = fw.file if isinstance(fw.file, str) else Path(fw.file).name
                future_to_info[future] = (orig_idx, label)

            processed = 0
            last_update = 0
            UPDATE_INTERVAL = 5

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
                        # Collect extracted images for later captioning
                        if doc.images:
                            images_by_file[filename] = doc.images
                except Exception as e:
                    results[orig_idx] = ParsedDocument(
                        display_text="", embedding_text="", filename=filename,
                        page_count=0, file_type="unknown", error=str(e),
                    )
                    file_errors[filename] = str(e)

                if processed - last_update >= UPDATE_INTERVAL or processed == total:
                    darwin.progress_update("parsing", processed)
                    last_update = processed
    finally:
        darwin.progress_end("parsing")
        darwin.stop_mp_listener()

    # Summary
    if completed_files:
        darwin.log(f"Parsed: {len(completed_files)} items", LogType.SUCCESS)
    if file_errors:
        darwin.log(f"Failed: {len(file_errors)} items", LogType.ERROR)
        for name, err in list(file_errors.items())[:5]:
            darwin.log(f"  • {name}: {err}", LogType.ERROR)
        if len(file_errors) > 5:
            darwin.log(f"  ... and {len(file_errors) - 5} more", LogType.ERROR)

    return [r for r in results if r is not None], images_by_file


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
                    darwin.log(f"file not found: {raw}", LogType.ERROR)
                    continue
                if not p.is_file():
                    darwin.log(f"file provided is not a file: {raw}", LogType.ERROR)
                    continue
                if detect_type(p) is None:
                    darwin.log(f"Unsupported file type: {p.suffix} ({p.name})", LogType.ERROR)
                    continue
                target_paths.append(p)
        if not target_paths and not target_urls and not args.blobs:
            darwin.log("No valid supported files, URLs, or blobs found.", LogType.ERROR)
            return []
    else:
        target_dir = Path(args.target).resolve()
        if not target_dir.exists():
            darwin.log(f"Directory not found: {args.target}", LogType.ERROR)
            return []
        target_paths = scan_directory(target_dir, recursive=args.recursive)

    if not target_paths and not target_urls and not args.blobs:
        darwin.log(f"No supported items found", LogType.ERROR)
        darwin.log("Supported: .pdf, .docx, .pptx, .txt, .md, URLs (http/https), or raw text blobs", LogType.INFO)
        return []

    darwin.log(
        f"Settings: Images={args.images} | Caption={args.caption} | "
        f"Recursive={args.recursive} | Mode={'Precise' if args.precise else 'Fast'}",
        LogType.INFO,
    )
    darwin.log("")

    # Register local files
    for path in target_paths:
        darwin.log(f"Registering file: {path.name}")
        store.register(path)

    # Register URLs (no store.register needed, but we log them)
    for url in target_urls:
        darwin.log(f"Registering URL: {url}")

    # Register blobs
    if args.blobs:
        for blob in args.blobs:
            darwin.log(f"Registering blob: {blob.filename}")

    # Skip already parsed local files unless --rerun
    to_process_local = []
    skipped_count = 0
    for path in target_paths:
        if store.has_markdown(path.name) and not args.rerun:
            skipped_count += 1
            continue
        to_process_local.append(path)

    # URLs are never "cached" in the store (for now)
    to_process_urls = target_urls

    # Blobs are never cached
    to_process_blobs = args.blobs

    if skipped_count > 0:
        darwin.log(f"Skipped {skipped_count} cached files (use --rerun to reprocess)", LogType.INFO)
    if not to_process_local and not to_process_urls and not to_process_blobs:
        darwin.log("All items already parsed!", LogType.SUCCESS)
        darwin.log("Use --rerun to reprocess with different settings", LogType.INFO)
        return []

    total_items = len(to_process_local) + len(to_process_urls) + len(to_process_blobs)
    darwin.log(f"Processing {total_items} item(s)...", LogType.INFO)

    # Build FileWithImages for both local files and URLs
    files = []
    for path in to_process_local:
        files.append(FileWithImages(
            file=path,
            image_dir=store.ensure_image_dir(path.name) if args.images else None,
        ))
    for url in to_process_urls:
        # URLs don't have local image dirs (images extracted from HTML go to a generic dir)
        files.append(FileWithImages(
            file=url,
            image_dir=store.ensure_image_dir("url_" + url.replace("://", "_").replace("/", "_")) if args.images else None,
        ))

    t0 = time.perf_counter()

    # ── Pass 1: parallel extraction for files/URLs (no captioning) ───────────
    parsed_docs, images_by_file = parse_documents(files, precise=args.precise)

    # ── Pass 1b: blob processing via BlobHandler (main process, immediate) ──
    blob_docs: List[ParsedDocument] = []
    if to_process_blobs:
        for blob in to_process_blobs:
            darwin.log(f"Processing blob: {blob.filename}...", LogType.INFO)
            doc = BlobHandler.process(blob)
            blob_docs.append(doc)
            if doc.error:
                darwin.log(f"Blob {blob.filename} failed: {doc.error}", LogType.WARNING)
            else:
                darwin.log(f"Blob {blob.filename}: {len(doc.embedding_text)} chars", LogType.SUCCESS)

    # Merge all parsed results: blobs first (user-facing), then files/URLs
    all_parsed_docs = blob_docs + parsed_docs

    # ── Pass 2: captioning (main process, once) ──────────────────────────
    if args.caption and args.images:
        all_images: List[ExtractedImage] = []
        for fname, imgs in images_by_file.items():
            all_images.extend(imgs)

        if all_images:
            darwin.log(f"Captioning {len(all_images)} images across {len(images_by_file)} items...", LogType.INFO)
            try:
                from src.core.Models import ModelManager, ModelCategory
                from src.features.Ingestion.Parsing.Captioning import caption_image, load_caption_cache, save_caption_cache
                from src.features.Ingestion.Parsing.lib import prepare_image, MAX_NEW_TOKENS

                loader = ModelManager.get_loader()
                vision_key = loader.get_model_key(ModelCategory.VISION)

                if vision_key == "not set":
                    darwin.log("No vision model configured. Skipping captioning.", LogType.WARNING)
                else:
                    darwin.log(f"Loading vision model '{vision_key}'...", LogType.INFO)
                    loaded = loader.get(ModelCategory.VISION)
                    if loaded is None:
                        darwin.log("Failed to load vision model", LogType.ERROR)
                        darwin.log("The operation will continue without vision")
                        darwin.log("[HELP]  try setting up your vision model / downloading one locally or managing your defaults")
                        return None

                    is_api_model = hasattr(loaded, 'caption')
                    processor = model = device = None

                    if is_api_model:
                        model_wrapper = loaded
                    else:
                        processor, model, device = loaded

                    # Caption each image, caching results
                    for img in all_images:
                        img_dir = img.path.parent
                        cache = load_caption_cache(img_dir)
                        if img.path.name in cache:
                            img.caption = cache[img.path.name]
                            continue
                        try:
                            if is_api_model:
                                img.caption = model_wrapper.caption(img.path, max_new_tokens=MAX_NEW_TOKENS)
                            else:
                                image = prepare_image(img.path)
                                img.caption = caption_image(
                                    img.path, processor, model, device, cache,
                                    preloaded_image=image, max_new_tokens=MAX_NEW_TOKENS
                                )
                            if img.caption:
                                cache[img.path.name] = img.caption
                                save_caption_cache(img_dir, cache)
                        except Exception as e:
                            darwin.log(f"Caption failed for {img.path.name}: {e}", LogType.WARNING)
                            img.caption = ""   # leave empty

                    # Unload model
                    try:
                        loader.unload(ModelCategory.VISION)
                        darwin.log("Vision model unloaded from memory", LogType.INFO)
                    except Exception:
                        pass

            except Exception as e:
                darwin.log(f"Captioning phase failed: {e}. Proceeding without captions.", LogType.ERROR)
                # Swallow the error — images will just show [Image] placeholders

    # ── Pass 3: apply captions to document text ──────────────────────────
    # Build a slot map per file using the now-captioned images
    for fname, imgs in images_by_file.items():
        slot_map = {}
        for img in imgs:
            idx = img.index
            rel_path = normalize_image_path(img.path, store.ensure_image_dir(fname) if args.images else None)
            slot_map[idx] = {
                "caption": img.caption,
                "rel_path": rel_path,
            }
        # Find the corresponding ParsedDocument and replace slots
        for doc in all_parsed_docs:
            if doc.filename == fname:
                doc.display_text, doc.embedding_text = replace_image_slots(
                    doc.display_text, doc.embedding_text, slot_map
                )
                break

    elapsed = (time.perf_counter() - t0) / 60
    darwin.log(f"Total time: {elapsed:.1f}min", LogType.INFO)

    succeeded = sum(1 for d in all_parsed_docs if not d.error)
    failed = sum(1 for d in all_parsed_docs if d.error)
    if succeeded:
        darwin.log(f"Successfully parsed {succeeded} item(s)", LogType.SUCCESS)
    if failed:
        darwin.log(f"Failed: {failed} item(s)", LogType.WARNING)

    # Save markdown (if required)
    if args.save:
        for doc in all_parsed_docs:
            if not doc.error:
                # For local files, use the original path
                if not is_url(doc.filename):
                    matching_path = next((p for p in to_process_local if p.name == doc.filename), None)
                    if matching_path:
                        store.register(matching_path, page_count=doc.page_count)
                # For URLs, register with a synthetic key
                elif doc.filename.startswith("http://") or doc.filename.startswith("https://"):
                    store.register_url(doc.filename, page_count=doc.page_count)
                # For blobs, register with blob filename
                else:
                    store.register_blob(doc.filename, page_count=doc.page_count)
                store.save_markdown(doc.filename, doc)

    return all_parsed_docs


# ── CLI entry point ──────────────────────────────────────────────────────────

async def main() -> int:
    from src.lib.Utils import ensure_writable_directory
    parser = argparse.ArgumentParser(description="Parse documents into .darwin format")
    parser.add_argument("path", nargs="?", default=".", help="File or directory to parse")
    parser.add_argument("--rerun", "-R", action="store_true", help="Reprocess cached files")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan directories recursively")
    parser.add_argument("--images", "-img", action="store_true", help="Extract images from documents")
    parser.add_argument("--caption", "-c", action="store_true", help="Generate captions for images (requires --images)")
    parser.add_argument("--file", "-f", nargs="+", default=None, help="Parse specific file(s) or URL(s)")
    parser.add_argument("--precise", action="store_true", help="Use precise extraction (layout-aware for PDFs, markdown output for HTML)")
    parser.add_argument("--no-save", action="store_true", help="Don't save markdown files (preview only)")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")
    parser.add_argument("--store-dir", default=None, help="Darwin store directory")
    parser.add_argument("--verbosity", choices=["normal", "quiet"], default="normal", help="Log verbosity level")
    args = parser.parse_args()

    # Validate dependencies
    if args.caption and not args.images:
        parser.error("--caption requires --images to be set")

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
                recursive=args.recursive,
                rerun=args.rerun,
                files=args.file,
                precise=args.precise,
                save=not args.no_save,
                blobs=[],  # CLI doesn't support blobs directly; use API
            )

            if args.store_dir and ensure_writable_directory(args.store_dir):
                store = DarwinStore(args.store_dir)
            else:
                store = DarwinStore()

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
        darwin.shutdown()


if __name__ == "__main__":
    asyncio.run(main())