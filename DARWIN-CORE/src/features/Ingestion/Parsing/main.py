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

from src.features.Ingestion.Parsing.Parser import parse as _parse, scan_directory
from src.features.Ingestion.Parsing.lib import (
    ParsedDocument,
    FileWithImages,
    CPU_COUNT,
    ParsingArgs,
    detect_type,
    ExtractedImage,
)
from src.features.Ingestion.Parsing.ParserUtils import (
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


# ── Worker function (NO CAPTIONING) ───────────────────────────────────────────

def parse_document_worker(
    path_with_images: FileWithImages,
    precise: bool,
) -> ParsedDocument:
    """
    Parse a single file. Captioning is disabled.
    Returns a ParsedDocument that may contain a list of extracted images.
    """
    file_path = Path(path_with_images.file)
    try:
        # Force caption=False inside the parser
        doc = _parse(
            file_path,
            images_dir=path_with_images.image_dir,
            caption=False,
            precise=precise,
        )
        return doc
    except Exception as e:
        return ParsedDocument(
            display_text="", embedding_text="", filename=file_path.name,
            page_count=0, file_type=detect_type(file_path) or "unknown",
            error=f"{type(e).__name__}: {e}",
        )


# ── Parallel document parser ─────────────────────────────────────────────────

def parse_documents(
    files: List[FileWithImages],
    precise: bool = False,
) -> Tuple[List[ParsedDocument], Dict[str, List[ExtractedImage]]]:
    """
    Extract text and images from multiple files in parallel.
    Returns:
        - List of ParsedDocument (without captions)
        - Dict mapping original filename -> list of ExtractedImage objects
    """
    if not files:
        return [], {}

    total = len(files)
    mode_label = "precise" if precise else "fast"
    darwin.log(f"Queuing {total} files ({mode_label})", LogType.INFO)

    # Sort by file size (smallest first) for better progress visibility
    indexed_files = sorted(enumerate(files), key=lambda x: Path(x[1].file).stat().st_size)
    n_workers = min(total, CPU_COUNT)

    results: List[Optional[ParsedDocument]] = [None] * total
    completed_files: set[str] = set()
    file_errors: Dict[str, str] = {}
    images_by_file: Dict[str, List[ExtractedImage]] = {}

    darwin.progress_start("parsing", total, f"Parsing {total} files")

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
                future_to_info[future] = (orig_idx, Path(fw.file).name)

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
        darwin.log(f"Parsed: {len(completed_files)} files", LogType.SUCCESS)
    if file_errors:
        darwin.log(f"Failed: {len(file_errors)} files", LogType.ERROR)
        for name, err in list(file_errors.items())[:5]:
            darwin.log(f"  • {name}: {err}", LogType.ERROR)
        if len(file_errors) > 5:
            darwin.log(f"  … and {len(file_errors) - 5} more", LogType.ERROR)

    return [r for r in results if r is not None], images_by_file


# ── Main pipeline handler ─────────────────────────────────────────────────────

def handle_parsing(args: ParsingArgs, store: DarwinStore) -> List[ParsedDocument]:
    target_paths: List[Path] = []

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

    darwin.log(
        f"Settings: Images={args.images} | Caption={args.caption} | "
        f"Recursive={args.recursive} | Mode={'Precise' if args.precise else 'Fast'}",
        LogType.INFO,
    )
    darwin.log("")

    # Register all files
    for path in target_paths:
        darwin.log(f"Registering file: {path.name}")
        store.register(path)

    # Skip already parsed files unless --rerun
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

    files = [
        FileWithImages(
            file=path,
            image_dir=store.ensure_image_dir(path.name) if args.images else None,
        )
        for path in to_process
    ]

    t0 = time.perf_counter()

    # ── Pass 1: parallel extraction (no captioning) ────────────────────
    parsed_docs, images_by_file = parse_documents(files, precise=args.precise)

    # ── Pass 2: captioning (main process, once) ────────────────────────
    if args.caption and args.images:
        all_images: List[ExtractedImage] = []
        for fname, imgs in images_by_file.items():
            all_images.extend(imgs)

        if all_images:
            darwin.log(f"Captioning {len(all_images)} images across {len(images_by_file)} files...", LogType.INFO)
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
                        darwin.log("The operation will contiune without vision")
                        darwin.log("[HELP]  try setting up your vision model / dowloading one locally or managing your defaults")
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

    # ── Pass 3: apply captions to document text ────────────────────────
    # Build a slot map per file using the now-captioned images
    for fname, imgs in images_by_file.items():
        slot_map = {}
        for img in imgs:
            # Determine slot index from img.index (or img_dict 'slot')
            idx = img.index
            rel_path = normalize_image_path(img.path, store.ensure_image_dir(fname) if args.images else None)
            slot_map[idx] = {
                "caption": img.caption,
                "rel_path": rel_path,
            }
        # Find the corresponding ParsedDocument and replace slots
        for doc in parsed_docs:
            if doc.filename == fname:
                doc.display_text, doc.embedding_text = replace_image_slots(
                    doc.display_text, doc.embedding_text, slot_map
                )
                break

    elapsed = (time.perf_counter() - t0) / 60
    darwin.log(f"Total time: {elapsed:.1f}min", LogType.INFO)

    succeeded = sum(1 for d in parsed_docs if not d.error)
    failed = sum(1 for d in parsed_docs if d.error)
    if succeeded:
        darwin.log(f"Successfully parsed {succeeded} file(s)", LogType.SUCCESS)
    if failed:
        darwin.log(f"Failed: {failed} file(s)", LogType.WARNING)

    # Save markdown (if required)
    if args.save:
        for doc in parsed_docs:
            if not doc.error:
                matching_path = next((p for p in to_process if p.name == doc.filename), None)
                if matching_path:
                    store.register(matching_path, page_count=doc.page_count)
                store.save_markdown(doc.filename, doc)

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
    parser.add_argument("--file", "-f", nargs="+", default=None, help="Parse specific file(s)")
    parser.add_argument("--precise", action="store_true", help="Use pymupdf4llm for layout-accurate extraction")
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
                target=args.path if not args.file else str(Path(args.file[0]).parent),
                images=args.images,
                caption=args.caption,
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