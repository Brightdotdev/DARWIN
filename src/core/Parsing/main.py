"""
main.py - Document parsing pipeline entry point
"""

import sys
import logging
import multiprocessing

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

# CRITICAL: Windows multiprocessing guard
if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Parsing.Parser import parse as _parse, scan_directory
from src.core.Parsing.ParserUtils import (
    ParsedDocument,
    FileWithImages,
    CPU_COUNT,
    ParsingArgs,
    detect_type,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI
from src.core.Models import setup
from src.lib.AppPaths import default_models_dir


# ── subprocess worker ──────────────────────────────────────────────────────────

def parse_document(
    path_with_images: FileWithImages,
    caption_images: bool,
    precise: bool,
) -> ParsedDocument:
    """
    Parse one document. Runs inside a subprocess on Windows (spawn).

    FIX: On Windows, ProcessPoolExecutor uses 'spawn' — each worker is a
    fresh Python interpreter with no inherited state. setup() must be called
    here so the global _loader is initialized before any code path that calls
    get_loader() (i.e. _run_captioning inside _parse_pdf_chunked / _parse_pdf_single).

    setup() is intentionally cheap when models are already downloaded:
    it only builds the ModelLoader datastructure and prints the ✅ check lines.
    The heavy model weights are loaded lazily on first loader.get() call.
    """
    # Re-initialize models inside this subprocess.
    # skip_download=True avoids redundant network checks — models were already
    # verified in the main process before workers were spawned.
    
    try:
        setup(default_models_dir(), skip_download=True)
    except Exception as e:
        # Non-fatal: captioning will be skipped inside _run_captioning if the
        # loader is still None, but text extraction will succeed regardless.
        TerminalUI.warn(f"parse_document: models.setup() failed in worker — {e}")

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
            display_text="",
            embedding_text="",
            filename=file_path.name,
            page_count=0,
            file_type=detect_type(file_path) or "unknown",
            error=f"{type(e).__name__}: {e}",
        )


# ── main orchestrator ──────────────────────────────────────────────────────────

def parse_documents(
    files: list[FileWithImages],
    caption_images: bool = False,
    precise: bool = False,
) -> list[ParsedDocument]:

    if not files:
        return []

    file_names = [Path(fw.file).name for fw in files]
    total_mb   = sum(Path(fw.file).stat().st_size / (1024 * 1024) for fw in files)
    mode_label = "precise" if precise else "fast"

    TerminalUI.dim(
        f"Queueing {len(files)} files ({total_mb:.1f} MB) [{mode_label} mode]"
    )

    indexed_files = sorted(
        enumerate(files),
        key=lambda x: Path(x[1].file).stat().st_size,
    )

    n_workers = min(
        len(files),
        max(1, CPU_COUNT // 2) if caption_images else CPU_COUNT,
    )

    results: list[Optional[ParsedDocument]] = [None] * len(files)
    completed_files: set[str] = set()
    file_errors:     dict[str, str] = {}

    try:
        with TerminalUI.multi_parse_progress(file_names) as get_callback:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:

                future_to_info: dict = {}
                for orig_idx, fw in indexed_files:
                    filename = Path(fw.file).name
                    future   = executor.submit(
                        parse_document,
                        fw, caption_images, precise,
                    )
                    future_to_info[future] = (orig_idx, filename)

                total_submitted = len(future_to_info)
                processed_count = 0

                TerminalUI.dim(
                    f"\nProcessing {total_submitted} files with {n_workers} workers…"
                )

                for future in as_completed(future_to_info):
                    orig_idx, filename = future_to_info[future]
                    update_progress    = get_callback(filename)

                    try:
                        doc = future.result()
                        results[orig_idx] = doc
                        processed_count  += 1

                        if doc.error:
                            file_errors[filename] = doc.error
                            update_progress(f" {doc.error}")
                        else:
                            completed_files.add(filename)
                            update_progress(
                                f"✓ {doc.page_count} pages · {len(doc.embedding_text):,} chars"
                            )

                    except Exception as e:
                        results[orig_idx] = ParsedDocument(
                            display_text="",
                            embedding_text="",
                            filename=filename,
                            page_count=0,
                            file_type="unknown",
                            error=f"{type(e).__name__}: {e}",
                        )
                        file_errors[filename] = str(e)
                        update_progress(f" {type(e).__name__}")
                        processed_count += 1

                TerminalUI.success(f"\n✨ All {processed_count} files processed!")

    except KeyboardInterrupt:
        TerminalUI.blank()
        TerminalUI.dim("Interrupted — exiting.")
        raise

    TerminalUI.blank()

    if completed_files:
        TerminalUI.success(f" Parsed: {len(completed_files)} files")
    if file_errors:
        TerminalUI.error(f" Failed: {len(file_errors)} files")
        for name, err in list(file_errors.items())[:5]:
            TerminalUI.dim(f"  • {name}: {err}")
        if len(file_errors) > 5:
            TerminalUI.dim(f"  … and {len(file_errors) - 5} more")

    return [r for r in results if r is not None]


# ── handle_parsing ─────────────────────────────────────────────────────────────

def handle_parsing(args: ParsingArgs) -> list[ParsedDocument]:
    """Main parsing handler."""

    TerminalUI.print_darwin()
    TerminalUI.blank()
    TerminalUI.bold(f" Parsing {args.target}")
    TerminalUI.dim(
        f"Settings: Images={args.images} | Caption={args.caption} "
        f"| Recursive={args.recursive} | Mode={'Precise' if args.precise else 'Fast'}"
    )
    TerminalUI.blank()
    target_paths: list[Path] = []

    if args.file:
        target_paths = scan_directory(args.file, recursive=False)
    else:
        target_paths = scan_directory(args.target, recursive=args.recursive)

    if not target_paths:
        TerminalUI.error(f"No supported files found in {args.target}")
        TerminalUI.dim("Exiting...")
        return []
    

    setup(default_models_dir())

    store = DarwinStore()
    for path in target_paths:
        store.registry.register(path)


    to_process    = []
    skipped_count = 0
    for path in target_paths:
        if store.has_markdown(path.name) and not args.rerun:
            skipped_count += 1
            continue
        to_process.append(path)

    if skipped_count > 0:
        TerminalUI.dim(f"Skipped {skipped_count} cached files (use --rerun to reprocess)")

    if not to_process:
        TerminalUI.success(" All files already parsed!")
        TerminalUI.dim("Use --rerun to reprocess with different settings")
        return []

    TerminalUI.bold(f"\n Processing {len(to_process)} file(s)...")

    files = [
        FileWithImages(
            file=path,
            image_dir=store.ensure_image_dir(path.name) if args.images else None,
        )
        for path in to_process
    ]
    TerminalUI.bold("Parsing files....this might take a while and the steps will be shown in the logs...try not to abrupt the process by closing the terminal or interrupting the process...")
    docs = parse_documents(
        files,
        caption_images=args.caption,
        precise=args.precise,
    )

    succeeded = failed = 0
   
    for doc in docs:
        if not doc:
            continue
        if doc.error:
            failed += 1
            TerminalUI.error(f" {doc.filename} - {doc.error}")
        else:
            if args.save:
                # Re-register with page count now that we know it
                matching_path = next(
                    (p for p in to_process if p.name == doc.filename), None
                )
                if matching_path:
                    store.registry.register(matching_path, page_count=doc.page_count)
                store.save_markdown(doc.filename, doc)
            succeeded += 1


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse documents into .darwin format",
    )
    parser.add_argument("path",        nargs="?", default=".",    help="File or directory to parse")
    parser.add_argument("--rerun",     "-R",  action="store_true", help="Reprocess cached files")
    parser.add_argument("--recursive", "-r",  action="store_true", help="Scan directories recursively")
    parser.add_argument("--images",    "-img", action="store_true", help="Extract images")
    parser.add_argument("--caption",   "-c",  action="store_true", help="Generate captions (requires --images)")
    parser.add_argument("--file",      "-f",  default=None,        help="Parse a single file instead of a directory")
    parser.add_argument("--precise",          action="store_true", help="Use pymupdf4llm for layout-accurate extraction")
    parser.add_argument("--no-save",          action="store_true", help="Don't save markdown files (preview only)")
    args = parser.parse_args()

    try:
        handle_parsing(ParsingArgs(
            target=args.path,
            images=args.images,
            caption=args.caption,
            recursive=args.recursive,
            rerun=args.rerun,
            file=args.file,
            precise=args.precise,
            save=not args.no_save,
        ))
        return 0
    except KeyboardInterrupt:
        TerminalUI.blank()
        TerminalUI.dim("\nGoodbye!")
        return 130
    except Exception as e:
        import traceback
        TerminalUI.error(f"\nFatal error: {e}")
        tb_str = ''.join(traceback.format_exception(e))
        TerminalUI.text(tb_str)
        return 1


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    sys.exit(main())