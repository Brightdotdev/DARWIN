"""
src/core/Parsing/main.py - Document parsing pipeline entry point

Windows multiprocessing note
-----------------------------
1. Never nest ProcessPoolExecutors — parse_document() is a plain function
   called inside the worker, not a launcher of more workers.
2. The if __name__ == "__main__" guard at the bottom is mandatory.
3. Never pass callbacks/closures to worker processes — they can't be pickled.
"""

import sys
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import Optional

from src.core.Storage.StoreUtils import sanitize

if __name__ == "__main__":
    multiprocessing.freeze_support()

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Parsing.Parser import parse as _parse, scan_directory
from src.core.Parsing.ParserUtils import (
    ParsedDocument,
    FileWithImages,
    CPU_COUNT,
    ParsingArgs,
    PARSE_TIMEOUT_SECONDS,
    detect_type,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI
from src.lib.Utils import open_in_explorer




def progress_callback(msg: str) -> None:
    TerminalUI.dim(msg)
# ── Single document parse (runs INSIDE a worker process) ──────────────────────

def parse_document(
    path_with_images: FileWithImages,
    caption_images:   bool = False,
    project_dir:      Optional[Path] = None,
) -> ParsedDocument:
    """
    Parse one document. Runs inside a subprocess.
    No callbacks — closures can't be pickled on Windows.
    """
    file_path = Path(path_with_images.file)

    try:
        return _parse(
            file_path,
            images_dir        = path_with_images.image_dir,
            caption           = caption_images,
            project_dir       = project_dir,
            progress_callback = progress_callback,
        )
    except Exception as e:
        return ParsedDocument(
            display_text   = "",
            embedding_text = "",
            filename       = file_path.name,
            page_count     = 0,
            file_type      = detect_type(file_path) or "unknown",
            error          = f"{type(e).__name__}: {e}",
        )


# ── Parallel document parsing ──────────────────────────────────────────────────

def parse_documents(
    files:          list[FileWithImages],
    caption_images: bool = False,
    timeout:        int  = PARSE_TIMEOUT_SECONDS,
    project_dir:    Optional[Path] = None,
) -> list[ParsedDocument]:
    """
    Parse a list of documents in parallel using a single ProcessPoolExecutor.
    Logs completion of each file as it finishes.
    """
    if not files:
        return []

    file_names = [Path(fw.file).name for fw in files]
    if len(file_names) <= 10:
        TerminalUI.dim(f"Working on: {', '.join(file_names)}")
    else:
        TerminalUI.dim(f"Working on: {', '.join(file_names[:10])} ...and {len(file_names) - 10} more")

    total_mb = sum(Path(fw.file).stat().st_size / (1024 * 1024) for fw in files)
    TerminalUI.dim(f"Queueing {len(files)} files ({total_mb:.1f} MB total)")

    indexed   = sorted(enumerate(files), key=lambda x: Path(x[1].file).stat().st_size)
    n_workers = min(
        len(files),
        max(1, CPU_COUNT // 2) if caption_images else CPU_COUNT,
    )

    results: list[Optional[ParsedDocument]] = [None] * len(files)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(parse_document, fw, caption_images, project_dir): orig_idx
            for orig_idx, fw in indexed
        }
        for future in as_completed(future_to_idx):
            orig_idx = future_to_idx[future]
            fw       = files[orig_idx]
            filename = Path(fw.file).name

            try:
                doc = future.result(timeout=timeout)
                results[orig_idx] = doc
                if doc.error:
                    TerminalUI.error(f"{filename} — {doc.error}")
                else:
                    TerminalUI.success(
                        f"{filename} — "
                        f"{doc.page_count} pages · "
                        f"{len(doc.embedding_text):,} chars"
                    )
            except TimeoutError:
                results[orig_idx] = ParsedDocument(
                    display_text="", embedding_text="",
                    filename=filename, page_count=0,
                    file_type=detect_type(Path(fw.file)) or "unknown",
                    error=f"timed out after {timeout}s",
                )
                TerminalUI.error(f"{filename} — timed out after {timeout}s")
            except Exception as e:
                results[orig_idx] = ParsedDocument(
                    display_text="", embedding_text="",
                    filename=filename, page_count=0,
                    file_type="unknown",
                    error=f"worker failed: {type(e).__name__}: {e}",
                )
                TerminalUI.error(f"{filename} — {type(e).__name__}: {e}")

    return results  # type: ignore


# ── High-level handler ─────────────────────────────────────────────────────────

def handle_parsing(args: ParsingArgs) -> None:
    if args.file:
        target_paths = scan_directory(args.file, recursive=False)
    else:
        target_paths = scan_directory(args.target, recursive=args.recursive)

    if not target_paths:
        TerminalUI.error(f"No supported files found in {args.target}")
        return

    store = DarwinStore(Path(args.project_dir) if args.project_dir else Path.cwd())

    # Filter cached files unless --rerun
    to_process = []
    for path in target_paths:
        if store.has_markdown(path.name) and not args.rerun:
            TerminalUI.dim(f"skipping {path.name} (cached)")
            continue
        to_process.append(path)

    if not to_process:
        TerminalUI.success("all files already parsed")
        TerminalUI.dim("use --rerun to reprocess with different settings")
        return

    # Single cached file shortcut
    if len(to_process) == 1 and not args.rerun:
        path = to_process[0]
        if store.has_markdown(path.name):
            TerminalUI.bold("Skipping (cached markdown found)")
            TerminalUI.dim("Opening existing markdown in explorer...")
            open_in_explorer(store._file_dir(path.name) / f"{sanitize(path.name)}.md")
            return

    TerminalUI.bold(f"Parsing {len(to_process)} file(s)...")

    files = [
        FileWithImages(
            file      = path,
            image_dir = store.ensure_image_dir(path.name) if args.images else None,
        )
        for path in to_process
    ]

    docs = parse_documents(
        files,
        caption_images = args.caption,
        project_dir    = store.project_dir,
    )

    # Persist
    succeeded = failed = 0
    for doc in docs:
        if not doc:
            continue
        if doc.error:
            failed += 1
        else:
            store.save_markdown(doc.filename, doc, rerun=args.rerun)
            succeeded += 1

    TerminalUI.blank()
    TerminalUI.success(f"done — {succeeded} parsed · {failed} failed")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Parse documents into .darwin")
    parser.add_argument("path",        nargs="?", default=".",    help="File or directory to parse")
    parser.add_argument("--rerun",     "-R", action="store_true", help="Reprocess cached files")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan directories recursively")
    parser.add_argument("--images",    "-i", action="store_true", help="Extract images")
    parser.add_argument("--caption",   "-c", action="store_true", help="Generate captions (requires --images)")
    parser.add_argument("--project",   "-p", default=None,        help="Project directory (default: cwd)")
    parser.add_argument("--file",      "-f", default=None,        help="Parse a single file instead of a directory")

    args = parser.parse_args()

    handle_parsing(ParsingArgs(
        target      = args.path,
        project_dir = args.project,
        images      = args.images,
        caption     = args.caption,
        recursive   = args.recursive,
        rerun       = args.rerun,
        file        = args.file,
    ))

    return 0


if __name__ == "__main__":
    sys.exit(main())