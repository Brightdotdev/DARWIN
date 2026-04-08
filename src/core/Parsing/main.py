"""
src/core/parser.py - Document parser integrated with DarwinStore
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError


from src.core.Storage.StoreUtils import sanitize

# Add parent to path if needed
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.Parsing.Parser import (parse as _parse ,scan_directory)
from src.core.Parsing.ParserUtils import (ParsedDocument, FileWithImages,CPU_COUNT,ParsingArgs,PARSE_TIMEOUT_SECONDS,detect_type)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI
from src.lib.Utils import open_in_explorer


def parse_document(
    path_with_images: FileWithImages,
    caption_images: bool = False,
    timeout: int = PARSE_TIMEOUT_SECONDS,
) -> ParsedDocument:
    """
    Parse a document with a hard timeout.
    If parsing exceeds timeout seconds the worker is killed
    and an error ParsedDocument is returned.
    """
    file_path = Path(path_with_images.file)
    image_dir = path_with_images.image_dir
    filename  = file_path.name

    # run the actual parse in a subprocess with a hard timeout
    # this works on Windows unlike signal-based approaches
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _parse_document_worker,
            path_with_images,
            caption_images,
        )
        try:
            doc = future.result(timeout=timeout)
        except TimeoutError:
            # kill the worker process
            executor.shutdown(wait=False, cancel_futures=True)
            return ParsedDocument(
                text       = "",
                filename   = filename,
                page_count = 0,
                file_type  = detect_type(file_path) or "unknown",
                strategy   = "light",
                error      = f"parsing timed out after {timeout}s — file may be corrupted or too complex",
            )
        except Exception as e:
            return ParsedDocument(
                text       = "",
                filename   = filename,
                page_count = 0,
                file_type  = detect_type(file_path) or "unknown",
                strategy   = "light",
                error      = f"{type(e).__name__}: {e}",
            )

    return doc


def _parse_document_worker(
    path_with_images: FileWithImages,
    caption_images: bool,
) -> ParsedDocument:
    """
    The actual parse logic — runs inside a subprocess.
    Separated so parse_document can enforce a timeout on it.
    """
    file_path = Path(path_with_images.file)
    image_dir = path_with_images.image_dir
    filename  = file_path.name

    doc = _parse(
        file_path,
        images_dir = image_dir,
        caption    = caption_images,
    )

    if doc.error:
        return doc

    return doc

def parse_documents(
    files:          list[FileWithImages],
    caption_images: bool = False,
    timeout:        int  = PARSE_TIMEOUT_SECONDS,
) -> list[ParsedDocument]:

    if not files:
        return []

    # 1. Grab the names safely from the dataclass attributes
    file_names = [fw.file.name for fw in files]
    
    # 2. UX: List the files (with a cap so we don't scroll-bomb the user)
    if len(file_names) <= 10:
        names_str = ", ".join(file_names)
    else:
        names_str = ", ".join(file_names[:10]) + f" ...and {len(file_names) - 10} more"
    
    TerminalUI.dim(f"Working on: {names_str}")

    # 3. Size Calculation
    total_mb = sum([fw.file.stat().st_size / (1024 * 1024) for fw in files])
    TerminalUI.dim(f"Queueing {len(files)} files ({total_mb:.1f} MB total)")

    indexed   = sorted(enumerate(files), key=lambda x: x[1].file.stat().st_size)

    
    n_workers = min(
        len(files),
        max(1, CPU_COUNT // 2) if caption_images else CPU_COUNT,
    )

    results: list[ParsedDocument | None] = [None] * len(files)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(
                parse_document, fw, caption_images, timeout
            ): original_idx
            for original_idx, fw in indexed
        }

        for future in as_completed(future_to_idx):
            original_idx = future_to_idx[future]
            fw           = files[original_idx]

            try:
                # outer timeout is timeout + 30 to give parse_document
                # time to handle its own timeout gracefully first
                results[original_idx] = future.result(timeout=timeout + 30)
            except Exception as e:
                results[original_idx] = ParsedDocument(
                    text       = "",
                    filename   = Path(fw.file).name,
                    page_count = 0,
                    file_type  = "unknown",
                    strategy   = "light",
                    error      = f"worker failed: {type(e).__name__}: {e}",
                )

    return results  # type: ignore


def handle_parsing(args: ParsingArgs):
    targetPath = None
    if args.file:
        targetPath = scan_directory(args.file, recursive=False)
    else:
        targetPath = scan_directory(args.target, recursive=args.recursive)
        

    if not targetPath:
        TerminalUI.error(f"No supported files found in {args.target}")
        return

    store = DarwinStore(Path(args.projectDir) if args.projectDir else Path.cwd())

    # resolve image dirs per file
    paths_with_images = [
        (path, store.ensure_image_dir(path.name) if args.images else None)
        for path in targetPath
    ]

    # filter cached — skip unless --rerun
    to_process = []
    for path, img_dir in paths_with_images:
        if store.has_markdown(path.name) and not args.rerun:
            TerminalUI.dim(f"skipping {path.name} (cached)")
            continue
        to_process.append((path, img_dir))

    if not to_process:
        TerminalUI.success("all files already parsed")
        TerminalUI.dim("use --rerun to reprocess with different parameters")
        return

    # single file — show the open in explorer shortcut
    if len(to_process) == 1 and not args.rerun:
        path, img_dir = to_process[0]
        if store.has_markdown(path.name):
            TerminalUI.bold("Skipping parsing (cached markdown found)")
            TerminalUI.dim("Opening existing markdown in explorer...")
            open_in_explorer(store._file_dir(path.name) / f"{sanitize(path.name)}.md")
            return

    TerminalUI.bold(f"Parsing {len(to_process)} file(s)...")

    files = [
        FileWithImages(file=path, image_dir=img_dir)
        for path, img_dir in to_process
    ]

    # delegate all parsing + parallelism to parse_documents
    docs = parse_documents(files, caption_images=args.caption)

    # save results and report
    succeeded = failed = 0
    for doc in docs:
        if doc.error:
            TerminalUI.error(f"{doc.filename} — {doc.error}")
            failed += 1
        else:
            store.save_markdown(doc.filename, doc, rerun=args.rerun)
            TerminalUI.success(
                f"{doc.filename} — "
                f"{doc.page_count} pages · "
                f"{len(doc.text):,} chars"
            )
            succeeded += 1

    TerminalUI.blank()
    TerminalUI.success(f"done — {succeeded} parsed · {failed} failed")




# ── CLI Integration ─────────────────────────────────────────────────────────

def main():
    """CLI entry point using DarwinStore"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse documents and store in .darwin")
    parser.add_argument("path", nargs="?", default=".", help="File or directory to parse")
    parser.add_argument("--rerun", "-R", action="store_true", help="Reprocess existing files")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan directories recursively")
    parser.add_argument("--images", "-i", action="store_true", help="Extract images")
    parser.add_argument("--caption", "-c", action="store_true", help="Generate image captions")
    parser.add_argument("--project", "-p", help="Project directory (default: current dir)")
    
    parser.add_argument("--file", "-f", default=None, help="Point to a file instead of the entire directory")
    
    args = parser.parse_args()
    
    handle_parsing(ParsingArgs(
        target=args.path,
        projectDir=args.project,
        images=args.images,
        caption=args.caption,
        recursive=args.recursive,
        rerun=args.rerun,
        file=args.file
    ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())