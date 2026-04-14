"""
src/core/Parsing/main.py - Document parsing pipeline entry point

Windows multiprocessing note
-----------------------------
1. Never nest ProcessPoolExecutors — parse_document() is a plain function
   called inside the worker, not a launcher of more workers.
2. The if __name__ == "__main__" guard at the bottom is mandatory.
3. Never pass callbacks/closures to worker processes — they can't be pickled.

Modes
-----
Default (fast):   fitz get_text("dict") — seconds for any PDF
--precise:        pymupdf4llm.to_markdown() — layout-accurate, slower
"""

import sys
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import Optional

import sys
sys.stdout.reconfigure(encoding='utf-8')




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





# ── Single document parse (runs INSIDE a worker process) ──────────────────────

def parse_document(
    path_with_images: FileWithImages,
    progress_queue: multiprocessing.Queue,
    caption_images: bool = False,
    project_dir: Optional[Path] = None,
    precise: bool = False,
    args: Optional[ParsingArgs] = None,
) -> ParsedDocument:
    """
    Parse one document. Runs inside a subprocess.
    Progress reported through multiprocessing.Queue (pickle-safe).
    """
    file_path = Path(path_with_images.file)
    
    def progress_callback(msg: str) -> None:
        """Send progress updates through the queue"""
        try:
            progress_queue.put((file_path.name, msg))
        except:
            pass  # Queue might be closed, ignore
    
    try:
        return _parse(
            file_path,
            images_dir=path_with_images.image_dir,
            caption=caption_images,
            project_dir=project_dir,
            progress_callback= progress_callback if args.inspect else progress_callback,
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
    


# ── Parallel document parsing ──────────────────────────────────────────────────
def parse_documents(
    files: list[FileWithImages],
    caption_images: bool = False,
    timeout: int = PARSE_TIMEOUT_SECONDS,
    project_dir: Optional[Path] = None,
    precise: bool = False,
    args: Optional[ParsingArgs] = None,
) -> list[ParsedDocument]:
    """
    Parse a list of documents in parallel using a single ProcessPoolExecutor.
    """
    if not files:
        return []
    
    time = TerminalUI.timer("Loading files")
    file_names = [Path(fw.file).name for fw in files]
    total_mb = sum(Path(fw.file).stat().st_size / (1024 * 1024) for fw in files)
    mode_label = "precise" if precise else "fast"
    
    TerminalUI.dim(f"Queueing {len(files)} files ({total_mb:.1f} MB total) [{mode_label} mode]")
    time.split()

    indexed = sorted(enumerate(files), key=lambda x: Path(x[1].file).stat().st_size)
    n_workers = min(
        len(files),
        max(1, CPU_COUNT // 2) if caption_images else CPU_COUNT,
    )

    results: list[Optional[ParsedDocument]] = [None] * len(files)
    
    # 🔥 Create a manager and queue for cross-process communication
    import multiprocessing as mp
    manager = mp.Manager()
    progress_queue = manager.Queue()
    
    # Track completion and errors
    completed_files = set()
    file_errors = {}

    # 🔥 Use multi-parse progress UI
    with TerminalUI.multi_parse_progress(file_names) as get_callback:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all jobs with the queue
            future_to_info = {}
            for orig_idx, fw in indexed:
                filename = Path(fw.file).name
                future = executor.submit(
                    parse_document, 
                    fw, 
                    progress_queue,  # Pass the queue
                    caption_images, 
                    project_dir, 
                    precise,
                    args
                )
                future_to_info[future] = (orig_idx, filename)
            
            # Process completions and queue messages
            futures_remaining = set(future_to_info.keys())
            
            while futures_remaining:
                # Check queue for progress updates
                try:
                    while True:  # Drain all available messages
                        filename, msg = progress_queue.get_nowait()
                        update_progress = get_callback(filename)
                        update_progress(msg)
                except:
                    pass  # Queue empty
                
                # Check for completed futures (non-blocking)
                done_futures = {f for f in futures_remaining if f.done()}
                
                for future in done_futures:
                    orig_idx, filename = future_to_info[future]
                    update_progress = get_callback(filename)
                    
                    try:
                        doc = future.result(timeout=0)  # Already done, no wait
                        results[orig_idx] = doc
                        
                        if doc.error:
                            file_errors[filename] = doc.error
                            update_progress(f"❌ {doc.error}")
                        else:
                            completed_files.add(filename)
                            update_progress(
                                f"✓ {doc.page_count} pages · {len(doc.embedding_text):,} chars"
                            )
                            
                    except TimeoutError:
                        results[orig_idx] = ParsedDocument(
                            display_text="", embedding_text="",
                            filename=filename, page_count=0,
                            file_type=detect_type(Path(fw.file)) or "unknown",
                            error=f"timed out after {timeout}s",
                        )
                        file_errors[filename] = f"timed out after {timeout}s"
                        update_progress(f"❌ timed out")
                        
                    except Exception as e:
                        results[orig_idx] = ParsedDocument(
                            display_text="", embedding_text="",
                            filename=filename, page_count=0,
                            file_type="unknown",
                            error=f"worker failed: {type(e).__name__}: {e}",
                        )
                        file_errors[filename] = f"{type(e).__name__}: {e}"
                        update_progress(f"❌ {type(e).__name__}")
                    
                    futures_remaining.remove(future)
                
                # Small sleep to prevent CPU spinning
                if futures_remaining:
                    time.sleep(0.05)
    
    # Final summary
    TerminalUI.blank()
    for filename in file_names:
        if filename in completed_files:
            TerminalUI.success(f"{filename} — parsed successfully")
        elif filename in file_errors:
            TerminalUI.error(f"{filename} — {file_errors[filename]}")
    
    return results  # type: ignore

# ── High-level handler ─────────────────────────────────────────────────────────

def handle_parsing(args: ParsingArgs) -> list[ParsedDocument]:


    TerminalUI.print_darwin()
    TerminalUI.blank()
    TerminalUI.bold(f"Parsing {args.target}")
    TerminalUI.dim(f"Images: {args.images} | Rerun: {args.rerun} | Recursive: {args.recursive} | Timeout: {args.timeout}")
    TerminalUI.blank()
    


    if args.file:
        target_paths = scan_directory(args.file, recursive=False)
    else:
        target_paths = scan_directory(args.target, recursive=args.recursive)

    if not target_paths:
        TerminalUI.error(f"No supported files found in {args.target}")
        return

    store = DarwinStore(Path(args.project_dir) if args.project_dir else Path.cwd())

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
        precise        = args.precise,     # ← thread through
        args=args

    )

    succeeded = failed = 0
    for doc in docs:
        if not doc:
            continue
        if doc.error:
            failed += 1
        else:
            if args.save : 
                store.save_markdown(doc.filename, doc, rerun=args.rerun)
            succeeded += 1

    TerminalUI.blank()
    TerminalUI.success(f"done — {succeeded} parsed · {failed} failed")
    TerminalUI.print_project_status(store)
    return docs



# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Parse documents into .darwin")
    parser.add_argument("path",        nargs="?", default=".",    help="File or directory to parse")
    parser.add_argument("--rerun",     "-R", action="store_true", help="Reprocess cached files")
    parser.add_argument("--recursive", "-r", action="store_true", help="Scan directories recursively")
    parser.add_argument("--images",    "-img", action="store_true", help="Extract images")
    parser.add_argument("--caption",   "-c", action="store_true", help="Generate captions (requires --images)")
    parser.add_argument("--project",   "-p", default=None,        help="Project directory (default: cwd)")
    parser.add_argument("--file",      "-f", default=None,        help="Parse a single file instead of a directory")
    parser.add_argument("--precise",         action="store_true", help="Use pymupdf4llm for layout-accurate extraction (slower)")
    parser.add_argument("--save",        default=True, help="Save markdown files")
    parser.add_argument("--inspect",  "-i",       default=True, help="Watch parsing progress")

    args = parser.parse_args()

    handle_parsing(ParsingArgs(
        target      = args.path,
        project_dir = args.project,
        images      = args.images,
        caption     = args.caption,
        recursive   = args.recursive,
        rerun       = args.rerun,
        file        = args.file,
        precise     = args.precise,    # ← new
        save        = args.save,
        inspect=args.inspect
    ))

    return 0


if __name__ == "__main__":
    sys.exit(main())