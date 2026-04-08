"""
src/core/Parser.py - Production document parser with image captioning
"""

import json
import re
import logging
import math
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime
from src.lib.Terminal import TerminalUI

from src.core.Parsing.ParserUtils import (
    SUPPORTED_TYPES, CPU_COUNT, ParsedDocument,
    PARSE_TIMEOUT_SECONDS, check_file_size,parse_markitdown,parse_pdf
)


# Setup logging
logger = logging.getLogger(__name__)


# ── Individual parsers ─────────────────────────────────────────────────────────





# ── Type detection ─────────────────────────────────────────────────────────────

def detect_type(path: Path) -> str | None:
    return SUPPORTED_TYPES.get(path.suffix.lower())


def is_supported(path: Path) -> bool:
    return detect_type(path) is not None


def is_good_enough(text: str) -> bool:
    if not text or len(text.strip()) < 100:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) > 0.6


# ── Main parse function ────────────────────────────────────────────────────────

def parse(
    file_path: str | Path,
    strategy: str = "light",
    images_dir: Path | None = None,
    caption: bool = False,
    timeout: int = PARSE_TIMEOUT_SECONDS,
    progress_callback: Optional[Callable] = None,
) -> ParsedDocument:
    """
    Parse a single file into a ParsedDocument.
    Never raises — errors go into .error field.

    Args:
        file_path: path to file
        strategy: "light" or "high"
        images_dir: where to save extracted images (None = skip images)
        caption: run SmolVLM captioning (only when images_dir is set)
        timeout: maximum seconds to spend parsing
        progress_callback: optional callback for progress updates
    """

    path = Path(file_path).resolve()
    start_time = datetime.now()

    if not path.exists():
        return ParsedDocument(
            text="", filename=path.name, page_count=0,
            file_type="unknown", strategy=strategy,
            error=f"file not found: {path}",
        )

    size_error = check_file_size(path)
    if size_error:
        return ParsedDocument(
            text="", filename=path.name, page_count=0,
            file_type="unknown", strategy=strategy,
            error=size_error,
        )

    file_type = detect_type(path)
    if file_type is None:
        supported = ", ".join(SUPPORTED_TYPES.keys())
        return ParsedDocument(
            text="", filename=path.name, page_count=0,
            file_type="unsupported", strategy=strategy,
            error=f"unsupported type '{path.suffix}' — supported: {supported}",
        )

    try:
        if progress_callback:
            progress_callback(f"Parsing {path.name}...")

        if file_type == "pdf":
            text, page_count, metadata, images = parse_pdf(
                path, images_dir=images_dir, caption=caption,
                progress_callback=progress_callback,
            )
        else:
            text, page_count, metadata, images = parse_markitdown(path)

        if not is_good_enough(text):
            return ParsedDocument(
                text="", filename=path.name, page_count=page_count,
                file_type=file_type, strategy=strategy, metadata=metadata,
                error="parsed text is empty or garbled",
            )

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Parsed {path.name} in {elapsed:.1f}s ({len(text):,} chars, {page_count} pages)")

        return ParsedDocument(
            text=text, filename=path.name, page_count=page_count,
            file_type=file_type, strategy=strategy,
            metadata=metadata, images=images, error=None,
        )

    except TimeoutError:
        return ParsedDocument(
            text="", filename=path.name, page_count=0,
            file_type=file_type, strategy=strategy,
            error=f"parsing timed out after {timeout} seconds",
        )
    except Exception as e:
        logger.exception(f"Failed to parse {path.name}")
        return ParsedDocument(
            text="", filename=path.name, page_count=0,
            file_type=file_type, strategy=strategy,
            error=f"{type(e).__name__}: {e}",
        )


# ── Multi-file parsing ─────────────────────────────────────────────────────────
def parse_many(
    file_paths: list[str | Path],
    strategy: str = "light",
    images_dir: Path | None = None,
    caption: bool = False,
    timeout: int = PARSE_TIMEOUT_SECONDS,
    progress_callback: Optional[Callable] = None,
) -> list[ParsedDocument]:
    """
    Parse multiple files in parallel, optimized for user experience:
    1. Sorts by size (Smallest -> Largest) for fast initial feedback.
    2. Adjusts worker count based on resource-heavy tasks (captioning).
    3. Handles results in sorted order.
    """
    
    # 1. Resolve and SORT by size (Ascending for the dopamine hit)
    paths = sorted(
        [Path(p).resolve() for p in file_paths], 
        key=lambda p: p.stat().st_size
    )
    
    results: list[ParsedDocument | None] = [None] * len(paths)
    to_parse: list[tuple[int, Path]] = []

    # 2. Filtering and UI Feedback
    total_mb = sum(p.stat().st_size for p in paths) / (1024 * 1024)
    if progress_callback:
        progress_callback(f"Queueing {len(paths)} files ({total_mb:.1f} MB total)")

    for idx, path in enumerate(paths):
        if not is_supported(path):
            warnings.warn(f"skipping unsupported file: {path.name}", stacklevel=2)
            TerminalUI.error(f"Unsupported: {path.name}")
            results[idx] = ParsedDocument(
                text="", filename=path.name, page_count=0,
                file_type="unsupported", strategy=strategy,
                error=f"unsupported type: {path.suffix}",
            )
        else:
            to_parse.append((idx, path))

    if not to_parse:
        return [r for r in results if r is not None]

    # 3. Handle Single File Case (Bypass Pool)
    if len(to_parse) == 1:
        idx, path = to_parse[0]
        file_imgs_dir = (images_dir / path.stem) if images_dir else None
        results[idx] = parse(
            path, strategy=strategy,
            images_dir=file_imgs_dir, caption=caption,
            timeout=timeout, progress_callback=progress_callback,
        )
        return results # type: ignore

    # 4. Determine Worker Count
    # If captioning is ON, we use fewer workers to avoid OOM (Out of Memory)
    # captioning involves loading a VLM per process.
    if caption:
        n_workers = min(len(to_parse), max(1, CPU_COUNT // 2))
    else:
        n_workers = min(len(to_parse), CPU_COUNT)
        
    logger.info(f"Parsing {len(to_parse)} files with {n_workers} workers")
    
    # 5. Parallel Execution
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        ftoi = {
            executor.submit(
                parse, path, strategy,
                (images_dir / path.stem) if images_dir else None,
                caption, timeout, None, # No nested callbacks
            ): idx
            for idx, path in to_parse
        }
        
        for future in as_completed(ftoi):
            idx = ftoi[future]
            try:
                results[idx] = future.result(timeout=timeout + 10)
                
                # Update UI immediately as each file finishes
                if progress_callback and results[idx]:
                    progress_callback(f"Finished {results[idx].filename}")
                
                TerminalUI.success(
                    f"Parsed {results[idx].filename}: "
                    f"{len(results[idx].text):,} chars, "
                    f"{results[idx].page_count} pages"
                )
                
            except TimeoutError:
                path = paths[idx]
                results[idx] = ParsedDocument(
                    text="", filename=path.name, page_count=0,
                    file_type=detect_type(path) or "unknown", strategy=strategy,
                    error=f"worker timed out after {timeout} seconds",
                )
            except Exception as e:
                path = paths[idx]
                results[idx] = ParsedDocument(
                    text="", filename=path.name, page_count=0,
                    file_type=detect_type(path) or "unknown", strategy=strategy,
                    error=f"worker failed: {type(e).__name__}: {e}",
                )

    return results 




def scan_directory(
    target: str | Path,
    recursive: bool = False,
) -> list[Path]:
    """
    Detects if the input is a single file or a directory.
    Returns a list of valid, supported Path objects.
    """
    path = Path(target).resolve()
    
    # 1. Check if it even exists
    if not path.exists():
        TerminalUI.error(f"Path does not exist: {path}")
        return []

    # 2. Case: Single File
    if path.is_file():
        if is_supported(path):
            return [path]
        else:
            TerminalUI.warning(f"File type not supported: {path.suffix}")
            return []

    # 3. Case: Directory
    if path.is_dir():
        pattern = "**/*" if recursive else "*"
        found_files = [
            p for p in path.glob(pattern)
            if p.is_file() 
            and is_supported(p)
            and not any(part.startswith(".") for part in p.parts)
        ]

        if found_files:
            TerminalUI.info(f"Scanner found {len(found_files)} supported files in {path.name}")
        else:
            TerminalUI.warning(f"No supported files found in directory: {path}")
            
        return found_files

    return []