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

from src.core.Parsing.ParserUtils import (
    SUPPORTED_TYPES, CPU_COUNT, LARGE_FILE_MB, MAX_FILE_SIZE_MB,
    MIN_IMAGE_SIZE_PX, ExtractedImage, ParsedDocument,
    PARSE_TIMEOUT_SECONDS, CACHE_CAPTIONS,CACHE_CAPTIONS, check_file_size,  save_caption_cache,workers_for_pages,split_page_ranges,parse_pdf_pages,extract_pdf_images,inject_captions,strip_image_markers,parse_markitdown,parse_pdf
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
    skip_unsupported: bool = True,
    images_dir: Path | None = None,
    caption: bool = False,
    timeout: int = PARSE_TIMEOUT_SECONDS,
    progress_callback: Optional[Callable] = None,
) -> list[ParsedDocument]:
    """Parse multiple files in parallel. Results in same order as input."""
    paths = [Path(p).resolve() for p in file_paths]
    results: list[ParsedDocument | None] = [None] * len(paths)
    to_parse: list[tuple[int, Path]] = []

    for idx, path in enumerate(paths):
        if not is_supported(path):
            if skip_unsupported:
                warnings.warn(f"skipping unsupported file: {path.name}", stacklevel=2)
                results[idx] = ParsedDocument(
                    text="", filename=path.name, page_count=0,
                    file_type="unsupported", strategy=strategy,
                    error=f"unsupported type: {path.suffix}",
                )
            else:
                to_parse.append((idx, path))
        else:
            to_parse.append((idx, path))

    if not to_parse:
        return [r for r in results if r is not None]

    if len(to_parse) == 1:
        idx, path = to_parse[0]
        file_imgs_dir = (images_dir / path.stem) if images_dir else None
        results[idx] = parse(
            path, strategy=strategy,
            images_dir=file_imgs_dir, caption=caption,
            timeout=timeout, progress_callback=progress_callback,
        )
        return results  # type: ignore

    n_workers = min(len(to_parse), CPU_COUNT)
    logger.info(f"Parsing {len(to_parse)} files with {n_workers} workers")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        ftoi = {
            executor.submit(
                parse, path, strategy,
                (images_dir / path.stem) if images_dir else None,
                caption, timeout, None,  # No progress callback in workers
            ): idx
            for idx, path in to_parse
        }
        for future in as_completed(ftoi):
            idx = ftoi[future]
            try:
                results[idx] = future.result(timeout=timeout + 10)
                if progress_callback:
                    progress_callback(f"Completed {results[idx].filename}")
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

    return results  # type: ignore


# ── Directory scanning ─────────────────────────────────────────────────────────

def scan_directory(
    directory: str | Path,
    recursive: bool = False,
    strategy: str = "light",
    images_dir: Path | None = None,
    caption: bool = False,
    timeout: int = PARSE_TIMEOUT_SECONDS,
    progress_callback: Optional[Callable] = None,
) -> list[ParsedDocument]:
    """Find and parse all supported files in a directory."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    all_files = [
        p for p in directory.glob(pattern)
        if p.is_file()
        and is_supported(p)
        and not any(part.startswith(".") for part in p.parts)
    ]

    if not all_files:
        return []

    logger.info(f"Found {len(all_files)} files in {directory}")
    
    return parse_many(
        all_files, strategy=strategy,
        images_dir=images_dir, caption=caption,
        timeout=timeout, progress_callback=progress_callback,
    )