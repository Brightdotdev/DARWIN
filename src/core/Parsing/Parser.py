"""
src/core/Parser.py - Production document parser with image captioning
"""

import json
import re
import logging
import math
import warnings
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime
from src.lib.Terminal import TerminalUI

from src.core.Parsing.ParserUtils import (
    SUPPORTED_TYPES, CPU_COUNT, ParsedDocument,
    PARSE_TIMEOUT_SECONDS, check_file_size,parse_markitdown,parse_pdf,detect_type,is_good_enough,is_supported
)


# Setup logging
logger = logging.getLogger(__name__)





# ── Main parse function ────────────────────────────────────────────────────────

def parse(
    file_path: str | Path,
    images_dir: Path | None = None,
    caption: bool = False,
    timeout: int = PARSE_TIMEOUT_SECONDS,
    progress_callback: Optional[Callable] = None,
    project_dir: Path | None = None,       # ← add this
) -> ParsedDocument:
    
    """
    Parse a single file into a ParsedDocument.
    Never raises — errors go into .error field.

    Args:
        file_path: path to file
        
        images_dir: where to save extracted images (None = skip images)
        caption: run SmolVLM captioning (only when images_dir is set)
        timeout: maximum seconds to spend parsing
        progress_callback: optional callback for progress updates
    """

    path = Path(file_path).resolve()
    start_time = datetime.now()

    if not path.exists():
        return ParsedDocument(
            display_text="",embedding_text="", filename=path.name, page_count=0,
            file_type="unknown",
            error=f"file not found: {path}",
        )

    size_error = check_file_size(path)
    if size_error:
        return ParsedDocument(
            display_text="",embedding_text="", filename=path.name, page_count=0,
            file_type="unknown",
            error=size_error,
        )

    file_type = detect_type(path)
    if file_type is None:
        supported = ", ".join(SUPPORTED_TYPES.keys())
        return ParsedDocument(
            display_text="",embedding_text="", filename=path.name, page_count=0,
            file_type="unsupported",
            error=f"unsupported type '{path.suffix}' — supported: {supported}",
        )

    try:
        if progress_callback:
            progress_callback(f"Parsing {path.name}...")

        if file_type == "pdf":
               display_text, embedding_text, page_count, metadata, all_images = parse_pdf(
        path,
        images_dir        = images_dir,
        caption           = caption,
        project_dir       = project_dir,   # ← add this
        progress_callback = progress_callback,
    )
        else:
            text, page_count, metadata, images = parse_markitdown(path)
            display_text = text
            embedding_text = text
            all_images = []

        if not is_good_enough(embedding_text):
            return ParsedDocument(
                 display_text="", embedding_text="", filename=path.name, page_count=page_count,
                file_type=file_type, metadata=metadata,
                error="parsed text is empty or garbled",
            )

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Parsed {path.name} in {elapsed:.1f}s ({len(display_text):,} chars, {page_count} pages)")

        return ParsedDocument(
           display_text=display_text, embedding_text=embedding_text, filename=path.name, page_count=page_count,
            file_type=file_type,
            images=all_images,
            metadata=metadata,  error=None,
        )

    except TimeoutError:
        return ParsedDocument(
            display_text="",embedding_text="", filename=path.name, page_count=0,
            file_type=file_type,
            error=f"parsing timed out after {timeout} seconds",
        )
    except Exception as e:
        logger.exception(f"Failed to parse {path.name}")
        return ParsedDocument(
            display_text="",embedding_text="", filename=path.name, page_count=0,
            file_type=file_type,
            error=f"{type(e).__name__}: {e}",
        )









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
            TerminalUI.info(
            label="Found Files",
            value=f"Scanner found {len(found_files)} supported files in {path.name}"
             )
        else:
            TerminalUI.warning(f"No supported files found in directory: {path}")
            
        return found_files

    return []