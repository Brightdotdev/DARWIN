"""
src/core/Parser.py - Production document parser with image captioning
"""

import logging
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime
from src.lib.Terminal import TerminalUI

from src.core.Parsing.ParserUtils import (
    SUPPORTED_TYPES, CPU_COUNT, ParsedDocument,
    PARSE_TIMEOUT_SECONDS, check_file_size, parse_markitdown,
    parse_pdf, detect_type, is_good_enough, is_supported,
)

logger = logging.getLogger(__name__)


def parse(
    file_path:         str | Path,
    images_dir:        Path | None = None,
    caption:           bool = False,
    timeout:           int = PARSE_TIMEOUT_SECONDS,
    progress_callback: Optional[Callable] = None,
    project_dir:       Path | None = None,
    precise:           bool = False,   # ← new: use pymupdf4llm instead of fitz
) -> ParsedDocument:
    """
    Parse a single file into a ParsedDocument.
    Never raises — errors go into .error field.

    Args:
        file_path:         path to file
        images_dir:        where to save extracted images (None = skip images)
        caption:           run SmolVLM captioning (requires images_dir)
        timeout:           maximum seconds to spend parsing
        progress_callback: optional callback for progress updates
        project_dir:       project root for relative image path normalization
        precise:           use pymupdf4llm for layout-accurate extraction
                           (slower but handles complex PDFs better)
    """
    path       = Path(file_path).resolve()
    time = TerminalUI.timer(f"Parsing {path.name}") 
    if not path.exists():
        return ParsedDocument(
            display_text="", embedding_text="", filename=path.name,
            page_count=0, file_type="unknown",
            error=f"file not found: {path}",
        )

    size_error = check_file_size(path)
    if size_error:
        return ParsedDocument(
            display_text="", embedding_text="", filename=path.name,
            page_count=0, file_type="unknown",
            error=size_error,
        )

    file_type = detect_type(path)
    if file_type is None:
        supported = ", ".join(SUPPORTED_TYPES.keys())
        return ParsedDocument(
            display_text="", embedding_text="", filename=path.name,
            page_count=0, file_type="unsupported",
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
                project_dir       = project_dir,
                progress_callback = progress_callback,
                precise           = precise,           # ← thread through
            )
        else:
            text, page_count, metadata, images = parse_markitdown(path)
            display_text   = text
            embedding_text = text
            all_images     = []
            TerminalUI.dim("Image extraction for other file types is not supported yet...¯\_(ツ)_/¯")


        if not is_good_enough(embedding_text):
            return ParsedDocument(
                display_text="", embedding_text="", filename=path.name,
                page_count=page_count, file_type=file_type, metadata=metadata,
                error="parsed text is empty or garbled",
            )
        
        time.done(msg=f"[green]✓[/green] Parsed {path.name} Successfully ")


        return ParsedDocument(
            display_text=display_text, embedding_text=embedding_text,
            filename=path.name, page_count=page_count,
            file_type=file_type, images=all_images,
            metadata=metadata, error=None,
        )

    except TimeoutError:
        return ParsedDocument(
            display_text="", embedding_text="", filename=path.name,
            page_count=0, file_type=file_type,
            error=f"parsing timed out after {timeout} seconds",
        )
    except Exception as e:
        logger.exception(f"Failed to parse {path.name}")
        return ParsedDocument(
            display_text="", embedding_text="", filename=path.name,
            page_count=0, file_type=file_type,
            error=f"{type(e).__name__}: {e}",
        )


def scan_directory(
    target:    str | Path,
    recursive: bool = False,
) -> list[Path]:
    """
    Detects if the input is a single file or a directory.
    Returns a list of valid, supported Path objects.
    """
    path = Path(target).resolve()

    if not path.exists():
        TerminalUI.error(f"Path does not exist: {path}")
        return []

    if path.is_file():
        if is_supported(path):
            return [path]
        else:
            TerminalUI.warn(f"File type not supported: {path.suffix} (Ծ_Ծ)")
            return []

    if path.is_dir():
        pattern     = "**/*" if recursive else "*"
        found_files = [
            p for p in path.glob(pattern)
            if p.is_file()
            and is_supported(p)
            and not any(part.startswith(".") for part in p.parts)
        ]

        if found_files:
            TerminalUI.info(
                label="Found Files",
                value=f"Scanner found {len(found_files)} supported files in {path.name} ",
            )
        else:
            TerminalUI.warn(f"No supported files found in directory: {path} ¯\_(ツ)_/¯")

        return found_files

    return []