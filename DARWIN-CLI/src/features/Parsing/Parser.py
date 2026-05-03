# ============================================================================
# src/core/Parsing/Parser.py
# ============================================================================

import logging
from pathlib import Path

from src.features.Parsing.lib import (ParsedDocument,
    SUPPORTED_TYPES,
    detect_type,
    is_supported)

from src.features.Parsing.ParserUtils import (
    check_file_size,
    parse_markitdown,
    parse_pdf,
    is_good_enough,
)
from src.core.Logger.DarwinLogger import darwin, LogType

logger = logging.getLogger(__name__)


def parse(
    file_path: str | Path,
    images_dir: Path | None = None,
    caption: bool = False,
    precise: bool = False,
) -> ParsedDocument:
    """Parse a single file into a ParsedDocument. Never raises — errors go into .error field."""
    path  = Path(file_path).resolve()

    if not path.exists():
        return ParsedDocument(
            display_text="", embedding_text="", filename=path.name,
            page_count=0, file_type="unknown",
            error=f"file not found: {path}",
        )

    file_type = detect_type(path)
    if file_type is None:
        supported = ", ".join(SUPPORTED_TYPES.keys())
        return ParsedDocument(
            display_text="", embedding_text="", filename=path.name,
            page_count=0, file_type="unsupported",
            error=f"unsupported file type '{path.suffix}' — supported: {supported}",
        )

    size_error = check_file_size(path)
    if size_error:
        return ParsedDocument(
            display_text="", embedding_text="", filename=path.name,
            page_count=0, file_type="unknown",
            error=size_error,
        )

    try:
        darwin.log(f"Parsing {path.name}...", LogType.INFO)

        if file_type == "pdf":
            display_text, embedding_text, page_count, metadata, all_images = parse_pdf(
                path,
                images_dir=images_dir,
                caption=caption,
                precise=precise,
            )
        else:
            display_text, embedding_text, page_count, metadata, all_images = parse_markitdown(path)
            if images_dir:
                darwin.log(f"Image extraction is not supported for {path.suffix} files", LogType.WARNING)

        if not is_good_enough(embedding_text):
            return ParsedDocument(
                display_text="", embedding_text="", filename=path.name,
                page_count=page_count, file_type=file_type, metadata=metadata,
                error="parsed text is empty or garbled",
            )

        darwin.log(f"Parsed {path.name} successfully", LogType.SUCCESS)

        return ParsedDocument(
            display_text=display_text,
            embedding_text=embedding_text,
            filename=path.name,
            page_count=page_count,
            file_type=file_type,
            images=all_images,
            metadata=metadata,
            error=None,
        )

    except TimeoutError:
        return ParsedDocument(
            display_text="", embedding_text="", filename=path.name,
            page_count=0, file_type=file_type,
            error=f"parsing timed out...double check your file or free up ram",
        )
    except Exception as e:
        logger.exception(f"Failed to parse {path.name}")
        return ParsedDocument(
            display_text="", embedding_text="", filename=path.name,
            page_count=0, file_type=file_type,
            error=f"{type(e).__name__}: {e}",
        )


def scan_directory(
    target: str | Path,
    recursive: bool = False,
) -> list[Path]:
    """Detects if the input is a single file or a directory. Returns list of valid Path objects."""
    path = Path(target).resolve()

    if not path.exists():
        darwin.log(f"Path does not exist: {path}", LogType.ERROR)
        return []

    if path.is_file():
        if is_supported(path):
            return [path]
        else:
            darwin.log(f"File type not supported: {path.suffix}", LogType.WARNING)
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
            darwin.log(f"Scanner found {len(found_files)} supported files in {path.name}", LogType.INFO)
        else:
            darwin.log(f"No supported files found in directory: {path}", LogType.WARNING)

        return found_files

    return []