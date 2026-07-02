import tempfile
from pathlib import Path
from urllib.parse import urlparse
from typing import Tuple, Optional, Union

from src.core.Logger.DarwinLogger import darwin, LogType
from src.pipleline.Ingestion.Parsing.UrlFetcher import fetch_url, validate_url
from src.pipleline.Ingestion.Parsing.HtmlProcessor import extract_content
from src.pipleline.Ingestion.Parsing.PdfExtractors import parse_pdf, parse_markitdown, check_file_size
from src.pipleline.Ingestion.Parsing.lib import (
    ParsedDocument,
    SUPPORTED_TYPES,
    detect_type,
    is_supported,
    URL_RE,
    is_good_enough,
)
from src.pipleline.Ingestion.Parsing.TextExtractors import (
    parse_html,
    parse_xml,
    parse_csv,
    parse_spreadsheet,
    parse_ebook,
    parse_email,
)


# ── Parser dispatch map ─────────────────────────────────────────────────────

PARSER_MAP = {
    "pdf":         parse_pdf,
    "markitdown":  parse_markitdown,
    "html":        parse_html,
    "xml":         parse_xml,
    "csv":         parse_csv,
    "spreadsheet": parse_spreadsheet,
    "ebook":       parse_ebook,
    "email":       parse_email,
}


def is_url(path_or_url: str | Path) -> bool:
    """Return True if the argument looks like an HTTP(S) URL."""
    return isinstance(path_or_url, str) and URL_RE.match(path_or_url) is not None


# ── Main entry point ──────────────────────────────────────────────────────

def parse(
    source: Union[str, Path, Tuple[str, bytes, str]],
    images_dir: Path | None = None,
    caption: bool = False,
    precise: bool = False,
    display_name: Optional[str] = None,
) -> ParsedDocument:
    """
    Parse a local file or URL.

    Args:
        source: Local file path, URL string, or tuple of (url, data, content_type).
        images_dir: Directory to store extracted images.
        caption: Whether to generate captions during extraction (ignored — done later).
        precise: Use precise extraction (layout‑aware for PDFs, markdown for HTML).
        display_name: Optional override for the returned document's filename.
    """
    if isinstance(source, tuple):
        url, data, content_type = source
        return _parse_url_with_data(
            url, data, content_type,
            images_dir=images_dir,
            caption=caption,
            precise=precise,
            display_name=display_name,
        )

    if is_url(source):
        return parse_url(
            str(source),
            images_dir=images_dir,
            caption=caption,
            precise=precise,
            display_name=display_name,
        )

    path = Path(source).resolve()
    if not path.exists():
        return ParsedDocument(
            display_text="", embedding_text="", filename=display_name or path.name,
            page_count=0, file_type="unknown",
            error=f"file not found: {path}",
        )

    file_type = detect_type(path)
    if file_type is None:
        supported = ", ".join(SUPPORTED_TYPES.keys())
        return ParsedDocument(
            display_text="", embedding_text="", filename=display_name or path.name,
            page_count=0, file_type="unsupported",
            error=f"unsupported file type '{path.suffix}' — supported: {supported}",
        )

    size_error = check_file_size(path)
    if size_error:
        return ParsedDocument(
            display_text="", embedding_text="", filename=display_name or path.name,
            page_count=0, file_type=file_type,
            error=size_error,
        )

    return _dispatch_parser(
        file_type,
        path,
        display_name or path.name,
        images_dir,
        caption,
        precise,
    )


# ── URL parsing ─────────────────────────────────────────────────────────────

def parse_url(
    url: str,
    images_dir: Path | None = None,
    caption: bool = False,
    precise: bool = False,
    timeout: float = 30.0,
    display_name: Optional[str] = None,
) -> ParsedDocument:
    """Fetch a URL and parse its content."""
    darwin.log(f"Fetching {url}...", LogType.INFO)

    err = validate_url(url)
    if err:
        return ParsedDocument(
            display_text="", embedding_text="", filename=display_name or url,
            page_count=0, file_type="unknown", error=err,
        )

    data, content_type, fetch_err = fetch_url(url, timeout=timeout)
    if fetch_err:
        return ParsedDocument(
            display_text="", embedding_text="", filename=display_name or url,
            page_count=0, file_type="unknown", error=fetch_err,
        )

    return _parse_url_with_data(
        url, data, content_type,
        images_dir=images_dir,
        caption=caption,
        precise=precise,
        display_name=display_name,
    )


def _parse_url_with_data(
    url: str,
    data: bytes,
    content_type: str,
    images_dir: Path | None = None,
    caption: bool = False,
    precise: bool = False,
    display_name: Optional[str] = None,
) -> ParsedDocument:
    """Parse already-fetched URL data."""
    assert data is not None
    content_type = content_type or ""

    # 1. Known binary types – write to temp file and dispatch
    if content_type == "application/pdf":
        suffix = Path(urlparse(url).path).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return _dispatch_parser(
                "pdf", tmp_path, display_name or url,
                images_dir, caption, precise,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    # 2. Other binary types
    if content_type not in ("text/html", "application/xhtml+xml", "text/plain", "text/markdown", ""):
        suffix = Path(urlparse(url).path).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            file_type = detect_type(tmp_path)
            if file_type is None:
                return ParsedDocument(
                    display_text="", embedding_text="", filename=display_name or url,
                    page_count=0, file_type="unsupported",
                    error=f"unsupported content from URL (type: {content_type})",
                )
            return _dispatch_parser(
                file_type, tmp_path, display_name or url,
                images_dir, caption, precise,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    # 3. HTML / plain text path
    if content_type in ("text/html", "application/xhtml+xml", "text/plain", "text/markdown", ""):
        text, metadata = extract_content(url, html_bytes=data, precise=precise)

        if not text or not is_good_enough(text, darwin):
            error_msg = metadata.get("error") or "extracted text is empty or garbled"
            darwin.error(error_msg)
            return ParsedDocument(
                display_text="", embedding_text="", filename=display_name or url,
                page_count=0, file_type="html", error=error_msg,
            )

        return ParsedDocument(
            display_text=text, embedding_text=text, filename=display_name or url,
            page_count=_words_to_pages(text), file_type="html",
            metadata=metadata, error=None,
        )
    darwin.error(f"unsupported content type '{content_type}'")
    return ParsedDocument(
        display_text="", embedding_text="", filename=display_name or url,
        page_count=0, file_type="unsupported",
        error=f"unsupported content type '{content_type}'",
    )


# ── Helper: dispatch a file path to the correct parser ─────────────────────

def _dispatch_parser(
    file_type: str,
    path: Path,
    display_name: str,
    images_dir: Optional[Path],
    caption: bool,
    precise: bool,
) -> ParsedDocument:
    parser = PARSER_MAP.get(file_type)
    if parser is None:
        return ParsedDocument(
            display_text="", embedding_text="", filename=display_name,
            page_count=0, file_type=file_type,
            error=f"no parser registered for type '{file_type}'",
        )
    try:
        darwin.log(f"Parsing {display_name} as {file_type}...", LogType.INFO)
        display_text, embedding_text, page_count, metadata, all_images = parser(
            path, images_dir=images_dir, caption=caption, precise=precise
        )
    except Exception as e:
        return ParsedDocument(
            display_text="", embedding_text="", filename=display_name,
            page_count=0, file_type=file_type,
            error=f"{type(e).__name__}: {e}",
        )

    if not is_good_enough(embedding_text, darwin):
        return ParsedDocument(
            display_text="", embedding_text="", filename=display_name,
            page_count=page_count, file_type=file_type, metadata=metadata,
            error="parsed text is empty or garbled",
        )

    return ParsedDocument(
        display_text=display_text, embedding_text=embedding_text,
        filename=display_name, page_count=page_count,
        file_type=file_type, images=all_images, metadata=metadata, error=None,
    )


def _words_to_pages(text: str, words_per_page: int = 500) -> int:
    return max(1, len(text.split()) // words_per_page)


# ── Directory scanner ────────────────────────────────────────────────────────

def scan_directory(target: str | Path, recursive: bool = False) -> list[Path]:
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
        pattern = "**/*" if recursive else "*"
        found = [
            p for p in path.glob(pattern)
            if p.is_file() and is_supported(p) and not any(part.startswith(".") for part in p.parts)
        ]
        if found:
            darwin.log(f"Scanner found {len(found)} supported files in {path.name}", LogType.INFO)
        else:
            darwin.log(f"No supported files found in directory: {path}", LogType.WARNING)
        return found
    return []