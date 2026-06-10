# ============================================================================
# src/core/Parsing/Parser.py
# ============================================================================

import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Tuple, Optional, List

import httpx

from src.features.Ingestion.Parsing.lib import (
    ParsedDocument,
    SUPPORTED_TYPES,
    detect_type,
    is_supported,
)

from src.features.Ingestion.Parsing.PdfExtractors import (
    check_file_size,
    is_good_enough,
    parse_pdf,
)

from src.features.Ingestion.Parsing.TextExtractors import (
    parse_html,
    parse_xml,
    parse_csv,
    parse_spreadsheet,
    parse_ebook,
    parse_email,
)

from src.features.Ingestion.Parsing.PdfExtractors import parse_markitdown

from src.core.Logger.DarwinLogger import darwin, LogType


# ── Blob input (kept for backward compat, prefer BlobHandler.BlobInput) ──────

@dataclass
class BlobInput:
    """Raw text content from clipboard, chat, or other non-file sources."""
    content: str                        # The raw text/HTML/markdown
    source: str                         # e.g. "clipboard", "chat", "web_clip"
    title: str = ""                     # Optional title/context
    mime_hint: str = "text/plain"       # text/plain, text/html, text/markdown
    turn_id: Optional[str] = None       # Conversation turn for unique filenames

    @property
    def filename(self) -> str:
        """Generate unique filename from turn + source."""
        prefix = f"{self.turn_id}_" if self.turn_id else ""
        return f"{prefix}{self.source}"


# ── URL validation ──────────────────────────────────────────────────────────

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

_BLOCKED_HOSTS = frozenset({
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
})

_PRIVATE_IP_PREFIXES = (
    "10.", "192.168.", "172.16.", "172.17.", "172.18.",
    "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
    "172.29.", "172.30.", "172.31.",
)


def is_url(path_or_url: str | Path) -> bool:
    """Return True if the argument looks like an HTTP(S) URL."""
    return isinstance(path_or_url, str) and _URL_RE.match(path_or_url) is not None


def _validate_url(url: str) -> Optional[str]:
    """
    Validate a URL for safety.
    Returns None if valid, otherwise an error string.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return f"unsupported URL scheme '{parsed.scheme}' — only http/https allowed"

    hostname = (parsed.hostname or "").lower()

    if not hostname:
        return "URL has no host"

    if hostname in _BLOCKED_HOSTS:
        return f"blocked host: {hostname}"

    if hostname.startswith(_PRIVATE_IP_PREFIXES):
        return f"blocked private IP range: {hostname}"

    return None


# ── HTTP fetch ────────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0",
}

def fetch_url(
    url: str,
    timeout: float = 30.0,
    max_redirects: int = 5,
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Fetch a URL and return (content_bytes, content_type, error).

    content_type is the normalized MIME type from the response headers,
    or None if the request failed.
    """
    validation_err = _validate_url(url)
    if validation_err:
        return None, None, validation_err

    try:
        with httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            max_redirects=max_redirects,
            headers=_HEADERS,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "").lower().split(";")[0].strip()
            return resp.content, content_type, None

    except httpx.TimeoutException:
        return None, None, f"request timed out after {timeout}s"
    except httpx.TooManyRedirects:
        return None, None, f"too many redirects (max {max_redirects})"
    except httpx.HTTPStatusError as e:
        return None, None, f"HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        return None, None, f"request failed: {e}"


# ── Content-type sniffing ───────────────────────────────────────────────────

# Map common MIME types to our internal file_type strings.
# `None` means "try detect_type on a temp file".
_MIME_TYPE_MAP = {
    "text/html": "html",
    "application/xhtml+xml": "html",
    "application/xml": "xml",
    "text/xml": "xml",
    "text/plain": "markitdown",
    "text/markdown": "markitdown",
    "text/csv": "csv",
    "application/csv": "csv",
    "application/pdf": "pdf",
    # Spreadsheets — let detect_type handle via temp file
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": None,
    "application/vnd.ms-excel": None,
    # E-books
    "application/epub+zip": None,
    # Email
    "message/rfc822": None,
}

# Content types we explicitly reject.
_REJECTED_MIME_PREFIXES = (
    "image/",
    "video/",
    "audio/",
    "application/zip",
    "application/x-zip",
    "application/x-rar",
    "application/x-7z",
    "application/gzip",
)


def _sniff_type(content_type: str, url: str, data: bytes) -> Tuple[Optional[str], Optional[str]]:
    """
    Determine the internal file_type from MIME type + URL + magic bytes.
    Returns (file_type, error).  file_type is None on error.
    """
    # Rejected broad categories
    for prefix in _REJECTED_MIME_PREFIXES:
        if content_type.startswith(prefix):
            return None, f"unsupported content type '{content_type}' — images, archives, and media are not supported"

    # Known mapping
    mapped = _MIME_TYPE_MAP.get(content_type)
    if mapped is not None:
        return mapped, None

    # Unknown — try to detect from URL extension or magic bytes
    return None, None  # signals "try detect_type on temp file"


# ── Parser dispatch map ─────────────────────────────────────────────────────

PARSER_MAP = {
    "pdf":         parse_pdf,
    "markitdown":  parse_markitdown,   # .docx, .pptx, .txt, .md
    "html":        parse_html,
    "xml":         parse_xml,
    "csv":         parse_csv,
    "spreadsheet": parse_spreadsheet,
    "ebook":       parse_ebook,
    "email":       parse_email,
}


# ── Core parse entry-point ──────────────────────────────────────────────────

def parse(
    file_path: str | Path,
    images_dir: Path | None = None,
    caption: bool = False,
    precise: bool = False,
) -> ParsedDocument:
    """Parse a single file or URL into a ParsedDocument. Never raises — errors go into .error field."""
    # ── URL branch ──────────────────────────────────────────────────────────
    if is_url(file_path):
        return parse_url(file_path, images_dir=images_dir, caption=caption, precise=precise)

    # ── Local file branch (original logic, unchanged) ─────────────────────
    path = Path(file_path).resolve()

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
        parser = PARSER_MAP.get(file_type)
        if parser is None:
            return ParsedDocument(
                display_text="", embedding_text="", filename=path.name,
                page_count=0, file_type=file_type,
                error=f"no parser registered for type '{file_type}'",
            )

        display_text, embedding_text, page_count, metadata, all_images = parser(
            path,
            images_dir=images_dir,
            caption=caption,
            precise=precise,
        )

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
            error="parsing timed out...double check your file or free up ram",
        )
    except Exception as e:
        darwin.log(f"Failed to parse {path.name}", LogType.ERROR)
        return ParsedDocument(
            display_text="", embedding_text="", filename=path.name,
            page_count=0, file_type=file_type,
            error=f"{type(e).__name__}: {e}",
        )


def parse_url(
    url: str,
    images_dir: Path | None = None,
    caption: bool = False,
    precise: bool = False,
    timeout: float = 30.0,
) -> ParsedDocument:
    """
    Fetch a URL and parse its content.

    Flow:
        1. Validate URL (block file://, localhost, private IPs).
        2. Fetch with httpx (follow redirects, respect timeout).
        3. Sniff content-type → decide parser.
        4. Text types (HTML, XML, plain text) → pass string directly.
        5. Binary types (PDF, spreadsheets, etc.) → temp file → existing parser.
        6. Rejected types (images, video, zip) → error immediately.
    """
    darwin.log(f"Fetching {url}...", LogType.INFO)

    data, content_type, fetch_err = fetch_url(url, timeout=timeout)
    if fetch_err:
        return ParsedDocument(
            display_text="", embedding_text="", filename=url,
            page_count=0, file_type="unknown",
            error=fetch_err,
        )

    assert data is not None
    content_type = content_type or ""

    # ── Sniff type ──────────────────────────────────────────────────────────
    file_type, sniff_err = _sniff_type(content_type, url, data)

    if sniff_err:
        return ParsedDocument(
            display_text="", embedding_text="", filename=url,
            page_count=0, file_type="unsupported",
            error=sniff_err,
        )

    # If MIME map returned None, try detect_type via temp file
    if file_type is None:
        # Write to temp file so detect_type can inspect extension / magic
        suffix = Path(urlparse(url).path).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        try:
            file_type = detect_type(tmp_path)
            if file_type is None:
                supported = ", ".join(SUPPORTED_TYPES.keys())
                return ParsedDocument(
                    display_text="", embedding_text="", filename=url,
                    page_count=0, file_type="unsupported",
                    error=f"unsupported content from URL — supported: {supported}",
                )

            # Now dispatch via PARSER_MAP on the temp file
            parser = PARSER_MAP.get(file_type)
            if parser is None:
                return ParsedDocument(
                    display_text="", embedding_text="", filename=url,
                    page_count=0, file_type=file_type,
                    error=f"no parser registered for type '{file_type}'",
                )

            darwin.log(f"Parsing {url} as {file_type}...", LogType.INFO)
            display_text, embedding_text, page_count, metadata, all_images = parser(
                tmp_path,
                images_dir=images_dir,
                caption=caption,
                precise=precise,
            )

        finally:
            tmp_path.unlink(missing_ok=True)

    # ── Text types: pass string directly (no temp file) ─────────────────────
    elif file_type in ("html", "xml", "markitdown"):
        # For HTML we use Trafilatura on the raw bytes/string
        if file_type == "html":
            darwin.log(f"Extracting article from {url}...", LogType.INFO)
            try:
                from trafilatura import extract

                html_str = data.decode("utf-8", errors="replace")
                extracted = extract(
                    html_str,
                    output_format="markdown" if precise else "txt",
                    url=url,
                    favor_recall=precise,
                    include_comments=False,
                    include_tables=True,
                    deduplicate=True,
                    with_metadata=True,
                )

                if extracted is None:
                    # Trafilatura failed — likely JS-rendered or empty page
                    darwin.log(
                        f"Page at {url} appears empty or JavaScript-rendered. "
                        "Install the Crawl4AI plugin for JS support.",
                        LogType.WARNING,
                    )
                    return ParsedDocument(
                        display_text="", embedding_text="", filename=url,
                        page_count=0, file_type="html",
                        error="page content is empty or requires JavaScript rendering",
                    )

                # Try to pull metadata from JSON output
                metadata: dict = {}
                try:
                    import json
                    meta = json.loads(extracted) if extracted.strip().startswith("{") else {}
                    if isinstance(meta, dict) and "title" in meta:
                        metadata = {k: v for k, v in meta.items() if k != "text"}
                        text = meta.get("text", extracted)
                    else:
                        text = extracted
                except (json.JSONDecodeError, TypeError):
                    text = extracted

                if not is_good_enough(text):
                    return ParsedDocument(
                        display_text="", embedding_text="", filename=url,
                        page_count=0, file_type="html", metadata=metadata,
                        error="extracted text is empty or garbled — page may require JavaScript rendering",
                    )

                darwin.log(f"Extracted {len(text)} chars from {url}", LogType.SUCCESS)
                return ParsedDocument(
                    display_text=text,
                    embedding_text=text,
                    filename=url,
                    page_count=_words_to_pages(text),
                    file_type="html",
                    metadata=metadata,
                    error=None,
                )

            except ImportError:
                return ParsedDocument(
                    display_text="", embedding_text="", filename=url,
                    page_count=0, file_type="html",
                    error="trafilatura is required for HTML parsing. Install with: pip install trafilatura",
                )

        elif file_type == "xml":
            # Pass raw string to parse_xml
            text = data.decode("utf-8", errors="replace")
            display_text, embedding_text, page_count, metadata, all_images = parse_xml(
                text,  # parse_xml needs to accept str too
                images_dir=images_dir,
                caption=caption,
                precise=precise,
            )

        else:  # markitdown — plain text / markdown
            text = data.decode("utf-8", errors="replace")
            display_text = text
            embedding_text = text
            page_count = _words_to_pages(text)
            metadata = {}
            all_images = []

        if not is_good_enough(embedding_text):
            return ParsedDocument(
                display_text="", embedding_text="", filename=url,
                page_count=page_count, file_type=file_type, metadata=metadata,
                error="parsed text is empty or garbled",
            )

        darwin.log(f"Parsed {url} successfully", LogType.SUCCESS)
        return ParsedDocument(
            display_text=display_text,
            embedding_text=embedding_text,
            filename=url,
            page_count=page_count,
            file_type=file_type,
            images=all_images,
            metadata=metadata,
            error=None,
        )

    # ── Binary types via temp file ──────────────────────────────────────────
    else:
        suffix = Path(urlparse(url).path).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        try:
            parser = PARSER_MAP.get(file_type)
            if parser is None:
                return ParsedDocument(
                    display_text="", embedding_text="", filename=url,
                    page_count=0, file_type=file_type,
                    error=f"no parser registered for type '{file_type}'",
                )

            darwin.log(f"Parsing {url} as {file_type}...", LogType.INFO)
            display_text, embedding_text, page_count, metadata, all_images = parser(
                tmp_path,
                images_dir=images_dir,
                caption=caption,
                precise=precise,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    # ── Final quality check ─────────────────────────────────────────────────
    if not is_good_enough(embedding_text):
        return ParsedDocument(
            display_text="", embedding_text="", filename=url,
            page_count=page_count, file_type=file_type, metadata=metadata,
            error="parsed text is empty or garbled",
        )

    darwin.log(f"Parsed {url} successfully", LogType.SUCCESS)
    return ParsedDocument(
        display_text=display_text,
        embedding_text=embedding_text,
        filename=url,
        page_count=page_count,
        file_type=file_type,
        images=all_images,
        metadata=metadata,
        error=None,
    )


def _words_to_pages(text: str, words_per_page: int = 500) -> int:
    """Rough word-count → page estimate."""
    return max(1, len(text.split()) // words_per_page)


# ── Directory scanner (unchanged) ────────────────────────────────────────────

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
        pattern = "**/*" if recursive else "*"
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