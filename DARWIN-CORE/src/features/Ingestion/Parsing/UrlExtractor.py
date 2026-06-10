"""
UrlFetcher.py — Safe, retrying HTTP fetcher with rotating User-Agents.
"""


import tempfile
from urllib.parse import urlparse
from typing import Optional
from src.core.Logger.DarwinLogger import darwin, LogType
from pathlib import Path
from src.features.Ingestion.Parsing.HtmlProcessor import extract_content
from src.features.Ingestion.Parsing.UrlFetcher import fetch_url

from src.features.Ingestion.Parsing.lib import (

    ParsedDocument,
    SUPPORTED_TYPES,
    detect_type,
    is_supported,
    is_good_enough,
URL_RE,
)
# ── Rotating User-Agent pool ───────────────────────────────────────────────


# ── URL validation constants ────────────────────────────────────────────────
BLOCKED_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})
PRIVATE_IP_PREFIXES = (
    "10.", "192.168.", "172.16.", "172.17.", "172.18.",
    "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
    "172.29.", "172.30.", "172.31.",
)



def _words_to_pages(text: str, words_per_page: int = 500) -> int:
    """Rough word-count → page estimate."""
    return max(1, len(text.split()) // words_per_page)


def validate_url(url: str) -> Optional[str]:
    """Return None if safe, otherwise an error string."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"unsupported scheme '{parsed.scheme}' — only http/https allowed"
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return "URL has no host"
    if hostname in BLOCKED_HOSTS:
        return f"blocked host: {hostname}"
    if hostname.startswith(PRIVATE_IP_PREFIXES):
        return f"blocked private IP range: {hostname}"
    return None






def parse_url(
    url: str,
    parser_map,
    images_dir: Path | None = None,
    caption: bool = False,
    precise: bool = False,
    timeout: float = 30.0,
) -> ParsedDocument:
    """
    Fetch a URL, extract its main content, and return a ParsedDocument.
    """
    darwin.log(f"Fetching {url}...", LogType.INFO)

    # 1. Validate the URL
    err = validate_url(url)
    if err:
        return ParsedDocument(
            display_text="", embedding_text="", filename=url,
            page_count=0, file_type="unknown",
            error=err,
        )

    # 2. Fetch raw bytes (with rotating UA, retries)
    data, content_type, fetch_err = fetch_url(url, timeout=timeout)
    if fetch_err:
        return ParsedDocument(
            display_text="", embedding_text="", filename=url,
            page_count=0, file_type="unknown",
            error=fetch_err,
        )

    assert data is not None
    content_type = content_type or ""

    # 3. Binary content handlers (PDF, spreadsheets, etc.)
    #    Write to a temp file and dispatch to the appropriate parser.
    binary_mime_types = {
        "application/pdf": "pdf",
        # Spreadsheets – we let detect_type decide based on file extension/magic
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": None,
        "application/vnd.ms-excel": None,
        "application/epub+zip": None,
        "message/rfc822": None,
    }

    # Check if the content-type is a known binary type
    file_type_hint = binary_mime_types.get(content_type)
    # If the hint is None, we'll try to detect the type from a temp file later
    # But for PDF we can go directly
    if content_type == "application/pdf":
        file_type_hint = "pdf"

    if file_type_hint is not None and file_type_hint in parser_map:
        # Direct dispatch
        suffix = Path(urlparse(url).path).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return _dispatch_parser(file_type_hint,parser_map, tmp_path, url, images_dir, caption, precise)
        finally:
            tmp_path.unlink(missing_ok=True)

    # For other binary content (or if we couldn't determine type), write to temp
    # and use detect_type
    if content_type not in ("text/html", "application/xhtml+xml", "text/plain", "text/markdown", ""):
        suffix = Path(urlparse(url).path).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            detected = detect_type(tmp_path)
            if detected is None:
                return ParsedDocument(
                    display_text="", embedding_text="", filename=url,
                    page_count=0, file_type="unsupported",
                    error=f"unsupported content from URL (type: {content_type})",
                )
            return _dispatch_parser(detected, tmp_path, url, images_dir, caption, precise)
        finally:
            tmp_path.unlink(missing_ok=True)

    # 4. HTML / plain text path → HtmlProcessor
    if content_type in ("text/html", "application/xhtml+xml", "text/plain", "text/markdown", ""):
        text, metadata = extract_content(url, html_bytes=data, precise=precise)

        if not text or not is_good_enough(text,darwin):
            error_msg = metadata.get("error") or "extracted text is empty or garbled"
            return ParsedDocument(
                display_text="", embedding_text="", filename=url,
                page_count=0, file_type="html",
                error=error_msg,
            )

        return ParsedDocument(
            display_text=text,
            embedding_text=text,
            filename=url,
            page_count=_words_to_pages(text),
            file_type="html",
            metadata=metadata,
            error=None,
        )

    # 5. Unknown / unsupported
    return ParsedDocument(
        display_text="", embedding_text="", filename=url,
        page_count=0, file_type="unsupported",
        error=f"unsupported content type '{content_type}'",
    )


# ── Helper: dispatch a file path to the correct parser ─────────────────────
def _dispatch_parser(
    file_type: str,
    parser_map,
    path: Path,
    display_name: str,
    images_dir: Optional[Path],
    caption: bool,
    precise: bool,
) -> ParsedDocument:
    parser = parser_map.get(file_type)
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

    if not is_good_enough(embedding_text,darwin):
        return ParsedDocument(
            display_text="", embedding_text="", filename=display_name,
            page_count=page_count, file_type=file_type, metadata=metadata,
            error="parsed text is empty or garbled",
        )

    return ParsedDocument(
        display_text=display_text,
        embedding_text=embedding_text,
        filename=display_name,
        page_count=page_count,
        file_type=file_type,
        images=all_images,
        metadata=metadata,
        error=None,
    )