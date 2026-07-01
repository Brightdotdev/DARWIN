"""
UrlFetcher.py — Safe, retrying HTTP fetcher with rotating User-Agents.
"""

import tempfile
from urllib.parse import urlparse
from typing import Optional
from pathlib import Path

from src.core.Logger.DarwinLogger import darwin, LogType
from src.pipleline.Ingestion.Parsing.lib import (
    ParsedDocument,
    detect_type,
    URL_RE,
)


# ── URL validation ────────────────────────────────────────────────

BLOCKED_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})
PRIVATE_IP_PREFIXES = (
    "10.", "192.168.", "172.16.", "172.17.", "172.18.",
    "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
    "172.29.", "172.30.", "172.31.",
)


def _words_to_pages(text: str, words_per_page: int = 500) -> int:
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


def fetch_url(url: str, timeout: float = 30.0) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Fetch URL content with retries and rotating User-Agents.
    Returns (data_bytes, content_type, error_string).
    """
    import requests

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    import random
    headers = {"User-Agent": random.choice(user_agents)}

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            return resp.content, resp.headers.get("Content-Type", "").split(";")[0].strip(), None
        except requests.exceptions.Timeout:
            if attempt < 2:
                continue
            return None, None, f"timeout after {timeout}s"
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                continue
            return None, None, str(e)

    return None, None, "all retries failed"


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

    err = validate_url(url)
    if err:
        return ParsedDocument(
            display_text="", embedding_text="", filename=url,
            page_count=0, file_type="unknown", error=err,
        )

    data, content_type, fetch_err = fetch_url(url, timeout=timeout)
    if fetch_err:
        return ParsedDocument(
            display_text="", embedding_text="", filename=url,
            page_count=0, file_type="unknown", error=fetch_err,
        )

    assert data is not None
    content_type = content_type or ""

    # Binary content handlers
    binary_mime_types = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": None,
        "application/vnd.ms-excel": None,
        "application/epub+zip": None,
        "message/rfc822": None,
    }

    file_type_hint = binary_mime_types.get(content_type)

    if content_type == "application/pdf" and file_type_hint in parser_map:
        suffix = Path(urlparse(url).path).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return _dispatch_parser(file_type_hint, parser_map, tmp_path, url, images_dir, caption, precise)
        finally:
            tmp_path.unlink(missing_ok=True)

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
            return _dispatch_parser(detected, parser_map, tmp_path, url, images_dir, caption, precise)
        finally:
            tmp_path.unlink(missing_ok=True)

    # HTML / plain text path
    if content_type in ("text/html", "application/xhtml+xml", "text/plain", "text/markdown", ""):
        from src.pipleline.Ingestion.Parsing.HtmlProcessor import extract_content
        from src.pipleline.Ingestion.Parsing.lib import is_good_enough

        text, metadata = extract_content(url, html_bytes=data, precise=precise)

        if not text or not is_good_enough(text, darwin):
            error_msg = metadata.get("error") or "extracted text is empty or garbled"
            return ParsedDocument(
                display_text="", embedding_text="", filename=url,
                page_count=0, file_type="html", error=error_msg,
            )

        return ParsedDocument(
            display_text=text, embedding_text=text, filename=url,
            page_count=_words_to_pages(text), file_type="html",
            metadata=metadata, error=None,
        )

    return ParsedDocument(
        display_text="", embedding_text="", filename=url,
        page_count=0, file_type="unsupported",
        error=f"unsupported content type '{content_type}'",
    )


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

    from src.pipleline.Ingestion.Parsing.lib import is_good_enough
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