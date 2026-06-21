"""
UrlFetcher.py — Safe, retrying HTTP fetcher with a descriptive User-Agent.
"""

from urllib.parse import urlparse
from typing import Optional, Tuple
import httpx
import time
from src.core.Logger.DarwinLogger import darwin, LogType

# ── URL validation constants ────────────────────────────────────────────────
BLOCKED_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})
PRIVATE_IP_PREFIXES = (
    "10.", "192.168.", "172.16.", "172.17.", "172.18.",
    "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
    "172.29.", "172.30.", "172.31.",
)

# ── Descriptive User-Agent (required by Wikipedia and other sites) ──────────
# Replace with your project name and a valid contact email/URL.
USER_AGENT = (
    "DarwinParser/1.0 (https://github.com/Brightdotdev; mail.bright.dev@gmail.com) "
    "Python-httpx/0.27"
)

# ── Content‑type mapping ────────────────────────────────────────────────────
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
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": None,
    "application/vnd.ms-excel": None,
    "application/epub+zip": None,
    "message/rfc822": None,
}

_REJECTED_MIME_PREFIXES = (
    "image/", "video/", "audio/",
    "application/zip", "application/x-zip",
    "application/x-rar", "application/x-7z",
    "application/gzip",
)

# ── Validation ──────────────────────────────────────────────────────────────
def validate_url(url: str) -> Optional[str]:
    """Return None if safe, otherwise an error string."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"unsupported URL scheme '{parsed.scheme}' — only http/https allowed"
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return "URL has no host"
    if hostname in BLOCKED_HOSTS:
        return f"blocked host: {hostname}"
    if hostname.startswith(PRIVATE_IP_PREFIXES):
        return f"blocked private IP range: {hostname}"
    return None

# ── Fetching ────────────────────────────────────────────────────────────────
def fetch_url(
    url: str,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Fetch a URL with retries for transient errors only (not 403).
    Returns (content_bytes, content_type, error).
    """
    err = validate_url(url)
    if err:
        return None, None, err

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(
                timeout=httpx.Timeout(timeout),
                follow_redirects=True,
                max_redirects=5,
                headers=headers,
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "").lower().split(";")[0].strip()
                return resp.content, content_type, None

        except httpx.HTTPStatusError as e:
            status = e.response.status_code

            # 403 Forbidden is permanent – do not retry
            if status == 403:
                return None, None, f"HTTP 403 (access denied)"

            # 429 / 503 are transient, retry with backoff
            if status in (429, 503) and attempt < max_retries:
                retry_after = e.response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = int(retry_after)
                else:
                    delay = 2 ** attempt * 1.5
                delay = min(delay, 10)
                darwin.log(f"HTTP {status} from {url}, retrying in {delay:.1f}s (attempt {attempt+1}/{max_retries})", LogType.WARNING)
                time.sleep(delay)
                last_error = f"HTTP {status}"
            else:
                return None, None, f"HTTP {status}"

        except (httpx.TimeoutException, httpx.RequestError) as e:
            if attempt < max_retries:
                delay = 2 ** attempt * 1.5
                time.sleep(min(delay, 10))
                last_error = str(e)
            else:
                return None, None, str(e)

    return None, None, last_error or "unknown fetch error"


# ── Content‑type resolution ─────────────────────────────────────────────────
def resolve_file_type(content_type: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert a MIME type into our internal file_type.
    Returns (file_type, error). file_type is None if unsupported/unknown.
    """
    for prefix in _REJECTED_MIME_PREFIXES:
        if content_type.startswith(prefix):
            return None, f"unsupported content type '{content_type}' — images, archives, and media are not supported"

    mapped = _MIME_TYPE_MAP.get(content_type)
    if mapped is not None:
        return mapped, None

    # Unknown, let the caller decide (e.g., try by URL extension)
    return None, None