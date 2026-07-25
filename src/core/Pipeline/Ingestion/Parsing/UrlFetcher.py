"""
UrlFetcher.py — Safe, retrying HTTP fetcher with rotating User-Agents.
"""

from typing import Optional
from urllib.parse import urlparse



# ── URL validation ────────────────────────────────────────────────

BLOCKED_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})
PRIVATE_IP_PREFIXES = (
    "10.", "192.168.", "172.16.", "172.17.", "172.18.",
    "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
    "172.29.", "172.30.", "172.31.",
)


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