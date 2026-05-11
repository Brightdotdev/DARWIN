# src/Utils/url_fetcher.py
"""
Bare‑minimum URL content fetcher with optional caching.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional
import json
import aiohttp
from src.features.Query.QueryTypes import URLContent

DEFAULT_CACHE_DIR = Path.home() / ".darwin" / "url_cache"
CACHE_TTL = 3600  # 1 hour


async def fetch_url(
    url: str,
    timeout: float = 10,
    cache_dir: Optional[Path] = None,
    allow_search: bool = False,
) -> Optional[URLContent]:
    """
    Fetch and extract readable text from a URL.
    Set allow_search=False to skip external fetches entirely.
    """
    if not allow_search:
        return None

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_file = cache_dir / url_hash
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if time.time() - data["fetched_at"] < CACHE_TTL:
                return URLContent(**data)
        except Exception:
            pass

    # Fetch
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()
                from trafilatura import extract
                clean_text = extract(html, output_format="txt")
                if not clean_text:
                    return None
                content = URLContent(
                    url=url,
                    title="",
                    fetched_text=clean_text.strip(),
                    fetched_at=time.time(),
                )
                # Cache
                cache_file.write_text(json.dumps(content.__dict__))
                return content
    except Exception:
        return None