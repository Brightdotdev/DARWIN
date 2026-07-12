import re
from typing import Tuple, Optional
from src.core.Logger.DarwinLogger import darwin, LogType

MIN_EXTRACTED_TEXT_LENGTH = 100


def _needs_js_rendering(html_bytes: bytes) -> bool:
    """Heuristic to detect pages that are mostly JavaScript shells."""
    try:
        text = html_bytes.decode("utf-8", errors="replace").lower()
    except Exception:
        return False

    indicators = [
        '<div id="root"', '<div id="app"',
        'window.__nuxt__', 'window.__next',
        'react-root', 'vue-app',
        '<noscript>you need to enable javascript',
        '<noscript>please enable javascript',
        '<script>window.__remixContext',
    ]
    if any(indicator in text for indicator in indicators):
        return True

    body_match = re.search(r'<body[^>]*>(.*?)</body>', text, re.DOTALL)
    if body_match and len(body_match.group(1).strip()) < 200:
        return True

    return False


def extract_content(
    url: str,
    html_bytes: Optional[bytes] = None,
    precise: bool = False,
) -> Tuple[str, dict]:
    """
    Extract main content from a URL or raw HTML bytes.

    Returns (text, metadata_dict).
    text will be empty if extraction fails, and metadata may contain 'error'.
    """
    if html_bytes is None:
        from src.core.Pipeline.Ingestion.Parsing.UrlFetcher import fetch_url
        html_bytes, _, err = fetch_url(url)
        if err:
            return "", {"error": err}

    # Step 1: Trafilatura
    try:
        from trafilatura import extract, bare_extraction
    except ImportError:
        darwin.warning("Trafilatura not installed, falling back to BeautifulSoup")
    else:
        extracted = extract(
            html_bytes,
            output_format="markdown" if precise else "txt",
            url=url,
            favor_recall=precise,
            include_comments=False,
            include_tables=True,
            deduplicate=True,
            with_metadata=False,
        )
        if extracted and len(extracted.strip()) >= MIN_EXTRACTED_TEXT_LENGTH:
            meta_dict = {}
            try:
                meta = bare_extraction(html_bytes, output_format="dict", url=url, with_metadata=True)
                if isinstance(meta, dict):
                    meta_dict = {k: v for k, v in meta.items() if k != "text"}
            except Exception:
                pass
            darwin.info(f"Trafilatura extracted {len(extracted)} chars from {url}")
            return extracted, meta_dict

    # Step 2: JS detection
    js_required = _needs_js_rendering(html_bytes)
    if js_required:
        darwin.warning(f"Page at {url} appears to require JavaScript rendering")

    # Step 3: BeautifulSoup fallback
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        darwin.error("BeautifulSoup not installed, cannot perform fallback extraction")
        return "", {"error": "extraction dependencies missing (install trafilatura or beautifulsoup4)"}

    soup = BeautifulSoup(html_bytes, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    if len(text) >= MIN_EXTRACTED_TEXT_LENGTH:
        darwin.info(f"BeautifulSoup extracted {len(text)} chars from {url}")
        return text, {}
    else:
        if js_required:
            msg = "Page is empty or requires JavaScript rendering (not yet supported)."
        else:
            msg = "Page content is empty or could not be extracted."
        return "", {"error": msg}