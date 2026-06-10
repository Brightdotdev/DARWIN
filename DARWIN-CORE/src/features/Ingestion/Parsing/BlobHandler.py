"""
BlobHandler.py — Process user-pasted blobs.

Links are extracted, fetched, and their content summarised inline.
HTML blobs are cleaned with Trafilatura via the shared HtmlProcessor.
Images are stubbed out — plug your OCR/captioning here later.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.Ingestion.Parsing.UrlFetcher import fetch_url
from src.features.Ingestion.Parsing.HtmlProcessor import extract_content
from src.features.Ingestion.Parsing.lib import ParsedDocument, BlobInput, ExtractedImage
from src.features.Ingestion.Parsing.lib import is_good_enough

# ── URL detection ──────────────────────────────────────────────────────────
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

def _is_url(s: str) -> bool:
    return bool(_URL_RE.match(s))

# ── Link extraction (unchanged) ────────────────────────────────────────────
_URL_PATTERN = re.compile(
    r'https?://'
    r'(?:[-\w.])+(?::\d+)?'
    r'(?:/[^\n\s<>"{}|\\^`\[\]]*'
    r'(?:\([^\n\s<>"{}|\\^`\[\]]*\))?'
    r'(?:/[^\n\s<>"{}|\\^`\[\]]*)*)?',
    re.IGNORECASE
)

def extract_links(text: str) -> List[Dict]:
    links = []
    seen = set()
    for match in _URL_PATTERN.finditer(text):
        url = match.group(0)
        while url and url[-1] in ".,;:!?":
            url = url[:-1]
        if url.endswith(")") and "(" not in url:
            url = url[:-1]
        if url not in seen and len(url) > 10:
            seen.add(url)
            links.append({
                "url": url,
                "start": match.start(),
                "end": match.start() + len(url),
                "domain": urlparse(url).netloc or "unknown",
                "status": "pending",
                "content": None,
                "error": None,
            })
    return links

# ── Placeholder helpers (slightly simplified) ──────────────────────────────
_PLACEHOLDER_TEMPLATE = '[Link:{idx} | domain:{domain} | status:{status} | content:{content} | url:{url}]'

def replace_links_with_placeholders(text: str, links: List[Dict]) -> str:
    """Replace each URL with a placeholder that will be resolved later."""
    # Sort descending by start to avoid offset issues
    for idx, link in sorted(enumerate(links), key=lambda x: x[1]["start"], reverse=True):
        placeholder = _PLACEHOLDER_TEMPLATE.format(
            idx=idx, domain=link["domain"], status="pending", content="...", url=link["url"]
        )
        text = text[:link["start"]] + placeholder + text[link["end"]:]
    return text

def resolve_placeholder(text: str, idx: int, link: Dict) -> str:
    """Insert resolved content (or error) into the placeholder."""
    status = "ok" if link["status"] == "resolved" else "failed"
    content = ""
    if link["status"] == "resolved" and link.get("content"):
        content = link["content"][:200].replace("[", "⟨").replace("]", "⟩")
        if len(link["content"]) > 200:
            content += "..."
    elif link["status"] == "failed":
        content = f"<failed: {link.get('error', 'unknown')}>"

    new = _PLACEHOLDER_TEMPLATE.format(
        idx=idx, domain=link["domain"], status=status, content=content, url=link["url"]
    )

    # Find the old placeholder pattern (status=pending) and replace
    old = _PLACEHOLDER_TEMPLATE.format(
        idx=idx, domain=link["domain"], status="pending", content="...", url=link["url"]
    )
    # Simple replacement – placeholder is unique enough
    return text.replace(old, new, 1)

# ── Link resolution (no longer uses parse_url) ─────────────────────────────
def resolve_link_content(url: str, timeout: float = 15.0) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch a URL and return a short text snippet (max 2000 chars).
    Uses the same URL‑agnostic pipeline as the main parser.
    """
    try:
        data, content_type, err = fetch_url(url, timeout=timeout)
        if err:
            return None, err

        # Extract content from the bytes (HTML, plain text, etc.)
        text, meta = extract_content(url, html_bytes=data, precise=False)
        if meta.get("error"):
            return None, meta["error"]

        if not text or len(text.strip()) < 50:
            return None, "insufficient content"

        return text[:2000], None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# ── BlobHandler ────────────────────────────────────────────────────────────
class BlobHandler:
    """Stateless handler for user‑pasted blobs."""

    @staticmethod
    def _detect_mime_hint(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("<") or "<!DOCTYPE" in stripped[:100].upper() or "<html" in stripped[:100].lower():
            return "text/html"
        md_patterns = ["# ", "## ", "### ", "- ", "* ", "> ", "```", "[", "](", "| ---", "**", "__"]
        md_score = sum(1 for p in md_patterns if p in text[:500])
        if md_score >= 2:
            return "text/markdown"
        return "text/plain"

    @classmethod
    def process(cls, blob: BlobInput) -> ParsedDocument:
        text = blob.content
        filename = blob.filename
        image_dir = blob.image_dir   # may be None – fine for now

        # Auto‑detect format
        mime_hint = blob.mime_hint
        if mime_hint == "text/plain":
            mime_hint = cls._detect_mime_hint(text)

        # ── Step 1: Extract links ──────────────────────────────────────
        links = extract_links(text)
        darwin.log(f"  Found {len(links)} link(s) in blob", LogType.INFO)

        # ── Step 2: Insert placeholders ─────────────────────────────────
        if links:
            text = replace_links_with_placeholders(text, links)

        # ── Step 3: Clean the blob itself if it's HTML ─────────────────
        if mime_hint in ("text/html", "application/xhtml+xml"):
            # Use the shared HtmlProcessor (Trafilatura → BeautifulSoup)
            cleaned, _ = extract_content("blob://", html_bytes=text.encode("utf-8", errors="replace"))
            if cleaned:
                text = cleaned
            else:
                darwin.log("  Could not extract blob HTML content", LogType.WARNING)

        # ── Step 4: Resolve links (optional) ────────────────────────────
        if blob.resolve_links and links:
            darwin.log(f"  Resolving {len(links)} link(s)...", LogType.INFO)
            for idx, link in enumerate(links):
                content, error = resolve_link_content(link["url"], timeout=blob.link_timeout)
                if content:
                    link["status"] = "resolved"
                    link["content"] = content
                    darwin.log(f"    [{idx+1}] {link['domain']}: {len(content)} chars", LogType.SUCCESS)
                else:
                    link["status"] = "failed"
                    link["error"] = error or "unknown error"
                    darwin.log(f"    [{idx+1}] {link['domain']} FAILED: {error}", LogType.WARNING)

                text = resolve_placeholder(text, idx, link)
        elif links and not blob.resolve_links:
            darwin.log("  Link resolution disabled", LogType.INFO)

        # ── Step 5: Mock image extraction ──────────────────────────────
        images: List[ExtractedImage] = []
        if image_dir:
            # TODO: Plug OCR / captioning here.
            # For now, this is a stub that would find images in the blob,
            # download them to image_dir, and create ExtractedImage entries.
            darwin.log("  Image extraction is not yet implemented – skipping", LogType.INFO)

        # ── Step 6: Quality check ──────────────────────────────────────
        if not is_good_enough(text,darwin):
            return ParsedDocument(
                display_text="", embedding_text="", filename=filename,
                page_count=0, file_type="blob",
                error="blob content is empty or garbled",
            )

        # ── Step 7: Assemble result ────────────────────────────────────
        metadata = {
            "source": blob.source,
            "title": blob.title,
            "mime_hint": mime_hint,
            "links": links,
        }
        page_count = max(1, len(text.split()) // 500)

        return ParsedDocument(
            display_text=text,
            embedding_text=text,
            filename=filename,
            page_count=page_count,
            file_type="blob",
            images=images,
            metadata=metadata,
            error=None,
        )