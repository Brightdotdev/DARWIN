# ============================================================================
# src/core/Parsing/BlobHandler.py
# ============================================================================

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

from src.features.Ingestion.Parsing.Parser import parse_url, is_url
from src.features.Ingestion.Parsing.lib import ParsedDocument, is_good_enough
from src.core.Logger.DarwinLogger import darwin, LogType


# ── Link extraction ─────────────────────────────────────────────────────────

_URL_PATTERN = re.compile(
    r'https?://(?:[-\w.])+(?::\d+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
    re.IGNORECASE
)


def extract_links(text: str) -> List[Dict]:
    """Extract all HTTP(S) URLs from text with position info."""
    links = []
    seen = set()
    for match in _URL_PATTERN.finditer(text):
        url = match.group(0)
        if url not in seen:
            seen.add(url)
            links.append({
                "url": url,
                "start": match.start(),
                "end": match.end(),
                "domain": urlparse(url).netloc or "unknown",
                "status": "pending",
                "content": None,
                "error": None,
            })
    return links


# ── Inline link replacement ───────────────────────────────────────────────

def replace_links_with_placeholders(text: str, links: List[Dict]) -> str:
    """Replace URLs in text with numbered placeholders."""
    # Sort by position descending to avoid offset shifts
    sorted_links = sorted(enumerate(links), key=lambda x: x[1]["start"], reverse=True)
    result = text
    for idx, link in sorted_links:
        placeholder = f'[link:{idx} | domain:{link["domain"]} | content:<pending> | url:{link["url"]}]'
        result = result[:link["start"]] + placeholder + result[link["end"]:]
    return result


def resolve_placeholder(text: str, idx: int, link: Dict) -> str:
    """Replace a pending placeholder with resolved content or failure reason."""
    old = f'[link:{idx} | domain:{link["domain"]} | content:<pending> | url:{link["url"]}]'

    if link["status"] == "resolved" and link["content"]:
        # Truncate content for inline display (keep it readable)
        content = link["content"][:200].replace("]", "⟩").replace("[", "⟨")
        if len(link["content"]) > 200:
            content += "..."
        new = f'[link:{idx} | domain:{link["domain"]} | content:{content} | url:{link["url"]}]'
    elif link["status"] == "failed":
        reason = link.get("error", "unknown error")[:100]
        new = f'[link:{idx} | domain:{link["domain"]} | content:<failed: {reason}> | url:{link["url"]}]'
    else:
        new = old  # unchanged

    return text.replace(old, new, 1)


# ── Link fetching (immediate, synchronous) ──────────────────────────────────

def fetch_link_content(url: str, timeout: float = 15.0) -> Tuple[Optional[str], Optional[str]]:
    """
    Immediately fetch and parse a link found in a blob.
    Returns (content, error).
    """
    try:
        doc = parse_url(url, precise=False, timeout=timeout)
        if doc.error:
            return None, doc.error

        # Use display_text as the extracted content
        text = doc.display_text or doc.embedding_text
        if not text or len(text.strip()) < 50:
            return None, "extracted content is empty or too short"

        # Truncate for inline embedding (full content available in metadata if needed)
        return text[:2000], None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ── BlobHandler ─────────────────────────────────────────────────────────────

@dataclass
class BlobInput:
    """Raw text content from clipboard, chat, or other non-file sources."""
    content: str                        # The raw text/HTML/markdown
    source: str                         # e.g. "clipboard", "chat", "web_clip"
    title: str = ""                     # Optional title/context
    mime_hint: str = "text/plain"       # text/plain, text/html, text/markdown
    turn_id: Optional[str] = None       # Conversation turn for unique filenames
    resolve_links: bool = True           # Immediately fetch linked URLs?
    link_timeout: float = 15.0          # Per-link fetch timeout

    @property
    def filename(self) -> str:
        """Generate unique filename from turn + source."""
        prefix = f"{self.turn_id}_" if self.turn_id else ""
        return f"{prefix}{self.source}"


class BlobHandler:
    """
    Handles user-pasted blobs: extracts links, fetches them immediately,
    replaces inline with placeholders, returns unified ParsedDocument.
    """

    @staticmethod
    def _detect_mime_hint(text: str) -> str:
        """Auto-detect if raw text is HTML, Markdown, or plain text."""
        stripped = text.strip()

        # HTML: starts with < or contains <!DOCTYPE or <html
        if stripped.startswith("<") or "<!DOCTYPE" in stripped[:100].upper() or "<html" in stripped[:100].lower():
            return "text/html"

        # Markdown: has significant markdown syntax
        md_patterns = ["# ", "## ", "### ", "- ", "* ", "> ", "```", "[", "](", "| ---", "**", "__"]
        md_score = sum(1 for p in md_patterns if p in text[:500])
        if md_score >= 2:
            return "text/markdown"

        return "text/plain"

    @classmethod
    def process(cls, blob: BlobInput) -> ParsedDocument:
        """
        Process a blob: extract links, fetch immediately, replace placeholders,
        return ParsedDocument ready for chunk + embed.
        """
        text = blob.content
        filename = blob.filename

        # Auto-detect mime hint if default
        mime_hint = blob.mime_hint
        if mime_hint == "text/plain":
            mime_hint = cls._detect_mime_hint(text)

        # ── Step 1: Extract links ───────────────────────────────────────────
        links = extract_links(text)

        # ── Step 2: Replace with placeholders ──────────────────────────────
        if links:
            text = replace_links_with_placeholders(text, links)
            darwin.log(f"Extracted {len(links)} links from {filename}", LogType.INFO)

        # ── Step 3: HTML extraction (if HTML blob) ──────────────────────────
        if mime_hint in ("text/html", "application/xhtml+xml"):
            try:
                from trafilatura import extract
                extracted = extract(
                    text,
                    output_format="txt",  # Keep placeholders intact
                    favor_recall=False,
                    include_comments=False,
                    include_tables=True,
                    deduplicate=True,
                )
                if extracted:
                    text = extracted
            except ImportError:
                darwin.log("trafilatura not installed, parsing blob as raw text", LogType.WARNING)

        # ── Step 4: Immediate link resolution (if enabled) ──────────────────
        if blob.resolve_links and links:
            for idx, link in enumerate(links):
                darwin.log(f"Fetching link {idx + 1}/{len(links)}: {link['url']}", LogType.INFO)
                content, error = fetch_link_content(link["url"], timeout=blob.link_timeout)

                if content:
                    link["status"] = "resolved"
                    link["content"] = content
                    darwin.log(f"  OK {link['domain']}: {len(content)} chars", LogType.SUCCESS)
                else:
                    link["status"] = "failed"
                    link["error"] = error or "unknown"
                    darwin.log(f"  FAIL {link['domain']}: {error}", LogType.WARNING)

                # Replace placeholder immediately
                text = resolve_placeholder(text, idx, link)

        # ── Step 5: Quality check ────────────────────────────────────────
        if not is_good_enough(text):
            return ParsedDocument(
                display_text="", embedding_text="", filename=filename,
                page_count=0, file_type="blob",
                error="blob content is empty or garbled after processing",
            )

        # ── Step 6: Build metadata ──────────────────────────────────────────
        metadata = {
            "source": blob.source,
            "title": blob.title,
            "mime_hint": mime_hint,
            "links": links,  # Full link data for downstream use
        }

        page_count = max(1, len(text.split()) // 500)

        darwin.log(f"Processed blob {filename}: {len(text)} chars, {len(links)} links", LogType.SUCCESS)

        return ParsedDocument(
            display_text=text,
            embedding_text=text,
            filename=filename,
            page_count=page_count,
            file_type="blob",
            metadata=metadata,
            error=None,
        )