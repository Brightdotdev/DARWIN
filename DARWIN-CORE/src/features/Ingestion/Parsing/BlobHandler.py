# ============================================================================
# src/core/Parsing/BlobHandler.py
# ============================================================================

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

from src.features.Ingestion.Parsing.Parser import parse_url, is_url,is_good_enough
from src.features.Ingestion.Parsing.lib import ParsedDocument, BlobInput
from src.core.Logger.DarwinLogger import darwin, LogType


# ── Link extraction ─────────────────────────────────────────────────────────

# Improved regex: captures parentheses in URLs (Wikipedia-style)
# Allows balanced parens at end, stops at whitespace or unambiguous URL terminators
_URL_PATTERN = re.compile(
    r'https?://'
    r'(?:[-\w.])+(?::\d+)?'  # host + optional port
    r'(?:/[^\n\s<>"{}|\\^`\[\]]*'  # path
    r'(?:\([^\n\s<>"{}|\\^`\[\]]*\))?'  # optional (parenthesized) suffix
    r'(?:/[^\n\s<>"{}|\\^`\[\]]*)*)?',  # more path after parens
    re.IGNORECASE
)


def extract_links(text: str) -> List[Dict]:
    """Extract all HTTP(S) URLs from text with position info."""
    links = []
    seen = set()
    for match in _URL_PATTERN.finditer(text):
        url = match.group(0)
        
        # Smart trailing punctuation strip: don't strip ) if URL has (
        while url and url[-1] in ".,;:!?":
            url = url[:-1]
        # Only strip ) if there's no ( in the URL
        if url.endswith(")") and "(" not in url:
            url = url[:-1]
        
        if url not in seen and len(url) > 10:
            seen.add(url)
            links.append({
                "url": url,
                "start": match.start(),
                "end": match.start() + len(url),  # adjust end to stripped length
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
    """
    Replace a pending placeholder with resolved content or failure reason.
    Handles trailing markdown syntax (like (mythology)) after placeholder.
    """
    # Find the placeholder — it may have trailing text immediately after ]
    old_pattern = f'[link:{idx} | domain:{link["domain"]} | content:<pending> | url:{link["url"]}]'

    # Try exact match first
    if old_pattern in text:
        if link["status"] == "resolved" and link.get("content"):
            content = link["content"][:200].replace("]", "⟩").replace("[", "⟨")
            if len(link["content"]) > 200:
                content += "..."
            new = f'[link:{idx} | domain:{link["domain"]} | content:{content} | url:{link["url"]}]'
        elif link["status"] == "failed":
            reason = link.get("error", "unknown error")[:100]
            new = f'[link:{idx} | domain:{link["domain"]} | content:<failed: {reason}> | url:{link["url"]}]'
        else:
            new = old_pattern  # unchanged

        return text.replace(old_pattern, new, 1)

    # Fallback: search for pattern with possible trailing chars
    # This handles cases where markdown syntax got appended
    escaped_url = re.escape(link["url"])
    pattern = re.compile(
        rf'\[link:{re.escape(str(idx))} \| domain:{re.escape(link["domain"])} \| content:<pending> \| url:{escaped_url}\]'
        r'([^\n\r]*)'  # capture trailing chars on same line
    )

    match = pattern.search(text)
    if match:
        full_match = match.group(0)
        trailing = match.group(1)

        if link["status"] == "resolved" and link.get("content"):
            content = link["content"][:200].replace("]", "⟩").replace("[", "⟨")
            if len(link["content"]) > 200:
                content += "..."
            new = f'[link:{idx} | domain:{link["domain"]} | content:{content} | url:{link["url"]}]{trailing}'
        elif link["status"] == "failed":
            reason = link.get("error", "unknown error")[:100]
            new = f'[link:{idx} | domain:{link["domain"]} | content:<failed: {reason}> | url:{link["url"]}]{trailing}'
        else:
            return text  # unchanged

        return text.replace(full_match, new, 1)

    # Could not find placeholder — log and return unchanged
    darwin.log(f"Could not find placeholder for link {idx} ({link['url']})", LogType.WARNING)
    return text


# ── Link fetching (immediate, synchronous) ──────────────────────────────────

def fetch_link_content(url: str, timeout: float = 15.0) -> Tuple[Optional[str], Optional[str]]:
    """
    Immediately fetch and parse a link found in a blob.
    Returns (content, error).
    """
    try:
        darwin.log(f"    Fetching {url}...", LogType.INFO)
        doc = parse_url(url, precise=False, timeout=timeout)

        if doc.error:
            return None, f"fetch failed: {doc.error}"

        text = doc.display_text or doc.embedding_text
        if not text or len(text.strip()) < 50:
            return None, "extracted content is empty or too short"

        return text[:2000], None

    except ImportError as e:
        return None, f"missing dependency: {e}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"



class BlobHandler:
    """
    Handles user-pasted blobs: extracts links, fetches them immediately,
    replaces inline with placeholders, returns unified ParsedDocument.
    """

    @staticmethod
    def _detect_mime_hint(text: str) -> str:
        """Auto-detect if raw text is HTML, Markdown, or plain text."""
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
        darwin.log(f"  Extracted {len(links)} link(s) from {filename}", LogType.INFO)

        # ── Step 2: Replace with placeholders ──────────────────────────────
        if links:
            text = replace_links_with_placeholders(text, links)
            darwin.log(f"  Replaced links with placeholders", LogType.INFO)

        # ── Step 3: HTML extraction (if HTML blob) ──────────────────────────
        if mime_hint in ("text/html", "application/xhtml+xml"):
            try:
                from trafilatura import extract
                extracted = extract(
                    text,
                    output_format="txt",
                    favor_recall=False,
                    include_comments=False,
                    include_tables=True,
                    deduplicate=True,
                )
                if extracted:
                    text = extracted
            except ImportError:
                darwin.log("  trafilatura not installed, parsing blob as raw text", LogType.WARNING)

        # ── Step 4: Immediate link resolution ───────────────────────────────
        if blob.resolve_links and links:
            darwin.log(f"  Resolving {len(links)} link(s)...", LogType.INFO)

            for idx, link in enumerate(links):
                darwin.log(f"    Link {idx + 1}/{len(links)}: {link['url']}", LogType.INFO)
                content, error = fetch_link_content(link["url"], timeout=blob.link_timeout)

                if content:
                    link["status"] = "resolved"
                    link["content"] = content
                    darwin.log(f"      OK: {len(content)} chars from {link['domain']}", LogType.SUCCESS)
                else:
                    link["status"] = "failed"
                    link["error"] = error or "unknown error"
                    darwin.log(f"      FAIL: {error}", LogType.WARNING)

                # Replace placeholder immediately
                text_before = text
                text = resolve_placeholder(text, idx, link)
                if text == text_before:
                    darwin.log(f"      WARNING: placeholder not replaced", LogType.WARNING)
        elif links and not blob.resolve_links:
            darwin.log(f"  Link resolution disabled (resolve_links=False)", LogType.INFO)
            for link in links:
                link["status"] = "skipped"
                link["error"] = "resolution disabled"
        else:
            darwin.log(f"  No links to resolve", LogType.INFO)

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
            "links": links,
        }

        page_count = max(1, len(text.split()) // 500)

        darwin.log(f"  Processed: {len(text)} chars, {len(links)} links", LogType.SUCCESS)

        return ParsedDocument(
            display_text=text,
            embedding_text=text,
            filename=filename,
            page_count=page_count,
            file_type="blob",
            metadata=metadata,
            error=None,
        )