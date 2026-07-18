import asyncio
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Pipeline.Ingestion.Parsing.UrlFetcher import fetch_url
from src.core.Pipeline.Ingestion.Parsing.HtmlProcessor import extract_content
from src.core.Pipeline.Ingestion.Parsing.lib import ParsedDocument, BlobInput, ExtractedImage
from src.core.Pipeline.Ingestion.Parsing.lib import is_good_enough

# ── Constants ──────────────────────────────────────────────────────────────
_MAX_BLOB_BYTES = 10 * 1024 * 1024       # 10 MB
_MAX_LINK_CONCURRENCY = 5
_MAX_LINK_FETCH_TIME = 15.0
_MAX_LINK_CONTENT_LEN = 2000
_MAX_LINK_SNIPPET_LEN = 200

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_URL_PATTERN = re.compile(
    r'https?://'
    r'(?:[-\w.])+(?::\d+)?'
    r'(?:/[^\n\s<>"{}|\\^`\[\]]*'
    r'(?:\([^\n\s<>"{}|\\^`\[\]]*\))?'
    r'(?:/[^\n\s<>"{}|\\^`\[\]]*)*)?',
    re.IGNORECASE
)


# ── Internal helpers ─────────────────────────────────────────────────────

def _is_url(s: str) -> bool:
    return bool(_URL_RE.match(s))

def _sanitize_filename(url: str) -> str:
    """Create a safe filename from a URL for image storage."""
    h = hashlib.sha256(url.encode()).hexdigest()[:12]
    parsed = urlparse(url)
    name = Path(parsed.path).name or "image"
    name = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
    if not name or name.startswith('.'):
        name = f"img_{h}"
    return f"{h}_{name}"

def _strip_trailing_punctuation(url: str) -> str:
    while url and url[-1] in ".,;:!?)":
        if url[-1] == ")" and "(" not in url:
            url = url[:-1]
        elif url[-1] in ".,;:!?":
            url = url[:-1]
        else:
            break
    return url


# ── Link token / placeholder system (index-based, no fragile string replace) ─

@dataclass
class _LinkToken:
    idx: int
    url: str
    domain: str
    status: str = "pending"
    content: Optional[str] = None
    error: Optional[str] = None

    def render(self) -> str:
        status_label = "ok" if self.status == "resolved" else "failed"
        snippet = ""
        if self.status == "resolved" and self.content:
            snippet = self.content[:_MAX_LINK_SNIPPET_LEN].replace("[", "⟨").replace("]", "⟩")
            if len(self.content) > _MAX_LINK_SNIPPET_LEN:
                snippet += "..."
        elif self.status == "failed":
            snippet = f"<failed: {self.error or 'unknown'}>"

        return (
            f"[Link:{self.idx} | domain:{self.domain} | "
            f"status:{status_label} | content:{snippet} | url:{self.url}]"
        )


@dataclass
class _LinkMatch:
    start: int
    end: int
    token: _LinkToken


def _extract_links(text: str) -> List[_LinkMatch]:
    """Extract URLs with positional info for robust segment assembly."""
    matches: List[_LinkMatch] = []
    seen: set[str] = set()
    for m in _URL_PATTERN.finditer(text):
        raw_url = m.group(0)
        url = _strip_trailing_punctuation(raw_url)
        if url not in seen and len(url) > 10:
            seen.add(url)
            matches.append(_LinkMatch(
                start=m.start(),
                end=m.start() + len(raw_url),
                token=_LinkToken(
                    idx=len(matches),
                    url=url,
                    domain=urlparse(url).netloc or "unknown",
                ),
            ))
    return matches


def _assemble_text(segments: List[Union[str, _LinkToken]]) -> str:
    """Join text segments and rendered link tokens into final string."""
    return "".join(seg if isinstance(seg, str) else seg.render() for seg in segments)


# ── Async link resolution ─────────────────────────────────────────────────

async def _resolve_links_async(
    tokens: List[_LinkToken],
    timeout: float = _MAX_LINK_FETCH_TIME,
) -> None:
    """Resolve all links concurrently with bounded concurrency and deduplication."""

    # Deduplicate by URL — share results across duplicates
    url_to_tokens: Dict[str, List[_LinkToken]] = {}
    for tok in tokens:
        url_to_tokens.setdefault(tok.url, []).append(tok)

    unique_urls = list(url_to_tokens.keys())
    if not unique_urls:
        return

    semaphore = asyncio.Semaphore(_MAX_LINK_CONCURRENCY)

    async def _fetch_one(url: str) -> Tuple[Optional[str], Optional[str]]:
        async with semaphore:
            try:
                return await asyncio.to_thread(resolve_link_content, url, timeout)
            except Exception as e:
                return None, f"{type(e).__name__}: {e}"

    darwin.info(
        f"  Resolving {len(unique_urls)} unique link(s) "
        f"({len(tokens)} total) with concurrency={_MAX_LINK_CONCURRENCY}..."
    )

    results = await asyncio.gather(*[_fetch_one(url) for url in unique_urls])

    for url, (content, error) in zip(unique_urls, results):
        for tok in url_to_tokens[url]:
            if content:
                tok.status = "resolved"
                tok.content = content
            else:
                tok.status = "failed"
                tok.error = error or "unknown error"

        domain = url_to_tokens[url][0].domain
        if content:
            darwin.success(f"    {domain}: {len(content)} chars")
        else:
            darwin.warning(f"    {domain} failed: {error}")


def resolve_link_content(url: str, timeout: float = _MAX_LINK_FETCH_TIME) -> Tuple[Optional[str], Optional[str]]:
    """Fetch a URL and return a short text snippet."""
    try:
        data, content_type, err = fetch_url(url, timeout=timeout)
        if err:
            return None, err

        text, meta = extract_content(url, html_bytes=data, precise=False)
        if meta.get("error"):
            return None, meta["error"]

        if not text or len(text.strip()) < 50:
            return None, "insufficient content"

        return text[:_MAX_LINK_CONTENT_LEN], None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ── Image extraction from HTML blobs ─────────────────────────────────────

def _extract_images_from_html(
    html_text: str,
    base_url: Optional[str] = None,
    image_dir: Optional[Path] = None,
) -> Tuple[List[ExtractedImage], List[str]]:
    """
    Parse HTML for <img> tags, download images, enrich via ImageHandler.
    Returns (ExtractedImage list, log messages).
    """
    images: List[ExtractedImage] = []
    logs: List[str] = []

    if not image_dir:
        return images, logs

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logs.append("BeautifulSoup not available — skipping image extraction")
        return images, logs

    soup = BeautifulSoup(html_text, "html.parser")
    img_tags = soup.find_all("img")

    if not img_tags:
        return images, logs

    logs.append(f"Found {len(img_tags)} <img> tag(s) in HTML blob")

    # Lazy import to avoid circular deps at module load
    from src.features.Images.ImageHandler import ImageHandler

    handler = ImageHandler()
    img_paths: List[Path] = []
    img_entries: List[ExtractedImage] = []

    for idx, tag in enumerate(img_tags):
        src = tag.get("src")
        if not src:
            continue

        # Resolve relative URLs
        if base_url and not _is_url(src):
            src = urljoin(base_url, src)
        elif not _is_url(src):
            logs.append(f"  Skipping relative image without base URL: {src}")
            continue

        # Download
        safe_name = _sanitize_filename(src)
        local_path = image_dir / safe_name

        try:
            data, _, err = fetch_url(src, timeout=10.0)
            if err or not data:
                logs.append(f"  Failed to download image {idx}: {err or 'empty response'}")
                continue
            local_path.write_bytes(data)
            img_paths.append(local_path)
            img_entries.append(ExtractedImage(
                path=local_path,
                page=0,
                index=idx,
                caption="",
                width=0,
                height=0,
            ))
        except Exception as e:
            logs.append(f"  Failed to download image {idx}: {e}")

    # Run enrichment if we got images
    if img_paths:
        logs.append(f"  Downloaded {len(img_paths)} image(s), enriching...")
        try:
            results = handler.process_batch(
                img_paths,
                stages="both",
                auto_ensure=True,
                unload_after=True,
                use_cache=True,
            )
            for entry, result in zip(img_entries, results):
                if result.error:
                    logs.append(f"    {entry.path.name}: enrichment failed")
                    entry.caption = ""
                else:
                    parts = []
                    if result.description:
                        parts.append(result.description)
                    if result.text:
                        parts.append(f"OCR: {result.text}")
                    entry.caption = " | ".join(parts) if parts else ""
        except Exception as e:
            logs.append(f"  Image enrichment failed: {e}")

    return img_entries, logs


# ── BlobHandler ────────────────────────────────────────────────────────────

class BlobHandler:
    """Stateless handler for user‑pasted blobs with async link resolution,
    image extraction, and deduplication."""

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
    async def process(cls, blob: BlobInput) -> ParsedDocument:
        text = blob.content
        filename = blob.filename
        image_dir = blob.image_dir

        # ── Size guard ─────────────────────────────────────────────────
        raw_bytes = text.encode("utf-8", errors="replace")
        if len(raw_bytes) > _MAX_BLOB_BYTES:
            darwin.warning(
                f"Blob exceeds {_MAX_BLOB_BYTES / 1_048_576:.1f} MB, truncating"
            )
            text = raw_bytes[:_MAX_BLOB_BYTES].decode("utf-8", errors="ignore")

        # ── Auto-detect format ─────────────────────────────────────────
        mime_hint = blob.mime_hint
        if mime_hint == "text/plain":
            mime_hint = cls._detect_mime_hint(text)

        # ── Step 1: Extract images from raw HTML (before cleaning) ─────
        images: List[ExtractedImage] = []
        if mime_hint == "text/html" and image_dir:
            base_url = blob.source if blob.source and _is_url(blob.source) else None
            extracted_imgs, img_logs = _extract_images_from_html(
                text, base_url=base_url, image_dir=image_dir
            )
            images.extend(extracted_imgs)
            for log_msg in img_logs:
                darwin.info(f"  {log_msg}")

        # ── Step 2: Clean HTML blobs ───────────────────────────────────
        if mime_hint in ("text/html", "application/xhtml+xml"):
            cleaned, _ = extract_content(
                "blob://", html_bytes=text.encode("utf-8", errors="replace")
            )
            if cleaned:
                text = cleaned
            else:
                darwin.warning("  Could not extract blob HTML content")

        # ── Step 3: Extract links (deduplicated) ─────────────────────
        link_matches = _extract_links(text)
        darwin.info(f"  Found {len(link_matches)} unique link(s) in blob")

        # Build segmented text (no fragile string replacement)
        segments: List[Union[str, _LinkToken]] = []
        last_end = 0
        for match in link_matches:
            segments.append(text[last_end:match.start])
            segments.append(match.token)
            last_end = match.end
        segments.append(text[last_end:])

        # ── Step 4: Resolve links (async, concurrent, deduplicated) ────
        if blob.resolve_links and link_matches:
            await _resolve_links_async(
                [m.token for m in link_matches],
                timeout=blob.link_timeout,
            )
        elif link_matches and not blob.resolve_links:
            darwin.info("  Link resolution disabled")

        # ── Step 5: Quality check ──────────────────────────────────────
        assembled = _assemble_text(segments)
        if not is_good_enough(assembled, darwin):
            return ParsedDocument(
                embedding_text="", filename=filename,
                page_count=0, file_type="blob",
                error="blob content is empty or garbled",
            )

        # ── Step 6: Assemble result ───────────────────────────────────
        metadata = {
            "source": blob.source,
            "title": blob.title,
            "mime_hint": mime_hint,
            "links": [
                {
                    "url": m.token.url,
                    "domain": m.token.domain,
                    "status": m.token.status,
                    "content": m.token.content,
                    "error": m.token.error,
                }
                for m in link_matches
            ],
        }
        page_count = max(1, len(assembled.split()) // 500)

        return ParsedDocument(
            embedding_text=assembled,
            filename=filename,
            page_count=page_count,
            file_type="blob",
            images=images,
            metadata=metadata,
            error=None,
        )