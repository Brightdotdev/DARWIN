"""
src/features/Query/InputPreprocessor.py
-----------------------------------
Sanitizes raw user input before classification or retrieval.

- Extracts URLs and image references
- Detects pasted text blobs (code blocks, tables, long documents)
- Produces a structured RawQuery for downstream consumers
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class PastedTextType(Enum):
    DOCUMENT = "document"
    CODE = "code"
    DATA = "data"
    LONG_QUESTION = "long_question"


@dataclass
class URLRef:
    url: str
    domain: str
    placeholder: str


@dataclass
class PastedText:
    content: str
    type: PastedTextType


@dataclass
class RawQuery:
    query: str                      # cleaned text with URLs/images stripped
    pasted_texts: List[PastedText] = field(default_factory=list)
    pinned_files: List[str] = field(default_factory=list)
    extracted_urls: List[URLRef] = field(default_factory=list)
    image_refs: List[str] = field(default_factory=list)


class InputPreprocessor:
    """Lightweight regex-based input sanitizer."""

    def process(self, raw_input: str, pinned_files: Optional[List[str]] = None) -> RawQuery:
        # 1. Extract structured elements
        urls = self._extract_urls(raw_input)
        images = self._extract_images(raw_input)
        pasted = self._detect_pasted_blobs(raw_input)

        # 2. Clean query text (remove URLs, images, code blocks, tables)
        clean = self._clean_query(raw_input, urls)

        return RawQuery(
            query=clean,
            pasted_texts=pasted,
            pinned_files=pinned_files or [],
            extracted_urls=urls,
            image_refs=images,
        )

    # ── Extraction helpers ────────────────────────────────────────────────

    def _extract_urls(self, text: str) -> List[URLRef]:
        pattern = r'https?://[^\s<<>"{}|\\^`\[\]]+'
        found = []
        for m in re.finditer(pattern, text):
            url = m.group(0).rstrip('.,;:!?)')
            domain = re.sub(r'^https?://', '', url).split('/')[0]
            found.append(URLRef(url=url, domain=domain, placeholder=f"[{domain}]"))
        # Deduplicate
        seen = set()
        deduped = []
        for u in found:
            if u.url not in seen:
                seen.add(u.url)
                deduped.append(u)
        return deduped

    def _extract_images(self, text: str) -> List[str]:
        refs = []
        # Markdown images: ![alt](url)
        for alt, url in re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', text):
            refs.append(url)
        # Base64 images
        refs.extend(re.findall(r'data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+', text))
        # File extensions
        refs.extend(re.findall(r'\b[^\s]+\.(?:png|jpg|jpeg|gif|webp|svg)\b', text, re.IGNORECASE))
        return list(dict.fromkeys(refs))

    def _detect_pasted_blobs(self, text: str) -> List[PastedText]:
        blobs: List[PastedText] = []

        # Code blocks (triple backtick)
        for block in re.findall(r'```(?:\w+)?\n?(.*?)```', text, re.DOTALL):
            if len(block) > 50:
                blobs.append(PastedText(content=block.strip(), type=PastedTextType.CODE))

        # Markdown tables
        for block in re.findall(r'((?:\|.*?\|[ \t]*\n)+)', text):
            if len(block) > 100:
                blobs.append(PastedText(content=block.strip(), type=PastedTextType.DATA))

        # Long paragraph blobs (>500 chars, not code/tables)
        paragraphs = re.split(r'\n\s*\n', text)
        for para in paragraphs:
            stripped = para.strip()
            if len(stripped) > 500 and not stripped.startswith('```') and '|' not in stripped[:100]:
                q_markers = stripped.count('?')
                # If it's mostly a question, tag differently
                if q_markers > 0 and len(stripped) < 2000:
                    blobs.append(PastedText(content=stripped, type=PastedTextType.LONG_QUESTION))
                else:
                    blobs.append(PastedText(content=stripped, type=PastedTextType.DOCUMENT))

        return blobs

    def _clean_query(self, text: str, urls: List[URLRef]) -> str:
        # Remove code blocks
        clean = re.sub(r'```(?:\w+)?\n?(.*?)```', '', text, flags=re.DOTALL)
        # Remove markdown tables
        clean = re.sub(r'((?:\|.*?\|[ \t]*\n)+)', '', clean)
        # Remove image markdown
        clean = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[\1]', clean)
        # Replace URLs with placeholders
        for u in urls:
            clean = clean.replace(u.url, u.placeholder)
        # Collapse whitespace
        clean = ' '.join(clean.split())
        return clean.strip()