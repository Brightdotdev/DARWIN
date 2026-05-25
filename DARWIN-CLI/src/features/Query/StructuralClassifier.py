"""
src/core/Knowledge/StructuralClassifier.py
------------------------------------------
Shared structural chunk detection.
Use at INGEST time to mark chunks, and at QUERY time for fallback filtering.
Document-agnostic — works for books, codebases, scraped HTML, etc.
"""

from typing import List, Optional, Counter
import re


# --- Tunables ---
STRUCTURAL_MIN_CHUNK_LENGTH = 200
STRUCTURAL_SIGNAL_RATIO = 0.30
STRUCTURAL_SHORT_LINE_RATIO = 0.5
STRUCTURAL_EARLY_CHUNK_RATIO = 0.05
STRUCTURAL_EARLY_MIN_WORDS = 20

# --- Signals ---
STRUCTURAL_SIGNALS = {
    "cover", "title", "copyright", "table of contents", "contents",
    "acknowledgments", "preface", "foreword", "introduction", "about",
    "appendix", "index", "glossary", "bibliography", "references",
    "page", "chapter", "book", "part", "section", "volume",
    "ebook", "edition", "published", "publisher", "isbn",
    "all rights reserved", "printed in", "first edition",
    "dedication", "prologue", "epilogue", "notes", "footnote",
    "abstract", "summary", "overview", "list of", "figure", "table",
    "extras", "chart", "works cited", "front matter", "back matter",
}

SECTION_PREFIX_RE = re.compile(
    r'^(Book|Chapter|Ch\.|Part|Section|Vol\.|Volume|Appendix|Excerpt|'
    r'Reference|Note|Figure|Table|Chart|Map|Illustration)s?\s*[\dIVX]+',
    re.IGNORECASE
)

TOC_ENTRY_RE = re.compile(
    r'([\.•\-–—]\s*\d+\s*$)|(^\d+[\.\)]\s+\w+)',
    re.IGNORECASE
)

VERSE_CITATION_RE = re.compile(r'\b\w+\s+\d+:\d+\b')
CROSS_REF_RE = re.compile(
    r'\b(?:cf\.|see|compare|v\.|vv\.|verses?|ch\.|chap\.|pp\.|p\.)\b',
    re.IGNORECASE
)


def is_structural_chunk(
    chunk,
    all_chunks: Optional[List] = None,
    chunk_hashes: Optional[Counter] = None,
    aggressive: bool = True,
) -> bool:
    """
    Detect structural chunks by content.
    
    Args:
        chunk: Any object with .clean (or .text), .index, .page
        all_chunks: Full chunk list for early-chunk and dedup checks
        chunk_hashes: Pre-computed hash Counter for dedup (performance)
        aggressive: False = light mode (>50% signals only)
    """
    text = getattr(chunk, 'clean', getattr(chunk, 'text', ''))
    text_lower = text.lower()
    words = set(re.findall(r'\b[a-z]+\b', text_lower))
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Light mode: only obvious structural chunks
    if not aggressive:
        structural_found = words & STRUCTURAL_SIGNALS
        return len(words) > 0 and len(structural_found) / len(words) > 0.5

    if len(text) < STRUCTURAL_MIN_CHUNK_LENGTH:
        return True

    structural_found = words & STRUCTURAL_SIGNALS
    if len(words) > 0 and len(structural_found) / len(words) > STRUCTURAL_SIGNAL_RATIO:
        return True

    if len(lines) > 3:
        short_numbered = sum(1 for l in lines if len(l) < 60 and re.search(r'\d', l))
        if short_numbered / len(lines) > STRUCTURAL_SHORT_LINE_RATIO:
            return True

    # Citation density (scripture indices, cross-reference tables)
    if len(lines) > 2:
        citation_lines = sum(
            1 for l in lines
            if VERSE_CITATION_RE.search(l) or CROSS_REF_RE.search(l)
        )
        if citation_lines / len(lines) > 0.3:
            return True

    # Cross-reference table pattern
    if len(lines) > 4:
        book_ref_lines = sum(
            1 for l in lines
            if re.match(r'^[A-Za-z]+\s+\d+[:.\-]\d*$', l.strip())
        )
        if book_ref_lines / len(lines) > 0.4:
            return True

    # Deduplication
    if chunk_hashes is not None:
        text_hash = hash(text_lower[:200])
        if chunk_hashes.get(text_hash, 0) > 1:
            return True
    elif all_chunks is not None:
        text_hash = hash(text_lower[:200])
        repeat_count = sum(
            1 for c in all_chunks
            if hash(getattr(c, 'clean', getattr(c, 'text', '')).lower()[:200]) == text_hash
        )
        if repeat_count > 2:
            return True

    # Early chunk with few words
    if all_chunks is not None:
        idx = getattr(chunk, 'index', 0)
        if idx <= len(all_chunks) * STRUCTURAL_EARLY_CHUNK_RATIO and len(words) < STRUCTURAL_EARLY_MIN_WORDS:
            return True

    section_headers = sum(1 for l in lines if SECTION_PREFIX_RE.match(l))
    if section_headers >= 3:
        return True

    if len(lines) > 3:
        toc_entries = sum(1 for l in lines if TOC_ENTRY_RE.search(l))
        if toc_entries / len(lines) > 0.40:
            return True

    if all_chunks is not None:
        idx = getattr(chunk, 'index', 0)
        if idx <= len(all_chunks) * 0.10 and len(lines) > 0:
            avg_line_len = sum(len(l) for l in lines) / len(lines)
            if (len(words) > 0 and
                len(structural_found) / len(words) > 0.15 and
                avg_line_len < 80):
                return True

    if len(lines) >= 3:
        prefixes = []
        for l in lines:
            m = re.match(r'^([A-Za-z]+)[\s\-–—]', l)
            if m:
                prefixes.append(m.group(1).lower())
        if prefixes:
            from collections import Counter
            most_common = Counter(prefixes).most_common(1)[0]
            if most_common[1] >= 3:
                return True

    return False