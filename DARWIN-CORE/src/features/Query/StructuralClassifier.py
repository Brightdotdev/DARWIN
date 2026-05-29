"""
src/core/Knowledge/StructuralClassifier.py
------------------------------------------
Shared structural chunk detection.
Use at INGEST time to mark chunks, and at QUERY time for penalty application.
Document-agnostic — works for books, codebases, scraped HTML, etc.
"""

from typing import List, Optional, Counter as CounterType
import re
from collections import Counter


# --- Tunables ---
STRUCTURAL_MIN_CHUNK_LENGTH = 200
STRUCTURAL_SIGNAL_RATIO = 0.30
STRUCTURAL_SHORT_LINE_RATIO = 0.5
STRUCTURAL_EARLY_CHUNK_RATIO = 0.05
STRUCTURAL_EARLY_MIN_WORDS = 20

# --- Penalty factors (multiplicative) ---
STRUCTURAL_PENALTY_FACTORS = {
    None: 1.0,
    "body": 1.0,
    "heading": 0.85,
    "toc": 0.4,
    "caption": 0.6,
    "cover": 0.3,
    "footnote": 0.75,
}

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


def classify_structural_type(
    chunk,
    all_chunks: Optional[List] = None,
    chunk_hashes: Optional[CounterType] = None,
    aggressive: bool = True,
) -> Optional[str]:
    """
    Classify a chunk's structural type. Returns None for body text.

    Types:
        None/"body"   — normal prose
        "heading"     — section header / chapter title
        "toc"         — table of contents / index
        "caption"     — image / figure caption
        "cover"       — cover page, copyright, metadata
        "footnote"    — footnotes / endnotes / citation tables
    """
    # Handle dict rows from storage (query_records returns dicts)
    if isinstance(chunk, dict):
        text = chunk.get('clean', chunk.get('text', ''))
        idx = chunk.get('index', 0)
        page = chunk.get('page', 0)
    else:
        text = getattr(chunk, 'clean', getattr(chunk, 'text', ''))
        idx = getattr(chunk, 'index', 0)
        page = getattr(chunk, 'page', 0)

    if not text:
        return None

    text_lower = text.lower()
    words = set(re.findall(r'\b[a-z]+\b', text_lower))
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Light mode: only obvious structural chunks
    if not aggressive:
        structural_found = words & STRUCTURAL_SIGNALS
        if len(words) > 0 and len(structural_found) / len(words) > 0.5:
            return "heading"
        return None

    # Caption detection (image descriptions from parser output)
    if text.startswith("[Caption:") or "the image depicts" in text_lower or "the image displays" in text_lower:
        return "caption"

    # Cover / front matter detection
    cover_signals = {"cover", "title", "copyright", "table of contents", "contents",
                     "acknowledgments", "preface", "foreword", "introduction - about",
                     "published", "publisher", "isbn", "all rights reserved",
                     "printed in", "first edition", "dedication", "prologue"}
    if words & cover_signals:
        ratio = len(words & cover_signals) / len(words) if words else 0
        if ratio > 0.15:
            return "cover"

    # TOC detection
    toc_signals = {"table of contents", "contents", "list of", "index", "glossary",
                   "bibliography", "references", "works cited"}
    if words & toc_signals:
        ratio = len(words & toc_signals) / len(words) if words else 0
        if ratio > 0.10:
            return "toc"

    # Footnote / note detection
    footnote_signals = {"footnote", "notes", "endnote", "abstract", "summary", "overview"}
    if words & footnote_signals:
        ratio = len(words & footnote_signals) / len(words) if words else 0
        if ratio > 0.15:
            return "footnote"

    # Short chunks — likely heading or cover fragment
    if len(text) < STRUCTURAL_MIN_CHUNK_LENGTH:
        section_headers = sum(1 for l in lines if SECTION_PREFIX_RE.match(l))
        if section_headers >= 1:
            return "heading"
        if all_chunks is not None and idx <= len(all_chunks) * STRUCTURAL_EARLY_CHUNK_RATIO:
            if len(words) < STRUCTURAL_EARLY_MIN_WORDS:
                return "cover"
        return "heading"

    # Structural signal ratio
    heading_signals = {"chapter", "book", "part", "section", "volume", "appendix",
                       "excerpt", "reference", "figure", "table", "chart", "map",
                       "illustration"}
    structural_found = words & (STRUCTURAL_SIGNALS | heading_signals)
    if len(words) > 0 and len(structural_found) / len(words) > STRUCTURAL_SIGNAL_RATIO:
        if words & toc_signals:
            return "toc"
        if words & cover_signals:
            return "cover"
        if words & footnote_signals:
            return "footnote"
        return "heading"

    # TOC entry pattern
    if len(lines) > 3:
        toc_entries = sum(1 for l in lines if TOC_ENTRY_RE.search(l))
        if toc_entries / len(lines) > 0.40:
            return "toc"

    # Section prefix density
    section_headers = sum(1 for l in lines if SECTION_PREFIX_RE.match(l))
    if section_headers >= 3:
        return "heading"

    # Verse/citation density
    if len(lines) > 2:
        citation_lines = sum(
            1 for l in lines
            if VERSE_CITATION_RE.search(l) or CROSS_REF_RE.search(l)
        )
        if citation_lines / len(lines) > 0.3:
            return "footnote"

    # Cross-reference table pattern
    if len(lines) > 4:
        book_ref_lines = sum(
            1 for l in lines
            if re.match(r'^[A-Za-z]+\s+\d+[:.\-]\d*$', l.strip())
        )
        if book_ref_lines / len(lines) > 0.4:
            return "toc"

    # Short numbered lines
    if len(lines) > 3:
        short_numbered = sum(1 for l in lines if len(l) < 60 and re.search(r'\d', l))
        if short_numbered / len(lines) > STRUCTURAL_SHORT_LINE_RATIO:
            return "toc"

    # Deduplication / repeated front matter
    if chunk_hashes is not None:
        text_hash = hash(text_lower[:200])
        if chunk_hashes.get(text_hash, 0) > 1:
            return "cover"
    elif all_chunks is not None:
        text_hash = hash(text_lower[:200])
        repeat_count = sum(
            1 for c in all_chunks
            if hash(getattr(c, 'clean', getattr(c, 'text', '')).lower()[:200]) == text_hash
        )
        if repeat_count > 2:
            return "cover"

    # Early chunk with few words
    if all_chunks is not None:
        if idx <= len(all_chunks) * STRUCTURAL_EARLY_CHUNK_RATIO and len(words) < STRUCTURAL_EARLY_MIN_WORDS:
            return "cover"

    # Prefix repetition (glossary/index style)
    if len(lines) >= 3:
        prefixes = []
        for l in lines:
            m = re.match(r'^([A-Za-z]+)[\s\-–—]', l)
            if m:
                prefixes.append(m.group(1).lower())
        if prefixes:
            most_common = Counter(prefixes).most_common(1)[0]
            if most_common[1] >= 3:
                return "toc"

    return None


def is_structural_chunk(
    chunk,
    all_chunks: Optional[List] = None,
    chunk_hashes: Optional[CounterType] = None,
    aggressive: bool = True,
) -> bool:
    """Backward-compatible boolean wrapper."""
    return classify_structural_type(chunk, all_chunks, chunk_hashes, aggressive) is not None


def get_penalty_factor(structural_type: Optional[str]) -> float:
    """Return the score penalty factor for a structural type."""
    return STRUCTURAL_PENALTY_FACTORS.get(structural_type, 1.0)