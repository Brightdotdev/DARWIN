"""
StructuralClassifier.py — Single source of truth for structural chunk classification.

Used by:
  - ChunkerUtils._make_chunk()      → sets structural_type during chunk creation
  - GraphExtractor                  → filters structural chunks from entity extraction
  - Embedding pipeline              → filters structural chunks from vector index
  - BM25 pipeline                   → filters structural chunks from sparse index
  - WikiGenerator                   → filters structural chunks from concept pages
  - RetrievalOrchestrator           → applies penalties using pre-computed metadata
  - KeywordSearch / SemanticSearch  → hard-gates structural chunks from results
"""

import re
from typing import Optional, Set

# ═══════════════════════════════════════════════════════════════════════════
# Keyword lists
# ═══════════════════════════════════════════════════════════════════════════

_TOC_KEYWORDS = [
    "table of contents", "contents", "list of figures", "list of tables",
    "list of illustrations", "index of", "outline",
]

_COPYRIGHT_KEYWORDS = [
    "copyright", "all rights reserved", "isbn", "published by",
    "no part of this", "reproduced in any form", "permission of the publisher",
    "printed in", "first edition",
]

_COVER_KEYWORDS = [
    "cover", "title page", "dedication", "acknowledgments", "acknowledgements",
    "preface", "foreword", "about the author", "about this book",
    "also by", "praise for",
]

_INDEX_KEYWORDS = [
    "index", "glossary", "bibliography", "references", "appendix",
    "notes", "endnotes", "works cited", "further reading",
]

# ═══════════════════════════════════════════════════════════════════════════
# Compiled patterns
# ═══════════════════════════════════════════════════════════════════════════

_COPYRIGHT_RE = re.compile(
    r"(copyright|©|all rights reserved|isbn|published by|"
    r"no part of this publication|reproduced in any form|"
    r"permission of the publisher|printed in)",
    re.IGNORECASE,
)

_FOOTNOTE_RE = re.compile(
    r"^\s*(?:\[\d+\]|\d+\.\s+|\*\d+|\†\d+|\⁎\d+)",
    re.MULTILINE,
)

_SEPARATOR_RE = re.compile(r"^[-=]{3,}\s*$", re.MULTILINE)


# ═══════════════════════════════════════════════════════════════════════════
# Structural type taxonomy
# ═══════════════════════════════════════════════════════════════════════════

# Types that should NEVER enter any search index (dense or sparse)
STRUCTURAL_EMBED_EXCLUDE: Set[str] = {
    "toc", "copyright", "cover", "index", "separator"
}

# Types that should NEVER enter knowledge graph extraction
STRUCTURAL_GRAPH_EXCLUDE: Set[str] = {
    "toc", "copyright", "cover", "index", "separator", "heading", "footnote"
}

# Types that receive a score penalty at retrieval time (but may still appear)
STRUCTURAL_PENALTY_TYPES: Set[str] = {
    "toc", "copyright", "cover", "index", "separator", "footnote"
}

# Penalty factor per type (0.0 = hard filter, 0.1 = heavy penalty, 1.0 = no penalty)
PENALTY_FACTORS = {
    "toc":        0.05,
    "copyright":  0.05,
    "cover":      0.05,
    "index":      0.05,
    "separator":  0.0,   # Hard filter — never appears
    "footnote":   0.3,
    "heading":    1.0,   # No penalty — semantic content
    "body":       1.0,
    "table":      1.0,
    "caption":    1.0,
}


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def classify_structural_type(
    text: str,
    heading: str,
    page: int,
    total_pages: int,
) -> str:
    """
    Classify chunk content as structural or body.

    Returns one of:
      toc | copyright | cover | index | footnote | separator | heading | body | table | caption
    """
    if not text or not text.strip():
        return "body"

    text_lower = text.lower().strip()
    heading_lower = heading.lower().strip()

    # ── Front matter (first 5% of pages) ───────────────────────────────────
    front_threshold = max(1, int(total_pages * 0.05)) if total_pages > 0 else 1
    if page <= front_threshold:
        if any(kw in heading_lower for kw in _TOC_KEYWORDS):
            return "toc"
        if any(kw in text_lower for kw in _TOC_KEYWORDS):
            return "toc"
        if any(kw in heading_lower for kw in _COVER_KEYWORDS):
            return "cover"
        if _COPYRIGHT_RE.search(text) or any(kw in text_lower for kw in _COPYRIGHT_KEYWORDS):
            return "copyright"
        # Very early pages with little text → likely cover material
        if page <= 2 and len(text) < 800:
            return "cover"

    # ── Back matter (last 5% of pages) ──────────────────────────────────────
    back_threshold = max(1, int(total_pages * 0.95)) if total_pages > 0 else 1
    if page >= back_threshold:
        if any(kw in heading_lower for kw in _INDEX_KEYWORDS):
            return "index"
        if any(kw in text_lower for kw in _INDEX_KEYWORDS):
            return "index"

    # ── Inline structural detection (any page) ──────────────────────────────

    # TOC inline (heading says "Contents" etc.)
    if any(kw in heading_lower for kw in ["contents", "table of contents"]):
        return "toc"

    # Separator / page-break artifacts
    if len(_SEPARATOR_RE.findall(text)) >= 3 and len(text) < 600:
        return "separator"

    # Footnotes
    if _FOOTNOTE_RE.search(text) and len(text) < 400:
        return "footnote"

    # Copyright inline (short legal text anywhere)
    if _COPYRIGHT_RE.search(text) and len(text) < 300:
        return "copyright"

    # Pure heading chunk (rare — happens when section has no body)
    if heading and text_lower == heading_lower and len(text) < 200:
        return "heading"

    return "body"


def is_structural_for_graph(structural_type: Optional[str]) -> bool:
    """
    Return True if chunks of this type should be excluded from
    knowledge-graph entity extraction.
    """
    return structural_type in STRUCTURAL_GRAPH_EXCLUDE


def is_structural_for_embedding(structural_type: Optional[str]) -> bool:
    """
    Return True if chunks of this type should be excluded from
    dense embedding index and BM25 sparse index.
    """
    return structural_type in STRUCTURAL_EMBED_EXCLUDE


def get_penalty_factor(structural_type: Optional[str]) -> float:
    """
    Return the score penalty factor for a given structural type.
    
    1.0   = no penalty (body, heading, table, caption)
    0.3   = light penalty (footnote)
    0.05  = heavy penalty (toc, copyright, cover, index)
    0.0   = hard filter (separator)
    """
    if structural_type is None:
        return 1.0
    return PENALTY_FACTORS.get(structural_type, 1.0)


def is_penalized(structural_type: Optional[str]) -> bool:
    """Return True if this structural type receives any penalty."""
    return get_penalty_factor(structural_type) < 1.0

