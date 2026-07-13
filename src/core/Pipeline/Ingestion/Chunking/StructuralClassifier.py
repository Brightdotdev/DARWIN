import re
import math
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

class StructuralClassifierConfig:
    """Configuration for the structural classifier."""
    # Front/back matter thresholds (as fraction of total pages)
    front_matter_ratio: float = 0.05
    back_matter_ratio: float = 0.05
    # Maximum pages to treat as front/back (safety caps)
    max_front_pages: int = 20
    max_back_pages: int = 20
    # N‑gram similarity threshold (0..1)
    ngram_similarity_threshold: float = 0.25
    # Minimum number of overlapping n‑grams to consider
    min_ngram_overlap: int = 5
    # Character n‑gram length range
    ngram_min: int = 3
    ngram_max: int = 5

    # Default penalties for retrieval (will be used by orchestrator)
    penalty_factors: Dict[str, float] = {
        "body":      1.0,
        "heading":   0.95,
        "table":     0.9,
        "caption":   0.9,
        "footnote":  0.7,
        "toc":       0.4,
        "index":     0.4,
        "copyright": 0.3,
        "cover":     0.2,
        "separator": 0.1,
        "unknown":   0.8,
    }

    # Hard overrides: source -> (start_page, end_page, type)
    # Inclusive ranges, pages are 1‑based.
    hard_overrides: Dict[str, List[Tuple[int, int, str]]] = {
        # Example:
        # "book_of_enoch.pdf": [(6, 20, "body")],
    }

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ──────────────────────────────────────────────────────────────────────────────
# N‑gram signature builder (pre‑computed from examples)
# ──────────────────────────────────────────────────────────────────────────────

def _build_ngram_signature(text: str, n_range: Tuple[int, int] = (3, 5)) -> Dict[str, int]:
    """
    Build a character n‑gram frequency dictionary from a piece of text.
    n_range: (min_n, max_n)
    """
    text = re.sub(r'\s+', ' ', text.lower().strip())
    counter: Dict[str, int] = {}
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            counter[ngram] = counter.get(ngram, 0) + 1
    return counter


# Pre‑built signatures for each structural type.
# These are trained on a small set of typical examples.
# You can extend these lists with more examples for better coverage.

_EXAMPLE_TEXTS: Dict[str, List[str]] = {
    "toc": [
        "Table of Contents",
        "Contents",
        "List of Figures",
        "List of Tables",
        "Chapter 1 Introduction",
        "Index",
        "Appendix A",
    ],
    "copyright": [
        "Copyright © 2024",
        "All rights reserved",
        "No part of this publication may be reproduced",
        "ISBN 978-3-16-148410-0",
        "Published by",
        "Permission of the publisher",
    ],
    "cover": [
        "Title Page",
        "Dedication",
        "Acknowledgments",
        "About the Author",
        "Praise for",
    ],
    "index": [
        "Index",
        "Glossary",
        "Bibliography",
        "References",
        "Works Cited",
        "Further Reading",
    ],
    "footnote": [
        "¹ See also",
        "² Ibid.",
        "³ op. cit.",
        "†",
        "‡",
        "[1]",
        "(1)",
    ],
}

# Build signatures once.
_NGRAM_SIGNATURES: Dict[str, Dict[str, int]] = {}
for stype, examples in _EXAMPLE_TEXTS.items():
    combined = " ".join(examples)
    _NGRAM_SIGNATURES[stype] = _build_ngram_signature(combined, (3, 5))

# Pre‑compute vector norms for cosine similarity.
_NGRAM_NORMS: Dict[str, float] = {
    stype: math.sqrt(sum(v*v for v in sig.values()))
    for stype, sig in _NGRAM_SIGNATURES.items()
}


# ──────────────────────────────────────────────────────────────────────────────
# Pattern matching (regex)
# ──────────────────────────────────────────────────────────────────────────────

# Compiled patterns for speed
_PATTERNS = {
    "toc": re.compile(r'(?:^|\n)\s*(?:table of contents|contents|list of (?:figures|tables)|index|appendix)', re.IGNORECASE),
    "copyright": re.compile(r'(?:^|\n)\s*(?:copyright|©|all rights reserved|isbn|published by|permission of)', re.IGNORECASE),
    "cover": re.compile(r'(?:^|\n)\s*(?:title page|dedication|acknowledgments|about the author|praise for)', re.IGNORECASE),
    "index": re.compile(r'(?:^|\n)\s*(?:index|glossary|bibliography|references|works cited|further reading)', re.IGNORECASE),
    "footnote": re.compile(r'(?:^|\n)\s*(?:\[\d+\]|\d+\.\s+|\*+\d+|†|‡|ibid\.|op\. cit\.)', re.IGNORECASE),
    "separator": re.compile(r'^[-=]{3,}\s*$', re.MULTILINE),
}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def classify_structural_type(
    text: str,
    heading: str,
    page: int,
    total_pages: int,
    source: Optional[str] = None,
    config: Optional[StructuralClassifierConfig] = None,
) -> str:
    """
    Classify a chunk's structural type.

    Args:
        text: The chunk text (raw, before cleaning)
        heading: The section heading (if any)
        page: Current page number (1‑based)
        total_pages: Total pages in the document
        source: Source filename (for hard overrides)
        config: Optional configuration

    Returns:
        Structural type string (e.g., "body", "toc", "footnote", etc.)
    """
    if config is None:
        config = StructuralClassifierConfig()

    if not text or not text.strip():
        return "body"

    # ── 1. Hard overrides ──────────────────────────────────────────────
    if source and source in config.hard_overrides:
        for start, end, stype in config.hard_overrides[source]:
            if start <= page <= end:
                return stype

    # ── 2. Position heuristics ─────────────────────────────────────────
    front_threshold = min(
        int(total_pages * config.front_matter_ratio) if total_pages > 0 else 0,
        config.max_front_pages
    )
    back_threshold = max(
        total_pages - int(total_pages * config.back_matter_ratio) if total_pages > 0 else total_pages,
        total_pages - config.max_back_pages
    )

    if page <= front_threshold:
        # Early pages often contain TOC, copyright, cover
        lower_text = text.lower()
        lower_heading = heading.lower()
        if any(kw in lower_heading for kw in ("contents", "table of contents", "list of")):
            return "toc"
        if any(kw in lower_text for kw in ("copyright", "all rights reserved", "isbn")):
            return "copyright"
        if any(kw in lower_heading for kw in ("title page", "dedication", "acknowledgments")):
            return "cover"
        # Very short text on first pages -> cover
        if page <= 2 and len(text) < 500:
            return "cover"

    if page >= back_threshold:
        lower_text = text.lower()
        lower_heading = heading.lower()
        if any(kw in lower_heading for kw in ("index", "glossary", "bibliography", "references")):
            return "index"
        if any(kw in lower_text for kw in ("index", "glossary", "bibliography", "references")):
            return "index"

    # ── 3. Pattern matching ─────────────────────────────────────────────
    # Separator detection (any page)
    sep_matches = len(_PATTERNS["separator"].findall(text))
    if sep_matches >= 3 and len(text) < 600:
        return "separator"

    # Footnote (any page, short text)
    if _PATTERNS["footnote"].search(text) and len(text) < 500:
        return "footnote"

    # Copyright (any page, short text)
    if _PATTERNS["copyright"].search(text) and len(text) < 400:
        return "copyright"

    # TOC (any page, heading says it)
    if any(kw in heading.lower() for kw in ("contents", "table of contents")):
        return "toc"

    # ── 4. N‑gram similarity ──────────────────────────────────────────
    # Only run if previous methods didn't return a type
    # Build n‑grams for this chunk
    chunk_ngrams = _build_ngram_signature(text, (config.ngram_min, config.ngram_max))
    if chunk_ngrams:
        best_type = "body"
        best_score = 0.0
        for stype, signature in _NGRAM_SIGNATURES.items():
            # Compute overlap
            overlap = set(chunk_ngrams.keys()) & set(signature.keys())
            if len(overlap) < config.min_ngram_overlap:
                continue
            # Dot product
            dot = sum(chunk_ngrams[k] * signature[k] for k in overlap)
            norm_chunk = math.sqrt(sum(v*v for v in chunk_ngrams.values()))
            norm_sig = _NGRAM_NORMS.get(stype, 1.0)
            if norm_chunk == 0 or norm_sig == 0:
                continue
            cos_sim = dot / (norm_chunk * norm_sig)
            if cos_sim > config.ngram_similarity_threshold and cos_sim > best_score:
                best_score = cos_sim
                best_type = stype
        if best_type != "body":
            return best_type

    # ── 5. Fallback ──────────────────────────────────────────────────────
    # If heading is not empty and text is short, treat as heading
    if heading and len(text) < 300 and text.strip() == heading.strip():
        return "heading"

    return "body"


# ──────────────────────────────────────────────────────────────────────────────
# Helper: get penalty factor for retrieval scoring
# ──────────────────────────────────────────────────────────────────────────────

def get_penalty_factor(structural_type: Optional[str]) -> float:
    """Return the retrieval score penalty factor for a structural type."""
    if structural_type is None:
        return 1.0
    return StructuralClassifierConfig.penalty_factors.get(structural_type, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Optionally, a class-based API for more flexibility
# ──────────────────────────────────────────────────────────────────────────────

class StructuralClassifier:
    """
    Class-based wrapper for configurable classification.
    """
    def __init__(self, config: Optional[StructuralClassifierConfig] = None):
        self.config = config or StructuralClassifierConfig()

    def classify(
        self,
        text: str,
        heading: str,
        page: int,
        total_pages: int,
        source: Optional[str] = None,
    ) -> str:
        return classify_structural_type(
            text, heading, page, total_pages,
            source=source,
            config=self.config
        )

    @staticmethod
    def penalty_factor(stype: str) -> float:
        return get_penalty_factor(stype)