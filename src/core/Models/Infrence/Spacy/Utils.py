"""
Darwin Wiki / Knowledge Module — Utilities (v5.1)
CHANGES (v5.1):
- get_base_name() / get_display_name() for consistent merging
- is_blacklisted() now checks head word (last token)
- custom_blacklist / custom_whitelist support
- compute_entity_signal uses log‑scaled frequency
"""

import re
import unicodedata
import math
from typing import Optional, List, Dict, Tuple, Set, Any


from src.core.Models.Infrence.Spacy.lib import VALID_ENTITY_TYPES, ENTITY_BLACKLIST

# ── Canonicalisation v2 ────────────────────────────────────────────────────

def get_base_name(text: str) -> str:
    """
    Return the stripped, lowercased, determiner‑free key for merging.
    'the Book of Enoch' → 'book of enoch'
    'The LORD' → 'lord'
    """
    if not text:
        return ""
    norm = _normalize_text(text)
    words = norm.split()
    determiners = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "their", "our"}
    if words and words[0] in determiners:
        words = words[1:]
    return " ".join(words).strip()


def get_display_name(text: str) -> str:
    """
    Return a human‑readable title‑cased version.
    'book of enoch' → 'Book of Enoch'
    'the lord' → 'The Lord'
    """
    if not text:
        return ""
    # Simple title case, but keep common acronyms like "C.S." or "U.S."
    # For now, just use .title() – can be improved later.
    return _normalize_text(text).title()


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    text = re.sub(r"^[^\w\s.]+", "", text)
    text = re.sub(r"[^\w\s.]+$", "", text)
    return text.strip()


def canonicalize(text: str) -> str:
    """Legacy alias – use get_base_name() for new code."""
    return get_base_name(text)


# ── Validation ────────────────────────────────────────────────────────────

_GARBAGE_PATTERNS = [
    re.compile(r"^\d+(:\d+)?$"),
    re.compile(r"^\d{1,4}(\s*(bc|ad|bce|ce))?$"),
    re.compile(r"^[ivxlc]+$"),
    re.compile(r"^\d+\s*(st|nd|rd|th)$"),
]


def is_blacklisted(text: str, custom_blacklist: Optional[Set[str]] = None, custom_whitelist: Optional[Set[str]] = None) -> bool:
    """
    Check if text is blacklisted.
    NEW: checks the HEAD word (last token) against the blacklist.
    Supports custom blacklist/whitelist.
    """
    norm = get_base_name(text)
    if not norm:
        return True

    # Whitelist takes precedence
    if custom_whitelist:
        if norm in custom_whitelist or text in custom_whitelist:
            return False

    # Combine blacklists
    combined = set(ENTITY_BLACKLIST)
    if custom_blacklist:
        combined.update(custom_blacklist)

    # Direct match
    if norm in combined:
        return True

    # Head‑word rule: last token
    tokens = norm.split()
    if not tokens:
        return True
    head = tokens[-1]
    if head in combined:
        return True

    # If phrase is just one word and it's blacklisted
    if len(tokens) == 1 and tokens[0] in combined:
        return True

    # Reject if > 50% of words are blacklisted (catch "the man" where "man" is blacklisted)
    blacklisted_words = sum(1 for w in tokens if w in combined)
    if blacklisted_words / len(tokens) > 0.5:
        return True

    return False


def is_valid_entity(text: str, label: str, custom_blacklist: Optional[Set[str]] = None, custom_whitelist: Optional[Set[str]] = None) -> bool:
    if not text or not text.strip():
        return False
    if label not in VALID_ENTITY_TYPES:
        return False
    if label in {"CARDINAL", "ORDINAL"}:
        return False

    if is_blacklisted(text, custom_blacklist, custom_whitelist):
        return False

    base = get_base_name(text)
    if len(base) < 2:
        return False
    if len(base) < 3 and not base.isalpha():
        return False

    for pat in _GARBAGE_PATTERNS:
        if pat.match(base):
            return False

    alpha_count = sum(1 for c in base if c.isalpha())
    if alpha_count / max(len(base), 1) < 0.4:
        return False

    if "@" in text or "http" in text.lower():
        return False

    punct_count = sum(1 for c in text if c in ".,;:!?-_—")
    if punct_count / max(len(text), 1) > 0.5:
        return False

    if base.isupper() and len(base) > 3:
        return False

    return True


def is_valid_noun_chunk(text: str, custom_blacklist: Optional[Set[str]] = None, custom_whitelist: Optional[Set[str]] = None) -> bool:
    if not text or not text.strip():
        return False
    base = get_base_name(text)
    if len(base) < 3:
        return False
    if is_blacklisted(text, custom_blacklist, custom_whitelist):
        return False

    tokens = base.split()
    function_words = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "into", "onto"}
    content_words = [w for w in tokens if w not in function_words]
    if len(content_words) < 1:
        return False
    if len(content_words) == 1 and len(tokens) > 3:
        return False

    avg_len = sum(len(w) for w in content_words) / max(len(content_words), 1)
    if avg_len < 3 and len(tokens) > 2:
        return False

    return True


# ── Signal Scoring ──────────────────────────────────────────────────────

def compute_entity_signal(
    mentions: List[Dict[str, Any]],
    total_chunks: int,
) -> Dict[str, float]:
    """
    v5.1: Uses log‑scaled frequency to dampen high‑count noise.
    """
    if not mentions:
        return {}

    freq = len(mentions)
    chunk_ids = {m["chunk_id"] for m in mentions}
    spread = len(chunk_ids)
    spread_ratio = spread / max(total_chunks, 1)

    ner_count = sum(1 for m in mentions if m.get("source") in ("ner", "both"))
    nc_count = sum(1 for m in mentions if m.get("source") in ("noun_chunk", "both"))
    source_diversity = nc_count / max(freq, 1)

    # Log‑scaled persistence: log(1+freq) * spread
    persistence = math.log(1 + freq) * spread

    # Prominence (unchanged)
    positions = []
    for m in mentions:
        span = m.get("span", [0, 0])
        if isinstance(span, (list, tuple)) and len(span) >= 2:
            positions.append(span[0] / max(span[1], 1))
    avg_position = sum(positions) / max(len(positions), 1) if positions else 0.5
    prominence = 1.0 - avg_position

    # Combine (weights adjusted for log scale)
    signal = (
        persistence * 0.5 +
        spread_ratio * 100 * 0.25 +
        source_diversity * 50 * 0.15 +
        prominence * 10 * 0.1
    )

    return {
        "frequency_score": freq,
        "spread_score": spread,
        "spread_ratio": round(spread_ratio, 4),
        "persistence_score": persistence,
        "source_diversity": round(source_diversity, 4),
        "syntactic_prominence": round(prominence, 4),
        "signal": round(signal, 4),
    }