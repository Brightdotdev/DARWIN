"""
Darwin Wiki / Knowledge Module — Utilities (v3.0)
Normalization, validation, and helper functions.
"""

import re
from typing import Optional

try:
    from .lib import VALID_ENTITY_TYPES, REJECTED_ENTITY_TYPES, ENTITY_STOPWORDS
except ImportError:
    from lib import VALID_ENTITY_TYPES, REJECTED_ENTITY_TYPES, ENTITY_STOPWORDS


def normalize(text: str) -> str:
    """
    Normalize entity text for canonical identity.

    Rules:
    - Lowercase
    - Collapse whitespace
    - Strip leading/trailing punctuation
    - Preserve internal punctuation (O'Connor, C.S. Lewis)
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    # Strip leading/trailing non-alphanumeric, but preserve internal
    text = re.sub(r"^[^\w]+", "", text)
    text = re.sub(r"[^\w]+$", "", text)
    return text


def is_valid_entity(text: str, label: str) -> bool:
    """
    Multi-layer entity validation.

    Rejects:
    1. Empty or whitespace-only text
    2. Rejected entity types (CARDINAL, ORDINAL)
    3. Stopword-only canonical names
    4. Pure numbers / verse references (e.g., "3:16", "1 Corinthians")
    5. Single-character entities
    6. Entities that are mostly non-alphabetic
    """
    if not text or not text.strip():
        return False

    if label in REJECTED_ENTITY_TYPES:
        return False

    norm = normalize(text)
    if not norm or len(norm) < 2:
        return False

    if norm in ENTITY_STOPWORDS:
        return False

    # Reject pure numbers, verse refs, chapter:verse patterns
    if re.match(r"^\d+(:\d+)?$", norm):
        return False

    # Reject if > 70% non-alphabetic (catches garbage like "!!!", "...")
    alpha_count = sum(1 for c in norm if c.isalpha())
    if alpha_count / len(norm) < 0.3:
        return False

    # Reject if it's just a year/date pattern
    if re.match(r"^\d{1,4}(\s*(bc|ad|bce|ce))?$", norm):
        return False

    return True


def is_garbage_entity(text: str) -> bool:
    """
    Extra aggressive filter for post-processing cleanup.
    Called after initial extraction to catch edge cases.
    """
    norm = normalize(text)

    # All-stopword check
    words = norm.split()
    if all(w in ENTITY_STOPWORDS for w in words):
        return True

    # Reject if looks like a URL or email fragment
    if "@" in text or "http" in text.lower():
        return True

    # Reject if contains excessive punctuation
    punct_count = sum(1 for c in text if c in ".,;:!?-_—")
    if punct_count / max(len(text), 1) > 0.4:
        return True

    return False