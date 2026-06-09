import re
from typing import Tuple
from src.features.Knowledge.lib import ENTITY_STOPWORDS, REJECTED_ENTITY_TYPES


_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "first", "second", "third",
    "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
}

def _is_garbage_date_time(norm: str) -> bool:
    """Return True if the normalised text is a meaningless DATE/TIME fragment."""
    tokens = norm.split()
    # Reject if every token is a stopword, number word, or digit
    if all(t in ENTITY_STOPWORDS or t in _NUMBER_WORDS or t.isdigit() for t in tokens):
        return True
    # Also reject if the first token is a stopword and the rest are too (catches "the day")
    return False  # the first condition already covers this


def normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    # Remove all non-word, non-space characters
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_entity_id(canonical_name: str, ent_type: str) -> str:
    return f"{canonical_name}|{ent_type.lower()}"


# Patterns for obvious noise (dates, verse references, pure numbers, etc.)
_NOISE_PATTERNS = re.compile(
    r"|".join([
        r"^\d+$",                     # pure number
        r"^\d+[:.\-]\d+([:.\-]\d+)*$", # verse / date ref like 12:9-10, 10.18
        r"^\d+[A-Za-z]?\d*$",         # 12910, 110172
        r"^[MDCLXVI]+$",              # roman numerals (optional)
    ])
)


def is_valid_entity(text: str, label: str) -> bool:
    """
    Decide whether a raw spaCy entity should be kept.
    Returns True if the entity passes all quality checks.
    """
    # Reject whole categories
    if label in REJECTED_ENTITY_TYPES:
        return False

    norm = normalize(text)

    # Minimum length
    if len(norm) < 2:
        return False

    # Stopword
    if norm in ENTITY_STOPWORDS:
        return False
    
    tokens = norm.split()
    if all(t in ENTITY_STOPWORDS or t.isdigit() for t in tokens):
        return False
    
    if label in {"DATE", "TIME"} and _is_garbage_date_time(norm):
        return False

    # Noise patterns
    if _NOISE_PATTERNS.search(norm):
        return False

    return True
