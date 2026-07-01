
import re
import numpy as np
from collections import Counter
from typing import Iterator, List, Set, Tuple,Dict, Any, Optional
from src.core.Logger.DarwinLogger import darwin, LogType

from src.pipleline.Indexing.BM25.lib import ( MIN_TOKEN_LENGTH, STANDARD_STOPS,DOMAIN_STOP_THRESHOLD)


# ── Tokenization ───────────────────────────────────────────────────────────────

def _blingfire_words(text: str) -> List[str]:
    """Return word tokens via BlingFire, falling back to regex."""
    try:
        from blingfire import text_to_words
        return text_to_words(text).split()
    except ImportError:
        darwin.log("BlingFire not available — using regex fallback", LogType.WARNING)
        return re.findall(r"[a-zA-Z]{2,}", text)


def _keep_token(word: str, standard_stops: Set[str], domain_stops: Set[str]) -> bool:
    w = word.lower().strip()
    return (
        len(w) >= MIN_TOKEN_LENGTH
        and not w.isdigit()
        and w not in standard_stops
        and w not in domain_stops
    )


def tokenize_text(
    text: str,
    standard_stops: Set[str] = STANDARD_STOPS,
    domain_stops: Set[str] = frozenset(),
) -> List[str]:
    """Tokenize text for BM25 — returns filtered lowercase tokens."""
    return [
        w.lower().strip()
        for w in _blingfire_words(text)
        if _keep_token(w, standard_stops, domain_stops)
    ]


def tokenize_chunks(
    chunks: List[Dict[str, Any]],
    domain_stops: Set[str] = frozenset(),
) -> List[List[str]]:
    return [tokenize_text(c.get("clean", ""), domain_stops=domain_stops) for c in chunks]


# ── Streaming ──────────────────────────────────────────────────────────────────

def stream_tokenize_chunks(
    chunks: List[Dict[str, Any]],
    domain_stops: Set[str] = frozenset(),
    batch_size: int = 100,
) -> Iterator[List[List[str]]]:
    """
    Yield batches of token lists without holding all chunks in memory.
    Activated automatically for corpora > 5 000 chunks.
    Each batch contains only non-empty token lists.
    """
    for i in range(0, len(chunks), batch_size):
        yield [
            tokens
            for chunk in chunks[i : i + batch_size]
            if (tokens := tokenize_text(chunk.get("clean", ""), domain_stops=domain_stops))
        ]


# ── Domain stopword detection ──────────────────────────────────────────────────

def detect_domain_stopwords(
    chunks: List[Dict[str, Any]],
    threshold: float = DOMAIN_STOP_THRESHOLD,
) -> Set[str]:
    """
    Return terms that appear in more than `threshold` fraction of chunks.

    These are document-specific noise — BM25 IDF would score them falsely
    high because they look globally rare but are locally ubiquitous.
    Removing them before indexing keeps IDF scores honest.
    """
    if not chunks:
        return set()

    doc_freq: Counter = Counter()
    for chunk in chunks:
        doc_freq.update(set(re.findall(r"[a-zA-Z]{3,}", chunk.get("clean", "").lower())))

    n = len(chunks)
    stops = {term for term, count in doc_freq.items() if count / n > threshold}

    if stops:
        preview = ", ".join(sorted(stops, key=lambda t: doc_freq[t], reverse=True)[:10])
        darwin.log(f"Domain stops ({len(stops)}): {preview}…", LogType.INFO)
    else:
        darwin.log("No domain stopwords detected", LogType.INFO)

    return stops



# ── Phrase matching ────────────────────────────────────────────────────────────

def check_phrase_match(phrase: List[str], chunk_tokens: List[str]) -> bool:
    """
    True if `phrase` appears as a consecutive run inside `chunk_tokens`.

    Sliding window over the token list already held in memory for BM25
    scoring — no separate position index required.
    """
    if not phrase:
        return False
    n = len(phrase)
    return any(chunk_tokens[i : i + n] == phrase for i in range(len(chunk_tokens) - n + 1))


def extract_phrases(query: str) -> Tuple[List[str], List[List[str]]]:
    """
    Split a query into individual terms and quoted phrase groups.

    Example
    -------
    'machine learning "neural networks"'
    → individual: ["machine", "learning"]
      phrases:    [["neural", "networks"]]
    """
    phrases: List[List[str]] = []

    def _collect_phrase(m: re.Match) -> str:
        tokens = tokenize_text(m.group(1))
        if len(tokens) > 1:
            phrases.append(tokens)
        return " "

    remaining = re.sub(r'"([^"]+)"', _collect_phrase, query)
    individual = tokenize_text(remaining)
    return individual, phrases