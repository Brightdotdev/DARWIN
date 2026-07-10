# BM25Utils.py
import re
from collections import Counter
from typing import Iterator, List, Optional, Set, Tuple, Dict, Any
from src.core.Logger.DarwinLogger import darwin
from src.core.Pipeline.Indexing.BM25.lib import (
    MIN_TOKEN_LENGTH, STANDARD_STOPS, DOMAIN_STOP_THRESHOLD
)

# ── Tokenization ───────────────────────────────────────────────────────────────

def _blingfire_words(text: str) -> List[str]:
    try:
        from blingfire import text_to_words
        return text_to_words(text).split()
    except ImportError:
        darwin.warning("BlingFire not available — using regex fallback")
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
    domain_stops: Optional[Set[str]] = None,
    min_length: int = 2,
    max_length: int = 40,
) -> List[str]:
    """
    Tokenize text into lowercased alphanumeric tokens, removing stopwords.
    """
    if not text:
        return []
    # Lowercase and split on non-alphanumeric
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    stopwords = STANDARD_STOPS.union(domain_stops or set())
    filtered = [
        t for t in tokens
        if t not in stopwords
        and min_length <= len(t) <= max_length
    ]
    return filtered



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
    Yield batches of token lists.
    Now yields ALL token lists (including empty) to preserve alignment with chunk_indices.
    """
    for i in range(0, len(chunks), batch_size):
        batch = []
        for chunk in chunks[i : i + batch_size]:
            tokens = tokenize_text(chunk.get("clean", ""), domain_stops=domain_stops)
            batch.append(tokens)          # always append, even if empty
        yield batch

# ── Domain stopword detection ──────────────────────────────────────────────────

def detect_domain_stopwords(
    chunks: List[Dict[str, Any]],
    threshold: float = DOMAIN_STOP_THRESHOLD,
) -> Set[str]:
    if not chunks:
        return set()
    doc_freq: Counter = Counter()
    for chunk in chunks:
        doc_freq.update(set(re.findall(r"[a-zA-Z]{3,}", chunk.get("clean", "").lower())))
    n = len(chunks)
    stops = {term for term, count in doc_freq.items() if count / n > threshold}
    if stops:
        preview = ", ".join(sorted(stops, key=lambda t: doc_freq[t], reverse=True)[:10])
        darwin.info(f"Domain stops ({len(stops)}): {preview}…")
    else:
        darwin.info("No domain stopwords detected")
    return stops

# ── Phrase matching ────────────────────────────────────────────────────────────

def check_phrase_match(phrase: List[str], chunk_tokens: List[str]) -> bool:
    if not phrase:
        return False
    n = len(phrase)
    return any(chunk_tokens[i : i + n] == phrase for i in range(len(chunk_tokens) - n + 1))

def extract_phrases(query: str) -> Tuple[List[str], List[List[str]]]:
    phrases: List[List[str]] = []
    def _collect_phrase(m: re.Match) -> str:
        tokens = tokenize_text(m.group(1))
        if len(tokens) > 1:
            phrases.append(tokens)
        return " "
    remaining = re.sub(r'"([^"]+)"', _collect_phrase, query)
    individual = tokenize_text(remaining)
    return individual, phrases