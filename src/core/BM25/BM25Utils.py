"""
src/core/BM25/BM25Utils.py
--------------------------
Utilities for BM25 indexing: tokenization, stopword detection,
phrase support, streaming, and IDF configuration.

Stopword strategy
-----------------
Two tiers:

    Standard stops  — hardcoded English function words (a, the, is …).
                      Never discriminative; always removed.

    Domain stops    — detected per-document at index time.
                      Terms appearing in > DOMAIN_STOP_THRESHOLD of chunks
                      are noise for this document (e.g. "photosynthesis" in
                      a biology textbook). BM25 IDF would score them falsely
                      high because they look rare globally but are ubiquitous
                      locally. Removing them before indexing is the correct
                      fix for sparse retrieval.

Phrase matching
---------------
Phrases use a sliding window over the already-stored token list — no
separate position index needed. We iterate every chunk for the BM25
score anyway, so the position-index skip optimisation never fired in
practice. Three lines beats a nested dict, four numpy arrays, and a
reconstruction pass on every load.

IDF smoothing
-------------
Formula: log((N - df + s) / (df + s) + 1)   where s = smoothing

    s = 0.5   standard BM25     — balanced
    s = 1.0   aggressive        — rare terms get a larger boost
    s = 0.1   conservative      — common terms penalised less
"""

import re
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set, Tuple

from src.core.Chunking.ChunkerUtils import Chunk
from src.core.Logger.DarwinLogger import darwin, LogType


# ── Constants ──────────────────────────────────────────────────────────────────

STANDARD_STOPS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "that", "these", "those",
    "or", "nor", "but", "not", "so", "yet", "if", "then", "than", "when",
    "i", "you", "we", "they", "my", "your", "our", "their", "me", "him",
    "her", "us", "them", "his", "hers", "ours", "yours", "theirs", "mine",
}

DOMAIN_STOP_THRESHOLD      = 0.30
MIN_TOKEN_LENGTH           = 2
DEFAULT_IDF_SMOOTHING      = 0.5
AGGRESSIVE_IDF_SMOOTHING   = 1.0
CONSERVATIVE_IDF_SMOOTHING = 0.1


# ── Args ───────────────────────────────────────────────────────────────────────

@dataclass
class BM25IndexingArgs:
    target:        str
    file:          Optional[str] = None
    project:       Optional[str] = None
    threshold:     float         = DOMAIN_STOP_THRESHOLD
    idf_smoothing: float         = DEFAULT_IDF_SMOOTHING
    inspect:       bool          = False
    rerun:         bool          = False


# ── BM25Data ───────────────────────────────────────────────────────────────────

@dataclass
class BM25Data:
    """
    All BM25 indexing state, ready to save and reload.

    Raw frequencies are stored (not scores) so k1, b, and IDF smoothing
    can be tuned at search time without rebuilding the index.

    Fields
    ------
    tokens           Tokenized chunks — list[list[str]]
    doc_freqs        term → number of chunks containing that term
    avgdl            Average chunk length in tokens
    doc_lengths      Token count per chunk
    domain_stopwords Terms removed as document-level noise (sorted list)
    threshold        The domain-stop threshold used at index time
    chunk_count      Number of chunks indexed
    vocab_size       Unique terms after stopword removal
    idf_smoothing    Smoothing value used for precomputed_idfs
    precomputed_idfs term → IDF score (populated in __post_init__)
    """
    tokens:           List[List[str]]
    doc_freqs:        Dict[str, int]
    avgdl:            float
    doc_lengths:      List[int]
    domain_stopwords: List[str]
    threshold:        float
    chunk_count:      int
    vocab_size:       int
    idf_smoothing:    float                      = DEFAULT_IDF_SMOOTHING
    precomputed_idfs: Optional[Dict[str, float]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.precomputed_idfs is None:
            self.precomputed_idfs = precompute_idfs(
                self.doc_freqs, self.chunk_count, self.idf_smoothing
            )

    def to_dict(self) -> dict:
        """Serialisable summary — tokens array is stored separately in npz."""
        return {
            "avgdl":            self.avgdl,
            "chunk_count":      self.chunk_count,
            "vocab_size":       self.vocab_size,
            "domain_stopwords": self.domain_stopwords,
            "threshold":        self.threshold,
            "idf_smoothing":    self.idf_smoothing,
            "doc_freqs":        self.doc_freqs,
            "doc_lengths":      self.doc_lengths,
            "precomputed_idfs": self.precomputed_idfs or {},
        }


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
    chunks: List[Chunk],
    domain_stops: Set[str] = frozenset(),
) -> List[List[str]]:
    return [tokenize_text(c.clean, domain_stops=domain_stops) for c in chunks]


# ── Streaming ──────────────────────────────────────────────────────────────────

def stream_tokenize_chunks(
    chunks: List[Chunk],
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
            if (tokens := tokenize_text(chunk.clean, domain_stops=domain_stops))
        ]


# ── Domain stopword detection ──────────────────────────────────────────────────

def detect_domain_stopwords(
    chunks: List[Chunk],
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
        doc_freq.update(set(re.findall(r"[a-zA-Z]{3,}", chunk.clean.lower())))

    n = len(chunks)
    stops = {term for term, count in doc_freq.items() if count / n > threshold}

    if stops:
        preview = ", ".join(sorted(stops, key=lambda t: doc_freq[t], reverse=True)[:10])
        darwin.log(f"Domain stops ({len(stops)}): {preview}…", LogType.INFO)
    else:
        darwin.log("No domain stopwords detected", LogType.INFO)

    return stops


# ── IDF ────────────────────────────────────────────────────────────────────────

def compute_idf(df: int, N: int, smoothing: float = DEFAULT_IDF_SMOOTHING) -> float:
    """IDF with configurable smoothing: log((N - df + s) / (df + s) + 1)"""
    return float(np.log((N - df + smoothing) / (df + smoothing) + 1))


def precompute_idfs(
    doc_freqs: Dict[str, int],
    N: int,
    smoothing: float = DEFAULT_IDF_SMOOTHING,
) -> Dict[str, float]:
    return {
        term: compute_idf(df, N, smoothing)
        for term, df in doc_freqs.items()
    }


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