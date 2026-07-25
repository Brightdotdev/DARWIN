from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ── Constants ──────────────────────────────────────────────────────────────────

STANDARD_STOPS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "that", "these", "those",
    "or", "nor", "but", "not", "so", "yet", "if", "then", "than", "when",
    "i", "you", "we", "they", "my", "your", "our", "their", "me", "him",
    "her", "us", "them", "his", "hers", "ours", "yours", "theirs", "mine",
}

# Terms appearing in more than this fraction of documents are treated as
# domain-specific stop words and excluded from indexing.
# Range: 0.0 - 1.0 (Typical: 0.20 - 0.50)
DOMAIN_STOP_THRESHOLD = 0.30

# Minimum token length (in characters) required for a term to be indexed.
# Shorter tokens are ignored.
# Range: 1 - 5 (Typical: 2)
MIN_TOKEN_LENGTH = 2

# Default IDF smoothing factor used when computing BM25 IDF.
# Balances the influence of rare and common terms.
# Range: 0.0 - 1.0 (Typical: 0.5)
DEFAULT_IDF_SMOOTHING = 0.5

# Higher IDF smoothing factor that reduces the weight difference between
# rare and common terms, producing more conservative rankings.
# Range: 0.5 - 2.0 (Typical: 1.0)
AGGRESSIVE_IDF_SMOOTHING = 1.0

# Lower IDF smoothing factor that preserves stronger weighting for rare
# terms, making rankings more sensitive to uncommon keywords.
# Range: 0.0 - 0.5 (Typical: 0.1)
CONSERVATIVE_IDF_SMOOTHING = 0.1


# ── IDF ────────────────────────────────────────────────────────────────────────

def compute_idf(df: int, N: int, smoothing: float = DEFAULT_IDF_SMOOTHING) -> float:
    """IDF with configurable smoothing: log((N - df + s) / (df + s) + 1)"""
    import numpy as np

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


# ── Args ───────────────────────────────────────────────────────────────────────

@dataclass
class BM25IndexingArgs:
    files: list[str] = field(default_factory=list)

    threshold: float = DOMAIN_STOP_THRESHOLD
    idf_smoothing: float = DEFAULT_IDF_SMOOTHING
    rerun: bool = False

    def __post_init__(self):
        # Clamp threshold to the valid range [0.0, 1.0].
        self.threshold = max(0.0, min(self.threshold, 1.0))

        # Clamp IDF smoothing to the valid range [0.0, 2.0].
        self.idf_smoothing = max(0.0, min(self.idf_smoothing, 2.0))

# ── BM25Data ───────────────────────────────────────────────────────────────────

@dataclass
class BM25Data:
    tokens:           List[List[str]]
    doc_freqs:        Dict[str, int]
    avgdl:            float
    doc_lengths:      List[int]
    domain_stopwords: List[str]
    threshold:        float
    chunk_count:      int
    vocab_size:       int
    idf_smoothing:    float                      = DEFAULT_IDF_SMOOTHING
    chunk_indices:    List[int]                  = field(default_factory=list)  # NEW
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
            "chunk_indices":    self.chunk_indices,  # NEW
            "precomputed_idfs": self.precomputed_idfs or {},
        }
      
   
    @classmethod
    def from_dict(cls, data: dict) -> "BM25Data":
        """
        Reconstruct BM25Data from a storage-layer dictionary.
        """
        return cls(
            tokens=[
                list(toks)
                for toks in data.get("tokens", [])
            ],

            doc_freqs={
                str(k): int(v)
                for k, v in data.get("doc_freqs", {}).items()
            },

            avgdl=float(
                data.get("avgdl", 0.0)
            ),

            doc_lengths=[
                int(x)
                for x in data.get("doc_lengths", [])
            ],

            domain_stopwords=list(
                data.get("domain_stopwords", [])
            ),

            threshold=float(
                data.get("threshold", 0.0)
            ),

            chunk_count=int(
                data.get(
                    "chunk_count",
                    len(data.get("tokens", []))
                )
            ),

            vocab_size=int(
                data.get("vocab_size", 0)
            ),

            idf_smoothing=float(
                data.get(
                    "idf_smoothing",
                    DEFAULT_IDF_SMOOTHING
                )
            ),

            chunk_indices=[
                int(x)
                for x in data.get("chunk_indices", [])
            ],  # NEW

            precomputed_idfs=(
                {
                    str(k): float(v)
                    for k, v in data.get(
                        "precomputed_idfs",
                        {}
                    ).items()
                }
                if data.get("precomputed_idfs") is not None
                else None
            ),
        )