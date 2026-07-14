"""
src/core/Retrival/RetrievalArgs.py
-----------------------------------
Unified data class for all retrieval methods.
"""

from dataclasses import dataclass, field
from typing import Optional, List

# All constants are defined in lib.py – import from there
from src.core.Retrival.lib import (
    RetrievalMethod,
    DEFAULT_TOP_K,
    DEFAULT_POOL_SIZE,
    DEFAULT_RRF_K,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_MIN_SCORE,
    DEFAULT_K1,
    DEFAULT_B,
    DEFAULT_RERANK_WEIGHT,
    DEFAULT_RERANK_TOP_K,
    _clamp_float,
    _warn_and_reset_int,
    _warn_and_reset_float,
)
from src.core.Logger.DarwinLogger import darwin


@dataclass
class RetrievalArgs:
    """Unified arguments for all retrieval methods."""

    query: str
    method: RetrievalMethod = RetrievalMethod.HYBRID
    files: List[str] = field(default_factory=list)

    top_k: int = DEFAULT_TOP_K

    # Hybrid specific
    pool_size: int = DEFAULT_POOL_SIZE
    rrf_k: int = DEFAULT_RRF_K
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT

    # Semantic specific
    min_score: float = DEFAULT_MIN_SCORE

    # Keyword specific
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B

    # Per‑document limiting (applies to all methods)
    per_doc_top_k: Optional[int] = None   # NEW

    # Reranking (optional)
    rerank: bool = False
    rerank_weight: float = DEFAULT_RERANK_WEIGHT
    rerank_top_k: int = DEFAULT_RERANK_TOP_K

    # Storage & logging
    store_dir: Optional[str] = None

    def __post_init__(self) -> None:
        """Runtime validation with safe moderation for all parameters."""
        self.top_k = _warn_and_reset_int(self.top_k, 1, DEFAULT_TOP_K, "top_k")
        self.pool_size = max(self.top_k, _warn_and_reset_int(
            self.pool_size, 1, DEFAULT_POOL_SIZE, "pool_size"
        ))
        self.rrf_k = _warn_and_reset_int(self.rrf_k, 1, DEFAULT_RRF_K, "rrf_k")
        self.k1 = _clamp_float(self.k1, 0.01, 3.0, "BM25 'k1'")
        self.b = _clamp_float(self.b, 0.0, 1.0, "BM25 'b'")
        self.semantic_weight = _warn_and_reset_float(
            self.semantic_weight, 0.0, DEFAULT_SEMANTIC_WEIGHT, "semantic_weight"
        )
        self.keyword_weight = _warn_and_reset_float(
            self.keyword_weight, 0.0, DEFAULT_KEYWORD_WEIGHT, "keyword_weight"
        )
        self.min_score = _clamp_float(self.min_score, 0.0, 1.0, "min_score")
        if self.per_doc_top_k is not None and self.per_doc_top_k < 1:
            darwin.warning(
                f"per_doc_top_k must be >= 1 or None. "
                f"Received {self.per_doc_top_k}; reset to None."
            )
            self.per_doc_top_k = None