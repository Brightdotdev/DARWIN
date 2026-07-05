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
)


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
    sse_port: int = 8765
    no_sse: bool = False