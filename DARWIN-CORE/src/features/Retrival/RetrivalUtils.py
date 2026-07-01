"""
src/core/Retrieval/RetrievalArgs.py
-----------------------------------
Unified data class for all retrieval methods.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class RetrievalMethod(str, Enum):
    """Available retrieval methods."""
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"


# Default values
DEFAULT_TOP_K = 5
DEFAULT_POOL_SIZE = 200
DEFAULT_RRF_K = 60
DEFAULT_SEMANTIC_WEIGHT = 1.0
DEFAULT_KEYWORD_WEIGHT = 1.0
DEFAULT_MIN_SCORE = 0.3
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_RERANK_WEIGHT = 0.3
DEFAULT_RERANK_TOP_K = 100
DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 0.5


@dataclass
class RetrievalArgs:
    """Unified arguments for all retrieval methods."""
    
    # Core
    query: str
    method: RetrievalMethod = RetrievalMethod.HYBRID
    files: list[str] = field(default_factory=list)
    
    # Output
    top_k: int = DEFAULT_TOP_K
    inspect: bool = False
    json_output: bool = False
    
    # Hybrid-specific
    pool_size: int = DEFAULT_POOL_SIZE
    rrf_k: int = DEFAULT_RRF_K
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    
    # Semantic-specific
    min_score: float = DEFAULT_MIN_SCORE
    
    # Keyword-specific
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    
    # Reranking
    rerank: bool = False
    rerank_weight: float = DEFAULT_RERANK_WEIGHT
    rerank_top_k: int = DEFAULT_RERANK_TOP_K
    
    # WebSocket
    sse_port: int = 8765
    no_ws: bool = False