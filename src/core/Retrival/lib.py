from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from enum import Enum

from src.core.Logger.DarwinLogger import darwin

# -------------------- ENUMS & BASE MODELS --------------------
class OverlapDirection(str, Enum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"

class RetrievalMethod(str, Enum):
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"

@dataclass
class ChunkLink:
    url: str
    text: str = ""

# -------------------- GOLD-STANDARD DEFAULTS --------------------
# BM25 Canonical (Elasticsearch/Lucene standard)
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75          

# Retrieval defaults
DEFAULT_TOP_K = 20
DEFAULT_RRF_K = 60        # The exact value from the original RRF paper
DEFAULT_POOL_SIZE = 200   # Standard 10x multiplier for RRF candidate pooling

# Blending weights (neutral baseline)
DEFAULT_SEMANTIC_WEIGHT = 1.0
DEFAULT_KEYWORD_WEIGHT = 1.0

# Thresholds & Reranking
DEFAULT_MIN_SCORE = 0.0
DEFAULT_RERANK_WEIGHT = 0.3
DEFAULT_RERANK_TOP_K = 100

# -------------------- HELPER FUNCTIONS FOR VALIDATION --------------------
def _clamp_float(value: float, min_val: float, max_val: float, param_name: str) -> float:
    """Clamp a float value to [min_val, max_val] and warn if modified."""
    if value < min_val:
        new_value = min_val
        darwin.warning(
            f"{param_name} must be >= {min_val}. "
            f"Received {value}; clamped to {new_value}."
        )
        return new_value
    if value > max_val:
        new_value = max_val
        darwin.warning(
            f"{param_name} must be <= {max_val}. "
            f"Received {value}; clamped to {new_value}."
        )
        return new_value
    return value

def _warn_and_reset_int(value: int, min_val: int, default: int, param_name: str) -> int:
    """Reset invalid int to default and warn."""
    if value < min_val:
        darwin.warning(
            f"{param_name} must be >= {min_val}. "
            f"Received {value}; reset to default ({default})."
        )
        return default
    return value

def _warn_and_reset_float(value: float, min_val: float, default: float, param_name: str) -> float:
    """Reset invalid float to default and warn."""
    if value < min_val:
        darwin.warning(
            f"{param_name} must be >= {min_val}. "
            f"Received {value}; reset to default ({default})."
        )
        return default
    return value

# -------------------- RESULT DATACLASSES (Unchanged, perfectly fine) --------------------
@dataclass
class SearchAudit:
    raw_score: float = 0.0
    final_score: float = 0.0
    structural_type: Optional[str] = None
    penalty_factor: float = 1.0
    was_penalized: bool = False
    penalty_reason: Optional[str] = None
    merge_reason: Optional[str] = None
    gap_filled: bool = False
    concept_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_score": self.raw_score,
            "final_score": self.final_score,
            "structural_type": self.structural_type,
            "penalty_factor": self.penalty_factor,
            "was_penalized": self.was_penalized,
            "penalty_reason": self.penalty_reason,
            "merge_reason": self.merge_reason,
            "gap_filled": self.gap_filled,
            "concept_source": self.concept_source,
        }

@dataclass
class HybridSearchResult:
    text: str
    clean: str
    rrf_score: float = 0.0
    final_score: float = 0.0
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    source: str = ""
    page: int = 0
    page_end: int = 0
    heading: str = ""
    chunk_index: int = 0
    links: List[ChunkLink] = field(default_factory=list)
    matched_terms: Set[str] = field(default_factory=set)
    overlap_direction: OverlapDirection = OverlapDirection.NONE
    audit: Optional[SearchAudit] = None

@dataclass
class SemanticSearchResult:
    text: str
    clean: str
    score: float
    source: str
    page: int
    page_end: int = 0
    heading: str = ""
    chunk_index: int = 0
    links: List[ChunkLink] = field(default_factory=list)
    matched_terms: Set[str] = field(default_factory=set)
    overlap_direction: OverlapDirection = OverlapDirection.NONE
    audit: Optional[SearchAudit] = None

@dataclass
class KeywordSearchResult:
    text: str
    clean: str
    score: float
    source: str
    page: int
    page_end: int = 0
    heading: str = ""
    chunk_index: int = 0
    links: List[ChunkLink] = field(default_factory=list)
    matched_terms: Set[str] = field(default_factory=set)
    overlap_direction: OverlapDirection = OverlapDirection.NONE
    audit: Optional[SearchAudit] = None

# -------------------- 10/10 ARGUMENT DATACLASSES (FIXED) --------------------

@dataclass
class HybridSearchArgs:
    query: str
    files: List[str] = field(default_factory=list)
    
    # Number of results to return. Controls recall/precision tradeoff.
    # Range: 1+ (Typical: 20)
    top_k: int = DEFAULT_TOP_K
    
    # RRF (Reciprocal Rank Fusion) constant. Higher values give more weight to
    # high-ranking results from each system.
    # Range: 1+ (Typical: 60)
    rrf_k: int = DEFAULT_RRF_K
    
    # Weight multiplier for semantic scores in hybrid blending.
    # Range: 0.0+ (Default: 1.0)
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT
    
    # Weight multiplier for keyword scores in hybrid blending.
    # Range: 0.0+ (Default: 1.0)
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
    
    # Number of candidates to pool from each retrieval system before RRF.
    # Must be >= top_k for RRF to function properly.
    # Range: >= top_k (Default: 200)
    pool_size: int = DEFAULT_POOL_SIZE
    
    # BM25 term-frequency saturation parameter. Higher values increase the
    # influence of repeated terms.
    # Range: 0.01-3.0 (Typical: 1.5)
    k1: float = DEFAULT_K1
    
    # BM25 document length normalization parameter. Controls how much document
    # length affects scoring. 0 = no length normalization, 1 = full normalization.
    # Range: 0.0-1.0 (Typical: 0.75)
    b: float = DEFAULT_B
    
    # Per-document top-k limit. If set, limits results per source file.
    # Useful for balancing results across documents.
    # Range: None or >= 1 (Default: None)
    per_doc_top_k: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Runtime validation with safe moderation for all parameters."""
        # Clamp BM25 k1
        self.k1 = _clamp_float(self.k1, 0.01, 3.0, "BM25 'k1'")
        
        # Clamp BM25 b
        self.b = _clamp_float(self.b, 0.0, 1.0, "BM25 'b'")
        
        # Validate and reset top_k
        self.top_k = _warn_and_reset_int(self.top_k, 1, DEFAULT_TOP_K, "top_k")
        
        # Validate and reset rrf_k
        self.rrf_k = _warn_and_reset_int(self.rrf_k, 1, DEFAULT_RRF_K, "rrf_k")
        
        # Validate and reset semantic_weight
        self.semantic_weight = _warn_and_reset_float(
            self.semantic_weight, 0.0, DEFAULT_SEMANTIC_WEIGHT, "semantic_weight"
        )
        
        # Validate and reset keyword_weight
        self.keyword_weight = _warn_and_reset_float(
            self.keyword_weight, 0.0, DEFAULT_KEYWORD_WEIGHT, "keyword_weight"
        )
        
        # Validate and reset per_doc_top_k
        if self.per_doc_top_k is not None and self.per_doc_top_k < 1:
            darwin.warning(
                f"per_doc_top_k must be >= 1 or None. "
                f"Received {self.per_doc_top_k}; reset to None."
            )
            self.per_doc_top_k = None
        
        # Validate pool_size >= top_k
        if self.pool_size < self.top_k:
            darwin.warning(
                f"pool_size ({self.pool_size}) must be >= top_k ({self.top_k}). "
                f"Reset pool_size to {self.top_k}."
            )
            self.pool_size = self.top_k

@dataclass
class KeywordSearchArgs:
    query: str
    files: List[str] = field(default_factory=list)
    
    # Number of results to return. Controls recall/precision tradeoff.
    # Range: 1+ (Typical: 20)
    top_k: int = DEFAULT_TOP_K
    
    # BM25 term-frequency saturation parameter. Higher values increase the
    # influence of repeated terms.
    # Range: 0.01-3.0 (Typical: 1.5)
    k1: float = DEFAULT_K1
    
    # BM25 document length normalization parameter. Controls how much document
    # length affects scoring. 0 = no length normalization, 1 = full normalization.
    # Range: 0.0-1.0 (Typical: 0.75)
    b: float = DEFAULT_B
    
    # Per-document top-k limit. If set, limits results per source file.
    # Useful for balancing results across documents.
    # Range: None or >= 1 (Default: None)
    per_doc_top_k: Optional[int] = None

    def __post_init__(self) -> None:
        """Runtime validation with safe moderation for all parameters."""
        # Clamp BM25 k1
        self.k1 = _clamp_float(self.k1, 0.01, 3.0, "BM25 'k1'")
        
        # Clamp BM25 b
        self.b = _clamp_float(self.b, 0.0, 1.0, "BM25 'b'")
        
        # Validate and reset top_k
        self.top_k = _warn_and_reset_int(self.top_k, 1, DEFAULT_TOP_K, "top_k")
        
        # Validate and reset per_doc_top_k
        if self.per_doc_top_k is not None and self.per_doc_top_k < 1:
            darwin.warning(
                f"per_doc_top_k must be >= 1 or None. "
                f"Received {self.per_doc_top_k}; reset to None."
            )
            self.per_doc_top_k = None

@dataclass
class SemanticSearchArgs:
    query: str
    files: List[str] = field(default_factory=list)
    
    # Number of results to return. Controls recall/precision tradeoff.
    # Range: 1+ (Typical: 20)
    top_k: int = DEFAULT_TOP_K
    
    # Minimum similarity score threshold. Results below this are filtered out.
    # Range: 0.0-1.0 (Default: 0.0)
    min_score: float = DEFAULT_MIN_SCORE
    
    # Enable verbose inspection output for debugging.
    inspect: bool = False
    
    # Output results as JSON instead of human-readable format.
    json_output: bool = False
    
    # Per-document top-k limit. If set, limits results per source file.
    # Useful for balancing results across documents.
    # Range: None or >= 1 (Default: None)
    per_doc_top_k: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Runtime validation with safe moderation for all parameters."""
        # Clamp min_score
        self.min_score = _clamp_float(self.min_score, 0.0, 1.0, "min_score")
        
        # Validate and reset top_k
        self.top_k = _warn_and_reset_int(self.top_k, 1, DEFAULT_TOP_K, "top_k")
        
        # Validate and reset per_doc_top_k
        if self.per_doc_top_k is not None and self.per_doc_top_k < 1:
            darwin.warning(
                f"per_doc_top_k must be >= 1 or None. "
                f"Received {self.per_doc_top_k}; reset to None."
            )
            self.per_doc_top_k = None