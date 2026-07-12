from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from enum import Enum

from src.core.Logger.DarwinLogger import darwin

# FIXED: Spelling error (Retrival -> Retrieval)
# from src.core.Retrieval.Keyword.KeywordUtilities import DEFAULT_B  # <-- Remove this import entirely

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
DEFAULT_B = 0.75          # Single source of truth (removed redundant import)

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
    top_k: int = DEFAULT_TOP_K
    rrf_k: int = DEFAULT_RRF_K
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
    pool_size: int = DEFAULT_POOL_SIZE
    k1: float = DEFAULT_K1   
    b: float = DEFAULT_B
    per_doc_top_k: Optional[int] = None  
    
    def __post_init__(self) -> None:
        """Runtime validation to ensure safe blending."""
        if self.semantic_weight < 0 or self.keyword_weight < 0:
            darwin.error("Weights cannot be negative.")
            return
        if self.pool_size < self.top_k:
            darwin.error(f"pool_size ({self.pool_size}) must be >= top_k ({self.top_k}) for RRF to function.")
            return
        if not 0.0 <= self.b <= 1.0:
            darwin.error(f"BM25 'b' parameter must be between 0.0 and 1.0. Got {self.b}.")
            return
        if self.k1 <= 0:
            darwin.error(f"BM25 'k1' parameter must be positive. Got {self.k1}.")
            return
        if self.per_doc_top_k is not None and self.per_doc_top_k < 1:
            darwin.error(f"per_doc_top_k must be >= 1 or None. Got {self.per_doc_top_k}.")
            return

@dataclass
class KeywordSearchArgs:
    query: str
    files: List[str] = field(default_factory=list)
    top_k: int = DEFAULT_TOP_K
    k1: float = DEFAULT_K1      
    b: float = DEFAULT_B 
    per_doc_top_k: Optional[int] = None  

    def __post_init__(self) -> None:
        if not 0.0 <= self.b <= 1.0:
            darwin.error(f"BM25 'b' must be between 0.0 and 1.0. Got {self.b}.")
            return
        if self.k1 <= 0:
            darwin.error(f"BM25 'k1' must be positive. Got {self.k1}.")
            return
        if self.per_doc_top_k is not None and self.per_doc_top_k < 1:
            darwin.error(f"per_doc_top_k must be >= 1 or None. Got {self.per_doc_top_k}.")
            return

@dataclass
class SemanticSearchArgs:
    query: str
    files: List[str] = field(default_factory=list)
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    inspect: bool = False
    json_output: bool = False
    per_doc_top_k: Optional[int] = None  
    
    def __post_init__(self) -> None:
        if self.per_doc_top_k is not None and self.per_doc_top_k < 1:
            darwin.error(f"per_doc_top_k must be >= 1 or None. Got {self.per_doc_top_k}.")
            return