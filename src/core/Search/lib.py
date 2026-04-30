
from enum import Enum
from dataclasses import dataclass, field
from typing import List,Optional,Any

# ============================================================================
# Data model
# ============================================================================


class OverlapDirection(str, Enum):
    """
    Describes which side(s) of a chunk carry overlap sentences.

    NONE   — no overlap; chunk is self-contained
    LEFT   — chunk *opens* with sentences repeated from the tail of the
             previous chunk  (this chunk "looks left")
    RIGHT  — chunk *closes* with sentences repeated at the head of the
             next chunk  (this chunk "donates right")
    BOTH   — interior window: received from the left AND donates to the right

    Use str-Enum so the value serialises cleanly to JSON as a plain string
    ("none", "left", "right", "both") without needing a custom encoder.
    """
    NONE  = "none"
    LEFT  = "left"
    RIGHT = "right"
    BOTH  = "both"



# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkLink:
    """A hyperlink scoped to a chunk."""
    url:  str
    text: str = ""



@dataclass
class SematicSearchResult:
    """Single search result with all metadata needed for RAG."""
    text: str           # Original text (for LLM context)
    clean: str          # Cleaned text (what was embedded)
    score: float        # Cosine similarity (0-1)
    source: str         # Document filename
    page: int           # Page number
    heading: str        # Section heading
    chunk_index: int    # Original chunk position
    page_end : int = 0
    overlap_direction : OverlapDirection = OverlapDirection.NONE
    links: List[tuple[str, str]] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return f"SematicSearchResult(score={self.score:.3f}, source='{self.source}', page={self.page})"

@dataclass
class KeywordSearchResult:
    """Single BM25 search result with all metadata."""
    text: str           # Original text (for LLM context)
    clean: str          # Cleaned text (what was indexed)
    score: float        # BM25 score (not normalized to 0-1)
    source: str         # Document filename
    page: int           # Page number
    heading: str        # Section heading
    chunk_index: int    # Original chunk position
    matched_terms: List[str]
    page_end : int = 0
    overlap_direction : OverlapDirection =  OverlapDirection.NONE 
    links:            List[ChunkLink]  = field(default_factory=list)
    
    
    def __repr__(self) -> str:
        return f"KeywordSearchResult(score={self.score:.3f}, source='{self.source}', page={self.page})"


# ============================================================================
# Data model
# ============================================================================

@dataclass
class HybridSearchResult:
    """Single hybrid search result with scores from both retrievers."""
    text:          str
    clean:         str
    rrf_score:     float = 0.0
    final_score:   float = 0.0
    semantic_score: Optional[float] = None
    keyword_score:  Optional[float] = None
    source:        str = ""
    page:          int = 0
    page_end:      int = 0
    heading:       str = ""
    chunk_index:   int = 0
    links:         List[Any] = field(default_factory=list) 
    matched_terms: List[str] | None = None
    overlap_direction : OverlapDirection =  OverlapDirection.NONE 
    links:            List[ChunkLink]  = field(default_factory=list)
    
    def __repr__(self) -> str:
        return (
    f"HybridSearchResult(rrf={self.rrf_score:.3f}, "
    f"final={self.final_score:.3f}, "
    f"sem={self.semantic_score}, kw={self.keyword_score}, "
    f"src='{self.source}', p={self.page})"
)
        
        