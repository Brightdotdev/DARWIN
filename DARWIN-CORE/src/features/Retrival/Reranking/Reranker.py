"""
src/core/Search/Reranking/Reranker.py
-------------------------------------
Cross-encoder reranking for improved search precision.

Why reranking?
--------------
Bi-encoders (semantic search) encode query and chunk separately — fast but
misses fine-grained query/chunk interaction. Cross-encoders process the pair
together with full attention — slower but much more accurate at judging
relevance. The standard pipeline is:

    fast retrieval (bi-encoder / BM25, pool_size=50-100)
        → cross-encoder rerank
            → top-k final results

Score blending
--------------
Raw cross-encoder logits are unbounded (roughly -10 to +10 for ms-marco
models). The original retrieval score lives in a completely different range
(cosine 0-1, BM25 0-∞, RRF ~0.01-0.03). Mixing them directly produces
garbage.

Fix: we sigmoid-normalize the rerank logit to [0,1] and also min-max
normalize the original score across the candidate set before blending:

    final = (1 - w) * norm_rerank + w * norm_original

This way both terms always live in [0,1] and the weight `w` actually means
something. Set w=0 to use pure cross-encoder ordering.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Union
import numpy as np


DEFAULT_TOP_K = 100

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

    Use str-Enum so the value serializes cleanly to JSON as a plain string
    ("none", "left", "right", "both") without needing a custom encoder.
    """
    NONE  = "none"
    LEFT  = "left"
    RIGHT = "right"
    BOTH  = "both"


@dataclass
class ChunkLink:
    """A hyperlink scoped to a chunk."""
    url:  str
    text: str = ""


@dataclass
class RerankedResult:
    """A result that has been reranked by a cross-encoder."""
    text:               str
    clean:              str
    rerank_score:       float                # raw cross-encoder logit
    original_score:     Optional[float]      # score from original retriever (pre-norm)
    final_score:        float                # blended normalized score
    source:             str
    page:               int
    page_end:           int
    heading:            str
    chunk_index:        int
    links:              List[ChunkLink] = field(default_factory=list)
    overlap_direction:  OverlapDirection = OverlapDirection.NONE
    matched_terms:      Optional[List[str]] = None
    
    def __repr__(self) -> str:
        return (
            f"RerankedResult(final={self.final_score:.3f}, "
            f"rerank={self.rerank_score:.3f}, "
            f"source='{self.source}', page={self.page})"
        )


# ============================================================================
# Score normalization
# ============================================================================

def _sigmoid(x: float) -> float:
    """Sigmoid to map an unbounded logit into (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def _get_original_score(candidate: Any) -> float:
    """
    Extract a normalized [0,1] score from any candidate type.
    
    Handles:
    - SemanticSearchResult: cosine similarity (0-1)
    - KeywordSearchResult: BM25 score (0-∞)
    - HybridSearchResult: RRF final_score (~0.01-0.03)
    - RerankedResult: existing reranker output
    """
    # Check for hybrid results first (most specific)
    if hasattr(candidate, 'final_score') and hasattr(candidate, 'rrf_score'):
        # RRF scores are typically very small (0.01-0.03)
        # Normalize to [0,1] range
        rrf = candidate.final_score
        # Assuming max RRF score ~0.033 (common with k=60)
        return min(1.0, rrf / 0.033)
    
    # Semantic search results have cosine similarity
    if hasattr(candidate, 'score') and 0 <= candidate.score <= 1:
        return candidate.score
    
    # Keyword search results have BM25 scores (unbounded)
    if hasattr(candidate, 'score'):
        # Use sigmoid to normalize unbounded BM25 scores
        return _sigmoid(candidate.score / 5.0)
    
    # Fallback for unknown types
    return 0.0


# ============================================================================
# Core reranking
# ============================================================================

def rerank_results(
    query: str,
    candidates: List[Any],
    top_k: int = DEFAULT_TOP_K,
    original_score_weight: float = 0.3,
) -> List[RerankedResult]:
    """
    Re-score candidate chunks with a cross-encoder.
    
    Args:
        query: Search query string
        candidates: List of search results (any type with .text attribute)
        top_k: Number of top results to return
        original_score_weight: Weight for original score in blending (0-1)
                              0 = pure cross-encoder, 1 = pure original
    
    Returns:
        List of RerankedResult sorted by final_score
    """
    from src.core.Logger.DarwinLogger import darwin, LogType
    from src.core.Models import ModelManager, ModelCategory
    
    if not candidates:
        darwin.log("No candidates to rerank", LogType.WARNING)
        return []

    darwin.log(f"Reranking {len(candidates)} candidates...", LogType.INFO)

    # Load cross-encoder model
    loader = ModelManager.get_loader()
    darwin.log(f"Loading Model")
    model = loader.get(ModelCategory.RERANKER)
    darwin.log(f"Loaded Successfully")

    # Prepare query-chunk pairs (truncate to ~512 tokens)
    pairs = [[query, c.text[:1500]] for c in candidates]
    
    # Get raw logits from cross-encoder
    logits = model.predict(pairs, show_progress_bar=False)
    
    # Normalize reranker scores to [0,1]
    norm_rerank = np.array([_sigmoid(logit) for logit in logits])

    # Build reranked results
    reranked = []
    for candidate, logit, nr in zip(candidates, logits, norm_rerank):
        # Get normalized original score
        norm_orig = _get_original_score(candidate)
        
        # Blend scores
        final = (1.0 - original_score_weight) * nr + original_score_weight * norm_orig
        
        # Extract original score for metadata
        original_score = None
        if hasattr(candidate, 'final_score'):
            original_score = candidate.final_score
        elif hasattr(candidate, 'score'):
            original_score = candidate.score
        elif hasattr(candidate, 'rrf_score'):
            original_score = candidate.rrf_score
        
        reranked.append(RerankedResult(
            text=candidate.text,
            clean=candidate.clean,
            rerank_score=float(logit),
            original_score=original_score,
            final_score=final,
            source=candidate.source,
            page=candidate.page,
            page_end=getattr(candidate, 'page_end', candidate.page),
            heading=getattr(candidate, 'heading', ''),
            chunk_index=getattr(candidate, 'chunk_index', 0),
            links=getattr(candidate, 'links', []),
            overlap_direction=getattr(candidate, 'overlap_direction', OverlapDirection.NONE),
            matched_terms=getattr(candidate, 'matched_terms', None),
        ))

    # Sort by final score descending
    reranked.sort(key=lambda x: x.final_score, reverse=True)
    
    result_count = min(top_k, len(reranked))
    loader.unload(ModelCategory.RERANKER)
    darwin.log(f"Freed Up Model Memory", LogType.SUCCESS)

    darwin.log(f"Reranked — returning top {result_count} results", LogType.SUCCESS)
    return reranked[:top_k]


