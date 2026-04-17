"""
src/core/Search/Reranking/Reranker.py
-------------------------------------
Cross-encoder reranking for improved search precision.

This module takes candidate chunks from any search method and
re-scores them using a cross-encoder that processes the query
and chunk together with full attention.

Why Reranking?
--------------
- Bi-encoders (semantic search) encode query/chunk separately → fast but misses nuance
- Cross-encoders process query+chunk together → slower but understands context
- Rerank top 50 candidates from fast retrieval → best of both worlds

Pipeline
--------
candidates (50-100) → cross-encoder scores → re-sort → top-k results

Integration
-----------
Can be used by:
- Semantic search: rerank_semantic(query, semantic_results)
- Keyword search: rerank_keyword(query, keyword_results)
- Hybrid search: rerank_hybrid(query, hybrid_results)
"""

from dataclasses import dataclass
from typing import List, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.core.Search.Sematic.SematicSearch import SearchResult
    from src.core.Search.Keyword.KeywordSearch import KeywordSearchResult
    from src.core.Search.Hybrid.HybridSearchUtils import HybridSearchResult

from src.core.Search.Reranking.RerankerUtils import (
    load_cross_encoder,
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_POOL_SIZE,
)
from src.lib.Terminal import TerminalUI


# ============================================================================
# Data model
# ============================================================================

@dataclass
class RerankedResult:
    """A result that has been reranked by cross-encoder."""
    text: str
    clean: str
    rerank_score: float           # Raw cross-encoder score
    original_score: Optional[float]  # Score from original retriever
    final_score: float            # Combined score (rerank + optional original)
    source: str
    page: int
    heading: str
    chunk_index: int
    
    def __repr__(self) -> str:
        return f"RerankedResult(final={self.final_score:.3f}, source='{self.source}', page={self.page})"


# ============================================================================
# Core reranking function
# ============================================================================

def rerank_results(
    query: str,
    candidates: List[any],  # Any result type with .text and .clean attributes
    model_name: str = DEFAULT_MODEL,
    top_k: int = DEFAULT_TOP_K,
    preserve_original_weight: float = 0.3,  # How much to blend original score
) -> List[RerankedResult]:
    """
    Rerank candidate chunks using cross-encoder.
    
    Args:
        query: Original search query
        candidates: List of search results (must have .text attribute)
        model_name: Cross-encoder model to use
        top_k: Number of final results to return
        preserve_original_weight: Weight for original score (0 = pure rerank)
    
    Returns:
        List[RerankedResult] sorted by final_score
    """
    if not candidates:
        TerminalUI.warn("No candidates to rerank")
        return []
    
    TerminalUI.dim(f"Reranking {len(candidates)} candidates...")
    
    # Load model
    model = load_cross_encoder(model_name)
    
    # Prepare pairs
    pairs = [[query, c.text[:1000]] for c in candidates]  # Truncate for speed
    
    # Get scores
    t = TerminalUI.timer("cross_encoder")
    scores = model.predict(pairs, show_progress_bar=False)
    t.done()
    
    # Build reranked results
    reranked = []
    for candidate, rerank_score in zip(candidates, scores):
        # Extract original score if available
        original_score = None
        if hasattr(candidate, 'score'):
            original_score = candidate.score
        elif hasattr(candidate, 'rrf_score'):
            original_score = candidate.rrf_score
        
        # Combine scores
        if original_score is not None:
            # Normalize rerank score to 0-1 range (approximate)
            norm_rerank = 1.0 / (1.0 + np.exp(-rerank_score))  # Sigmoid
            final_score = (1 - preserve_original_weight) * norm_rerank + preserve_original_weight * original_score
        else:
            final_score = float(rerank_score)
        
        reranked.append(RerankedResult(
            text=candidate.text,
            clean=candidate.clean,
            rerank_score=float(rerank_score),
            original_score=original_score,
            final_score=final_score,
            source=candidate.source,
            page=candidate.page,
            heading=candidate.heading,
            chunk_index=candidate.chunk_index,
        ))
    
    # Sort by final score
    reranked.sort(key=lambda x: x.final_score, reverse=True)
    
    TerminalUI.success(f"Reranked to top {min(top_k, len(reranked))} results")
    
    return reranked[:top_k]


# ============================================================================
# Convenience wrappers for different search types
# ============================================================================

def rerank_semantic(
    query: str,
    semantic_results: List['SearchResult'],
    model_name: str = DEFAULT_MODEL,
    top_k: int = DEFAULT_TOP_K,
) -> List[RerankedResult]:
    """Rerank semantic search results."""
    return rerank_results(query, semantic_results, model_name, top_k)


def rerank_keyword(
    query: str,
    keyword_results: List['KeywordSearchResult'],
    model_name: str = DEFAULT_MODEL,
    top_k: int = DEFAULT_TOP_K,
) -> List[RerankedResult]:
    """Rerank keyword search results."""
    return rerank_results(query, keyword_results, model_name, top_k)


def rerank_hybrid(
    query: str,
    hybrid_results: List['HybridSearchResult'],
    model_name: str = DEFAULT_MODEL,
    top_k: int = DEFAULT_TOP_K,
) -> List[RerankedResult]:
    """Rerank hybrid search results."""
    return rerank_results(query, hybrid_results, model_name, top_k)






# ============================================================================
# Full search with reranking (integrated pipeline)
# ============================================================================

def semantic_search_with_rerank(
    query: str,
    store,
    filename: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    top_k: int = DEFAULT_TOP_K,
    pool_size: int = DEFAULT_POOL_SIZE,
) -> List[RerankedResult]:
    """
    Semantic search with automatic reranking.
    
    Pipeline: semantic_search(pool_size) → rerank(top_k)
    """
    from src.core.Search.Sematic.SematicSearch import (
        search_single_document as semantic_single,
        search_across_documents as semantic_all,
    )
    
    # Get candidates
    if filename:
        candidates = semantic_single(query, filename, store, top_k=pool_size)
    else:
        candidates = semantic_all(query, store, top_k=pool_size)
    
    # Rerank
    return rerank_semantic(query, candidates, model_name, top_k)


def hybrid_search_with_rerank(
    query: str,
    store,
    filename: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    top_k: int = DEFAULT_TOP_K,
    pool_size: int = DEFAULT_POOL_SIZE,
) -> List[RerankedResult]:
    """
    Hybrid search with automatic reranking.
    
    Pipeline: hybrid_search(pool_size) → rerank(top_k)
    """
    from src.core.Search.Hybrid.HybridSearch import (
        search_single_document_hybrid,
        search_across_documents_hybrid,
    )
    
    # Get candidates
    if filename:
        candidates = search_single_document_hybrid(
            query, filename, store, top_k=pool_size
        )
    else:
        candidates = search_across_documents_hybrid(
            query, store, top_k=pool_size
        )
    
    # Rerank
    return rerank_hybrid(query, candidates, model_name, top_k)



