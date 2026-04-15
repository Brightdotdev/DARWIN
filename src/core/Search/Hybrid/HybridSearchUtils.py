"""
src/core/HybridSearch/HybridSearchUtils.py
------------------------------------------
Utilities for hybrid search: RRF fusion, result merging, and formatting.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================================
# Constants
# ============================================================================

DEFAULT_TOP_K = 5
DEFAULT_RRF_K = 60
DEFAULT_SEMANTIC_WEIGHT = 1.0
DEFAULT_KEYWORD_WEIGHT = 1.0
DEFAULT_POOL_SIZE = 50


# ============================================================================
# Data model
# ============================================================================

@dataclass
class HybridSearchResult:
    """Single hybrid search result with scores from both retrievers."""
    text: str
    clean: str
    rrf_score: float = 0.0                    # Combined RRF score (default 0.0)
    final_score: float = 0.0                  # Hybrid score (RRF + weighted original scores)
    semantic_score: Optional[float] = None    # Original semantic score
    keyword_score: Optional[float] = None     # Original BM25 score
    source: str = ""
    page: int = 0
    heading: str = ""
    chunk_index: int = 0
    
    def __repr__(self) -> str:
        return (f"HybridSearchResult(rrf={self.rrf_score:.3f}, "
                f"final={self.final_score:.3f}, source='{self.source}', page={self.page})")


# ============================================================================
# Args
# ============================================================================

@dataclass
class HybridSearchArgs:
    query: str
    file: Optional[str] = None
    project: Optional[str] = None
    top_k: int = DEFAULT_TOP_K
    rrf_k: int = DEFAULT_RRF_K
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
    pool_size: int = DEFAULT_POOL_SIZE
    parallel: bool = True          # Enable parallel retrieval
    inspect: bool = False
    json_output: bool = False


# ============================================================================
# Result building and merging
# ============================================================================

def build_unified_result(
    result,
    semantic_score: Optional[float] = None,
    keyword_score: Optional[float] = None,
) -> HybridSearchResult:
    """Create a HybridSearchResult from a semantic or keyword result."""
    return HybridSearchResult(
        text=result.text,
        clean=result.clean,
        rrf_score=0.0,              # Will be set by RRF
        final_score=0.0,            # Will be set by hybrid scoring
        semantic_score=semantic_score,
        keyword_score=keyword_score,
        source=result.source,
        page=result.page,
        heading=result.heading,
        chunk_index=result.chunk_index,
    )


def merge_results(semantic_results: List, keyword_results: List) -> List[HybridSearchResult]:
    """
    Merge semantic and keyword results, deduplicating by (source, chunk_index).
    """
    seen_items: Dict[tuple, HybridSearchResult] = {}
    
    def get_key(r):
        return (r.source, r.chunk_index)
    
    # Semantic pass
    for r in semantic_results:
        key = get_key(r)
        if key not in seen_items:
            seen_items[key] = build_unified_result(r, semantic_score=r.score)
        else:
            seen_items[key].semantic_score = r.score
    
    # Keyword pass
    for r in keyword_results:
        key = get_key(r)
        if key not in seen_items:
            seen_items[key] = build_unified_result(r, keyword_score=r.score)
        else:
            seen_items[key].keyword_score = r.score
    
    return list(seen_items.values())


def build_ranked_lists(items: List[HybridSearchResult]) -> List[List[HybridSearchResult]]:
    """
    Build separate ranked lists for semantic and keyword results.
    Each list is sorted by its respective score descending.
    """
    semantic_list = [x for x in items if x.semantic_score is not None]
    keyword_list = [x for x in items if x.keyword_score is not None]
    
    semantic_list.sort(key=lambda x: x.semantic_score, reverse=True)
    keyword_list.sort(key=lambda x: x.keyword_score, reverse=True)
    
    return [semantic_list, keyword_list]


# ============================================================================
# Reciprocal Rank Fusion (Optimized)
# ============================================================================

def reciprocal_rank_fusion(
    result_lists: List[List[HybridSearchResult]],
    rrf_k: int = DEFAULT_RRF_K,
    weights: Optional[List[float]] = None,
) -> List[HybridSearchResult]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    Optimized: Uses dictionary lookups O(n) instead of list.index() O(n²).
    """
    if not result_lists:
        return []
    
    if weights is None:
        weights = [1.0] * len(result_lists)
    
    if len(weights) != len(result_lists):
        raise ValueError(f"weights length ({len(weights)}) must match lists ({len(result_lists)})")
    
    # Build rank lookup maps for O(1) rank access
    rank_maps = []
    for lst in result_lists:
        rank_map = {}
        for rank, item in enumerate(lst, start=1):
            key = (item.source, item.chunk_index)
            rank_map[key] = rank
        rank_maps.append(rank_map)
    
    # Collect all unique items
    items_by_key: Dict[tuple, HybridSearchResult] = {}
    for lst in result_lists:
        for item in lst:
            key = (item.source, item.chunk_index)
            items_by_key[key] = item
    
    # Compute RRF scores
    for key, item in items_by_key.items():
        rrf_score = 0.0
        for list_idx, rank_map in enumerate(rank_maps):
            if key in rank_map:
                rank = rank_map[key]
                rrf_score += weights[list_idx] / (rrf_k + rank)
        item.rrf_score = rrf_score
    
    # Sort by RRF score
    return sorted(items_by_key.values(), key=lambda x: x.rrf_score, reverse=True)


# ============================================================================
# Hybrid Scoring (Optimized)
# ============================================================================

def apply_hybrid_scoring(
    items: List[HybridSearchResult],
    alpha: float = 0.3,
    beta: float = 0.2,
) -> List[HybridSearchResult]:
    """
    Apply hybrid scoring: final = RRF + alpha * semantic + beta * keyword.
    
    Optimized: Single pass, no lambda in sort key (pre-compute).
    """
    if not items:
        return []
    
    # Normalize semantic and keyword scores to 0-1 range for fair combination
    sem_scores = [item.semantic_score for item in items if item.semantic_score is not None]
    kw_scores = [item.keyword_score for item in items if item.keyword_score is not None]
    
    sem_max = max(sem_scores) if sem_scores else 1.0
    kw_max = max(kw_scores) if kw_scores else 1.0
    
    for item in items:
        # Normalize scores
        norm_sem = (item.semantic_score / sem_max) if item.semantic_score and sem_max > 0 else 0.0
        norm_kw = (item.keyword_score / kw_max) if item.keyword_score and kw_max > 0 else 0.0
        
        # Compute final score
        item.final_score = item.rrf_score + alpha * norm_sem + beta * norm_kw
    
    # Sort by final_score (pre-computed, no lambda overhead)
    items.sort(key=lambda x: x.final_score, reverse=True)
    return items


# ============================================================================
# Parallel Retrieval
# ============================================================================

def parallel_retrieval(
    query: str,
    filename: str,
    store,
    pool_size: int,
    semantic_search_fn,
    keyword_search_fn,
) -> tuple:
    """
    Run semantic and keyword retrieval in parallel.
    
    Returns:
        (semantic_results, keyword_results)
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        semantic_future = executor.submit(
            semantic_search_fn,
            query=query,
            filename=filename,
            store=store,
            top_k=pool_size,
        )
        keyword_future = executor.submit(
            keyword_search_fn,
            query=query,
            filename=filename,
            store=store,
            top_k=pool_size,
        )
        
        # Wait for both to complete
        semantic_results = semantic_future.result()
        keyword_results = keyword_future.result()
    
    return semantic_results, keyword_results


def parallel_retrieval_multi(
    query: str,
    store,
    filenames: List[str],
    pool_size: int,
    semantic_search_fn,
    keyword_search_fn,
) -> tuple:
    """
    Run multi-document semantic and keyword retrieval in parallel.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        semantic_future = executor.submit(
            semantic_search_fn,
            query=query,
            store=store,
            filenames=filenames,
            top_k=pool_size,
        )
        keyword_future = executor.submit(
            keyword_search_fn,
            query=query,
            store=store,
            filenames=filenames,
            top_k=pool_size,
        )
        
        semantic_results = semantic_future.result()
        keyword_results = keyword_future.result()
    
    return semantic_results, keyword_results


# ============================================================================
# Result formatting
# ============================================================================

def format_hybrid_results(results: List[HybridSearchResult], json_output: bool = False) -> str:
    """Format hybrid search results for display or JSON export."""
    import json
    
    if json_output:
        output = []
        for r in results:
            output.append({
                "rrf_score": round(r.rrf_score, 4),
                "final_score": round(r.final_score, 4),
                "semantic_score": round(r.semantic_score, 4) if r.semantic_score else None,
                "keyword_score": round(r.keyword_score, 4) if r.keyword_score else None,
                "source": r.source,
                "page": r.page,
                "heading": r.heading,
                "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
            })
        return json.dumps(output, indent=2)
    
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] Final={r.final_score:.3f} | RRF={r.rrf_score:.3f} | {r.source} | p{r.page}")
        if r.semantic_score:
            lines.append(f"    └─ semantic: {r.semantic_score:.3f}")
        if r.keyword_score:
            lines.append(f"    └─ keyword:  {r.keyword_score:.3f}")
        if r.heading:
            lines.append(f"    └─ {r.heading}")
        preview = r.text[:200].replace('\n', ' ').strip()
        lines.append(f"       {preview}...")
        lines.append("")
    
    return "\n".join(lines)


def to_llm_context(results: List[HybridSearchResult], max_chunks: int = 5) -> str:
    """Convert hybrid results to LLM-ready context string."""
    context_parts = []
    
    for r in results[:max_chunks]:
        part = f"[Document: {r.source}, Page: {r.page}]"
        if r.heading:
            part += f"\nHeading: {r.heading}"
        part += f"\nContent: {r.text}\n---"
        context_parts.append(part)
    
    return "\n".join(context_parts)