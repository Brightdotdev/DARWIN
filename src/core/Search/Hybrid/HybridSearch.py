"""
src/core/HybridSearch/HybridSearch.py
---------------------------------------
Hybrid retrieval combining semantic and keyword search via RRF.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.core.Search.Sematic.SematicSearch import (
    search_single_document as semantic_search_single,
    search_across_documents as semantic_search_all,
)
from src.core.Search.Sematic.SematicSearchUtils import SematicSearchResult

from src.core.Search.Keyword.KeywordSearch import (
    search_single_document_keyword,
    search_across_documents_keyword,
    KeywordSearchResult,
)

from src.core.Search.Hybrid.HybridSearchUtils import (
    reciprocal_rank_fusion,
    merge_results,
    build_ranked_lists,
    apply_hybrid_scoring,
    parallel_retrieval,
    parallel_retrieval_multi,
    HybridSearchResult,
    DEFAULT_TOP_K,
    DEFAULT_RRF_K,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_POOL_SIZE,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


# ============================================================================
# Core hybrid search
# ============================================================================

def hybrid_search(
    semantic_results: List[SematicSearchResult],
    keyword_results: List[KeywordSearchResult],
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> List[HybridSearchResult]:
    """
    Combine semantic and keyword results using RRF.
    """
    # Step 1: Merge and deduplicate
    merged_items = merge_results(semantic_results, keyword_results)
    
    # Step 2: Build ranked lists
    ranked_lists = build_ranked_lists(merged_items)
    
    # Step 3: RRF fusion
    rrf_results = reciprocal_rank_fusion(ranked_lists, rrf_k, [semantic_weight, keyword_weight])
    
    # Step 4: Hybrid scoring (RRF + weighted original scores)
    final_results = apply_hybrid_scoring(rrf_results)
    
    return final_results[:top_k]


# ============================================================================
# Single document search (with parallel support)
# ============================================================================

def search_single_document_hybrid(
    query: str,
    filename: str,
    store: DarwinStore,
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    pool_size: int = DEFAULT_POOL_SIZE,
    parallel: bool = True,
) -> List[HybridSearchResult]:
    """
    Hybrid search within a single document.
    """
    if parallel:
        # Parallel retrieval
        semantic_results, keyword_results = parallel_retrieval(
            query=query,
            filename=filename,
            store=store,
            pool_size=pool_size,
            semantic_search_fn=semantic_search_single,
            keyword_search_fn=search_single_document_keyword,
        )
    else:
        # Sequential retrieval
        semantic_results = semantic_search_single(
            query=query, filename=filename, store=store, top_k=pool_size
        )
        keyword_results = search_single_document_keyword(
            query=query, filename=filename, store=store, top_k=pool_size
        )
    
    TerminalUI.dim(f"Semantic: {len(semantic_results)} results, Keyword: {len(keyword_results)} results")
    
    return hybrid_search(
        query=query,
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        top_k=top_k,
        rrf_k=rrf_k,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
    )


# ============================================================================
# Multi-document search (with parallel support)
# ============================================================================

def search_across_documents_hybrid(
    query: str,
    store: DarwinStore,
    filenames: Optional[List[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    pool_size: int = DEFAULT_POOL_SIZE,
    parallel: bool = True,
) -> List[HybridSearchResult]:
    """
    Hybrid search across multiple documents.
    """
    # Get documents with both indexes
    if filenames is None:
        status = store.status()
        filenames = [
            fname for fname, info in status.get("files", {}).items()
            if info["components"]["embeddings"] and info["components"]["bm25"]
        ]
    
    if not filenames:
        TerminalUI.warn("No documents with both embeddings and BM25 index found")
        TerminalUI.dim("Try running 'darwin embeddings' and 'darwin index-bm25' first")
        return []
    
    if parallel:
        semantic_results, keyword_results = parallel_retrieval_multi(
            query=query,
            store=store,
            filenames=filenames,
            pool_size=pool_size,
            semantic_search_fn=semantic_search_all,
            keyword_search_fn=search_across_documents_keyword,
        )
    else:
        semantic_results = semantic_search_all(
            query=query, store=store, filenames=filenames, top_k=pool_size
        )
        keyword_results = search_across_documents_keyword(
            query=query, store=store, filenames=filenames, top_k=pool_size
        )
    
    TerminalUI.dim(f"Semantic: {len(semantic_results)} results, Keyword: {len(keyword_results)} results")
    
    return hybrid_search(
        query=query,
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        top_k=top_k,
        rrf_k=rrf_k,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
    )