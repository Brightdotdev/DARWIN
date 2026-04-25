"""
src/core/HybridSearch/HybridSearch.py
---------------------------------------
Hybrid retrieval combining semantic and keyword search via RRF.

Retrieval is intentionally sequential — both retrievers are CPU-bound
(numpy cosine similarity + BM25 scoring), so threading offers no real
speedup and adds context-switch overhead.
"""

from typing import List, Optional

from src.core.Search.Sematic.SematicSearch import (
    search_single_document  as semantic_search_single,
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
# Core fusion
# ============================================================================

def hybrid_search(
    semantic_results: List[SematicSearchResult],
    keyword_results:  List[KeywordSearchResult],
    top_k:            int   = DEFAULT_TOP_K,
    rrf_k:            int   = DEFAULT_RRF_K,
    semantic_weight:  float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight:   float = DEFAULT_KEYWORD_WEIGHT,
) -> List[HybridSearchResult]:
    """
    Combine semantic and keyword results using Reciprocal Rank Fusion.
    """
    merged      = merge_results(semantic_results, keyword_results)
    ranked      = build_ranked_lists(merged)
    rrf_results = reciprocal_rank_fusion(ranked, rrf_k, [semantic_weight, keyword_weight])
    final       = apply_hybrid_scoring(rrf_results)
    return final[:top_k]


# ============================================================================
# Single document
# ============================================================================

def search_single_document_hybrid(
    query:            str,
    filename:         str,
    store:            DarwinStore,
    top_k:            int   = DEFAULT_TOP_K,
    rrf_k:            int   = DEFAULT_RRF_K,
    semantic_weight:  float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight:   float = DEFAULT_KEYWORD_WEIGHT,
    pool_size:        int   = DEFAULT_POOL_SIZE,
) -> List[HybridSearchResult]:
    """
    Hybrid search within a single document.
    Runs semantic then keyword retrieval sequentially.
    """
    TerminalUI.dim(f"  searching semantics...  ({filename})")
    semantic_results = semantic_search_single(
        query=query, filename=filename, store=store, top_k=pool_size
    )
    TerminalUI.dim(f"  ✓ semantic  — {len(semantic_results)} candidates")

    TerminalUI.dim(f"  searching keywords...")
    keyword_results = search_single_document_keyword(
        query=query, filename=filename, store=store, top_k=pool_size
    )
    TerminalUI.dim(f"  ✓ keyword   — {len(keyword_results)} candidates")

    if not semantic_results and not keyword_results:
        TerminalUI.warn("no results from either retriever")
        return []

    TerminalUI.dim(f"  fusing results via RRF...")
    results = hybrid_search(
        semantic_results = semantic_results,
        keyword_results  = keyword_results,
        top_k            = top_k,
        rrf_k            = rrf_k,
        semantic_weight  = semantic_weight,
        keyword_weight   = keyword_weight,
    )
    TerminalUI.dim(f"  ✓ done      — {len(results)} final results")

    return results


# ============================================================================
# Multi-document
# ============================================================================

def search_across_documents_hybrid(
    query:            str,
    store:            DarwinStore,
    filenames:        Optional[List[str]] = None,
    top_k:            int   = DEFAULT_TOP_K,
    rrf_k:            int   = DEFAULT_RRF_K,
    semantic_weight:  float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight:   float = DEFAULT_KEYWORD_WEIGHT,
    pool_size:        int   = DEFAULT_POOL_SIZE,
) -> List[HybridSearchResult]:
    """
    Hybrid search across multiple documents.
    Runs semantic then keyword retrieval sequentially.
    """
    if filenames is None:
        status    = store.status()
        filenames = [
            fname for fname, info in status.get("files", {}).items()
            if info["components"]["embeddings"] and info["components"]["bm25"]
        ]

    if not filenames:
        TerminalUI.warn("no documents with both embeddings and BM25 index found")
        TerminalUI.dim("run 'darwin embeddings' and 'darwin index-bm25' first")
        return []

    TerminalUI.dim(f"  searching semantics across {len(filenames)} document(s)...")
    semantic_results = semantic_search_all(
        query=query, store=store, filenames=filenames, top_k=pool_size
    )
    TerminalUI.dim(f"  ✓ semantic  — {len(semantic_results)} candidates")

    TerminalUI.dim(f"  searching keywords...")
    keyword_results = search_across_documents_keyword(
        query=query, store=store, filenames=filenames, top_k=pool_size
    )
    TerminalUI.dim(f"  ✓ keyword   — {len(keyword_results)} candidates")

    if not semantic_results and not keyword_results:
        TerminalUI.warn("no results from either retriever")
        return []

    TerminalUI.dim(f"  fusing results via RRF...")
    results = hybrid_search(
        semantic_results = semantic_results,
        keyword_results  = keyword_results,
        top_k            = top_k,
        rrf_k            = rrf_k,
        semantic_weight  = semantic_weight,
        keyword_weight   = keyword_weight,
    )
    TerminalUI.dim(f"  ✓ done      — {len(results)} final results")

    return results