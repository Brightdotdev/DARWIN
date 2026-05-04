# src/core/HybridSearch/HybridSearch.py - Fixed (no DEBUG)
# ---------------------------------------
# Hybrid retrieval combining semantic and keyword search via RRF.

import sys
import json
from pathlib import Path
from typing import List, Optional

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.features.Retrival.Semantic.SematicSearch import (
    search_single_document as semantic_search_single,
    search_across_documents as semantic_search_all,
)
from src.features.Retrival.Semantic.SematicSearchUtils import SematicSearchResult

from src.features.Retrival.Keyword.KeywordSearch import (
    search_single_document_keyword as keyword_search_single,
    search_across_documents_keyword as keyword_search_all,
    KeywordSearchResult,
)

from src.features.Retrival.Hybrid.HybridSearchUtils import (
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
    DEFAULT_ALPHA,
    DEFAULT_BETA,
)
from src.features.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.Utils import scan_folder


# ============================================================================
# Core fusion
# ============================================================================

def _normalize_scores(results: List[HybridSearchResult]) -> None:
    """Min-max normalize keyword scores in-place."""
    kw_scores = [r.keyword_score for r in results if r.keyword_score is not None]
    if not kw_scores:
        return
    min_s, max_s = min(kw_scores), max(kw_scores)
    span = max_s - min_s or 1.0
    for r in results:
        if r.keyword_score is not None:
            r.keyword_score = (r.keyword_score - min_s) / span

def hybrid_search(
    semantic_results: List[SematicSearchResult],
    keyword_results: List[KeywordSearchResult],
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA
) -> List[HybridSearchResult]:
    """Combine semantic and keyword results using Reciprocal Rank Fusion."""

    # Step 1: Merge into unified HybridSearchResult objects (deduped by source/page/chunk)
    merged = merge_results(semantic_results, keyword_results)

    # Step 2: Build separate ranked lists (by semantic score / keyword score)
    ranked_lists = build_ranked_lists(merged)  # [semantic_list, keyword_list]

    # Step 3: RRF over the unified typed lists
    weights = [semantic_weight, keyword_weight]
    results = reciprocal_rank_fusion(ranked_lists, rrf_k, weights)

    _normalize_scores(results)

        # Step 4: Apply final scoring blend
    for r in results:
        if r.semantic_score is not None and r.keyword_score is not None and r.keyword_score > 0:
            r.final_score = (alpha * r.semantic_score + beta * r.keyword_score) / (alpha + beta)
        elif r.semantic_score is not None:
            r.final_score = r.semantic_score
        elif r.keyword_score is not None and r.keyword_score > 0:
            r.final_score = r.keyword_score
        else:
            r.final_score = r.rrf_score

    # Re-sort by final_score after blending
    results.sort(key=lambda x: x.final_score, reverse=True)

    return results[:top_k]

# ============================================================================
# Single document
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
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA
) -> List[HybridSearchResult]:
    """
    Hybrid search within a single document.
    Runs semantic then keyword retrieval sequentially.
    """
    darwin.log(f"searching semantics... ({filename})", LogType.INFO)
    semantic_results = semantic_search_single(
        query=query, filename=filename, store=store, top_k=pool_size
    )
    darwin.log(f"semantic — {len(semantic_results)} candidates", LogType.INFO)

    darwin.log(f"searching keywords...", LogType.INFO)
    keyword_results = keyword_search_single(
        query=query, filename=filename, store=store, top_k=pool_size
    )
    darwin.log(f"keyword — {len(keyword_results)} candidates", LogType.INFO)

    if not semantic_results and not keyword_results:
        darwin.log("no results from either retriever", LogType.WARNING)
        return []

    darwin.log(f"fusing results via RRF...", LogType.INFO)
    semantic_results = semantic_results[:pool_size]
    keyword_results = keyword_results[:pool_size]
    results = hybrid_search(
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        top_k=top_k,
        rrf_k=rrf_k,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        alpha=alpha,
        beta=beta
    )
    darwin.log(f"done — {len(results)} final results", LogType.SUCCESS)

    return results


# ============================================================================
# Multi-document
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
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA
) -> List[HybridSearchResult]:
    """
    Hybrid search across multiple documents.
    Runs semantic then keyword retrieval sequentially.
    """
    if not filenames:
        darwin.log("no documents with both embeddings and BM25 index found", LogType.WARNING)
        darwin.log("run 'darwin embeddings' and 'darwin index-bm25' first", LogType.INFO)
        return []

    darwin.log(f"searching semantics across {len(filenames)} document(s)...", LogType.INFO)
    semantic_results = semantic_search_all(
        query=query, store=store, filenames=filenames, top_k=pool_size
    )
    darwin.log(f"semantic — {len(semantic_results)} candidates", LogType.INFO)

    darwin.log(f"searching keywords...", LogType.INFO)
    keyword_results = keyword_search_all(
        query=query, store=store, filenames=filenames, top_k=pool_size
    )
    darwin.log(f"keyword — {len(keyword_results)} candidates", LogType.INFO)

    if not semantic_results and not keyword_results:
        darwin.log("no results from either retriever", LogType.WARNING)
        return []

    darwin.log(f"fusing results via RRF...", LogType.INFO)

    semantic_results = semantic_results[:pool_size]
    keyword_results = keyword_results[:pool_size]
    results = hybrid_search(
        semantic_results=semantic_results,
        keyword_results=keyword_results,
        top_k=top_k,
        rrf_k=rrf_k,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        alpha=alpha,
        beta=beta
    )
    darwin.log(f"done — {len(results)} final results", LogType.SUCCESS)

    return results

