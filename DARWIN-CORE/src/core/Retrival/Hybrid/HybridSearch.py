"""
src/features/Retrival/Hybrid/HybridSearch.py
--------------------------------------------
Hybrid search: combine semantic and keyword results using RRF.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from src.core.Retrival.Semantic.SemanticSearch import (
    search_single_document as semantic_search_single,
    search_across_documents as semantic_search_all,
)

from src.core.Retrival.Keyword.KeywordSearch import (
    search_single_document_keyword as keyword_search_single,
    search_across_documents_keyword as keyword_search_all,
)
from src.core.Retrival.lib import (
    KeywordSearchResult,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_POOL_SIZE,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    HybridSearchResult,
    SemanticSearchResult,
)

from src.core.Retrival.Hybrid.HybridSearchUtils import (
    reciprocal_rank_fusion,
    merge_results,
    build_ranked_lists,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin


def hybrid_search(
    semantic_results: List[SemanticSearchResult],
    keyword_results: List[KeywordSearchResult],
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> List[HybridSearchResult]:
    """
    Combine semantic and keyword results using Reciprocal Rank Fusion (RRF).
    If one list is empty, the other is used alone.
    """
    if not semantic_results and not keyword_results:
        return []

    # If one side is empty, convert the other to HybridSearchResult and return
    if not semantic_results:
        # Only keyword results
        results = []
        for r in keyword_results:
            results.append(
                HybridSearchResult(
                    text=r.text,
                    clean=r.clean,
                    semantic_score=None,
                    keyword_score=r.score,
                    source=r.source,
                    page=r.page,
                    page_end=r.page_end,
                    heading=r.heading,
                    chunk_index=r.chunk_index,
                    links=r.links,
                    matched_terms=r.matched_terms,
                    overlap_direction=r.overlap_direction,
                    rrf_score=r.score,  # use keyword score as RRF score (no fusion)
                    final_score=r.score,
                )
            )
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:top_k]

    if not keyword_results:
        # Only semantic results
        results = []
        for r in semantic_results:
            results.append(
                HybridSearchResult(
                    text=r.text,
                    clean=r.clean,
                    semantic_score=r.score,
                    keyword_score=None,
                    source=r.source,
                    page=r.page,
                    page_end=r.page_end,
                    heading=r.heading,
                    chunk_index=r.chunk_index,
                    links=r.links,
                    matched_terms=r.matched_terms or set(),
                    overlap_direction=r.overlap_direction,
                    rrf_score=r.score,  # use semantic score as RRF score
                    final_score=r.score,
                )
            )
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:top_k]

    # Both non‑empty – proceed with normal RRF
    merged = merge_results(semantic_results, keyword_results)
    ranked_lists = build_ranked_lists(merged)
    weights = [semantic_weight, keyword_weight]
    results = reciprocal_rank_fusion(ranked_lists, rrf_k, weights)

    for r in results:
        r.final_score = r.rrf_score

    # Normalize RRF scores to [0,1] so orchestrator thresholds work correctly
    if results:
        scores = [r.final_score for r in results]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s
        if range_s > 1e-9:
            for r in results:
                norm = (r.final_score - min_s) / range_s
                r.final_score = norm
                r.rrf_score = norm
        else:
            for r in results:
                r.final_score = 1.0
                r.rrf_score = 1.0

    results.sort(key=lambda x: x.final_score, reverse=True)
    return results[:top_k]


def search_single_document_hybrid(
    query: str,
    model_name: str,
    filename: str,
    store: DarwinStore,
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    pool_size: int = DEFAULT_POOL_SIZE,
    per_doc_top_k: Optional[int] = None,
) -> List[HybridSearchResult]:
    """Hybrid search within a single document (parallelized)."""

    def _semantic():
        return semantic_search_single(
            query=query,
            model_name=model_name,
            filename=filename,
            store=store,
            top_k=pool_size,
            per_doc_top_k=per_doc_top_k,
        )

    def _keyword():
        return keyword_search_single(
            query=query,
            filename=filename,
            store=store,
            top_k=pool_size,
            per_doc_top_k=per_doc_top_k,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        sem_future = executor.submit(_semantic)
        kw_future = executor.submit(_keyword)
        semantic_results = sem_future.result()
        keyword_results = kw_future.result()

    if not semantic_results and not keyword_results:
        darwin.warning("No results from either semantic or keyword search")
        return []

    darwin.info(f"Semantic: {len(semantic_results)} candidates")
    darwin.info(f"Keyword: {len(keyword_results)} candidates")

    darwin.info("Fusing results via RRF")
    results = hybrid_search(
        semantic_results=semantic_results[:pool_size],
        keyword_results=keyword_results[:pool_size],
        top_k=top_k,
        rrf_k=rrf_k,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
    )
    darwin.success(f"Hybrid search complete: {len(results)} results")
    return results


def search_across_documents_hybrid(
    query: str,
    model_name: str,
    store: DarwinStore,
    filenames: Optional[List[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    pool_size: int = DEFAULT_POOL_SIZE,
    per_doc_top_k: Optional[int] = None,
) -> List[HybridSearchResult]:
    """Hybrid search across multiple documents (parallelized)."""
    if not filenames:
        darwin.warning(
            "No documents with both embeddings and BM25 index found"
        )
        darwin.info(
            "Run 'darwin embeddings' and 'darwin index-bm25' first"
        )
        return []

    def _semantic():
        return semantic_search_all(
            query=query,
            model_name=model_name,
            store=store,
            filenames=filenames,
            top_k=pool_size,
            per_doc_top_k=per_doc_top_k,
        )

    def _keyword():
        return keyword_search_all(
            query=query,
            filenames=filenames,
            store=store,
            top_k=pool_size,
            per_doc_top_k=per_doc_top_k,
        )

    darwin.info(
        f"Searching semantics across {len(filenames)} document(s)"
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        sem_future = executor.submit(_semantic)
        kw_future = executor.submit(_keyword)
        semantic_results = sem_future.result()
        keyword_results = kw_future.result()

    darwin.info(f"Semantic: {len(semantic_results)} candidates")
    darwin.info(f"Keyword: {len(keyword_results)} candidates")

    if not semantic_results and not keyword_results:
        darwin.warning("No results from either semantic or keyword search")
        return []

    darwin.info("Fusing results via RRF")
    results = hybrid_search(
        semantic_results=semantic_results[:pool_size],
        keyword_results=keyword_results[:pool_size],
        top_k=top_k,
        rrf_k=rrf_k,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
    )
    darwin.success(f"Hybrid search complete: {len(results)} results")
    return results
