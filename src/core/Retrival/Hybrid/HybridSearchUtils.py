from typing import List, Dict, Optional, Tuple,Any
import copy

from core.Retrival.lib import DEFAULT_RRF_K, ChunkLink, HybridSearchResult,SemanticSearchResult,KeywordSearchResult, OverlapDirection
from core.Logger.DarwinLogger import darwin



def reciprocal_rank_fusion(
    result_lists: List[List[HybridSearchResult]],
    rrf_k:        int                  = DEFAULT_RRF_K,
    weights:      Optional[List[float]] = None,
) -> List[HybridSearchResult]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    score(d) = sum(weight_i / (rrf_k + rank_i(d)))

    O(n) rank lookup via pre‑built dicts instead of list.index().

    FIX: Uses copy.deepcopy() instead of copy.copy() to prevent
    shared mutable state between results.
    """
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    if len(weights) != len(result_lists):
        darwin.error(
            f"weights length ({len(weights)}) must match result_lists ({len(result_lists)})"
        )
        return []

    if any(w < 0 for w in weights):
        darwin.error("weights must be non-negative")
        return []

    rank_maps = [
        {(item.source, item.page, item.chunk_index): rank
         for rank, item in enumerate(lst, start=1)}
        for lst in result_lists
    ]

    # FIX: Use deepcopy to prevent shared mutable state
    items_by_key: Dict[Tuple[str, int, int], HybridSearchResult] = {}
    for lst in result_lists:
        for item in lst:
            key = (item.source, item.page, item.chunk_index)
            if key not in items_by_key:
                items_by_key[key] = copy.deepcopy(item)

    for key, item in items_by_key.items():
        item.rrf_score = sum(
            weights[i] / (rrf_k + rank_map[key])
            for i, rank_map in enumerate(rank_maps)
            if key in rank_map
        )

    return sorted(items_by_key.values(), key=lambda x: x.rrf_score, reverse=True)

def build_unified_result(
    result: Any,
    semantic_score: Optional[float] = None,
    keyword_score: Optional[float] = None,
) -> HybridSearchResult:
    """Create a HybridSearchResult from a semantic or keyword result."""
    return HybridSearchResult(
        text=result.text,
        clean=result.clean,
        rrf_score=0.0,
        final_score=0.0,
        semantic_score=semantic_score,
        keyword_score=keyword_score,
        source=result.source,
        page=result.page,
        page_end=getattr(result, "page_end", result.page),
        heading=result.heading,
        chunk_index=result.chunk_index,
        links=getattr(result, "links", []),
        matched_terms=set(getattr(result, "matched_terms", [])),
        overlap_direction=getattr(result, "overlap_direction", OverlapDirection.NONE),
        structural_type=getattr(result, "structural_type", None),
        audit=getattr(result, "audit", None),
    )



def merge_results(
    semantic_results: List[SemanticSearchResult],
    keyword_results:  List[KeywordSearchResult],
) -> List[HybridSearchResult]:
    """
    Merge semantic and keyword results, deduplicating by (source, page, chunk_index).
    """
    seen: Dict[Tuple[str, int, int], HybridSearchResult] = {}

    for r in semantic_results:
        k = (r.source, r.page, r.chunk_index)
        if k not in seen:
            seen[k] = build_unified_result(r, semantic_score=r.score)
        else:
            seen[k].semantic_score = r.score

    for r in keyword_results:
        k = (r.source, r.page, r.chunk_index)
        if k not in seen:
            seen[k] = build_unified_result(r, keyword_score=r.score)
        else:
            seen[k].keyword_score = r.score
            if hasattr(r, "matched_terms") and r.matched_terms:
                seen[k].matched_terms.update(r.matched_terms)

    return list(seen.values())

def build_ranked_lists(
    items: List[HybridSearchResult],
    min_score_threshold: float = 0.0,  # FIX: Added threshold parameter
) -> List[List[HybridSearchResult]]:
    """
    Build separate ranked lists for semantic and keyword results,
    each sorted by its respective score descending.

    FIX: Filters out results with score below min_score_threshold
    to prevent poor matches from influencing RRF ranking.
    """
    semantic_list = sorted(
        [x for x in items if x.semantic_score is not None and x.semantic_score >= min_score_threshold],
        key=lambda x: x.semantic_score,
        reverse=True,
    )
    keyword_list = sorted(
        [x for x in items if x.keyword_score is not None and x.keyword_score >= min_score_threshold],
        key=lambda x: x.keyword_score,
        reverse=True,
    )
    return [semantic_list, keyword_list]




# Legacy function – kept for backward compatibility
def apply_hybrid_scoring(
    rrf_scores: Dict[str, float],
    merged: Dict[str, dict] = None,
    alpha: float = 0.5,
    beta: float = 0.5
) -> List[HybridSearchResult]:
    """Apply hybrid scoring to RRF results. (Legacy/compat)"""
    results = []
    for chunk_id, rrf_score in rrf_scores.items():
        data = merged.get(chunk_id, {}) if merged else {}

        semantic_score = data.get("semantic_score")
        keyword_score = data.get("keyword_score")

        final_score = rrf_score
        if semantic_score is not None and keyword_score is not None:
            final_score = (alpha * semantic_score + beta * keyword_score) / (alpha + beta)
        elif semantic_score is not None:
            final_score = semantic_score
        elif keyword_score is not None:
            final_score = keyword_score

        links = data.get("links", [])
        chunk_links = []
        for l in links:
            if isinstance(l, dict):
                chunk_links.append(ChunkLink(url=l.get("url", ""), text=l.get("text", "")))
            else:
                chunk_links.append(l)

        results.append(HybridSearchResult(
            text=data.get("text", ""),
            clean=data.get("clean", ""),
            rrf_score=rrf_score,
            final_score=final_score,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            source=data.get("source", ""),
            page=data.get("page", 0),
            page_end=data.get("page_end", data.get("page", 0)),
            heading=data.get("heading", ""),
            chunk_index=int(chunk_id) if str(chunk_id).isdigit() else 0,
            links=chunk_links,
            matched_terms=data.get("matched_terms", set()),
            overlap_direction=data.get("overlap_direction", OverlapDirection.NONE),
            audit=data.get("audit"),
        ))

    results.sort(key=lambda x: x.final_score, reverse=True)
    return results