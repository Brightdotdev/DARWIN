"""
src/core/HybridSearch/HybridSearchUtils.py
------------------------------------------
Utilities for hybrid search: RRF fusion, result merging, and formatting.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


# ============================================================================
# Constants
# ============================================================================

DEFAULT_TOP_K    = 5
DEFAULT_RRF_K    = 60       # magic constant from Cormack et al. 2009 RRF paper
DEFAULT_SEMANTIC_WEIGHT = 1.0
DEFAULT_KEYWORD_WEIGHT  = 1.0
DEFAULT_POOL_SIZE = 50


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
    links:         List[Any] = field(default_factory=list)  # List[ChunkLink]
    is_overlap:    bool = False
    matched_terms: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"HybridSearchResult(rrf={self.rrf_score:.3f}, "
            f"final={self.final_score:.3f}, source='{self.source}', page={self.page})"
        )


# ============================================================================
# Args
# ============================================================================

@dataclass
class HybridSearchArgs:
    query:            str
    file:             Optional[str]   = None
    project:          Optional[str]   = None
    top_k:            int             = DEFAULT_TOP_K
    rrf_k:            int             = DEFAULT_RRF_K
    semantic_weight:  float           = DEFAULT_SEMANTIC_WEIGHT
    keyword_weight:   float           = DEFAULT_KEYWORD_WEIGHT
    pool_size:        int             = DEFAULT_POOL_SIZE
    inspect:          bool            = False
    json_output:      bool            = False


# ============================================================================
# Result building and merging
# ============================================================================

def build_unified_result(
    result,
    semantic_score: Optional[float] = None,
    keyword_score:  Optional[float] = None,
) -> HybridSearchResult:
    """Create a HybridSearchResult from a semantic or keyword result."""
    return HybridSearchResult(
        text           = result.text,
        clean          = result.clean,
        rrf_score      = 0.0,
        final_score    = 0.0,
        semantic_score = semantic_score,
        keyword_score  = keyword_score,
        source         = result.source,
        page           = result.page,
        page_end       = getattr(result, "page_end", result.page),
        heading        = result.heading,
        chunk_index    = result.chunk_index,
        links          = getattr(result, "links", []),
        is_overlap     = getattr(result, "is_overlap", False),
        matched_terms  = list(getattr(result, "matched_terms", [])),
    )


def merge_results(
    semantic_results: List,
    keyword_results:  List,
) -> List[HybridSearchResult]:
    """
    Merge semantic and keyword results, deduplicating by (source, chunk_index).
    """
    seen: Dict[tuple, HybridSearchResult] = {}

    for r in semantic_results:
        k = (r.source, r.chunk_index)
        if k not in seen:
            seen[k] = build_unified_result(r, semantic_score=r.score)
        else:
            seen[k].semantic_score = r.score

    for r in keyword_results:
        k = (r.source, r.chunk_index)
        if k not in seen:
            seen[k] = build_unified_result(r, keyword_score=r.score)
        else:
            seen[k].keyword_score = r.score
            if hasattr(r, "matched_terms") and r.matched_terms:
                existing = set(seen[k].matched_terms)
                seen[k].matched_terms.extend(
                    t for t in r.matched_terms if t not in existing
                )

    return list(seen.values())


def build_ranked_lists(
    items: List[HybridSearchResult],
) -> List[List[HybridSearchResult]]:
    """
    Build separate ranked lists for semantic and keyword results,
    each sorted by its respective score descending.
    """
    semantic_list = sorted(
        [x for x in items if x.semantic_score is not None],
        key=lambda x: x.semantic_score,
        reverse=True,
    )
    keyword_list = sorted(
        [x for x in items if x.keyword_score is not None],
        key=lambda x: x.keyword_score,
        reverse=True,
    )
    return [semantic_list, keyword_list]


# ============================================================================
# Reciprocal Rank Fusion
# ============================================================================

def reciprocal_rank_fusion(
    result_lists: List[List[HybridSearchResult]],
    rrf_k:        int                  = DEFAULT_RRF_K,
    weights:      Optional[List[float]] = None,
) -> List[HybridSearchResult]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    score(d) = Σ  weight_i / (rrf_k + rank_i(d))

    O(n) rank lookup via pre-built dicts instead of list.index().
    rrf_k=60 is the value from the original paper (Cormack et al. 2009).
    """
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    if len(weights) != len(result_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match result_lists ({len(result_lists)})"
        )

    rank_maps = [
        {(item.source, item.chunk_index): rank
         for rank, item in enumerate(lst, start=1)}
        for lst in result_lists
    ]

    items_by_key: Dict[tuple, HybridSearchResult] = {}
    for lst in result_lists:
        for item in lst:
            items_by_key[(item.source, item.chunk_index)] = item

    for key, item in items_by_key.items():
        item.rrf_score = sum(
            weights[i] / (rrf_k + rank_map[key])
            for i, rank_map in enumerate(rank_maps)
            if key in rank_map
        )

    return sorted(items_by_key.values(), key=lambda x: x.rrf_score, reverse=True)


# ============================================================================
# Hybrid Scoring
# ============================================================================

def apply_hybrid_scoring(
    items: List[HybridSearchResult],
    alpha: float = 0.3,
    beta:  float = 0.2,
) -> List[HybridSearchResult]:
    """
    final = rrf_score + alpha * norm_semantic + beta * norm_keyword

    Scores are normalized to [0, 1] before weighting so neither retriever's
    raw score magnitude dominates the other.
    """
    if not items:
        return []

    sem_scores = [x.semantic_score for x in items if x.semantic_score is not None]
    kw_scores  = [x.keyword_score  for x in items if x.keyword_score  is not None]

    sem_max = max(sem_scores) if sem_scores else 1.0
    kw_max  = max(kw_scores)  if kw_scores  else 1.0

    for item in items:
        norm_sem = (item.semantic_score / sem_max) if item.semantic_score is not None and sem_max > 0 else 0.0
        norm_kw  = (item.keyword_score  / kw_max)  if item.keyword_score  is not None and kw_max  > 0 else 0.0
        item.final_score = item.rrf_score + alpha * norm_sem + beta * norm_kw

    items.sort(key=lambda x: x.final_score, reverse=True)
    return items


# ============================================================================
# Result formatting
# ============================================================================

def format_hybrid_results(
    results:     List[HybridSearchResult],
    json_output: bool = False,
) -> str:
    """Format hybrid search results for display or JSON export."""
    import json

    if json_output:
        return json.dumps([
            {
                "rrf_score":      round(r.rrf_score,      4),
                "final_score":    round(r.final_score,    4),
                "semantic_score": round(r.semantic_score, 4) if r.semantic_score is not None else None,
                "keyword_score":  round(r.keyword_score,  4) if r.keyword_score  is not None else None,
                "source":         r.source,
                "page":           r.page,
                "page_end":       r.page_end,
                "heading":        r.heading,
                "matched_terms":  r.matched_terms,
                "text":           r.text[:500] + "..." if len(r.text) > 500 else r.text,
            }
            for r in results
        ], indent=2)

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] Final={r.final_score:.3f} | RRF={r.rrf_score:.3f} | {r.source} | p{r.page}")
        if r.semantic_score is not None:
            lines.append(f"    └─ semantic: {r.semantic_score:.3f}")
        if r.keyword_score is not None:
            lines.append(f"    └─ keyword:  {r.keyword_score:.3f}")
        if r.matched_terms:
            lines.append(f"    └─ matched: {', '.join(r.matched_terms[:5])}")
        if r.heading:
            lines.append(f"    └─ {r.heading}")
        if r.links:
            lines.append(f"    └─ {len(r.links)} link(s)")
        preview = r.text[:200].replace("\n", " ").strip()
        lines.append(f"       {preview}...")
        lines.append("")
    return "\n".join(lines)


def to_llm_context(results: List[HybridSearchResult], max_chunks: int = 5) -> str:
    """Convert hybrid results to an LLM-ready context string."""
    parts = []
    for r in results[:max_chunks]:
        part = f"[Document: {r.source}, Page: {r.page}]"
        if r.heading:
            part += f"\nHeading: {r.heading}"
        part += f"\nContent: {r.text}"
        if r.links:
            part += "\nReferences:"
            for link in r.links:
                label = link.text if link.text else link.url
                part += f"\n  - {label}: {link.url}"
        part += "\n---"
        parts.append(part)
    return "\n".join(parts)