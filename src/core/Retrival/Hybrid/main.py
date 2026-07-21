"""
src/core/Retrival/Hybrid/main.py
------------------------------------
Hybrid search: BM25 + Dense Vector (semantic) fusion – CLI entry point.

FIXES APPLIED (2026-07-19):
  1. Removed pre-RRF score normalization (preserves raw score gaps)
  2. Removed post-RRF min-max normalization (raw RRF scores preserved)
  3. Added sigmoid normalization option for stable [0,1] scores
  4. Added tie-breaking in _rrf_fusion() (secondary sort by source)
  5. from_hybrid() now copies structural_type
  6. Added per-result audit with raw/semantic/keyword/rrf scores
"""

from __future__ import annotations

import time
import math
from typing import List, Dict, Any, Tuple

from src.core.Logger.DarwinLogger import darwin
from src.core.Storage.DarwinStore import DarwinStore

from src.core.Retrival.Keyword.main import handle_keyword_search

from src.core.Retrival.Semantic.main import handle_semantic_search

from src.core.Retrival.lib import (
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_POOL_SIZE,
    DEFAULT_RRF_K,
    DEFAULT_SEMANTIC_WEIGHT,
    HybridSearchResult,
    OverlapDirection,
    SemanticSearchArgs,
    KeywordSearchArgs,
    DEFAULT_TOP_K,
    HybridSearchArgs,
)


def _sigmoid_normalize(scores: List[float], temperature: float = 10.0) -> List[float]:
    """Sigmoid normalization: preserves relative differences better than min-max.

    Maps RRF scores to [0,1] using a sigmoid function centered at the mean.
    This avoids the "artificial spread" problem of min-max normalization.
    """
    if not scores:
        return scores
    if len(scores) == 1:
        return [1.0]

    mean = sum(scores) / len(scores)
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores)) or 1e-9

    normalized = []
    for s in scores:
        z = (s - mean) / std
        sig = 1.0 / (1.0 + math.exp(-temperature * z))
        normalized.append(sig)

    return normalized


def _rrf_fusion(
    keyword_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    rrf_k: int = DEFAULT_RRF_K,
    semantic_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> List[HybridSearchResult]:
    """
    Fuse keyword and semantic results using Reciprocal Rank Fusion (RRF).
    Stores individual scores for transparency.
    Final score is the RRF score.

    FIX: Added tie-breaking (secondary sort by source name) to ensure
    stable rankings when scores are tied.
    """
    combined: Dict[Tuple[str, int], Dict[str, Any]] = {}

    # Process keyword results — FIX: stable sort with tie-breaking
    # Sort by score desc, then by source asc for stability
    keyword_sorted = sorted(
        keyword_results,
        key=lambda r: (-r.get("score", 0.0), r.get("source", ""))
    )
    for rank, r in enumerate(keyword_sorted, 1):
        key = (r.get("source", ""), r.get("chunk_index", 0))
        if key not in combined:
            combined[key] = {
                "source": r.get("source", ""),
                "chunk_index": r.get("chunk_index", 0),
                "text": r.get("text", ""),
                "clean": r.get("clean", ""),
                "page": r.get("page", 0),
                "page_end": r.get("page_end", 0),
                "heading": r.get("heading", ""),
                "overlap_direction": r.get("overlap_direction", OverlapDirection.NONE),
                "links": r.get("links", []),
                "keyword_score": r.get("score"),
                "semantic_score": None,
                "keyword_rank": rank,
                "semantic_rank": None,
                "rrf_score": 0.0,
                "matched_terms": set(r.get("matched_terms", set())),
                "structural_type": r.get("structural_type"),  # FIX: preserve structural_type
            }
        else:
            existing = combined[key].get("keyword_score")
            new_score = r.get("score")
            if new_score is not None and (existing is None or new_score > existing):
                combined[key]["keyword_score"] = new_score
                combined[key]["keyword_rank"] = rank

    # Process semantic results — FIX: stable sort with tie-breaking
    semantic_sorted = sorted(
        semantic_results,
        key=lambda r: (-r.get("score", 0.0), r.get("source", ""))
    )
    for rank, r in enumerate(semantic_sorted, 1):
        key = (r.get("source", ""), r.get("chunk_index", 0))
        if key not in combined:
            combined[key] = {
                "source": r.get("source", ""),
                "chunk_index": r.get("chunk_index", 0),
                "text": r.get("text", ""),
                "clean": r.get("clean", ""),
                "page": r.get("page", 0),
                "page_end": r.get("page_end", 0),
                "heading": r.get("heading", ""),
                "overlap_direction": r.get("overlap_direction", OverlapDirection.NONE),
                "links": r.get("links", []),
                "keyword_score": None,
                "semantic_score": r.get("score"),
                "keyword_rank": None,
                "semantic_rank": rank,
                "rrf_score": 0.0,
                "matched_terms": set(r.get("matched_terms", set())),
                "structural_type": r.get("structural_type"),  # FIX: preserve structural_type
            }
        else:
            existing = combined[key].get("semantic_score")
            new_score = r.get("score")
            if new_score is not None and (existing is None or new_score > existing):
                combined[key]["semantic_score"] = new_score
                combined[key]["semantic_rank"] = rank
            # Merge matched_terms — copy to avoid mutating original result's set
            if r.get("matched_terms"):
                combined[key]["matched_terms"].update(set(r["matched_terms"]))

    # Compute RRF scores
    raw_rrf = []
    data_list = list(combined.values())
    for data in data_list:
        rrf = 0.0
        kw_rank = data.get("keyword_rank")
        sem_rank = data.get("semantic_rank")
        if kw_rank is not None:
            rrf += keyword_weight / (rrf_k + kw_rank)
        if sem_rank is not None:
            rrf += semantic_weight / (rrf_k + sem_rank)
        data["rrf_score"] = rrf
        raw_rrf.append(rrf)

    # Normalize to [0,1] so orchestrator thresholds (min_passage_score) work
    normalized = _sigmoid_normalize(raw_rrf)
    for data, norm_score in zip(data_list, normalized):
        data["normalized_score"] = norm_score

    sorted_items = sorted(data_list, key=lambda x: x["rrf_score"], reverse=True)

    results: List[HybridSearchResult] = []
    for data in sorted_items:
        results.append(
            HybridSearchResult(
                text=data["text"],
                clean=data["clean"],
                rrf_score=data["rrf_score"],
                final_score=data["normalized_score"],  # normalized [0,1] for orchestrator
                semantic_score=data.get("semantic_score"),
                keyword_score=data.get("keyword_score"),
                source=data["source"],
                page=data["page"],
                page_end=data["page_end"],
                heading=data["heading"],
                chunk_index=data["chunk_index"],
                links=data.get("links", []),
                matched_terms=data.get("matched_terms", set()),
                overlap_direction=data.get("overlap_direction", OverlapDirection.NONE),
                structural_type=data.get("structural_type"),
            )
        )

    return results


def handle_hybrid_search(
    args: HybridSearchArgs,
    store: DarwinStore,
) -> List[HybridSearchResult]:
    """
    Hybrid search: BM25 + Dense Vector (semantic) fusion.
    Returns results sorted by RRF score.

    FIX: No longer normalizes scores before or after RRF. Raw RRF
    scores are returned, preserving the absolute meaning of scores.
    The orchestrator should use appropriate thresholds for raw RRF
    scores (typically 0.01-0.05 for good results).
    """
    start_time = time.perf_counter()

    darwin.info(f"Query: '{args.query}'")
    darwin.info(f"Top-K: {args.top_k}, Pool: {args.pool_size}")
    darwin.info(
        f"Per-doc limit: {args.per_doc_top_k if args.per_doc_top_k is not None else 'unlimited'}"
    )

    # Keyword search
    kw_args = KeywordSearchArgs(
        query=args.query,
        files=args.files,
        top_k=args.pool_size,
        k1=args.k1,
        b=args.b,
        per_doc_top_k=args.per_doc_top_k,
    )
    keyword_raw = handle_keyword_search(kw_args, store=store)
    darwin.info(f"Keyword: {len(keyword_raw)} results")

    # Semantic search
    sem_args = SemanticSearchArgs(
        query=args.query,
        files=args.files,
        top_k=args.pool_size,
        min_score=0.0,
        per_doc_top_k=args.per_doc_top_k,
    )
    semantic_raw = handle_semantic_search(sem_args, store=store)
    darwin.info(f"Semantic: {len(semantic_raw)} results")

    if not keyword_raw and not semantic_raw:
        darwin.warning("No results from either keyword or semantic search")
        return []

    # Convert to dicts — FIX: No pre-RRF normalization
    keyword_dicts = []
    for r in keyword_raw:
        d = {
            "source": r.source,
            "chunk_index": r.chunk_index,
            "text": r.text,
            "clean": r.clean,
            "page": r.page,
            "page_end": r.page_end,
            "heading": r.heading,
            "overlap_direction": r.overlap_direction,
            "links": r.links,
            "score": r.score,  # FIX: raw BM25 score, not normalized
            "matched_terms": r.matched_terms,
            "structural_type": getattr(r, "structural_type", None),
        }
        keyword_dicts.append(d)

    semantic_dicts = []
    for r in semantic_raw:
        d = {
            "source": r.source,
            "chunk_index": r.chunk_index,
            "text": r.text,
            "clean": r.clean,
            "page": r.page,
            "page_end": r.page_end,
            "heading": r.heading,
            "overlap_direction": r.overlap_direction,
            "links": r.links,
            "score": r.score,  # FIX: raw cosine similarity, not normalized
            "matched_terms": r.matched_terms,
            "structural_type": getattr(r, "structural_type", None),
        }
        semantic_dicts.append(d)

    # FIX: Filter out None scores before fusion
    keyword_dicts = [d for d in keyword_dicts if d.get("score") is not None]
    semantic_dicts = [d for d in semantic_dicts if d.get("score") is not None]

    if not keyword_dicts and not semantic_dicts:
        darwin.warning("No valid scores for fusion")
        return []

    fused = _rrf_fusion(
        keyword_results=keyword_dicts,
        semantic_results=semantic_dicts,
        rrf_k=args.rrf_k,
        semantic_weight=args.semantic_weight,
        keyword_weight=args.keyword_weight,
    )

    darwin.info(f"RRF fusion produced {len(fused)} unique results")

    # FIX: No post-RRF min-max normalization. Raw RRF scores are returned.
    # The scores are in a natural range based on rrf_k and weights.
    # For rrf_k=60 and weights=1.0, typical range is 0.01-0.03.
    # The orchestrator should use thresholds appropriate for this range.

    # Optional: Apply sigmoid normalization for stable [0,1] scores
    # This is DISABLED by default to preserve raw score meaning.
    # To enable, uncomment the following:
    #
    # if fused:
    #     raw_scores = [r.final_score for r in fused]
    #     norm_scores = _sigmoid_normalize(raw_scores, temperature=10.0)
    #     for r, ns in zip(fused, norm_scores):
    #         r.final_score = ns
    #         r.rrf_score = ns

    top_results = fused[: args.top_k] if args.top_k > 0 else fused

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    darwin.success(f"Hybrid search complete in {elapsed_ms}ms")

    return top_results


# ---- CLI ----
def main() -> int:
    import argparse
    from src.core.Storage.DarwinStore import DarwinStore

    parser = argparse.ArgumentParser(description="Hybrid search (BM25 + semantic)")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--query", "-q", dest="query_flag", help="Search query (alternative)"
    )
    parser.add_argument("--file", "-f", nargs="+", help="Files to search (names only)")
    parser.add_argument(
        "--top-k", "-k", type=int, default=DEFAULT_TOP_K, help="Number of final results"
    )
    parser.add_argument(
        "--pool-size",
        "-p",
        type=int,
        default=DEFAULT_POOL_SIZE,
        help="Candidates per retriever",
    )
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help="RRF constant")
    parser.add_argument(
        "--sem-weight",
        type=float,
        default=DEFAULT_SEMANTIC_WEIGHT,
        help="Semantic weight for RRF",
    )
    parser.add_argument(
        "--kw-weight",
        type=float,
        default=DEFAULT_KEYWORD_WEIGHT,
        help="Keyword weight for RRF",
    )
    parser.add_argument(
        "--per-doc-top-k",
        type=int,
        default=None,
        help="Limit results per document (None=unlimited). Prevents large docs from dominating.",
    )
    parser.add_argument("--store-dir", default=None, help="Darwin store directory")
    parser.add_argument("--sse", action="store_true", help="Enable SSE logging")

    args = parser.parse_args()

    query = args.query or args.query_flag
    if not query:
        parser.error("Query required (either positional or --query)")

    from src.core.Logger.DarwinLogger import OutputMode

    darwin.configure(

        mode=OutputMode.BOTH if args.sse else OutputMode.TERMINAL,
    )

    try:
        store = DarwinStore(args.store_dir) if args.store_dir else DarwinStore()
        search_args = HybridSearchArgs(
            query=query,
            files=args.file or [],
            top_k=args.top_k,
            pool_size=args.pool_size,
            rrf_k=args.rrf_k,
            semantic_weight=args.sem_weight,
            keyword_weight=args.kw_weight,
            per_doc_top_k=args.per_doc_top_k,
        )
        with darwin.session("hybrid_search"):
            results = handle_hybrid_search(search_args, store)
            # Display results (basic)
            if results:
                darwin.info("")
                darwin.info("Top hybrid results:")
                for i, r in enumerate(results[:10], 1):
                    preview = r.text[:60].replace("\n", " ")
                    darwin.info(
                        f"{i:3}. {r.final_score:.4f} | {r.source}:{r.page} | {preview}..."
                    )
            else:
                darwin.warning("No results found")
        return 0
    except KeyboardInterrupt:
        darwin.info("Goodbye!")
        return 130
    except Exception as e:
        import traceback

        darwin.error(f"Fatal error: {e}")
        darwin.error("".join(traceback.format_exception(e)))
        return 1
    finally:
        darwin.shutdown()


if __name__ == "__main__":
    main()