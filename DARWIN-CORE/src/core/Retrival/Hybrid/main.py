"""
src/core/Retrival/Hybrid/main.py
------------------------------------
Hybrid search: BM25 + Dense Vector (semantic) fusion – CLI entry point.
"""

from __future__ import annotations

import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

from src.core.Logger.DarwinLogger import darwin, LogType
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


def _normalize_scores(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Min-max normalize scores of a list of result dicts to [0,1]."""
    if not results:
        return results

    valid_results = [r for r in results if r.get("score") is not None]
    if not valid_results:
        return results

    scores = [r["score"] for r in valid_results]
    min_score = min(scores)
    max_score = max(scores)
    range_score = max_score - min_score

    if range_score < 1e-9:
        for r in results:
            if r.get("score") is not None:
                r["score"] = 0.5
        return results

    for r in results:
        raw = r.get("score")
        if raw is not None:
            r["score"] = (raw - min_score) / range_score

    return results


def _rrf_fusion(
    keyword_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    semantic_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> List[HybridSearchResult]:
    """
    Fuse keyword and semantic results using Reciprocal Rank Fusion (RRF).
    Stores individual scores for transparency.
    Final score is the RRF score.
    """
    combined: Dict[Tuple[str, int], Dict[str, Any]] = {}

    # Process keyword results
    for rank, r in enumerate(keyword_results, 1):
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
                "matched_terms": r.get("matched_terms", set()),
            }
        else:
            existing = combined[key].get("keyword_score")
            new_score = r.get("score")
            if new_score is not None and (existing is None or new_score > existing):
                combined[key]["keyword_score"] = new_score

    # Process semantic results
    for rank, r in enumerate(semantic_results, 1):
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
                "matched_terms": r.get("matched_terms", set()),
            }
        else:
            existing = combined[key].get("semantic_score")
            new_score = r.get("score")
            if new_score is not None and (existing is None or new_score > existing):
                combined[key]["semantic_score"] = new_score
                combined[key]["semantic_rank"] = rank

    # Compute RRF scores
    for data in combined.values():
        rrf = 0.0
        kw_rank = data.get("keyword_rank")
        sem_rank = data.get("semantic_rank")
        if kw_rank is not None:
            rrf += keyword_weight / (rrf_k + kw_rank)
        if sem_rank is not None:
            rrf += semantic_weight / (rrf_k + sem_rank)
        data["rrf_score"] = rrf

    sorted_items = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)

    results: List[HybridSearchResult] = []
    for data in sorted_items:
        results.append(
            HybridSearchResult(
                text=data["text"],
                clean=data["clean"],
                rrf_score=data["rrf_score"],
                final_score=data["rrf_score"],  # final_score = RRF score
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
    """
    start_time = time.perf_counter()

    darwin.log(f"[Hybrid] Query: '{args.query}'", LogType.INFO)
    darwin.log(f"[Hybrid] Top-K: {args.top_k}, Pool: {args.pool_size}", LogType.INFO)
    darwin.log(
        f"[Hybrid] Per‑doc limit: {args.per_doc_top_k if args.per_doc_top_k is not None else 'unlimited'}",
        LogType.INFO,
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
    keyword_raw = handle_keyword_search(kw_args, store=store) or []
    darwin.log(f"[Hybrid] Keyword: {len(keyword_raw)} results", LogType.INFO)

    # Semantic search
    sem_args = SemanticSearchArgs(
        query=args.query,
        files=args.files,
        top_k=args.pool_size,
        min_score=0.0,
        per_doc_top_k=args.per_doc_top_k,
    )
    semantic_raw = handle_semantic_search(sem_args, store=store) or []
    darwin.log(f"[Hybrid] Semantic: {len(semantic_raw)} results", LogType.INFO)

    if not keyword_raw and not semantic_raw:
        darwin.log("[Hybrid] No results from either search", LogType.WARNING)
        return []

    # Convert to dicts for fusion
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
            "score": r.score,
            "matched_terms": r.matched_terms,
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
            "score": r.score,
            "matched_terms": r.matched_terms,
        }
        semantic_dicts.append(d)

    # Normalise scores before fusion (to avoid scale dominance)
    if keyword_dicts:
        keyword_dicts = _normalize_scores(keyword_dicts)
    if semantic_dicts:
        semantic_dicts = _normalize_scores(semantic_dicts)

    # Filter out any remaining None scores before fusion
    keyword_dicts = [d for d in keyword_dicts if d.get("score") is not None]
    semantic_dicts = [d for d in semantic_dicts if d.get("score") is not None]

    if not keyword_dicts and not semantic_dicts:
        darwin.log("[Hybrid] No valid scores after normalization", LogType.WARNING)
        return []

    fused = _rrf_fusion(
        keyword_results=keyword_dicts,
        semantic_results=semantic_dicts,
        rrf_k=args.rrf_k,
        semantic_weight=args.semantic_weight,
        keyword_weight=args.keyword_weight,
    )

    darwin.log(f"[Hybrid] Fused: {len(fused)} unique results", LogType.INFO)

    # Normalize RRF scores to [0,1] so orchestrator thresholds work correctly
    if fused:
        scores = [r.final_score for r in fused]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s
        if range_s > 1e-9:
            for r in fused:
                norm = (r.final_score - min_s) / range_s
                r.final_score = norm
                r.rrf_score = norm
        else:
            for r in fused:
                r.final_score = 1.0
                r.rrf_score = 1.0

    top_results = fused[: args.top_k] if args.top_k > 0 else fused

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    darwin.log(f"[Hybrid] Done in {elapsed_ms}ms", LogType.SUCCESS)

    return top_results


# ---- CLI ----
def main() -> int:
    import sys
    import argparse
    from src.core.Storage.DarwinStore import DarwinStore
    from src.lib.Utils import ensure_writable_directory

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
    parser.add_argument(
        "--sse-port", type=int, default=8765, help="WebSocket server port"
    )

    args = parser.parse_args()

    query = args.query or args.query_flag
    if not query:
        parser.error("Query required (either positional or --query)")

    darwin.configure(sse_port=args.sse_port, show_banner=True)

    try:
        store = DarwinStore(args.store_dir) if args.store_dir else DarwinStore()
        search_args = HybridSearchArgs(
            query=query,
            files=args.file,
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
                darwin.log("")
                darwin.log("Top hybrid results:")
                for i, r in enumerate(results[:10], 1):
                    preview = r.text[:60].replace("\n", " ")
                    darwin.log(
                        f"{i:3}. {r.final_score:.4f} | {r.source}:{r.page} | {preview}..."
                    )
            else:
                darwin.log("No results found.", LogType.WARNING)
        return 0
    except KeyboardInterrupt:
        darwin.log("Goodbye!")
        return 130
    except Exception as e:
        import traceback

        darwin.log(f"Fatal error: {e}", LogType.ERROR)
        darwin.log("".join(traceback.format_exception(e)), LogType.ERROR)
        return 1
    finally:
        darwin.shutdown()


if __name__ == "__main__":
    main()
