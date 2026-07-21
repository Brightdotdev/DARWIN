"""
src/features/Retrival/main.py
------------------------------
Unified retrieval entry point.

Default method is HYBRID. All paths return List[RetrievalResult].

FIXES APPLIED (2026-07-19):
  1. _rerank(): preserves rrf_score, semantic_score, keyword_score 
     in the reranked RetrievalResult (previously lost via getattr defaults)
  2. _rerank(): uses getattr with proper defaults to prevent None 
     scores from being silently discarded
  3. handle_retrieval(): added per_doc_top_k logging for clarity
"""

from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path
from typing import List, Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


from src.core.Retrival.RetrievalArgs import RetrievalArgs

from src.core.Retrival.lib import (
    RetrievalMethod,
    DEFAULT_TOP_K,
    DEFAULT_POOL_SIZE,
    DEFAULT_RRF_K,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_MIN_SCORE,
    DEFAULT_K1,
    DEFAULT_B,
    DEFAULT_RERANK_WEIGHT,
    DEFAULT_RERANK_TOP_K,
)

from src.core.Retrival.RetrievalResult import (
    RetrievalResult,
    OverlapDirection,
    from_hybrid,
    from_semantic,
    from_keyword,
)

from src.core.Logger.DarwinLogger import darwin

from src.core.Retrival.Hybrid.main import handle_hybrid_search
from src.core.Retrival.Semantic.main import handle_semantic_search
from src.core.Retrival.Keyword.main import handle_keyword_search

from src.core.Retrival.lib import (
    SemanticSearchArgs,
    HybridSearchArgs,
    KeywordSearchArgs,
)


def _run_hybrid(args: RetrievalArgs, store) -> List[Any]:
    hybrid_args = HybridSearchArgs(
        query=args.query,
        files=args.files,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        semantic_weight=args.semantic_weight,
        keyword_weight=args.keyword_weight,
        pool_size=args.pool_size,
        k1=args.k1,
        b=args.b,
        per_doc_top_k=args.per_doc_top_k,  # <-- propagate
    )
    return handle_hybrid_search(hybrid_args, store=store)


def _run_semantic(args: RetrievalArgs, store) -> List[Any]:
    sem_args = SemanticSearchArgs(
        query=args.query,
        files=args.files,
        top_k=args.top_k,
        min_score=args.min_score,
        per_doc_top_k=args.per_doc_top_k,  # <-- propagate
    )
    return handle_semantic_search(sem_args, store=store)


def _run_keyword(args: RetrievalArgs, store) -> List[Any]:
    kw_args = KeywordSearchArgs(
        query=args.query,
        files=args.files,
        top_k=args.top_k,
        k1=args.k1,
        b=args.b,
        per_doc_top_k=args.per_doc_top_k,  # <-- propagate
    )
    return handle_keyword_search(kw_args, store=store)


def _rerank(
    query: str,
    candidates: List[RetrievalResult],
    args: RetrievalArgs,
) -> List[RetrievalResult]:
    if not args.rerank or not candidates:
        return candidates

    try:
        from src.core.Retrival.Reranking.Reranker import rerank_results

        darwin.info(
            f"Reranking {len(candidates)} results (weight={args.rerank_weight})"
        )
        reranked_raw = rerank_results(
            query=query,
            candidates=candidates,
            top_k=min(args.top_k, len(candidates)),
            original_score_weight=args.rerank_weight,
        )

        out: List[RetrievalResult] = []
        for r in reranked_raw:
            # FIX: Preserve original scores from the candidate, not just getattr defaults
            # Find the matching original candidate to preserve its scores
            orig = next(
                (c for c in candidates if getattr(c, 'chunk_index', None) == getattr(r, 'chunk_index', None) 
                 and c.source == getattr(r, 'source', None)),
                None
            )

            out.append(
                RetrievalResult(
                    result_type=orig.result_type if orig else candidates[0].result_type,
                    score=r.final_score,
                    source=r.source,
                    text=r.text,
                    clean=getattr(r, "clean", None),
                    page=r.page,
                    page_end=getattr(r, "page_end", None),
                    heading=getattr(r, "heading", None),
                    chunk_index=r.chunk_index,
                    overlap_direction=r.overlap_direction,
                    links=r.links,
                    rrf_score=getattr(orig, "rrf_score", None) if orig else None,
                    semantic_score=getattr(orig, "semantic_score", None) if orig else None,
                    keyword_score=getattr(orig, "keyword_score", None) if orig else None,
                    rerank_score=r.rerank_score,
                    rerank_original_score=getattr(r, "original_score", None),
                    matched_terms=set(r.matched_terms) if r.matched_terms else (set(orig.matched_terms) if orig and orig.matched_terms else None),
                )
            )
        darwin.success(f"Reranking complete: {len(out)} results")
        return out

    except ImportError:
        darwin.warning("Reranker not available, skipping")
        return candidates
    except Exception as e:
        darwin.error(f"Reranking failed: {type(e).__name__}: {e}")
        darwin.info("Returning original order")
        return candidates


def _display_results(results: List[RetrievalResult], query: str, method: str) -> None:
    """Display search results in a formatted table."""
    sep = "=" * 70
    darwin.info(f"\n{sep}")
    darwin.info(f"Results: '{query}'  [{method.upper()}]")
    darwin.success(
        f"Total: {len(results)} results from {len(set(r.source for r in results))} sources"
    )
    if results:
        darwin.info(f"Top score: {results[0].score:.4f}")
    darwin.info(f"{sep}\n")


def handle_retrieval(args: RetrievalArgs, store) -> List[RetrievalResult]:
    """
    Main retrieval entry point. Always returns List[RetrievalResult].
    """
    darwin.info("")
    darwin.info("=" * 70)
    darwin.info("Retrieval")
    darwin.info("=" * 70)
    darwin.info(f"Query  : '{args.query}'")
    darwin.info(f"Method : {args.method.value}")
    darwin.info(f"Top-K  : {args.top_k}")
    if args.per_doc_top_k is not None:
        darwin.info(f"Per-doc limit: {args.per_doc_top_k}")
    if args.rerank:
        darwin.info(f"Rerank : enabled (weight={args.rerank_weight})")

    start = time.perf_counter()

    raw: List[Any] = []
    if args.method == RetrievalMethod.HYBRID:
        darwin.info(f"Pool size: {args.pool_size}  |  rrf_k={args.rrf_k}")
        raw = _run_hybrid(args, store=store)
    elif args.method == RetrievalMethod.SEMANTIC:
        darwin.info(f"Min score: {args.min_score}")
        raw = _run_semantic(args, store=store)
    elif args.method == RetrievalMethod.KEYWORD:
        darwin.info(f"BM25: k1={args.k1}  b={args.b}")
        raw = _run_keyword(args, store=store)

    elapsed = time.perf_counter() - start
    darwin.info(f"Search: {len(raw)} results in {elapsed:.2f}s")

    results: List[RetrievalResult] = []
    for r in raw:
        if args.method == RetrievalMethod.HYBRID:
            results.append(from_hybrid(r))
        elif args.method == RetrievalMethod.SEMANTIC:
            results.append(from_semantic(r))
        elif args.method == RetrievalMethod.KEYWORD:
            results.append(from_keyword(r))

    if args.rerank and results:
        rerank_start = time.perf_counter()
        results = _rerank(args.query, results, args)
        darwin.info(
            f"Reranking: {time.perf_counter() - rerank_start:.2f}s"
        )

    if not results:
        darwin.warning(f"No results for '{args.query}'")
        darwin.info(
            "Try: different keywords | --method | --min-score | --pool-size"
        )
        return []

    total_elapsed = time.perf_counter() - start
    darwin.success(
        f"Total: {len(results)} results in {total_elapsed:.2f}s"
    )

    _display_results(results, args.query, args.method.value)

    return results


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified retrieval -- hybrid (default), semantic, or keyword",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default hybrid search
  %(prog)s "photosynthesis"

  # Explicit method
  %(prog)s "cell division" --method keyword
  %(prog)s "mitochondria"  --method semantic

  # With reranking
  %(prog)s "DNA replication" --rerank

  # Prevent large docs from dominating (limit to 3 per doc)
  %(prog)s "ATP synthesis" --per-doc-top-k 3
        """,
    )

    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--query", "-q", dest="query_flag")
    parser.add_argument(
        "--method",
        "-m",
        choices=["hybrid", "semantic", "keyword"],
        default="hybrid",
    )
    parser.add_argument("--file", "-f", nargs="+", default=None)
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--per-doc-top-k",
        type=int,
        default=None,
        help="Limit results per document (None=unlimited).",
    )
    parser.add_argument("--json", "-j", action="store_true")

    hybrid = parser.add_argument_group("Hybrid")
    hybrid.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    hybrid.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    hybrid.add_argument(
        "--semantic-weight", type=float, default=DEFAULT_SEMANTIC_WEIGHT
    )
    hybrid.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT)

    sem = parser.add_argument_group("Semantic")
    sem.add_argument("--min-score", "-s", type=float, default=DEFAULT_MIN_SCORE)

    kw = parser.add_argument_group("Keyword")
    kw.add_argument("--k1", type=float, default=DEFAULT_K1)
    kw.add_argument("--b", type=float, default=DEFAULT_B)

    rerank = parser.add_argument_group("Reranking")
    rerank.add_argument("--rerank", "-r", action="store_true")
    rerank.add_argument("--rerank-weight", type=float, default=DEFAULT_RERANK_WEIGHT)
    rerank.add_argument("--rerank-top-k", type=int, default=DEFAULT_RERANK_TOP_K)

    parser.add_argument("--store-dir", default=None)
    parser.add_argument(
        "--sse", action="store_true", help="Enable SSE logging (requires SSE server)"
    )

    return parser


def main() -> int:
    from src.core.Storage.DarwinStore import DarwinStore
    from src.lib.Utils import ensure_writable_directory

    parser = create_parser()
    cli = parser.parse_args()

    query = cli.query or cli.query_flag
    if not query:
        parser.error("Query required")

    from src.core.Logger.DarwinLogger import OutputMode

    darwin.configure(

        mode=OutputMode.BOTH if cli.sse else OutputMode.TERMINAL,
    )

    try:
        args = RetrievalArgs(
            query=query,
            method=RetrievalMethod(cli.method),
            files=cli.file or [],
            top_k=cli.top_k,
            pool_size=cli.pool_size,
            rrf_k=cli.rrf_k,
            semantic_weight=cli.semantic_weight,
            keyword_weight=cli.keyword_weight,
            min_score=cli.min_score,
            k1=cli.k1,
            b=cli.b,
            per_doc_top_k=cli.per_doc_top_k,  # <-- new
            rerank=cli.rerank,
            rerank_weight=cli.rerank_weight,
            rerank_top_k=cli.rerank_top_k,
            store_dir=cli.store_dir,
        )

        with darwin.session("retrieval"):
            if cli.store_dir and ensure_writable_directory(cli.store_dir):
                store = DarwinStore(cli.store_dir)
            else:
                store = DarwinStore()
            handle_retrieval(args, store=store)
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