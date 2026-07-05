"""
src/features/Retrival/main.py
------------------------------
Unified retrieval entry point.

Default method is HYBRID. All paths return List[RetrievalResult].
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

from src.core.Logger.DarwinLogger import darwin, LogType

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
    return handle_hybrid_search(hybrid_args, store=store) or []


def _run_semantic(args: RetrievalArgs, store) -> List[Any]:
    sem_args = SemanticSearchArgs(
        query=args.query,
        files=args.files,
        top_k=args.top_k,
        min_score=args.min_score,
        per_doc_top_k=args.per_doc_top_k,  # <-- propagate
    )
    return handle_semantic_search(sem_args, store=store) or []


def _run_keyword(args: RetrievalArgs, store) -> List[Any]:
    kw_args = KeywordSearchArgs(
        query=args.query,
        files=args.files,
        top_k=args.top_k,
        k1=args.k1,
        b=args.b,
        per_doc_top_k=args.per_doc_top_k,  # <-- propagate
    )
    return handle_keyword_search(kw_args, store=store) or []


def _rerank(
    query: str,
    candidates: List[RetrievalResult],
    args: RetrievalArgs,
) -> List[RetrievalResult]:
    if not args.rerank or not candidates:
        return candidates

    try:
        from src.core.Retrival.Reranking.Reranker import rerank_results

        darwin.log(
            f"Reranking {len(candidates)} results (weight={args.rerank_weight})",
            LogType.INFO,
        )
        reranked_raw = rerank_results(
            query=query,
            candidates=candidates,
            top_k=min(args.top_k, len(candidates)),
            original_score_weight=args.rerank_weight,
        )

        out: List[RetrievalResult] = []
        for r in reranked_raw:
            out.append(
                RetrievalResult(
                    result_type=candidates[0].result_type,
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
                    rrf_score=getattr(r, "rrf_score", None),
                    semantic_score=getattr(r, "semantic_score", None),
                    keyword_score=getattr(r, "keyword_score", None),
                    matched_terms=set(r.matched_terms) if r.matched_terms else None,
                    rerank_score=r.rerank_score,
                    rerank_original_score=getattr(r, "original_score", None),
                )
            )
        darwin.log(f"Reranking complete -- {len(out)} results", LogType.SUCCESS)
        return out

    except ImportError:
        darwin.log("Reranker not available -- skipping", LogType.WARNING)
        return candidates
    except Exception as e:
        darwin.log(f"Reranking failed: {e} -- returning original order", LogType.ERROR)
        return candidates


def _display_results(results: List[RetrievalResult], query: str, method: str) -> None:
    """Display search results in a formatted table."""
    darwin.log(f"\n{'=' * 70}", LogType.INFO)
    darwin.log(f"Results: '{query}'  [{method.upper()}]", LogType.INFO)
    darwin.log(f"\n{'=' * 70}", LogType.INFO)
    darwin.log(
        f"Total: {len(results)}  |  sources: {len(set(r.source for r in results))}",
        LogType.SUCCESS,
    )
    if results:
        darwin.log(f"Top score: {results[0].score:.4f}", LogType.INFO)
    darwin.log(f"{'=' * 70}\n", LogType.INFO)


def handle_retrieval(args: RetrievalArgs, store) -> List[RetrievalResult]:
    """
    Main retrieval entry point. Always returns List[RetrievalResult].
    """
    darwin.log("")
    darwin.log("=" * 70, LogType.INFO)
    darwin.log("Retrieval", LogType.INFO)
    darwin.log("=" * 70, LogType.INFO)
    darwin.log(f"Query  : '{args.query}'", LogType.INFO)
    darwin.log(f"Method : {args.method.value}", LogType.INFO)
    darwin.log(f"Top-K  : {args.top_k}", LogType.INFO)
    if args.per_doc_top_k is not None:
        darwin.log(f"Per‑doc limit: {args.per_doc_top_k}", LogType.INFO)
    if args.rerank:
        darwin.log(f"Rerank : enabled (weight={args.rerank_weight})", LogType.INFO)

    start = time.perf_counter()

    raw: List[Any] = []
    if args.method == RetrievalMethod.HYBRID:
        darwin.log(f"Pool size: {args.pool_size}  |  rrf_k={args.rrf_k}", LogType.INFO)
        raw = _run_hybrid(args, store=store)
    elif args.method == RetrievalMethod.SEMANTIC:
        darwin.log(f"Min score: {args.min_score}", LogType.INFO)
        raw = _run_semantic(args, store=store)
    elif args.method == RetrievalMethod.KEYWORD:
        darwin.log(f"BM25: k1={args.k1}  b={args.b}", LogType.INFO)
        raw = _run_keyword(args, store=store)

    elapsed = time.perf_counter() - start
    darwin.log(f"Search: {len(raw)} results in {elapsed:.2f}s", LogType.INFO)

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
        darwin.log(
            f"Reranking: {time.perf_counter() - rerank_start:.2f}s", LogType.INFO
        )

    if not results:
        darwin.log(f"\nNo results for '{args.query}'", LogType.WARNING)
        darwin.log(
            "Try: different keywords | --method | --min-score | --pool-size",
            LogType.INFO,
        )
        return []

    total_elapsed = time.perf_counter() - start
    darwin.log(
        f"Total: {len(results)} results in {total_elapsed:.2f}s", LogType.SUCCESS
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
    parser.add_argument("--sse-port", type=int, default=8765)
    parser.add_argument(
        "--no-ws", action="store_true", help="Disable WebSocket logging"
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

    darwin.configure(sse_port=cli.sse_port, show_banner=True)

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
            sse_port=cli.sse_port,
            no_ws=cli.no_ws,
        )

        with darwin.session("retrieval"):
            if cli.store_dir and ensure_writable_directory(cli.store_dir):
                store = DarwinStore(cli.store_dir)
            else:
                store = DarwinStore()
            handle_retrieval(args, store=store)
        return 0

    except KeyboardInterrupt:
        darwin.log("Goodbye!", LogType.INFO)
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
