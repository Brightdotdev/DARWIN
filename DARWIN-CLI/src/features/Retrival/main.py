"""
src/features/Retrival/main.py
------------------------------
Unified retrieval entry point.

Default method is HYBRID. All paths return List[RetrievalResult].

    HYBRID   — RRF fusion of semantic + keyword search
    SEMANTIC — dense vector search only
    KEYWORD  — BM25 keyword search only

Usage:
    from src.features.Retrival.main import handle_retrieval
    from src.features.Retrival.RetrivalUtils import RetrievalArgs, RetrievalMethod

    results = handle_retrieval(RetrievalArgs(query="ATP synthesis"))

    from src.features.Retrival.RetrievalResult import ResultType
    for r in results:
        print(r.score, r.source, r.text[:80])
"""

from __future__ import annotations

import sys
import os
import warnings
import argparse
import asyncio
import time
from pathlib import Path
from typing import List, Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# CHANGED: suppress LiteLLM Bedrock warning spam
warnings.filterwarnings("ignore", module="litellm")
os.environ["LITELLM_SUPPRESS_DEBUG_INFO"] = "True"

from src.features.Retrival.RetrivalUtils import (
    RetrievalArgs,
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
    DEFAULT_ALPHA,
    DEFAULT_BETA,
)

from src.features.Retrival.RetrievalResult import (
    RetrievalResult,
    from_hybrid,
    from_semantic,
    from_keyword,
)

from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity

from src.features.Retrival.Hybrid.main import handle_hybrid_search
from src.features.Retrival.Semantic.main import handle_semantic_search
from src.features.Retrival.Keyword.main import handle_keyword_search

from src.features.Retrival.Hybrid.HybridSearchUtils import (
    HybridSearchArgs,
)
from src.features.Retrival.Semantic.SematicSearchUtils import (
    SemanticSearchArgs,
)
from src.features.Retrival.Keyword.KeywordUtilities import (
    KeywordSearchArgs,
)


# ============================================================================
# Search dispatchers
# ============================================================================

def _run_hybrid(args: RetrievalArgs,store) -> List[Any]:
    hybrid_args = HybridSearchArgs(
        query           = args.query,
        files           = args.files,
        top_k           = args.top_k,
        rrf_k           = args.rrf_k,
        semantic_weight = args.semantic_weight,
        keyword_weight  = args.keyword_weight,
        pool_size       = args.pool_size,
        inspect         = args.inspect,
        json_output     = args.json_output,
        alpha           = args.alpha,
        beta            = args.beta,
    )
    return handle_hybrid_search(hybrid_args,store=store) or []


def _run_semantic(args: RetrievalArgs,store) -> List[Any]:
    sem_args = SemanticSearchArgs(
        query       = args.query,
        files       = args.files,
        top_k       = args.top_k,
        min_score   = args.min_score,
        inspect     = args.inspect,
        json_output = args.json_output,
    )
    return handle_semantic_search(sem_args,store=store) or []


def _run_keyword(args: RetrievalArgs,store) -> List[Any]:
    kw_args = KeywordSearchArgs(
        query       = args.query,
        files       = args.files,
        top_k       = args.top_k,
        k1          = args.k1,
        b           = args.b,
        inspect     = args.inspect,
        json_output = args.json_output,
    )
    return handle_keyword_search(kw_args,store=store) or []


# ============================================================================
# Reranking
# ============================================================================

def _rerank(
    query:      str,
    candidates: List[RetrievalResult],
    args:       RetrievalArgs,
) -> List[RetrievalResult]:
    if not args.rerank or not candidates:
        return candidates

    try:
        from src.features.Retrival.Reranking.Reranker import rerank_results

        darwin.log(f"Reranking {len(candidates)} results (weight={args.rerank_weight})", LogType.INFO)
        reranked_raw = rerank_results(
            query                 = query,
            candidates            = candidates,
            top_k                 = min(args.top_k, len(candidates)),
            original_score_weight = args.rerank_weight,
        )
        out: List[RetrievalResult] = []
        for r in reranked_raw:
            rr = r if isinstance(r, RetrievalResult) else candidates[0]
            rr.rerank_score          = getattr(r, "rerank_score", None)
            rr.rerank_original_score = getattr(r, "original_score", None)
            if rr.rerank_score is not None:
                rr.score = getattr(r, "final_score", rr.score)
            out.append(rr)
        darwin.log(f"Reranking complete — {len(out)} results", LogType.SUCCESS)
        return out

    except ImportError:
        darwin.log("Reranker not available — skipping", LogType.WARNING)
        return candidates
    except Exception as e:
        darwin.log(f"Reranking failed: {e} — returning original order", LogType.ERROR)
        return candidates


# ============================================================================
# Display
# ============================================================================

def _display_results(results: List[RetrievalResult], args: RetrievalArgs) -> None:
    if args.json_output:
        import json
        out = []
        for r in results:
            d = {
                "result_type":   r.result_type.value,
                "score":         round(r.score, 4),
                "source":        r.source,
                "text":          r.text,
                "page":          r.page,
                "heading":       r.heading,
                "chunk_index":   r.chunk_index,
                "rrf_score":     round(r.rrf_score, 4)     if r.rrf_score     else None,
                "semantic_score":round(r.semantic_score, 4) if r.semantic_score else None,
                "keyword_score": round(r.keyword_score, 4)  if r.keyword_score  else None,
                "matched_terms": r.matched_terms,
                "rerank_score":  round(r.rerank_score, 4)  if r.rerank_score  else None,
            }
            out.append(d)
        print(json.dumps(out, indent=2))
        return

    darwin.log(f"\n{'='*70}", LogType.INFO)
    darwin.log(f"Results: '{args.query}'  [{args.method.value.upper()}]", LogType.INFO)
    darwin.log(f"{'='*70}", LogType.INFO)
    darwin.log(f"  #    Score       Source                    Page  Preview", LogType.INFO)
    darwin.log(f"  ───  ──────────  ────────────────────────  ────  ───────", LogType.INFO)

    for i, r in enumerate(results[:args.top_k], 1):
        preview = (r.text or "")[:60].replace("\n", " ").strip()
        darwin.log(
            f"  {i:<4} {r.score:.3f}      {r.source[:25]:<25}  {r.page:<4}  {preview}...",
            LogType.INFO,
        )
        if r.was_reranked:
            darwin.log(
                f"       rerank={r.rerank_score:.3f}  orig={r.rerank_original_score}",
                LogType.INFO,
            )
        elif r.is_hybrid:
            parts = []
            if r.semantic_score: parts.append(f"sem={r.semantic_score:.3f}")
            if r.keyword_score:  parts.append(f"kw={r.keyword_score:.3f}")
            if parts:
                darwin.log(f"       {' | '.join(parts)}", LogType.INFO)

    darwin.log(f"\n{'='*70}", LogType.INFO)
    darwin.log(f"Total: {len(results)}  |  sources: {len(set(r.source for r in results))}", LogType.SUCCESS)
    if results:
        darwin.log(f"Top score: {results[0].score:.4f}", LogType.INFO)
    darwin.log(f"{'='*70}\n", LogType.INFO)


def _print_inspection(results: List[RetrievalResult], args: RetrievalArgs) -> None:
    darwin.log(f"\n{'='*60}", LogType.INFO)
    darwin.log(f"Inspection — '{args.query}'", LogType.INFO)
    darwin.log(f"{'='*60}", LogType.INFO)

    for i, r in enumerate(results[:5], 1):
        darwin.log(f"\n[{i}]  type={r.result_type.value}  score={r.score:.4f}  source={r.source}", LogType.INFO)
        darwin.log(f"     page      : {r.page}", LogType.INFO)
        darwin.log(f"     heading   : {r.heading}", LogType.INFO)
        darwin.log(f"     chunk_idx : {r.chunk_index}", LogType.INFO)
        if r.was_reranked:
            darwin.log(f"     rerank    : {r.rerank_score:.4f}  orig={r.rerank_original_score}", LogType.INFO)
        if r.rrf_score is not None:
            darwin.log(f"     rrf       : {r.rrf_score:.4f}", LogType.INFO)
        if r.semantic_score is not None:
            darwin.log(f"     semantic  : {r.semantic_score:.4f}", LogType.INFO)
        if r.keyword_score is not None:
            darwin.log(f"     keyword   : {r.keyword_score:.4f}", LogType.INFO)
        if r.matched_terms:
            darwin.log(f"     matched   : {', '.join(list(r.matched_terms)[:10])}", LogType.INFO)
        preview = (r.text or "")[:400].replace("\n", " ").strip()
        darwin.log(f"     preview   : {preview}{'...' if len(r.text or '') > 400 else ''}", LogType.INFO)
        darwin.log(f"     {'─'*40}", LogType.INFO)

    darwin.log(f"{'='*60}", LogType.INFO)


# ============================================================================
# Core handler
# ============================================================================

def handle_retrieval(args: RetrievalArgs,store) -> List[RetrievalResult]:
    """
    Main retrieval entry point. Always returns List[RetrievalResult].

    Default method is HYBRID. Switch via args.method:
        RetrievalMethod.HYBRID   — RRF fusion (default)
        RetrievalMethod.SEMANTIC — dense vector only
        RetrievalMethod.KEYWORD  — BM25 only
    """
    darwin.log("")
    darwin.log("=" * 70, LogType.INFO)
    darwin.log("Retrieval", LogType.INFO)
    darwin.log("=" * 70, LogType.INFO)
    darwin.log(f"Query  : '{args.query}'", LogType.INFO)
    darwin.log(f"Method : {args.method.value}", LogType.INFO)
    darwin.log(f"Top-K  : {args.top_k}", LogType.INFO)
    if args.rerank:
        darwin.log(f"Rerank : enabled (weight={args.rerank_weight})", LogType.INFO)

    start = time.perf_counter()

    # ── Execute search ──────────────────────────────────────────────────────
    raw: List[Any] = []
    if args.method == RetrievalMethod.HYBRID:
        darwin.log(f"Pool size: {args.pool_size}  |  rrf_k={args.rrf_k}", LogType.INFO)
        raw = _run_hybrid(args,store=store)
    elif args.method == RetrievalMethod.SEMANTIC:
        darwin.log(f"Min score: {args.min_score}", LogType.INFO)
        raw = _run_semantic(args,store=store)
    elif args.method == RetrievalMethod.KEYWORD:
        darwin.log(f"BM25: k1={args.k1}  b={args.b}", LogType.INFO)
        raw = _run_keyword(args,store=store)

    elapsed = time.perf_counter() - start
    darwin.log(f"Search: {len(raw)} results in {elapsed:.2f}s", LogType.INFO)

    # ── Convert to RetrievalResult ──────────────────────────────────────────
    results: List[RetrievalResult] = []
    for r in raw:
        if args.method == RetrievalMethod.HYBRID:
            results.append(from_hybrid(r))
        elif args.method == RetrievalMethod.SEMANTIC:
            results.append(from_semantic(r))
        elif args.method == RetrievalMethod.KEYWORD:
            results.append(from_keyword(r))

    # ── Reranking ───────────────────────────────────────────────────────────
    if args.rerank and results:
        rerank_start = time.perf_counter()
        results = _rerank(args.query, results, args)
        darwin.log(f"Reranking: {time.perf_counter() - rerank_start:.2f}s", LogType.INFO)

    if not results:
        darwin.log(f"\nNo results for '{args.query}'", LogType.WARNING)
        darwin.log("Try: different keywords | --method | --min-score | --pool-size", LogType.INFO)
        return []

    total_elapsed = time.perf_counter() - start
    darwin.log(f"Total: {len(results)} results in {total_elapsed:.2f}s", LogType.SUCCESS)

    _display_results(results, args)
    if args.inspect:
        _print_inspection(results, args)

    return results


# ============================================================================
# CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified retrieval — hybrid (default), semantic, or keyword",
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

  # Full example
  %(prog)s "ATP synthesis" --method hybrid --file bio.pdf --top-k 10 --inspect
        """
    )

    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--query", "-q", dest="query_flag")
    parser.add_argument(
        "--method", "-m",
        choices=["hybrid", "semantic", "keyword"],
        default="hybrid",
    )
    parser.add_argument("--file", "-f", nargs="+", default=None)
    parser.add_argument("--top-k",  "-k", type=int,  default=DEFAULT_TOP_K)
    parser.add_argument("--inspect","-i", action="store_true")
    parser.add_argument("--json",   "-j", action="store_true")

    hybrid = parser.add_argument_group("Hybrid")
    hybrid.add_argument("--pool-size",       type=int,   default=DEFAULT_POOL_SIZE)
    hybrid.add_argument("--rrf-k",           type=int,   default=DEFAULT_RRF_K)
    hybrid.add_argument("--semantic-weight", type=float, default=DEFAULT_SEMANTIC_WEIGHT)
    hybrid.add_argument("--keyword-weight",  type=float, default=DEFAULT_KEYWORD_WEIGHT)
    hybrid.add_argument("--alpha",           type=float, default=DEFAULT_ALPHA)
    hybrid.add_argument("--beta",            type=float, default=DEFAULT_BETA)

    sem = parser.add_argument_group("Semantic")
    sem.add_argument("--min-score", "-s", type=float, default=DEFAULT_MIN_SCORE)

    kw = parser.add_argument_group("Keyword")
    kw.add_argument("--k1", type=float, default=DEFAULT_K1)
    kw.add_argument("--b",  type=float, default=DEFAULT_B)

    rerank = parser.add_argument_group("Reranking")
    rerank.add_argument("--rerank",       "-r", action="store_true")
    rerank.add_argument("--rerank-weight",      type=float, default=DEFAULT_RERANK_WEIGHT)
    rerank.add_argument("--rerank-top-k",       type=int,   default=DEFAULT_RERANK_TOP_K)

    parser.add_argument("--verbosity", choices=["normal", "quiet"], default="normal", help="Log verbosity")
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--no-ws",   action="store_true")

    return parser


async def main() -> int:
    from src.core.Storage.DarwinStore import DarwinStore
    from src.lib.Utils import ensure_writable_directory

    parser = create_parser()
    cli    = parser.parse_args()

    query = cli.query or cli.query_flag
    if not query:
        parser.error("Query required")

    darwin.configure(
        enable_websocket=not cli.no_ws,
        ws_port=cli.ws_port,
        enable_history=True,
        verbosity_default=Verbosity.QUIET if cli.verbosity == "quiet" else Verbosity.NORMAL,
        show_banner=True,
    )

    try:
        args = RetrievalArgs(
            query           = query,
            method          = RetrievalMethod(cli.method),
            files           = cli.file or [],
            top_k           = cli.top_k,
            inspect         = cli.inspect,
            json_output     = cli.json,
            pool_size       = cli.pool_size,
            rrf_k           = cli.rrf_k,
            semantic_weight = cli.semantic_weight,
            keyword_weight  = cli.keyword_weight,
            alpha           = cli.alpha,
            beta            = cli.beta,
            min_score       = cli.min_score,
            k1              = cli.k1,
            b               = cli.b,
            rerank          = cli.rerank,
            rerank_weight   = cli.rerank_weight,
            rerank_top_k    = cli.rerank_top_k,
            ws_port         = cli.ws_port,
            no_ws           = cli.no_ws,
        )

        with darwin.session("retrieval"):
            if args.store_dir and ensure_writable_directory(args.store_dir):
                store = DarwinStore(args.store_dir)
            else:
                store = DarwinStore()
            handle_retrieval(args,store=store)
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
    asyncio.run(main())