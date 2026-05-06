import sys
import argparse
import asyncio
import time
from pathlib import Path
from typing import List, Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

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

from src.features.Storage.DarwinStore import DarwinStore
from src.lib.Utils import scan_folder
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket


# ============================================================================
# Import search functions
# ============================================================================

from src.features.Retrival.Hybrid.main import handle_hybrid_search
from src.features.Retrival.Semantic.main import handle_semantic_search
from src.features.Retrival.Keyword.main import handle_keyword_search


from src.features.Retrival.Hybrid.HybridSearchUtils import (
    HybridSearchArgs,
    format_hybrid_results,
    OverlapDirection,
)

from src.features.Retrival.Semantic.SematicSearch import (
    search_single_document as semantic_search_single,
    search_across_documents as semantic_search_across,
)
from src.features.Retrival.Semantic.SematicSearchUtils import (
    format_results as format_semantic_results,
    SemanticSearchArgs
)

from src.features.Retrival.Keyword.KeywordSearch import (
    search_across_documents_keyword,
)
from src.features.Retrival.Keyword.KeywordUtilities import (
    format_keyword_results,KeywordSearchArgs
)

# ============================================================================
# Reranking wrapper
# ============================================================================

def _rerank_results(
    query: str,
    candidates: List[Any],
    args: RetrievalArgs,
) -> List[Any]:
    """Apply cross-encoder reranking if requested."""
    if not args.rerank or not candidates:
        return candidates
    
    try:
        from src.features.Retrival.Reranking.Reranker import rerank_results
        
        darwin.log(f"\n{'='*50}", LogType.INFO)
        darwin.log("Applying cross-encoder reranking...", LogType.INFO)
        darwin.log(f"Rerank weight: {args.rerank_weight} | Top-K: {args.rerank_top_k}", LogType.INFO)
        
        reranked = rerank_results(
            query=query,
            candidates=candidates,
            top_k=min(args.top_k, len(candidates)),
            original_score_weight=args.rerank_weight,
        )
        
        darwin.log(f"Reranking complete — {len(reranked)} results", LogType.SUCCESS)
        return reranked
        
    except ImportError:
        darwin.log("Reranker module not available. Skipping reranking.", LogType.WARNING)
        return candidates
    except Exception as e:
        darwin.log(f"Reranking failed: {e}. Returning original results.", LogType.ERROR)
        return candidates


# ============================================================================
# Search dispatcher
# ============================================================================

def _execute_search(
    args: RetrievalArgs,
) -> List[Any]:
    """Execute the appropriate search based on method."""
    
    if args.method == RetrievalMethod.HYBRID:

        hybridSearchArgs =  HybridSearchArgs(
            query=args.query,
            files=args.files,
            top_k=args.top_k,
            rrf_k=args.rrf_k,
            semantic_weight=args.semantic_weight,
            keyword_weight=args.keyword_weight,
            pool_size=args.pool_size,
            inspect=args.inspect,
            json_output=args.json_output,
            alpha=args.alpha,
            beta=args.beta,)
        
        handle_hybrid_search(hybridSearchArgs)
        
    elif args.method == RetrievalMethod.SEMANTIC:
        darwin.log(f"Method: Semantic (dense retrieval)", LogType.INFO)
        darwin.log(f"Min score threshold: {args.min_score}", LogType.INFO)
       

        semantic_search_args = SemanticSearchArgs(
            query=args.query,
            files=args.files,
            top_k=args.top_k,
            min_score=args.min_score,
            inspect=args.inspect,
            json_output=args.json_output,
        )
        return handle_semantic_search(semantic_search_args)
    
    elif args.method == RetrievalMethod.KEYWORD:
        darwin.log(f"Method: Keyword (BM25)", LogType.INFO)
        darwin.log(f"BM25 parameters: k1={args.k1}, b={args.b}", LogType.INFO)
        search_args = KeywordSearchArgs(
            query=args.query,
            files=args.files,
            top_k=args.top_k,
            k1=args.k1,
            b=args.b,
            inspect=args.inspect,
            json_output=args.json_output,
        )
        return handle_keyword_search(search_args)
    
    return []


# ============================================================================
# Results display
# ============================================================================

def _display_results(results: List[Any], args: RetrievalArgs) -> None:
    """Display search results with appropriate formatting."""
    
    if args.json_output:
        # Use appropriate formatter based on result type
        if args.rerank:
            # Reranked results have their own format
            import json
            output = []
            for r in results:
                output.append({
                    "final_score": r.final_score,
                    "rerank_score": r.rerank_score,
                    "original_score": r.original_score,
                    "text": r.text,
                    "source": r.source,
                    "page": r.page,
                    "heading": r.heading,
                })
            print(json.dumps(output, indent=2))
        elif args.method == RetrievalMethod.HYBRID:
            print(format_hybrid_results(results, json_output=True))
        elif args.method == RetrievalMethod.SEMANTIC:
            print(format_semantic_results(results, json_output=True))
        elif args.method == RetrievalMethod.KEYWORD:
            print(format_keyword_results(results, json_output=True))
        return
    
    # Console display
    darwin.log(f"\n{'='*70}", LogType.INFO)
    darwin.log(f"Search Results: '{args.query}'", LogType.INFO)
    darwin.log(f"{'='*70}\n", LogType.INFO)
    
    darwin.log(f"  #    Score       Source                  Page  Preview", LogType.INFO)
    darwin.log(f"  ───  ──────────  ──────────────────────  ────  ───────", LogType.INFO)
    
    for i, r in enumerate(results[:args.top_k], 1):
        # Extract score based on result type
        if hasattr(r, 'final_score'):
            score = r.final_score
        elif hasattr(r, 'score'):
            score = r.score
        else:
            score = 0.0
        
        preview = r.text[:60].replace('\n', ' ').strip()
        darwin.log(
            f"  {i:<4} {score:.3f}      {r.source[:25]:<25}  {r.page:<4}  {preview}...",
            LogType.INFO
        )
        
        # Show extra info for hybrid/reranked results
        if args.rerank and hasattr(r, 'rerank_score'):
            darwin.log(
                f"       {'':>10}  Rerank: {r.rerank_score:.3f} | Orig: {r.original_score}",
                LogType.INFO
            )
        elif args.method == RetrievalMethod.HYBRID and hasattr(r, 'rrf_score'):
            details = []
            if r.semantic_score:
                details.append(f"Sem: {r.semantic_score:.3f}")
            if r.keyword_score:
                details.append(f"KW: {r.keyword_score:.3f}")
            if details:
                darwin.log(f"       {'':>10}  {' | '.join(details)}", LogType.INFO)
    
    # Stats
    darwin.log(f"\n{'='*70}", LogType.INFO)
    darwin.log(f"Found: {len(results)} results", LogType.SUCCESS)
    darwin.log(f"Sources: {len(set(r.source for r in results))} files", LogType.INFO)
    if results:
        top_score = getattr(results[0], 'final_score', getattr(results[0], 'score', 0))
        darwin.log(f"Top score: {top_score:.4f}", LogType.INFO)
    darwin.log(f"{'='*70}\n", LogType.INFO)


def _print_inspection(results: List[Any], args: RetrievalArgs) -> None:
    """Print detailed inspection of results."""
    darwin.log(f"\n{'='*60}", LogType.INFO)
    darwin.log(f"Search Inspection — '{args.query}'", LogType.INFO)
    darwin.log(f"{'='*60}", LogType.INFO)
    
    darwin.log(f"Total results: {len(results)}", LogType.INFO)
    if results:
        top_score = getattr(results[0], 'final_score', getattr(results[0], 'score', 0))
        darwin.log(f"Top score: {top_score:.4f}", LogType.INFO)
    
    darwin.log(f"\n--- Top Results ---", LogType.INFO)
    
    for i, r in enumerate(results[:min(5, len(results))], 1):
        darwin.log(f"\n[{i}] Rank {i}", LogType.INFO)
        
        # Show all available scores
        if hasattr(r, 'final_score'):
            darwin.log(f"    Final score: {r.final_score:.4f}", LogType.INFO)
        if hasattr(r, 'rerank_score'):
            darwin.log(f"    Rerank score: {r.rerank_score:.4f}", LogType.INFO)
        if hasattr(r, 'original_score') and r.original_score:
            darwin.log(f"    Original score: {r.original_score}", LogType.INFO)
        if hasattr(r, 'rrf_score'):
            darwin.log(f"    RRF score: {r.rrf_score:.4f}", LogType.INFO)
        if hasattr(r, 'semantic_score') and r.semantic_score:
            darwin.log(f"    Semantic score: {r.semantic_score:.4f}", LogType.INFO)
        if hasattr(r, 'keyword_score') and r.keyword_score:
            darwin.log(f"    Keyword score: {r.keyword_score:.4f}", LogType.INFO)
        if hasattr(r, 'score'):
            darwin.log(f"    Score: {r.score:.4f}", LogType.INFO)
        
        darwin.log(f"    Source: {r.source}", LogType.INFO)
        darwin.log(f"    Page: {r.page}", LogType.INFO)
        darwin.log(f"    Heading: {getattr(r, 'heading', '(none)')}", LogType.INFO)
        
        if hasattr(r, 'matched_terms') and r.matched_terms:
            darwin.log(f"    Matched: {', '.join(list(r.matched_terms)[:10])}", LogType.INFO)
        
        if hasattr(r, 'overlap_direction') and r.overlap_direction != OverlapDirection.NONE:
            darwin.log(f"    Overlap: {r.overlap_direction.value}", LogType.WARNING)
        
        preview = r.text[:500].replace("\n", " ").strip()
        if len(r.text) > 500:
            preview += "..."
        darwin.log(f"    Preview: {preview}", LogType.INFO)
        darwin.log(f"    {'-'*40}", LogType.INFO)
    
    darwin.log(f"{'='*60}", LogType.INFO)


# ============================================================================
# Core handler
# ============================================================================

def handle_retrieval(args: RetrievalArgs) -> None:
    """Main handler for unified retrieval."""
    
    # Log search configuration
    darwin.log("")
    darwin.log("=" * 70, LogType.INFO)
    darwin.log("Unified Retrieval Search", LogType.INFO)
    darwin.log("=" * 70, LogType.INFO)
    darwin.log(f"Query: '{args.query}'", LogType.INFO)
    darwin.log(f"Method: {args.method.value}", LogType.INFO)
    darwin.log(f"Top-K: {args.top_k}", LogType.INFO)
    
    if args.rerank:
        darwin.log(f"Reranking: enabled (weight={args.rerank_weight})", LogType.INFO)
    else:
        darwin.log(f"Reranking: disabled", LogType.INFO)
    
    
    start_time = time.perf_counter()
    
    results = _execute_search(args)
    
    elapsed = time.perf_counter() - start_time
    darwin.log(f"Initial retrieval: {len(results)} results in {elapsed:.2f}s", LogType.SUCCESS)
    
    # Apply reranking if requested
    if args.rerank and results:
        rerank_start = time.perf_counter()
        results = _rerank_results(args.query, results, args)
        rerank_elapsed = time.perf_counter() - rerank_start
        darwin.log(f"Reranking completed in {rerank_elapsed:.2f}s", LogType.INFO)
    

    # Display results
    if not results:
        darwin.log(f"\nNo results found for '{args.query}'", LogType.WARNING)
        darwin.log("\nTry:", LogType.INFO)
        darwin.log("  • Using different keywords", LogType.INFO)
        darwin.log("  • Changing search method (--method)", LogType.INFO)
        darwin.log("  • Lowering thresholds (--min-score for semantic)", LogType.INFO)
        darwin.log("  • Increasing pool size with --pool-size", LogType.INFO)
        return
    
    _display_results(results, args)
    
    if args.inspect:
        _print_inspection(results, args)
    return results


# ============================================================================
# CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for unified retrieval CLI."""
    parser = argparse.ArgumentParser(
        description="Unified retrieval search — hybrid, semantic, or keyword",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default hybrid search
  %(prog)s "photosynthesis"
  
  # Explicit method
  %(prog)s "cell division" --method keyword
  %(prog)s "mitochondria" --method semantic
  
  # With reranking
  %(prog)s "DNA replication" --rerank
  
  # Full example
  %(prog)s "ATP synthesis" --method hybrid --file bio.pdf --rerank --top-k 10 --inspect
        """
    )
    
    # Core arguments
    parser.add_argument("query", nargs="?", default=None, help="Search query string")
    parser.add_argument("--query", "-q", dest="query_flag", help="Search query (alternative)")
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["hybrid", "semantic", "keyword"],
        default="hybrid",
        help="Retrieval method (default: hybrid)"
    )
    
    # Scope
    parser.add_argument("--file", "-f", nargs="+", default=None, help="One or more files to index")
    
    # Output
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K,
                       help=f"Number of results (default: {DEFAULT_TOP_K})")
    parser.add_argument("--inspect", "-i", action="store_true",
                       help="Show detailed results")
    parser.add_argument("--json", "-j", action="store_true",
                       help="Output as JSON")
    
    # Hybrid parameters
    hybrid_group = parser.add_argument_group("Hybrid search parameters")
    hybrid_group.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE,
                             help=f"Candidates per retriever (default: {DEFAULT_POOL_SIZE})")
    hybrid_group.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K,
                             help=f"RRF dampening constant (default: {DEFAULT_RRF_K})")
    hybrid_group.add_argument("--semantic-weight", type=float, default=DEFAULT_SEMANTIC_WEIGHT,
                             help=f"Semantic weight (default: {DEFAULT_SEMANTIC_WEIGHT})")
    hybrid_group.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT,
                             help=f"Keyword weight (default: {DEFAULT_KEYWORD_WEIGHT})")
    hybrid_group.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                             help=f"Alpha parameter (default: {DEFAULT_ALPHA})")
    hybrid_group.add_argument("--beta", type=float, default=DEFAULT_BETA,
                             help=f"Beta parameter (default: {DEFAULT_BETA})")
    
    # Semantic parameters
    semantic_group = parser.add_argument_group("Semantic search parameters")
    semantic_group.add_argument("--min-score", "-s", type=float, default=DEFAULT_MIN_SCORE,
                               help=f"Minimum score threshold (default: {DEFAULT_MIN_SCORE})")
    
    # Keyword parameters
    keyword_group = parser.add_argument_group("Keyword search parameters")
    keyword_group.add_argument("--k1", type=float, default=DEFAULT_K1,
                              help=f"BM25 k1 parameter (default: {DEFAULT_K1})")
    keyword_group.add_argument("--b", type=float, default=DEFAULT_B,
                              help=f"BM25 b parameter (default: {DEFAULT_B})")
    
    # Reranking parameters
    rerank_group = parser.add_argument_group("Reranking parameters")
    rerank_group.add_argument("--rerank", "-r", action="store_true",
                             help="Enable cross-encoder reranking")
    rerank_group.add_argument("--rerank-weight", type=float, default=DEFAULT_RERANK_WEIGHT,
                             help=f"Original score weight in reranking (default: {DEFAULT_RERANK_WEIGHT})")
    rerank_group.add_argument("--rerank-top-k", type=int, default=DEFAULT_RERANK_TOP_K,
                             help=f"Rerank candidates count (default: {DEFAULT_RERANK_TOP_K})")
    
    # WebSocket
    parser.add_argument("--ws-port", type=int, default=8765,
                       help="WebSocket server port (default: 8765)")
    parser.add_argument("--no-ws", action="store_true",
                       help="Disable WebSocket server")
    
    return parser


async def main() -> int:
    """Entry point for unified retrieval CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Resolve query
    query = args.query or args.query_flag
    if not query:
        parser.error("Query required (either as positional argument or --query flag)")
    
    # Setup WebSocket
    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)
    
    try:
        # Build retrieval args
        retrieval_args = RetrievalArgs(
            query=query,
            method=RetrievalMethod(args.method),
            files=args.file,
            top_k=args.top_k,
            inspect=args.inspect,
            json_output=args.json,
            pool_size=args.pool_size,
            rrf_k=args.rrf_k,
            semantic_weight=args.semantic_weight,
            keyword_weight=args.keyword_weight,
            alpha=args.alpha,
            beta=args.beta,
            min_score=args.min_score,
            k1=args.k1,
            b=args.b,
            rerank=args.rerank,
            rerank_weight=args.rerank_weight,
            rerank_top_k=args.rerank_top_k,
            ws_port=args.ws_port,
            no_ws=args.no_ws,
        )
        
        handle_retrieval(retrieval_args)
        return 0
    
    except KeyboardInterrupt:
        darwin.log("")
        darwin.log("Goodbye!", LogType.INFO)
        return 130
    except Exception as e:
        import traceback
        darwin.log(f"Fatal error: {e}", LogType.ERROR)
        darwin.log(''.join(traceback.format_exception(e)), LogType.ERROR)
        return 1
    finally:
        shutdown_websocket(ws_server)


if __name__ == "__main__":
    asyncio.run(main())