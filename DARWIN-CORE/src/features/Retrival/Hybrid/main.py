"""
src/core/HybridSearch/main.py - Hybrid search
"""

import sys
import os
import warnings
import argparse
import asyncio
from pathlib import Path
from typing import List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


from src.features.Retrival.Hybrid.HybridSearch import (
    search_single_document_hybrid,
    search_across_documents_hybrid,
)
from src.features.Retrival.lib import (HybridSearchResult)
from src.features.Retrival.Hybrid.HybridSearchUtils import (
    HybridSearchArgs,
    DEFAULT_TOP_K,
    DEFAULT_RRF_K,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_POOL_SIZE,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    OverlapDirection,
    format_hybrid_results
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Utils import scan_folder
from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity


# ============================================================================
# Inspect
# =========================================================================

def _print_inspection(results: List[HybridSearchResult], query: str) -> None:
    """Print detailed inspection of hybrid search results."""
    darwin.log(f"\n{'='*60}", LogType.INFO)
    darwin.log(f"Hybrid Search Inspection — '{query}'", LogType.INFO)
    darwin.log(f"{'='*60}", LogType.INFO)
    
    darwin.log(f"Total results: {len(results)}", LogType.INFO)
    if results:
        darwin.log(f"Top RRF score: {results[0].rrf_score:.4f}", LogType.INFO)
        darwin.log(f"Top final score: {results[0].final_score:.4f}", LogType.INFO)
    
    darwin.log(f"\n--- Top Results ---", LogType.INFO)
    
    for i, r in enumerate(results[:5], 1):
        darwin.log(f"\n[{i}] Rank {i}", LogType.INFO)
        darwin.log(f"    Final score: {r.final_score:.4f}", LogType.INFO)
        darwin.log(f"    RRF score: {r.rrf_score:.4f}", LogType.INFO)
        if r.semantic_score:
            darwin.log(f"    Semantic: {r.semantic_score:.4f}", LogType.INFO)
        if r.keyword_score:
            darwin.log(f"    Keyword: {r.keyword_score:.4f}", LogType.INFO)
        if r.matched_terms:
            darwin.log(f"    Matched terms: {', '.join(list(r.matched_terms)[:10])}", LogType.INFO)
        darwin.log(f"    Source: {r.source}", LogType.INFO)
        darwin.log(f"    Page: {r.page}" + (f"-{r.page_end}" if r.page_end != r.page else ""), LogType.INFO)
        darwin.log(f"    Heading: {r.heading or '(none)'}", LogType.INFO)
        if r.links:
            darwin.log(f"    Links: {len(r.links)} found", LogType.INFO)
        if r.overlap_direction != OverlapDirection.NONE:
            darwin.log(f"    Overlap: ⚠️ This chunk has {r.overlap_direction.value} overlap", LogType.WARNING)
        
        preview = r.text[:500].replace("\n", " ").strip()
        if len(r.text) > 500:
            preview += "..."
        darwin.log(f"    Preview: {preview}", LogType.INFO)
        darwin.log(f"    {'-'*40}", LogType.INFO)
    
    darwin.log(f"{'='*60}", LogType.INFO)


def _is_hybrid_searchable(filename: str | None, status: dict) -> bool:
    """Determine which files to search based on args and registry status."""
    if filename is None:
        return False

    key = Path(filename).name
    file_data = status.get("files", {}).get(key)
    if file_data is None:
        return False
    return file_data.get("is_hybrid_searchable", False)


def _validate_files(
    files: List[str], 
    store: DarwinStore
) -> List[str]:
    """Validate files exist and are searchable, returning valid filenames."""
    status = store.status()
    valid_files = []
    invalid_files = []
    
    for filename in files:
        if _is_hybrid_searchable(filename, status):
            valid_files.append(filename)
        else:
            invalid_files.append(filename)
    
    if invalid_files:
        for f in invalid_files:
            darwin.log(f"{f} is not searchable", LogType.WARNING)
    return valid_files


# ============================================================================
# Core handler
# ============================================================================

def handle_hybrid_search(args: HybridSearchArgs, store: DarwinStore) -> List[HybridSearchResult] | None:
    """Main handler for hybrid search CLI."""
    status = store.status()
    tobe_searched = []


    if args.files:
        tobe_searched = _validate_files(args.files, store)
        
        if not tobe_searched:
            darwin.log("No valid files provided for search. Search aborted.", LogType.ERROR)
            return
        
        darwin.log(f"Searching {len(tobe_searched)} valid file(s)  for '{args.query}'", LogType.INFO)
    
    else:
        tobe_searched = [
            fname for fname, info in status.get("files", {}).items()
             if info.get("is_hybrid_searchable", False)
        ]
        darwin.log(f"Searching {len(tobe_searched)} files for '{args.query}'", LogType.INFO)

    if not tobe_searched:
        darwin.log("No hybrid-searchable files found. Run both keyword and embedding indexing pipelines first.", LogType.WARNING)
        return

    darwin.log(f"\n{'='*50}", LogType.INFO)
    darwin.log(f"Hybrid Search: '{args.query}'", LogType.INFO)
    darwin.log(f"RRF k={args.rrf_k} | Semantic weight={args.semantic_weight} | Keyword weight={args.keyword_weight}", LogType.INFO)
    darwin.log(f"Pool size={args.pool_size} | Top-K={args.top_k}", LogType.INFO)
    
    
    darwin.log(f"{'='*50}", LogType.INFO)

    if args.top_k == 0:
        darwin.log("top_k=0 requested, using default", LogType.WARNING)
        args.top_k = DEFAULT_TOP_K

    results = []
    if len(tobe_searched) == 1:
        results = search_single_document_hybrid(
            query=args.query,
            filename=tobe_searched[0],
            store=store,
            top_k=args.top_k,
            rrf_k=args.rrf_k,
            semantic_weight=args.semantic_weight,
            keyword_weight=args.keyword_weight,
            pool_size=args.pool_size,
            alpha=args.alpha,
            beta=args.beta
        )
    else:
        results = search_across_documents_hybrid(
            query=args.query,
            store=store,
            filenames=tobe_searched if tobe_searched else None,
            top_k=args.top_k,
            rrf_k=args.rrf_k,
            semantic_weight=args.semantic_weight,
            keyword_weight=args.keyword_weight,
            pool_size=args.pool_size,
            alpha=args.alpha,
            beta=args.beta
        )

    darwin.log(f"Found {len(results)} results", LogType.SUCCESS)

    if not results:
        darwin.log("No results found", LogType.WARNING)
        darwin.log("")
        darwin.log("Try:", LogType.INFO)
        darwin.log("  • Using different or more specific keywords", LogType.INFO)
        darwin.log("  • Adjusting RRF parameters (--rrf-k, --semantic-weight, --keyword-weight)", LogType.INFO)
        darwin.log("  • Increasing pool size (--pool-size)", LogType.INFO)
        darwin.log("  • Searching a specific file with --file", LogType.INFO)
        return

    # Output results
    if args.json_output:
        print(format_hybrid_results(results, json_output=True))
    else:
        # Console output with proper formatting
        darwin.log(f"\n--- Hybrid Search Results ({len(results)} found) ---", LogType.INFO)
        
        for i, r in enumerate(results[:args.top_k], 1):
            darwin.log(f"\n[{i}] Score: {r.final_score:.4f} | Source: {r.source} (p{r.page})", LogType.INFO)
            preview = r.text[:200].replace('\n', ' ')
            if len(r.text) > 200:
                preview += "..."
            darwin.log(f"    {preview}", LogType.INFO)
            
            # Show additional info
            details = []
            if r.rrf_score:
                details.append(f"RRF: {r.rrf_score:.3f}")
            if r.semantic_score:
                details.append(f"Sem: {r.semantic_score:.3f}")
            if r.keyword_score:
                details.append(f"KW: {r.keyword_score:.3f}")
            if details:
                darwin.log(f"    ({' | '.join(details)})", LogType.INFO)
        
        # Summary stats
        darwin.log(f"\n{'='*50}", LogType.INFO)
        darwin.log(f"Sources: {len(set(r.source for r in results))} files", LogType.INFO)
        darwin.log(f"Top RRF score: {results[0].rrf_score:.4f}" if results else "N/A", LogType.INFO)
        darwin.log(f"Top final score: {results[0].final_score:.4f}" if results else "N/A", LogType.INFO)

    if args.inspect:
        _print_inspection(results, args.query)
    return results


# ============================================================================
# CLI
# ============================================================================

def create_hybrid_search_parser():
    """Create parser for hybrid search CLI."""
    parser = argparse.ArgumentParser(
        description="Hybrid search combining semantic and keyword retrieval"
    )

    # Flexible query - positional OR flag
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query string (positional)"
    )
    parser.add_argument(
        "--query", "-q",
        dest="query_flag",
        help="Search query string (flag alternative)"
    )
    
    parser.add_argument("--file", "-f", nargs="+", default=None, help="One or more files to index")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K, help="Final results to return")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help="RRF dampening constant")
    parser.add_argument("--semantic-weight", type=float, default=DEFAULT_SEMANTIC_WEIGHT, help="Weight for semantic results")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Alpha parameter for hybrid scoring")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Beta parameter for hybrid scoring")
    parser.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT, help="Weight for keyword results")
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE, help="Candidates per retriever")
    parser.add_argument("--inspect", "-i", action="store_true", help="Show detailed results")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--verbosity", choices=["normal", "quiet"], default="normal", help="Log verbosity")
    parser.add_argument("--store-dir", default=None, help="Darwin store directory")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port (default: 8765)")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")

    return parser


async def main() -> int:
    """Entry point for hybrid search CLI."""
    from src.lib.Utils import ensure_writable_directory

    parser = create_hybrid_search_parser()
    args = parser.parse_args()



    # Resolve query
    query = args.query or args.query_flag
    if not query:
        parser.error("❌ Query required (either as positional argument or --query flag)")

    if args.query and args.query_flag:
        darwin.log(f"Both positional and --query provided. Using positional: '{args.query}'", LogType.WARNING)
        query = args.query


    # Setup WebSocket
    darwin.configure(
        enable_websocket=not args.no_ws,
        ws_port=args.ws_port,
        enable_history=True,
        verbosity_default=Verbosity.QUIET if args.verbosity == "quiet" else Verbosity.NORMAL,
        show_banner=True,
    )

    try:
        search_args = HybridSearchArgs(
            query=query,
            files=args.file,
            top_k=args.top_k,
            rrf_k=args.rrf_k,
            semantic_weight=args.semantic_weight,
            keyword_weight=args.keyword_weight,
            pool_size=args.pool_size,
            inspect=args.inspect,
            json_output=args.json,
            alpha=args.alpha,
            beta=args.beta,
        )
        
        with darwin.session("hybrid_search"):
            if args.store_dir and ensure_writable_directory(args.store_dir):
                store = DarwinStore(args.store_dir)
            else:
                store = DarwinStore()
            handle_hybrid_search(search_args, store)
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
        darwin.shutdown()


if __name__ == "__main__":
    asyncio.run(main())