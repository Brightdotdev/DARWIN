"""
src/core/HybridSearch/main.py - Hybrid search CLI

Combine semantic and keyword retrieval via Reciprocal Rank Fusion.

Usage examples
--------------
# Search all documents
uv run src/core/HybridSearch/main.py "photosynthesis"

# Search specific document
uv run src/core/HybridSearch/main.py "david covenant" --file bible.pdf

# Use query flag
uv run src/core/HybridSearch/main.py --query "cell division" --file bio.pdf

# Adjust RRF parameters
uv run src/core/HybridSearch/main.py "cell division" --rrf-k 30 --semantic-weight 1.5

# More candidates from each retriever
uv run src/core/HybridSearch/main.py "mitochondria" --pool-size 100 --top-k 10

# Inspect results
uv run src/core/HybridSearch/main.py "ATP" --inspect

# JSON export
uv run src/core/HybridSearch/main.py "enzymes" --json
"""

import sys
import argparse
import asyncio
from pathlib import Path
from typing import List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Search.Hybrid.HybridSearch import (
    search_single_document_hybrid,
    search_across_documents_hybrid,
)
from src.core.Search.lib import (HybridSearchResult)
from src.core.Search.Hybrid.HybridSearchUtils import (
    HybridSearchArgs,
    DEFAULT_TOP_K,
    DEFAULT_RRF_K,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_POOL_SIZE,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    OverlapDirection,
    format_hybrid_results,
    to_llm_context,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Utils import scan_folder
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket


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


# ============================================================================
# Core handler
# ============================================================================

def handle_hybrid_search(args: HybridSearchArgs) -> None:
    """Main handler for hybrid search CLI."""
    store = DarwinStore()
    status = store.status()
    tobe_searched = []

    if args.file:
        if not _is_hybrid_searchable(args.file, status):
            darwin.log(f"Cannot search {args.file} — file is not searchable", LogType.WARNING)
            return

        tobe_searched = [args.file]
        darwin.log(f"Searching {args.file} for '{args.query}'", LogType.INFO)
    else:
        raw_files = [str(f) for f in scan_folder()]
        darwin.log(f"Found {len(raw_files)} supported files in {Path.cwd()}", LogType.INFO)
        tobe_searched = [
            r.original_name
            for r in store.registry.list_kw_searchable()
        ]
        darwin.log(f"Searching {len(tobe_searched)} files for '{args.query}'", LogType.INFO)

    darwin.log(f"\n{'='*50}", LogType.INFO)
    darwin.log(f"Hybrid Search: '{args.query}'", LogType.INFO)
    darwin.log(f"RRF k={args.rrf_k} | Semantic weight={args.semantic_weight} | Keyword weight={args.keyword_weight}", LogType.INFO)
    darwin.log(f"Pool size={args.pool_size} | Top-K={args.top_k}", LogType.INFO)
    darwin.log(f"{'='*50}", LogType.INFO)

    if args.top_k == 0:
        darwin.log("top_k=0 requested, using default", LogType.WARNING)
        args.top_k = DEFAULT_TOP_K

    results = []
    if args.file:
        results = search_single_document_hybrid(
            query=args.query,
            filename=args.file,
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

    # LLM context preview
    if results:
        darwin.log(f"\n--- LLM Context Preview (first 3 chunks) ---", LogType.INFO)
        context = to_llm_context(results, max_chunks=3)
        darwin.log(context, LogType.INFO)

    if args.inspect:
        _print_inspection(results, args.query)


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
    parser.add_argument("--file", "-f", default=None, help="Search specific file")
    parser.add_argument("--project", "-p", default=None, help="Project directory")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K, help="Final results to return")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help="RRF dampening constant")
    parser.add_argument("--semantic-weight", type=float, default=DEFAULT_SEMANTIC_WEIGHT, help="Weight for semantic results")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Alpha parameter for hybrid scoring")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Beta parameter for hybrid scoring")
    parser.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT, help="Weight for keyword results")
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE, help="Candidates per retriever")
    parser.add_argument("--inspect", "-i", action="store_true", help="Show detailed results")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port (default: 8765)")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")

    return parser


async def main() -> int:
    """Entry point for hybrid search CLI."""
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
    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        search_args = HybridSearchArgs(
            query=query,
            file=args.file,
            project=args.project,
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
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, handle_hybrid_search, search_args)
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