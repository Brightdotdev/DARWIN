"""
src/core/Search/Sematic/main.py - Semantic search CLI
"""

import sys
import argparse
import asyncio
from pathlib import Path
from typing import List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.features.Retrival.Semantic.SematicSearch import (
    search_single_document,
    search_across_documents,
)
from src.features.Retrival.Semantic.SematicSearchUtils import (
    SemanticSearchArgs,
    DEFAULT_TOP_K,
    DEFAULT_MIN_SCORE,
    format_results)

from src.features.Retrival.lib import SemanticSearchResult
from src.features.Storage.DarwinStore import DarwinStore
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket
from src.core.Logger.DarwinLogger import darwin, LogType


# ============================================================================
# Inspect — detailed view of results
# ============================================================================

def _print_inspection(results: list[SemanticSearchResult,], query: str) -> None:
    """Print detailed inspection of search results."""
    darwin.log("")
    darwin.log(f"── search inspection — '{query}' ──", LogType.INFO)
    darwin.log("")
    
    darwin.log("  metric              value", LogType.INFO)
    darwin.log("  ──────────────────  ─────", LogType.INFO)
    darwin.log(f"  total results       {len(results)}", LogType.INFO)
    top_score = f"{results[0].score:.4f}" if results else "N/A"
    darwin.log(f"  top score           {top_score}", LogType.INFO)
    low_score = f"{results[-1].score:.4f}" if results else "N/A"
    darwin.log(f"  lowest score        {low_score}", LogType.INFO)
    darwin.log("")
    
    darwin.log("── top results ──", LogType.INFO)
    darwin.log("")
    
    for i, r in enumerate(results[:5], 1):
        darwin.log(f"  rank:     {i}", LogType.INFO)
        darwin.log(f"  score:    {r.score:.4f}", LogType.INFO)
        darwin.log(f"  source:   {r.source}", LogType.INFO)
        darwin.log(f"  page:     {r.page}", LogType.INFO)
        darwin.log(f"  heading:  {r.heading or '(none)'}", LogType.INFO)
        darwin.log("")
        
        preview = r.text[:500].replace("\n", " ").strip()
        if len(r.text) > 500:
            preview += "..."
        darwin.log(f"  {preview}", LogType.INFO)
        darwin.log("")
        darwin.log("─" * 40, LogType.INFO)
        darwin.log("")
    
    darwin.log("─" * 60, LogType.INFO)


def _is_embedding_searchable(filename: str | None, status: dict) -> bool:
    """Determine which files to search based on args and registry status."""
    if filename is None:
        return False
        
    key = Path(filename).name
    file_data = status.get("files", {}).get(key)
    if file_data is None:
        return False
    return file_data.get("is_embedding_searchable", False)


def _validate_files(
    files: List[str], 
    store: DarwinStore
) -> List[str]:
    """Validate files exist and are searchable, returning valid filenames."""
    status = store.status()
    valid_files = []
    invalid_files = []
    
    for filename in files:
        if _is_embedding_searchable(filename, status):
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

def handle_semantic_search(args: SemanticSearchArgs) -> List[SemanticSearchResult] | None:
    store = DarwinStore()
    config = store.load_config()
    model_name = config.get("embedding_model", "unknown")
    
    darwin.log("")
    darwin.log("Semantic Search", LogType.INFO)
    darwin.log(f"Query: '{args.query}'", LogType.INFO)
    darwin.log(f"Model: {model_name}", LogType.INFO)
    
    status = store.status()
    tobe_searched = []
    
   
    if args.files:
        
        tobe_searched = _validate_files(args.files, store)
        
        if not tobe_searched:
            darwin.log("No valid files to search.. Search aborted.", LogType.ERROR)
            return
        
        darwin.log(f"Searching {len(tobe_searched)} valid files to search. for '{args.query}'", LogType.INFO)

    else:
        darwin.log(f"File: all files | Top-K: {args.top_k} | Min-Score: {args.min_score}", LogType.INFO)
        tobe_searched = [
            fname for fname, info in status.get("files", {}).items()
             if info.get("is_embedding_searchable", False)
        ]
        
        darwin.log(f"Searching {len(tobe_searched)} embedding-searchable files for '{args.query}'", LogType.INFO)
    
    darwin.log("")
    
    if not tobe_searched:
        darwin.log("No embedding-searchable files found. Run the embedding pipeline first.", LogType.WARNING)
        return
    
    results = []
    if len(tobe_searched) == 1:
        results = search_single_document(
            query=args.query,
            filename=tobe_searched[0],
            store=store,
            top_k=args.top_k,
            min_score=args.min_score,
        )
    else:
        results = search_across_documents(
            query=args.query,
            filenames=tobe_searched,
            store=store,
            top_k=args.top_k,
            min_score=args.min_score,
        )

    if not results:
        darwin.log(f"No results found for '{args.query}'", LogType.WARNING)
        darwin.log("")
        darwin.log("Try:", LogType.INFO)
        darwin.log("  • Using different or more specific keywords", LogType.INFO)
        darwin.log("  • Lowering the minimum score threshold (--min-score)", LogType.INFO)
        darwin.log("  • Searching a specific file with --file", LogType.INFO)
        return
    
    darwin.log(f"Found {len(results)} results", LogType.SUCCESS)
    darwin.log("")
    
    # Results table
    darwin.log("  #    Score       Source                Page  Preview", LogType.INFO)
    darwin.log("  ───  ──────────  ────────────────────  ────  ───────", LogType.INFO)
    
    for i, r in enumerate(results[:10], 1):
        preview = r.text[:60].replace('\n', ' ')
        darwin.log(f"  {i:<4} {r.score:.3f}      {r.source[:20]:<20}  {r.page:<4}  {preview}...", LogType.INFO)
    
    # Search stats
    darwin.log("")
    darwin.log(f"  Min score threshold: {args.min_score}", LogType.INFO)
    darwin.log(f"  Top score: {results[0].score:.3f}", LogType.INFO)
    darwin.log(f"  Sources: {len(set(r.source for r in results))} files", LogType.INFO)
    darwin.log("")
    
    if args.inspect:
        _print_inspection(results, args.query)
    
    if args.json_output:
        print(format_results(results, json_output=True))

    return results


# ============================================================================
# CLI
# ============================================================================

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Semantic search over embedded documents",
        usage="darwin search [QUERY] [options]"
    )
    
    parser.add_argument("query", nargs="?", default=None, help="Search query string (positional)")
    parser.add_argument("--query", "-q", dest="query_flag", help="Search query string (flag alternative)")
    parser.add_argument("--file", "-f", nargs="+", default=None, help="One or more files to index")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K, help="Number of results")
    parser.add_argument("--min-score", "-s", type=float, default=DEFAULT_MIN_SCORE, help="Minimum score threshold (0-1)")
    parser.add_argument("--inspect", "-i", action="store_true", help="Show detailed results")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")
    
    args = parser.parse_args()
    
    query = args.query or args.query_flag
    
    if not query:
        parser.error("Query required (either as positional argument or --query flag)")
    
    
    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        search_args = SemanticSearchArgs(
            query=query,
            files=args.file,
            top_k=args.top_k,
            min_score=args.min_score,
            inspect=args.inspect,
            json_output=args.json,
        )
        
        handle_semantic_search(search_args)
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