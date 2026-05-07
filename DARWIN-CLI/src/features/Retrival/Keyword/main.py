import sys
import argparse
import time
import asyncio
from pathlib import Path
from typing import List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.features.Retrival.Keyword.KeywordSearch import (
    search_across_documents_keyword,
)

from src.features.Retrival.lib import KeywordSearchResult

from src.features.Retrival.Keyword.KeywordUtilities import (
    KeywordSearchArgs,
    DEFAULT_TOP_K,
    DEFAULT_K1,
    DEFAULT_B,
    format_keyword_results,
    
)
from src.lib.Utils import scan_folder
from src.features.Storage.DarwinStore import DarwinStore
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket
from src.core.Logger.DarwinLogger import darwin, LogType


# ============================================================================
# Inspect — detailed view of results
# ============================================================================

def _print_inspection(results: List[KeywordSearchResult], query: str) -> None:
    """Print detailed inspection of keyword search results."""
    darwin.log("")
    darwin.log(f"── keyword search inspection — '{query}' ──", LogType.INFO)
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
        darwin.log(f"  matches:  {', '.join(r.matched_terms)}", LogType.INFO)
        darwin.log("")
        
        preview = r.text[:500].replace("\n", " ").strip()
        if len(r.text) > 500:
            preview += "..."
        darwin.log(f"  {preview}", LogType.INFO)
        darwin.log("")
        darwin.log("─" * 40, LogType.INFO)
        darwin.log("")
    
    darwin.log("─" * 60, LogType.INFO)


def _is_kw_searchable(filename: str | None, status: dict) -> bool:
    """Determine which files to search based on args and registry status."""
    if filename is None:
        return False
        
    key = Path(filename).name
    file_data = status.get("files", {}).get(key)
    if file_data is None:
        return False
    return file_data.get("is_kw_searchable", False)


def _validate_files(
    files: List[str], 
    store: DarwinStore
) -> List[str]:
    """Validate  files exist and are searchable, returning valid filenames."""
    status = store.status()
    valid_files = []
    invalid_files = []
    
    for filename in files:
        if _is_kw_searchable(filename, status):
            valid_files.append(filename)
        else:
            invalid_files.append(filename)
    
    if invalid_files:
        for f in invalid_files:
            darwin.log(f"{f} is not searchable", LogType.WARNING)

    return valid_files


# ============================================================================
# Search wrapper
# ============================================================================

def _search_wrapper(
    store: DarwinStore,
    args: KeywordSearchArgs,
) -> List[KeywordSearchResult]:
    """Execute search and return results."""
    status = store.status()
    tobe_searched = []
    
    if args.files:
        darwin.log(f"Searching across {' ,'.join(args.files)}", LogType.INFO)
        tobe_searched = _validate_files(args.files, store)
        
        if not tobe_searched:
            darwin.log("No valid files provided Search aborted.", LogType.ERROR)
            return []
        
        darwin.log(f"Searching {len(tobe_searched)} valid files for '{args.query}'", LogType.INFO)

    else:
        tobe_searched = [
            fname for fname, info in status.get("files", {}).items()
             if info.get("is_kw_searchable", False)
        ]
        darwin.log(f"Searching {len(tobe_searched)} files for '{args.query}'", LogType.INFO)
    
    if not tobe_searched:
        darwin.log("No searchable files found. Run indexing first.", LogType.WARNING)
        return []
    
    start_time = time.perf_counter()
    
    results = search_across_documents_keyword(
        filenames=tobe_searched,
        query=args.query,
        store=store,
        top_k=args.top_k,
        k1=args.k1,
        b=args.b,
    )
    
    elapsed = time.perf_counter() - start_time
    
    if results:
        darwin.log(f"Found {len(results)} results in {elapsed:.2f}s", LogType.SUCCESS)
    else:
        darwin.log(f"No results found for '{args.query}'", LogType.WARNING)
    
    return results


# ============================================================================
# Results Display
# ============================================================================

def _display_search_results(results: List[KeywordSearchResult], args: KeywordSearchArgs) -> None:
    """Display results with formatting."""
    
    if not results:
        darwin.log("")
        darwin.log(f"No results found for '{args.query}'", LogType.WARNING)
        darwin.log("")
        darwin.log("Try:", LogType.INFO)
        darwin.log("  • Using fewer or different keywords", LogType.INFO)
        darwin.log("  • Checking spelling", LogType.INFO)
        darwin.log("  • Searching a specific file with --file", LogType.INFO)
        return
    
    darwin.log("")
    darwin.log(f"Found {len(results)} results for '{args.query}'", LogType.SUCCESS)
    darwin.log("")
    
    # Results table
    darwin.log("  #    Score       Source                  Page  Preview", LogType.INFO)
    darwin.log("  ───  ──────────  ──────────────────────  ────  ───────", LogType.INFO)
    
    for i, r in enumerate(results[:args.top_k], 1):
        preview = r.text[:60].replace('\n', ' ').strip()
        darwin.log(f"  {i:<4} {r.score:.3f}      {r.source[:25]:<25}  {r.page:<4}  {preview}...", LogType.INFO)
    
    # LLM context preview
    darwin.log("")
    darwin.log("── LLM Context Preview ──", LogType.INFO)
    darwin.log("")
    

    # Search stats
    darwin.log("")
    darwin.log(f"  BM25 parameters: k1={args.k1}, b={args.b}", LogType.INFO)
    darwin.log(f"  Top score: {results[0].score:.3f}", LogType.INFO)
    darwin.log(f"  Sources: {len(set(r.source for r in results))} files", LogType.INFO)
    darwin.log("")


# ============================================================================
# Core handler
# ============================================================================

def handle_keyword_search(args: KeywordSearchArgs) -> List[KeywordSearchResult] | None:
    store = DarwinStore()
    
    darwin.log("")
    darwin.log("Keyword Search", LogType.INFO)
    darwin.log(f"Query: {args.query}", LogType.INFO)
    darwin.log(f"BM25: k1={args.k1}, b={args.b}", LogType.INFO)
    darwin.log(f"Top-K: {args.top_k}", LogType.INFO)
    darwin.log(f"Scope: {' ,'.join(args.files) if args.files else 'all documents'}", LogType.INFO)


    darwin.log("")
    
    results = _search_wrapper(store, args)
    _display_search_results(results, args)
    
    if not results:
        return

    if args.inspect and results:
        _print_inspection(results, args.query)
    
    if args.json_output:
        darwin.log("── JSON Output ──", LogType.INFO)
        print(format_keyword_results(results, json_output=True))
    
    return results


# ============================================================================
# CLI
# ============================================================================

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="BM25 keyword search over indexed documents"
    )
    parser.add_argument("query", nargs="?", default=None, help="Search query string")
    parser.add_argument("--query", "-q", dest="query_flag", help="Search query (alternative)")
    
    parser.add_argument("--file", "-f", nargs="+", default=None, help="One or more files to index")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    parser.add_argument("--inspect", "-i", action="store_true")
    parser.add_argument("--json", "-j", action="store_true")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")

    args = parser.parse_args()
    
    query = args.query or args.query_flag
    
    if not query:
        parser.error("Query required (either as positional or --query)")
        

    # Setup WebSocket
    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        search_args = KeywordSearchArgs(
            query=query,
            files=args.file,
            top_k=args.top_k,
            k1=args.k1,
            b=args.b,
            inspect=args.inspect,
            json_output=args.json,
        )
        
        handle_keyword_search(search_args)
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