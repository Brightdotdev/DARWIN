"""
src/core/Search/Sematic/main.py - Semantic search CLI

Search embedded documents using pure semantic (dense) retrieval.

Usage examples
--------------
# Search all documents in current project
uv run src/core/Search/Sematic/main.py "photosynthesis"

# Search a specific document
uv run src/core/Search/Sematic/main.py "cell division" --file biology.pdf

# Get more results
uv run src/core/Search/Sematic/main.py "mitochondria" --top-k 10

# Set minimum score threshold
uv run src/core/Search/Sematic/main.py "ATP synthesis" --min-score 0.5

# Export as JSON for scripting
uv run src/core/Search/Sematic/main.py "enzymes" --json

# Inspect — show full chunks with metadata
uv run src/core/Search/Sematic/main.py "DNA replication" --inspect
"""

import sys
import argparse
import asyncio
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Search.Sematic.SematicSearch import (
    search_single_document,
    search_across_documents,
)
from src.core.Search.Sematic.SematicSearchUtils import (
    SearchArgs,
    DEFAULT_TOP_K,
    DEFAULT_MIN_SCORE,
    format_results,
    SematicSearchResult,
    to_llm_context,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Utils import scan_folder
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket
from src.core.Logger.DarwinLogger import darwin, LogType


# ============================================================================
# Inspect — detailed view of results
# ============================================================================

def _print_inspection(results: list[SematicSearchResult], query: str) -> None:
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


# ============================================================================
# Core handler
# ============================================================================

def handle_search(args: SearchArgs) -> None:
    store = DarwinStore()
    config = store.load_config()
    model_name = config.get("embedding_model", "unknown")
    
    darwin.log("")
    darwin.log("Semantic Search", LogType.INFO)
    darwin.log(f"Query: '{args.query}'", LogType.INFO)
    darwin.log(f"File: {args.file if args.file else 'all files'} | Top-K: {args.top_k}", LogType.INFO)
    darwin.log(f"Model: {model_name}", LogType.INFO)
    darwin.log("")

    status = store.status()
    tobe_searched = []
    
    if args.file:
        if not _is_embedding_searchable(args.file, status):
            darwin.log(f"Cannot search {args.file} — the file is not embedding-searchable", LogType.ERROR)
            return
        tobe_searched = [args.file]
        darwin.log(f"Searching {args.file} for '{args.query}'", LogType.INFO)
    else:
        raw_files = [str(f) for f in scan_folder()]
        darwin.log(f"Found {len(raw_files)} supported files in {Path.cwd()}", LogType.INFO)
        tobe_searched = [
            r.original_name
            for r in store.registry.list_all()
            if _is_embedding_searchable(r.original_name, status)
        ]
        darwin.log(f"Searching {len(tobe_searched)} embedding-searchable files for '{args.query}'", LogType.INFO)
    
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
        darwin.log("No results found", LogType.WARNING)
        return
    
    darwin.log(f"Found {len(results)} results", LogType.SUCCESS)
    darwin.log("")
    
    # Results table
    darwin.log("  #    Score       Source                Page  Preview", LogType.INFO)
    darwin.log("  ───  ──────────  ────────────────────  ────  ───────", LogType.INFO)
    
    for i, r in enumerate(results[:10], 1):
        preview = r.text[:60].replace('\n', ' ')
        darwin.log(f"  {i:<4} {r.score:.3f}      {r.source[:20]:<20}  {r.page:<4}  {preview}...", LogType.INFO)
    
    # LLM context preview
    darwin.log("")
    darwin.log("── LLM context preview (first 3 chunks) ──", LogType.INFO)
    darwin.log("")
    context = to_llm_context(results, max_chunks=3)
    darwin.log(context, LogType.INFO)
    
    if args.inspect:
        _print_inspection(results, args.query)
    
    if args.json_output:
        print(format_results(results, json_output=True))


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
    parser.add_argument("--file", "-f", default=None, help="Search specific file")
    parser.add_argument("--project", "-p", default=None, help="Project directory")
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
        search_args = SearchArgs(
            query=query,
            file=args.file,
            project=args.project,
            top_k=args.top_k,
            min_score=args.min_score,
            inspect=args.inspect,
            json_output=args.json,
        )
        
        handle_search(search_args)
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