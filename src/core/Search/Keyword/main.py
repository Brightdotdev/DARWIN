"""
src/core/KeywordSearch/main.py - BM25 keyword search CLI

Search indexed documents using pure BM25 (sparse) retrieval.

Usage examples
--------------
# Search all documents in current project
uv run src/core/KeywordSearch/main.py "photosynthesis"

# Search a specific document
uv run src/core/KeywordSearch/main.py "cell division" --file biology.pdf

# Get more results
uv run src/core/KeywordSearch/main.py "mitochondria" --top-k 10

# Tune BM25 parameters
uv run src/core/KeywordSearch/main.py "ATP synthesis" --k1 1.2 --b 0.5

# Export as JSON for scripting
uv run src/core/KeywordSearch/main.py "enzymes" --json

# Inspect — show full chunks with metadata
uv run src/core/KeywordSearch/main.py "DNA replication" --inspect
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Search.Keyword.KeywordSearch import (
    search_single_document_keyword,
    search_across_documents_keyword,
    KeywordSearchResult,
)
from src.core.Search.Keyword.KeywordUtilities import (
    KeywordSearchArgs,
    DEFAULT_TOP_K,
    DEFAULT_K1,
    DEFAULT_B,
    format_keyword_results,
    to_llm_context,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


# ============================================================================
# Inspect — detailed view of results
# ============================================================================

def _print_inspection(results: List[KeywordSearchResult], query: str) -> None:
    """Print detailed inspection of keyword search results."""
    TerminalUI.blank()
    TerminalUI.rule(f"keyword search inspection — '{query}'")
    TerminalUI.blank()
    
    TerminalUI.table(
        headers=["metric", "value"],
        rows=[
            ["total results", str(len(results))],
            ["top score", f"{results[0].score:.4f}" if results else "N/A"],
            ["lowest score", f"{results[-1].score:.4f}" if results else "N/A"],
        ],
    )
    
    TerminalUI.blank()
    TerminalUI.dim("── top results ──")
    TerminalUI.blank()
    
    for i, r in enumerate(results[:5], 1):
        TerminalUI.info("rank", str(i))
        TerminalUI.info("score", f"{r.score:.4f}")
        TerminalUI.info("source", r.source)
        TerminalUI.info("page", str(r.page))
        TerminalUI.info("heading", r.heading or "(none)")
        TerminalUI.info("matches", ", ".join(r.matched_terms))
        TerminalUI.blank()
        
        # Show first 500 chars
        preview = r.text[:500].replace("\n", " ").strip()
        if len(r.text) > 500:
            preview += "..."
        TerminalUI.dim(f"  {preview}")
        TerminalUI.blank()
        TerminalUI.divider()
        TerminalUI.blank()
    
    TerminalUI.rule()


# ============================================================================
# Fancy Search with Spinner
# ============================================================================

def _search_wrapper(
    store: DarwinStore,
    args: KeywordSearchArgs,
) -> List[KeywordSearchResult]:
    """
    Execute search with fancy spinner animation.
    """
    # Determine search scope for spinner message
    if args.file:
        search_scope = f"in {args.file}"
    else:
        # Count available files with BM25 indexes
        status = store.status()
        files_with_bm25 = sum(
            1 for fname, info in status.get("files", {}).items()
            if info.get("components", {}).get("bm25")
        )
        search_scope = f"across {files_with_bm25} document(s)"
    
    results = []
    
    
    with TerminalUI.spinner_progress(
        description=f"[cyan]Searching[/cyan] '{args.query}' {search_scope}"
    ) as (progress, task_id):
        
        # Simulate search time (optional - makes spinner visible for fast searches)
        start_time = time.perf_counter()
        
        # Execute search
        if args.file:
            bm25Data = store.load_bm25(args.file)
            if not bm25Data:
                TerminalUI.error(f"Could not find BM25 data for '{args.file}' exiting...")
                return
            chunks = store.load_chunks(args.file)
            if not chunks:
                TerminalUI.error(f"Could not find chunks for '{args.file}' exiting...")
                return
            
            results = search_single_document_keyword(
                filename=args.file,
                query=args.query,
                data=bm25Data,
                chunks=chunks,
                top_k=args.top_k,
                k1=args.k1,
                b=args.b,
            )
        else:
            results = search_across_documents_keyword(
                query=args.query,
                store=store,
                top_k=args.top_k,
                k1=args.k1,
                b=args.b,
            )
        
        # Ensure spinner shows for at least 0.5s (feels more satisfying)
        elapsed = time.perf_counter() - start_time
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        
        # Update spinner with completion message
        if results:
            progress.update(
                task_id,
                description=f"[green]✓ Found {len(results)} results[/green] for '[cyan]{args.query}[/cyan]'"
            )
        else:
            progress.update(
                task_id,
                description=f"[yellow]⚠ No results found[/yellow] for '[cyan]{args.query}[/cyan]'"
            )
    
    return results


# ============================================================================
# Results Display with Animation
# ============================================================================

def _display_search_results(results: List[KeywordSearchResult], args: KeywordSearchArgs) -> None:
    """Display results with fancy formatting."""
    
    if not results:
        TerminalUI.blank()
        TerminalUI.panel(
            f"No results found for '[cyan]{args.query}[/cyan]'\n\n"
            "Try:\n"
            "  • Using fewer or different keywords\n"
            "  • Checking spelling\n"
            "  • Searching a specific file with --file",
            title=" Search Results",
            border_style="yellow"
        )
        return
    
    TerminalUI.blank()
    
    # Animated count reveal (optional - feels premium)
    TerminalUI.success(f"Found {len(results)} results for '[cyan]{args.query}[/cyan]'")
    TerminalUI.blank()
    
    # Table view with color-coded scores
    table_rows = []
    for i, r in enumerate(results[:args.top_k], 1):
        preview = r.text[:60].replace('\n', ' ').strip()
        
        # Color code scores
        if r.score > 10:
            score_display = f"[green]{r.score:.3f}[/green]"
        elif r.score > 5:
            score_display = f"[yellow]{r.score:.3f}[/yellow]"
        else:
            score_display = f"[dim]{r.score:.3f}[/dim]"
        
        table_rows.append([
            str(i),
            score_display,
            r.source[:25],
            str(r.page),
            preview + "..."
        ])
    
    TerminalUI.table(
        headers=["#", "Score", "Source", "Page", "Preview"],
        rows=table_rows,
        title=f"BM25 Search Results (k1={args.k1}, b={args.b})"
    )
    
    # Show LLM context preview with animation feel
    TerminalUI.blank()
    TerminalUI.divider("LLM Context Preview")
    TerminalUI.blank()
    
    context = to_llm_context(results, max_chunks=3)
    TerminalUI.panel(
        context, 
        title="Ready for LLM", 
        border_style="cyan"
    )
    
    # Show search stats
    TerminalUI.blank()
    TerminalUI.dim(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    TerminalUI.dim(f"  BM25 parameters: k1={args.k1}, b={args.b}")
    TerminalUI.dim(f"  Top score: {results[0].score:.3f}")
    TerminalUI.dim(f"  Sources: {len(set(r.source for r in results))} files")
    TerminalUI.dim(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ============================================================================
# Core handler
# ============================================================================

def handle_keyword_search(args: KeywordSearchArgs) -> None:
    store = DarwinStore()
    
    
    TerminalUI.print_darwin()
    TerminalUI.blank()
    
    # Show search configuration
    config_panel = (
        f"[cyan]Query:[/cyan] {args.query}\n"
        f"[cyan]BM25:[/cyan] k1={args.k1}, b={args.b}\n"
        f"[cyan]Top-K:[/cyan] {args.top_k}\n"
        f"[cyan]Scope:[/cyan] {args.file if args.file else 'all documents'}"
    )
    TerminalUI.panel(config_panel, title="Keyword Search", border_style="cyan")
    TerminalUI.blank()
    
    
    results = _search_wrapper(store, args)
    
    
    _display_search_results(results, args)
    
    # Inspection mode
    if args.inspect and results:
        _print_inspection(results, args.query)
    
    # JSON output if requested
    if args.json_output:
        TerminalUI.blank()
        TerminalUI.dim("── JSON Output ──")
        print(format_keyword_results(results, json_output=True))


# ============================================================================
# CLI
# =============
def main() -> int:
    parser = argparse.ArgumentParser(
        description="BM25 keyword search over indexed documents"
    )
    
    # Make query flexible - positional OR flag
    parser.add_argument(
        "query", 
        nargs="?",  # Optional positional
        default=None,
        help="Search query string"
    )
    parser.add_argument(
        "--query", "-q", 
        dest="query_flag",
        help="Search query (alternative to positional)"
    )
    parser.add_argument("--file", "-f", default=None, help="Search specific file")
    parser.add_argument("--project", "-p", default=None, help="Project directory")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    parser.add_argument("--inspect", "-i", action="store_true")
    parser.add_argument("--json", "-j", action="store_true")
    
    args = parser.parse_args()
    
    # Resolve query from either source
    query = args.query or args.query_flag
    
    if not query:
        parser.error("Query required (either as positional or --query)")


    handle_keyword_search(KeywordSearchArgs(
        query=query,
        file=args.file,
        project=args.project,
        top_k=args.top_k,
        k1=args.k1,
        b=args.b,
        inspect=args.inspect,
        json_output=args.json,
    ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())