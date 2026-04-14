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
# Core handler
# ============================================================================

def handle_keyword_search(args: KeywordSearchArgs) -> None:
    store = DarwinStore(Path(args.project) if args.project else Path.cwd())
    
    TerminalUI.print_darwin()
    TerminalUI.blank()
    TerminalUI.bold(f"Keyword search: '{args.query}'")
    TerminalUI.dim(f"BM25: k1={args.k1}, b={args.b} | Top-K: {args.top_k}")
    TerminalUI.blank()
    
    # Execute search
    if args.file:
        results = search_single_document_keyword(
            query=args.query,
            filename=args.file,
            store=store,
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
    
    # Display results
    if not results:
        TerminalUI.warn("No results found")
        return
    
    TerminalUI.success(f"Found {len(results)} results")
    TerminalUI.blank()
    
    # Table view
    table_rows = []
    for i, r in enumerate(results[:10], 1):
        preview = r.text[:60].replace('\n', ' ')
        table_rows.append([
            str(i),
            f"{r.score:.3f}",
            r.source[:20],
            str(r.page),
            preview + "..."
        ])
    
    TerminalUI.table(
        headers=["#", "Score", "Source", "Page", "Preview"],
        rows=table_rows,
        title="keyword search results"
    )
    
    # Show LLM context preview
    TerminalUI.blank()
    TerminalUI.dim("── LLM context preview (first 3 chunks) ──")
    TerminalUI.blank()
    context = to_llm_context(results, max_chunks=3)
    TerminalUI.panel(context, title="context", border_style="dim")
    
    if args.inspect:
        _print_inspection(results, args.query)
    
    # JSON output if requested
    if args.json_output:
        print(format_keyword_results(results, json_output=True))


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="BM25 keyword search over indexed documents"
    )
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--file", "-f", default=None, help="Search specific file")
    parser.add_argument("--project", "-p", default=None, help="Project directory")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K, help="Number of results")
    parser.add_argument("--k1", type=float, default=DEFAULT_K1, help="BM25 term frequency saturation")
    parser.add_argument("--b", type=float, default=DEFAULT_B, help="BM25 length normalization")
    parser.add_argument("--inspect", "-i", action="store_true", help="Show detailed results")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    handle_keyword_search(KeywordSearchArgs(
        query=args.query,
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