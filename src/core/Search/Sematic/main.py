"""
src/core/Search/main.py - Semantic search CLI

Search embedded documents using pure semantic (dense) retrieval.

Usage examples
--------------
# Search all documents in current project
uv run src/core/Search/main.py "photosynthesis"

# Search a specific document
uv run src/core/Search/main.py "cell division" --file biology.pdf

# Get more results
uv run src/core/Search/main.py "mitochondria" --top-k 10

# Set minimum score threshold
uv run src/core/Search/main.py "ATP synthesis" --min-score 0.5

# Export as JSON for scripting
uv run src/core/Search/main.py "enzymes" --json

# Inspect — show full chunks with metadata
uv run src/core/Search/main.py "DNA replication" --inspect
"""

import sys
import argparse
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
    SearchResult,

    to_llm_context,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI
from src.core.Embedding.EmbedderUtils import DEFAULT_MODEL


# ============================================================================
# Inspect — detailed view of results
# ============================================================================

def _print_inspection(results: list[SearchResult], query: str) -> None:
    """Print detailed inspection of search results."""
    TerminalUI.blank()
    TerminalUI.rule(f"search inspection — '{query}'")
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

def handle_search(args: SearchArgs) -> None:
    store = DarwinStore(Path(args.project) if args.project else Path.cwd())
    
    # Load config for model name
    config = store.load_config()
    model_name = config.get("embedding_model", args.model)
    
    TerminalUI.print_darwin()
    TerminalUI.blank()
    TerminalUI.bold(f"Searching: '{args.query}'")
    TerminalUI.dim(f"Model: {model_name} | Top-K: {args.top_k}")
    TerminalUI.blank()
    
    # Execute search
    if args.file:
        results = search_single_document(
            query=args.query,
            filename=args.file,
            store=store,
            top_k=args.top_k,
            model_name=model_name,
            min_score=args.min_score,
        )
    else:
        results = search_across_documents(
            query=args.query,
            store=store,
            top_k=args.top_k,
            model_name=model_name,
            min_score=args.min_score,
        )
    
    # Display results
    if not results:
        TerminalUI.warn("No results found")
        return
    
    TerminalUI.success(f"Found {len(results)} results")
    TerminalUI.blank()
    
    # Simple table view
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
        title="search results"
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
        print(format_results(results, json_output=True))


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Semantic search over embedded documents",
        usage="darwin search QUERY [options]"
    )
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--file", "-f", default=None, help="Search specific file")
    parser.add_argument("--project", "-p", default=None, help="Project directory")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Embedding model")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K, help="Number of results")
    parser.add_argument("--min-score", "-s", type=float, default=DEFAULT_MIN_SCORE, help="Minimum score threshold (0-1)")
    parser.add_argument("--inspect", "-i", action="store_true", help="Show detailed results")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    handle_search(SearchArgs(
        query=args.query,
        file=args.file,
        project=args.project,
        model=args.model,
        top_k=args.top_k,
        min_score=args.min_score,
        inspect=args.inspect,
        json_output=args.json,
    ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())