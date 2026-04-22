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
from pathlib import Path
from typing import List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.Search.Hybrid.HybridSearch import (
    search_single_document_hybrid,
    search_across_documents_hybrid,
)
from src.core.Search.Hybrid.HybridSearchUtils import (
    HybridSearchArgs,
    HybridSearchResult,
    DEFAULT_TOP_K,
    DEFAULT_RRF_K,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_POOL_SIZE,
    format_hybrid_results,
    to_llm_context,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


# ============================================================================
# Inspect
# ============================================================================

def _print_inspection(results: List[HybridSearchResult], query: str) -> None:
    """Print detailed inspection of hybrid search results."""
    TerminalUI.blank()
    TerminalUI.rule(f"hybrid search inspection — '{query}'")
    TerminalUI.blank()
    
    TerminalUI.table(
        headers=["metric", "value"],
        rows=[
            ["total results", str(len(results))],
            ["top RRF score", f"{results[0].rrf_score:.4f}" if results else "N/A"],
            ["top final score", f"{results[0].final_score:.4f}" if results else "N/A"],
        ],
    )
    
    TerminalUI.blank()
    TerminalUI.dim("── top results ──")
    TerminalUI.blank()
    
    for i, r in enumerate(results[:5], 1):
        TerminalUI.info("rank", str(i))
        TerminalUI.info("final score", f"{r.final_score:.4f}")
        TerminalUI.info("RRF score", f"{r.rrf_score:.4f}")
        if r.semantic_score:
            TerminalUI.info("semantic", f"{r.semantic_score:.4f}")
        if r.keyword_score:
            TerminalUI.info("keyword", f"{r.keyword_score:.4f}")
        if r.matched_terms:
            TerminalUI.info("matched terms", ", ".join(r.matched_terms[:10]))
        TerminalUI.info("source", r.source)
        TerminalUI.info("page", f"{r.page}" + (f"-{r.page_end}" if r.page_end != r.page else ""))
        TerminalUI.info("heading", r.heading or "(none)")
        if r.links:
            TerminalUI.info("links", f"{len(r.links)} found")
        if r.is_overlap:
            TerminalUI.info("overlap", "⚠️ This chunk overlaps with adjacent chunks")
        TerminalUI.blank()
        
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

def handle_hybrid_search(args: HybridSearchArgs) -> None:
    store = DarwinStore(Path(args.project) if args.project else Path.cwd())
    
    TerminalUI.print_darwin()
    TerminalUI.blank()
    TerminalUI.bold(f"Hybrid search: '{args.query}'")
    TerminalUI.dim(f"RRF k={args.rrf_k} | Semantic weight={args.semantic_weight} | Keyword weight={args.keyword_weight}")
    TerminalUI.dim(f"Pool size={args.pool_size} | Top-K={args.top_k} | Parallel={args.parallel}")
    TerminalUI.blank()

    if args.top_k == 0:
        TerminalUI.dim("Return no results?? lmao okay...defaulting to default topk")
        args.top_k = DEFAULT_TOP_K
    
    results = []
    with TerminalUI.spinner_progress("Searching...") as (progress, task_id):
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
                parallel=args.parallel,
            )
        else:
            results = search_across_documents_hybrid(
                query=args.query,
                store=store,
                top_k=args.top_k,
                rrf_k=args.rrf_k,
                semantic_weight=args.semantic_weight,
                keyword_weight=args.keyword_weight,
                pool_size=args.pool_size,
                parallel=args.parallel,
            )
        
        progress.update(task_id, description=f"[green]✓ Found {len(results)} results[/green]")
    
    if not results:
        TerminalUI.warn("No results found")
        return
    
    TerminalUI.success(f"Found {len(results)} results")
    TerminalUI.blank()
    
    # Table view
    table_rows = []
    for i, r in enumerate(results[:10], 1):
        preview = r.text[:60].replace('\n', ' ')
        source_str = f"{r.source[:15]} (p{r.page})"
        table_rows.append([
            str(i),
            f"{r.final_score:.3f}",
            f"{r.rrf_score:.3f}",
            f"{r.semantic_score:.3f}" if r.semantic_score else "—",
            f"{r.keyword_score:.3f}" if r.keyword_score else "—",
            source_str,
            preview + "..."
        ])
    
    TerminalUI.table(
        headers=["#", "Final", "RRF", "Sem", "KW", "Source", "Preview"],
        rows=table_rows,
        title="hybrid search results"
    )
    
    # LLM context preview
    TerminalUI.blank()
    TerminalUI.dim("── LLM context preview (first 3 chunks) ──")
    TerminalUI.blank()
    context = to_llm_context(results, max_chunks=3)
    TerminalUI.panel(context, title="context", border_style="dim")
    
    if args.inspect:
        _print_inspection(results, args.query)
    
    if args.json_output:
        print(format_hybrid_results(results, json_output=True))


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
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
    parser.add_argument("--keyword-weight", type=float, default=DEFAULT_KEYWORD_WEIGHT, help="Weight for keyword results")
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE, help="Candidates per retriever")
    parser.add_argument("--inspect", "-i", action="store_true", help="Show detailed results")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--parallel", action="store_true", default=True, 
                   help="Enable parallel retrieval (default: True)")
    parser.add_argument("--no-parallel", action="store_false", dest="parallel",
                   help="Disable parallel retrieval")
    
    args = parser.parse_args()
    
    # Resolve query
    query = args.query or args.query_flag
    if not query:
        parser.error("❌ Query required (either as positional argument or --query flag)")
    
    if args.query and args.query_flag:
        TerminalUI.warn(f"Both positional and --query provided. Using positional: '{args.query}'")
        query = args.query
    
    handle_hybrid_search(HybridSearchArgs(
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
        parallel=args.parallel
    ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())