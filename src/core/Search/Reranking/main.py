"""
src/core/Search/Reranking/main.py - Reranking CLI

Test reranking on search results. Useful for comparing before/after.

Usage examples
--------------
# Rerank across all documents
uv run src/core/Search/Reranking/main.py "photosynthesis" --search-type semantic

# Rerank within a specific file
uv run src/core/Search/Reranking/main.py "david covenant" --file bible.pdf --search-type hybrid

# Rerank with fast model on single file
uv run src/core/Search/Reranking/main.py "covenant" --file bible.pdf --model fast --inspect
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.core.Search.Reranking.Reranker import (
    rerank_semantic,
    rerank_keyword,
    rerank_hybrid,
    RerankedResult,
)
from src.core.Search.Reranking.RerankerUtils import (
    MODEL_OPTIONS,
    DEFAULT_TOP_K,
    DEFAULT_POOL_SIZE,
    format_reranked_results,
    to_llm_context,
    RerankingArgs
)
from src.core.Search.Sematic.SematicSearch import (
    search_single_document as semantic_single,
    search_across_documents as semantic_all,
)
from src.core.Search.Keyword.KeywordSearch import (
    search_single_document_keyword as keyword_single,
    search_across_documents_keyword as keyword_all,
)
from src.core.Search.Hybrid.HybridSearch import (
    search_single_document_hybrid as hybrid_single,
    search_across_documents_hybrid as hybrid_all,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


# ============================================================================
# Inspect
# ============================================================================

def _print_inspection(results: List[RerankedResult], query: str) -> None:
    """Print detailed inspection of reranked results."""
    TerminalUI.blank()
    TerminalUI.rule(f"reranking inspection — '{query}'")
    TerminalUI.blank()
    
    TerminalUI.table(
        headers=["metric", "value"],
        rows=[
            ["total results", str(len(results))],
            ["top final score", f"{results[0].final_score:.4f}" if results else "N/A"],
            ["top rerank score", f"{results[0].rerank_score:.4f}" if results else "N/A"],
        ],
    )
    
    TerminalUI.blank()
    TerminalUI.dim("── top results (before vs after) ──")
    TerminalUI.blank()
    
    for i, r in enumerate(results[:5], 1):
        TerminalUI.info("rank", str(i))
        TerminalUI.info("final score", f"{r.final_score:.4f}")
        TerminalUI.info("rerank score", f"{r.rerank_score:.4f}")
        if r.original_score:
            TerminalUI.info("original score", f"{r.original_score:.4f}")
        TerminalUI.info("source", f"{r.source} (p{r.page})")
        TerminalUI.info("heading", r.heading or "(none)")
        TerminalUI.blank()
        
        preview = r.text[:300].replace('\n', ' ').strip()
        TerminalUI.dim(f"  {preview}...")
        TerminalUI.blank()
        TerminalUI.divider()
        TerminalUI.blank()
    
    TerminalUI.rule()


# ============================================================================
# Core handler
# ============================================================================

def handle_reranking(args : RerankingArgs) -> None:
    store = DarwinStore(Path(args.project) if args.project else Path.cwd())
    
    TerminalUI.print_darwin()
    TerminalUI.blank()
    TerminalUI.bold(f"Reranking: '{args.query}'")
    if args.file:
        TerminalUI.dim(f"File: {args.file}")
    TerminalUI.dim(f"Model: {args.model} | Top-K: {args.top_k} | Pool: {args.pool_size}")
    TerminalUI.blank()
    
    # Get candidates based on search type and file
    if args.search_type == "semantic":
        if args.file:
            candidates = semantic_single(args.query, args.file, store, top_k=args.pool_size)
        else:
            candidates = semantic_all(args.query, store, top_k=args.pool_size)
        results = rerank_semantic(args.query, candidates, args.model, args.top_k)
        
    elif args.search_type == "keyword":
        if args.file:
            candidates = keyword_single(args.query, args.file, store, top_k=args.pool_size)
        else:
            candidates = keyword_all(args.query, store, top_k=args.pool_size)
        results = rerank_keyword(args.query, candidates, args.model, args.top_k)
        
    else:  # hybrid
        if args.file:
            candidates = hybrid_single(
                args.query, args.file, store, top_k=args.pool_size
            )
        else:
            candidates = hybrid_all(args.query, store, top_k=args.pool_size)
        results = rerank_hybrid(args.query, candidates, args.model, args.top_k)
    
    if not results:
        TerminalUI.warn("No results found")
        return
    
    TerminalUI.success(f"Found {len(results)} results after reranking")
    TerminalUI.blank()
    
    # Table view
    table_rows = []
    for i, r in enumerate(results[:10], 1):
        preview = r.text[:50].replace('\n', ' ')
        table_rows.append([
            str(i),
            f"{r.final_score:.3f}",
            f"{r.rerank_score:.3f}",
            r.source[:20],
            str(r.page),
            preview + "..."
        ])
    
    TerminalUI.table(
        headers=["#", "Final", "Rerank", "Source", "Page", "Preview"],
        rows=table_rows,
        title="reranked results"
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
        print(format_reranked_results(results, json_output=True))


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-encoder reranking for improved search precision"
    )
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--file", "-f", default=None, help="Search within specific file")
    parser.add_argument("--project", "-p", default=None, help="Project directory")
    parser.add_argument("--model", "-m", default="balanced", 
                       choices=["fast", "balanced", "accurate"],
                       help="Reranking model (fast/balanced/accurate)")
    parser.add_argument("--search-type", "-s", default="hybrid",
                       choices=["semantic", "keyword", "hybrid"],
                       help="Search method to rerank")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K,
                       help="Final results to return")
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE,
                       help="Candidates to fetch before reranking")
    parser.add_argument("--inspect", "-i", action="store_true",
                       help="Show detailed before/after comparison")
    parser.add_argument("--json", "-j", action="store_true",
                       help="Output as JSON")
    
    args = parser.parse_args()
    
    # Resolve model shortcut
    args.model = MODEL_OPTIONS.get(args.model, args.model)
    
    handle_reranking(RerankingArgs(
        query=args.query,
        file=args.file,
        project=args.project,
        model=args.model,
        search_type=args.search_type,
        top_k=args.top_k,
        pool_size=args.pool_size,
        inspect=args.inspect,
        json_output=args.json,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())