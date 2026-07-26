"""
Search command handler for darwin-admin.

Commands: search (retrieval with auto-save to CWD)
"""



from src.admin.lib import SearchArgs, ns_to
from src.admin.renderer import _r


def _str2bool(v):
    """Convert string to bool for argparse (true/false/1/0/yes/no)."""
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise ValueError(f"Boolean value expected, got {v!r}")


def cmd_search(args: SearchArgs, _get_store) -> None:
    from pathlib import Path
    from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
    from src.core.Orchestrators.lib import RetrievalParameters

    if not args.query:
        _r.err("Query is required")
        return

    store = _get_store(args.store)
    orch = RetrievalOrchestrator(store=store)

    kwargs = {"query": args.query}
    if args.method is not None:
        kwargs["method"] = args.method
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.max_passages is not None:
        kwargs["max_passages"] = args.max_passages
    if args.max_chars is not None:
        kwargs["max_chars"] = args.max_chars
    if args.min_passage_score is not None:
        kwargs["min_passage_score"] = args.min_passage_score
    if args.merge_strategy is not None:
        kwargs["merge_strategy"] = args.merge_strategy
    if args.files is not None:
        kwargs["files"] = args.files
    if args.rerank:
        kwargs["rerank"] = True
    if args.structural_penalty_factor is not None:
        kwargs["structural_penalty_factor"] = args.structural_penalty_factor
    if args.diversity_rerank is not None:
        kwargs["diversity_rerank"] = args.diversity_rerank
    if args.include_audit is not None:
        kwargs["include_audit"] = args.include_audit

    _should_save = not args.no_save
    if _should_save:
        kwargs["save"] = True
        if args.results_dir is not None:
            kwargs["results_dir"] = args.results_dir
        else:
            kwargs["results_dir"] = str(Path.cwd())

    params = RetrievalParameters(**kwargs)

    _r.header(f"Search: {args.query}")
    _r.info(f"Method: {args.method} | Top K: {args.top_k} | Max passages: {args.max_passages}")
    if _should_save:
        save_path = args.results_dir or str(Path.cwd())
        _r.info(f"Save: {save_path}")
    _r._out()

    ctx = orch.retrieve_and_resolve(params)

    if ctx is None or ctx.is_empty():
        _r.warn("No results found")
        return

    _r.ok(f"Found {ctx.total_passages} passage(s) in {ctx.elapsed_ms}ms")
    _r._out(f"  Max score: {ctx.max_score:.3f} | Avg score: {ctx.avg_score:.3f}")
    _r._out(f"  Sources: {', '.join(ctx.sources_used) if ctx.sources_used else 'none'}")
    if ctx.fallback_used:
        _r.warn(f"  Fallback used: {ctx.fallback_reason}")
    _r._out()

    for i, p in enumerate(ctx.passages, 1):
        meta = p.metadata
        _r._out(f"  [{i}] {meta.source} (score: {meta.avg_score:.3f}, pages {meta.page_start}-{meta.page_end})")
        text_preview = p.text[:200].replace("\n", " ")
        _r._out(f"      {text_preview}...")
        _r._out()

    if _should_save:
        _r.ok("Results saved to current directory")


def add_args(sub, _get_store) -> None:
    """Register the search subcommand."""
    p = sub.add_parser("search", help="Search the knowledge base")
    p.add_argument("query", help="Search query")
    p.add_argument("--method", choices=["hybrid", "semantic", "keyword"], default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-passages", type=int, default=None)
    p.add_argument("--max-chars", type=int, default=None)
    p.add_argument("--min-score", type=float, default=None, dest="min_passage_score")
    p.add_argument("--merge-strategy", choices=["none", "adjacent", "aggressive"], default=None)
    p.add_argument("--files", "-f", nargs="+", help="Restrict to specific files")
    p.add_argument("--store", help="Store ID, name, or path")
    p.add_argument("--no-save", action="store_true", help="Disable auto-save to current directory")
    p.add_argument("--results-dir", help="Override save directory (default: current directory)")
    p.add_argument("--rerank", action="store_true", help="Enable reranking")
    p.add_argument("--structural-penalty-factor", type=float, default=None,
                   help="Penalty for non-body content (0.0-1.0, default: 0.5). Lower = harsher penalty.")
    p.add_argument("--diversity-rerank", type=_str2bool, default=None, nargs="?",
                   const=True, help="Enable source diversity penalty (default: true). Set false for deep dives into one file.")
    p.add_argument("--include-audit", type=_str2bool, default=None, nargs="?",
                   const=True, help="Include per-passage scoring breakdown (default: false)")
    p.set_defaults(func=lambda a: cmd_search(ns_to(SearchArgs, a), _get_store))
