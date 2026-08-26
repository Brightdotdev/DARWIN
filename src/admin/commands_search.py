"""
Search command handler for darwin-admin.

Commands: search, search-lists
"""



from admin.lib import SearchArgs, SearchListsArgs, ns_to
from admin.renderer import _r


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
    from core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
    from core.Orchestrators.lib import RetrievalParameters

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


def _parse_filters(raw_filters: list[str] | None) -> dict | None:
    """Parse CLI filter args into a dict.

    Accepts: ["status=open", "priority=high"] → {"status": "open", "priority": "high"}
    Values are auto-typed: int → int, float → float, else string.
    """
    if not raw_filters:
        return None
    filters = {}
    for item in raw_filters:
        if "=" not in item:
            _r.warn(f"Skipping invalid filter (no '='): {item}")
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        # Auto-type conversion
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # keep as string
        filters[key] = value
    return filters if filters else None


def cmd_search_lists(args: SearchListsArgs, _get_store) -> None:
    """Handle ``darwin-admin search-lists`` commands.

    Supports three modes:
      1. Schema inspection: --schema <file> [file2 ...] or --list-all
      2. Filter validation: --validate-filters status=open priority=high
      3. Structured search: --filters / --query  →  filter-by-fields with hybrid fallback
    """
    from core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
    from core.Orchestrators.lib import StructuredSearchParams

    store = _get_store(args.store)

    # ── List all schemas mode ─────────────────────────────────────────
    if args.list_all:
        all_schemas = store.get_all_schemas()
        overlapping = store.find_overlapping_keys()

        if not all_schemas:
            _r.warn("No structured files found in store")
            return

        _r.header(f"Structured Files ({len(all_schemas)})")
        for filename, schema in all_schemas.items():
            keys = schema.get("keys", [])
            types = schema.get("types", {})
            record_count = schema.get("record_count", 0)
            file_type = schema.get("file_type", "unknown")
            _r.ok(f"{filename}")
            _r._out(f"    Type: {file_type} | Records: {record_count} | Fields: {len(keys)}")
            _r._out(f"    Keys: {', '.join(keys)}")
            _r._out()

        if overlapping:
            _r.warn(f"Overlapping keys found across files:")
            for key, file_list in overlapping.items():
                _r._out(f"  '{key}' → {', '.join(file_list)}")
        return

    # ── Schema inspection mode ────────────────────────────────────────
    if args.schema_file:
        schema_files = args.schema_file if isinstance(args.schema_file, list) else [args.schema_file]

        for sf in schema_files:
            schema = store.load_schema(sf)
            if schema is None:
                _r.err(f"No schema found for '{sf}'")
                _r.info("File may not be indexed in structured mode.")
                _r.info("Run: darwin-admin pipeline run <file> --index-mode structured")
                continue

            _r.header(f"Schema: {sf}")
            _r.ok(f"Record count: {schema.get('record_count', 0)}")
            _r.info(f"File type: {schema.get('file_type', 'unknown')}")
            _r.info(f"Index mode: {schema.get('index_mode', 'unknown')}")
            _r._out()

            types = schema.get("types", {})
            keys = schema.get("keys", [])
            _r.info(f"Fields ({len(keys)}):")
            for key in keys:
                field_type = types.get(key, "unknown")
                _r._out(f"  {key}: {field_type}")
            _r._out()

        # If multiple files, check for overlapping keys
        if len(schema_files) > 1:
            overlapping = store.find_overlapping_keys()
            all_keys = set()
            for sf in schema_files:
                s = store.load_schema(sf)
                if s:
                    all_keys.update(s.get("keys", []))
            relevant = {k: v for k, v in overlapping.items() if k in all_keys}
            if relevant:
                _r.warn("Overlapping keys across selected files:")
                for key, file_list in relevant.items():
                    _r._out(f"  '{key}' → {', '.join(file_list)}")
        return

    # ── Validate filters mode ─────────────────────────────────────────
    if args.validate_filters:
        parsed_validate = _parse_filters(args.validate_filters) if isinstance(args.validate_filters, list) else args.validate_filters
        if not parsed_validate:
            _r.err("No valid filters to validate")
            return
        report = store.validate_filters(parsed_validate, args.files)
        _r.header("Filter Validation")
        if not report["schemas_found"]:
            _r.err("No structured schemas found")
            for w in report["warnings"]:
                _r._out(f"  {w}")
            return

        _r.info(f"Files checked: {', '.join(report['files_checked'])}")
        _r._out()

        if report["valid"]:
            _r.ok(f"Valid filters:")
            for k, v in report["valid"].items():
                _r._out(f"  {k} = {v}")
        _r._out()

        if report["warnings"]:
            _r.warn("Warnings:")
            for w in report["warnings"]:
                _r._out(f"  {w}")
        _r._out()

        if report["overlapping"]:
            _r.warn("Overlapping keys:")
            for k, files in report["overlapping"].items():
                _r._out(f"  '{k}' → {', '.join(files)}")
        return

    # ── Search mode ───────────────────────────────────────────────────
    if not args.query and not args.filters:
        _r.err("Provide --query and/or --filters, or use --schema/--list-all/--validate-filters")
        return

    orch = RetrievalOrchestrator(store=store)

    # Pre-validate filters
    validated_filters = args.filters
    if args.filters:
        report = store.validate_filters(args.filters, args.files)
        validated_filters = report.get("valid", args.filters)
        if report["warnings"]:
            _r.warn("Filter validation warnings:")
            for w in report["warnings"]:
                _r._out(f"  {w}")
            _r._out()
        if not validated_filters:
            _r.err("No valid filters after validation")
            return

    params = StructuredSearchParams(
        query=args.query,
        filters=validated_filters,
        mode=args.mode,
        top_k=args.top_k,
        files=args.files,
        filter_boost_factor=args.filter_boost_factor,
        rerank=args.rerank,
    )

    _r.header(f"Structured Search")
    if validated_filters:
        filter_str = " ".join(f"{k}={v}" for k, v in validated_filters.items())
        _r.info(f"Filters: {filter_str}")
    if args.query:
        _r.info(f"Query: '{args.query}' (fallback)")
    _r.info(f"Mode: {args.mode} | Top K: {args.top_k} | Boost: {args.filter_boost_factor}")
    if args.files:
        _r.info(f"Files: {', '.join(args.files)}")
    _r._out()

    ctx = orch.search_lists(params)

    if ctx is None or ctx.is_empty():
        _r.warn("No results found")
        if validated_filters:
            _r.info("Tip: check that the file is indexed in structured mode")
            _r.info("     and that the filter field names match the schema.")
        return

    # Determine mode from context
    is_structured = "structured" in (ctx.method or "")

    if is_structured:
        _r.ok(f"Found {ctx.total_passages} record(s) (structured mode)")
    else:
        _r.ok(f"Found {ctx.total_passages} passage(s) (text fallback)")
        if ctx.fallback_used:
            _r.warn(f"  Fallback reason: {ctx.fallback_reason}")
    _r._out(f"  Max score: {ctx.max_score:.3f} | Avg score: {ctx.avg_score:.3f}")
    _r._out(f"  Sources: {', '.join(ctx.sources_used) if ctx.sources_used else 'none'}")
    _r._out()

    for i, p in enumerate(ctx.passages, 1):
        meta = p.metadata
        fields = getattr(meta, "_structured_fields", {})
        if fields:
            # Structured record — show fields
            _r._out(f"  [{i}] {meta.source} (score: {meta.avg_score:.3f})")
            for k, v in fields.items():
                _r._out(f"      {k}: {v}")
        else:
            # Text passage — show preview
            _r._out(f"  [{i}] {meta.source} (score: {meta.avg_score:.3f}, pages {meta.page_start}-{meta.page_end})")
            text_preview = p.text[:200].replace("\n", " ")
            _r._out(f"      {text_preview}...")
        _r._out()


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

    # ── search-lists ──────────────────────────────────────────────
    sl = sub.add_parser("search-lists", help="Structured search (filter-by-fields with hybrid fallback)")
    sl.add_argument("--query", "-q", default="", help="Text query for hybrid fallback")
    sl.add_argument("--filters", "-F", nargs="+", metavar="KEY=VALUE",
                    help="Field filters: status=open priority=high")
    sl.add_argument("--mode", choices=["auto", "structured", "text"], default="auto",
                    help="Search mode (default: auto)")
    sl.add_argument("--files", "-f", nargs="+", help="Restrict to specific files")
    sl.add_argument("--top-k", "-k", type=int, default=5, help="Max results (default: 5)")
    sl.add_argument("--filter-boost-factor", type=float, default=0.2,
                    help="Score boost for filter matches (default: 0.2)")
    sl.add_argument("--rerank", action="store_true", help="Enable reranking")
    sl.add_argument("--store", help="Store ID, name, or path")
    sl.add_argument("--schema", dest="schema_file", nargs="+", metavar="FILE",
                    help="Show schema for one or more structured files")
    sl.add_argument("--list-all", action="store_true",
                    help="List all structured files and their schemas")
    sl.add_argument("--validate-filters", nargs="+", metavar="KEY=VALUE",
                    help="Validate filter keys/types without searching")
    sl.set_defaults(func=lambda a: cmd_search_lists(ns_to(SearchListsArgs, a), _get_store))
