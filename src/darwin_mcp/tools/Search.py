import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations

from core.Orchestrators.lib import RetrievalParameters, resolve_store
from core.Logger.DarwinLogger import darwin
from darwin_mcp.logging_bridge import darwin_bridge

_MAX_QUERY_LEN = 4096

_orch_cache: dict = {}  # store_key -> RetrievalOrchestrator


def _filter_passage(p: dict) -> dict:
    """Strip clean, audit, token_count from a passage dict."""
    meta = p.get("metadata", {})
    for k in ("clean", "token_count"):
        meta.pop(k, None)
    p["metadata"] = meta
    p.pop("audit", None)
    return p


def _filter_context(data: dict) -> dict:
    """Strip context-level bloat from a RetrievalContext dict."""
    for k in ("candidates_found", "penalized_count", "source_diversity", "audit_summary", "raw_results"):
        data.pop(k, None)
    if "passages" in data:
        data["passages"] = [_filter_passage(p) for p in data["passages"]]
    return data


def _validate_query(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if len(query) > _MAX_QUERY_LEN:
        raise ValueError(f"query exceeds {_MAX_QUERY_LEN} chars")
    return query


def _get_results_dir() -> Path:
    from lib.AppPaths import set_darwin_base_dir
    return set_darwin_base_dir() / "search-results"


def register(mcp, retrieval_orch):
    @mcp.tool(
        annotations=ToolAnnotations(
            title="Search Darwin",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def search_darwin(
        ctx: Context,
        query: str,
        method: Optional[str] = None,
        store: Optional[str] = None,
        files: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        max_passages: Optional[int] = None,
        max_chars: Optional[int] = None,
        min_passage_score: Optional[float] = None,
        merge_strategy: Optional[str] = None,
        fill_gaps: Optional[bool] = None,
        max_per_source: Optional[int] = None,
        prefer_sources: Optional[list[str]] = None,
        exclude_sources: Optional[list[str]] = None,
        filter_structural: Optional[bool] = None,
        structural_penalty_factor: Optional[float] = None,
        diversity_rerank: Optional[bool] = None,
        include_audit: Optional[bool] = None,
        fallback_strategy: Optional[str] = None,
        rerank: Optional[bool] = None,
        save: Optional[bool] = None,
        results_dir: Optional[str] = None,
        filter: Optional[bool] = None,
    ) -> dict:
        """Search the DARWIN knowledge base for relevant passages.

        Parameters with enums (pass exact string):
          method: "hybrid" (default) | "semantic" | "keyword"
          merge_strategy: "adjacent" (default) | "none" | "aggressive"
          fallback_strategy: "both" (default) | "expand" | "relax"

        Defaults (applied by RetrievalParameters if not provided):
          top_k: 5, max_passages: 5, max_chars: 6000,
          min_passage_score: 0.15, fill_gaps: false, max_per_source: 2,
          filter_structural: true, structural_penalty_factor: 0.5,
          diversity_rerank: true, include_audit: false, rerank: false

        Usage tips:
        - Check 'summary' in the result for a quick overview before reading passages.
        - method="hybrid" (default) combines keyword + semantic; best for most queries.
        - method="semantic" is better for conceptual/fuzzy queries.
        - method="keyword" is better for exact terms or code references.
        - If results are sparse, try lowering min_passage_score (e.g. 0.15).
        - fallback_used=true means hybrid search broadened — check fallback_reason.
        - Pass store name/path/ID to search a specific store.
        - Use prefer_sources/exclude_sources to target or skip specific files.
        - save=true persists results as JSON; fetch later via darwin://search/results resource.
        - include_audit=true adds per-passage scoring breakdown (raw_score, penalty_factor, etc.).
        - filter=true strips clean, audit, token_count from passages (text is canonical).
        - diversity_rerank=false disables source diversity penalty — useful for deep dives into one file.
        - structural_penalty_factor controls how aggressively non-body content is penalized. Lower = harsher.
        - max_per_source caps passages per file. Raise for deeper coverage, lower for more diversity.
        """
        async with darwin_bridge(ctx):
            try:
                query = _validate_query(query)
                kwargs = {"query": query}
                if method is not None:
                    kwargs["method"] = method
                if store is not None:
                    kwargs["store"] = store
                if files is not None:
                    kwargs["files"] = files
                if top_k is not None:
                    kwargs["top_k"] = top_k
                if max_passages is not None:
                    kwargs["max_passages"] = max_passages
                if max_chars is not None:
                    kwargs["max_chars"] = max_chars
                if min_passage_score is not None:
                    kwargs["min_passage_score"] = min_passage_score
                if merge_strategy is not None:
                    kwargs["merge_strategy"] = merge_strategy
                if fill_gaps is not None:
                    kwargs["fill_gaps"] = fill_gaps
                if max_per_source is not None:
                    kwargs["max_per_source"] = max_per_source
                if prefer_sources is not None:
                    kwargs["prefer_sources"] = prefer_sources
                if exclude_sources is not None:
                    kwargs["exclude_sources"] = exclude_sources
                if filter_structural is not None:
                    kwargs["filter_structural"] = filter_structural
                if structural_penalty_factor is not None:
                    kwargs["structural_penalty_factor"] = structural_penalty_factor
                if diversity_rerank is not None:
                    kwargs["diversity_rerank"] = diversity_rerank
                if include_audit is not None:
                    kwargs["include_audit"] = include_audit
                if fallback_strategy is not None:
                    kwargs["fallback_strategy"] = fallback_strategy
                if rerank is not None:
                    kwargs["rerank"] = rerank
                if save is not None:
                    kwargs["save"] = save
                if results_dir is not None:
                    kwargs["results_dir"] = results_dir
                params = RetrievalParameters(**kwargs)
                _filter = filter is True

                def _blocking():
                    if store:
                        if store not in _orch_cache:
                            from core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
                            resolved = resolve_store(store)
                            resolved.bootstrap(quiet=True)
                            _orch_cache[store] = RetrievalOrchestrator(store=resolved)
                        return _orch_cache[store].retrieve_and_resolve(params)
                    return retrieval_orch.retrieve_and_resolve(params)

                import asyncio
                ctx_result = await asyncio.get_event_loop().run_in_executor(None, _blocking)
                if ctx_result is None or ctx_result.is_empty():
                    return {"success": True, "data": {"passages": [], "total": 0}}

                passages = [p.to_dict() for p in ctx_result.passages]
                if _filter:
                    passages = [_filter_passage(p) for p in passages]

                return {
                    "success": True,
                    "data": {
                        "passages": passages,
                        "total": ctx_result.total_passages,
                        "total_chars": ctx_result.total_chars,
                        "sources": ctx_result.sources_used,
                        "max_score": ctx_result.max_score,
                        "avg_score": ctx_result.avg_score,
                        "elapsed_ms": ctx_result.elapsed_ms,
                        "fallback_used": ctx_result.fallback_used,
                        "fallback_reason": ctx_result.fallback_reason,
                    },
                }
            except Exception as e:
                darwin.error(f"search_darwin failed: {type(e).__name__}: {e}")
                return {"success": False, "error": str(e)}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Search Results",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def get_search_results(
        ctx: Context,
        results_dir: Optional[str] = None,
    ) -> dict:
        """List all saved search results.

        Returns a list of saved result files with metadata (query, method,
        passage count, max score, timestamp). This listing is already minimal —
        no filter param needed. Use get_search_result_by_id to load a specific
        result (pass filter=true for a stripped-down view).

        Tip: The darwin://search/results resource provides the same data
        without a tool call — use it for quick context lookups.
        """
        async with darwin_bridge(ctx):
            try:
                import asyncio

                def _blocking():
                    search_dir = Path(results_dir) if results_dir else _get_results_dir()
                    if not search_dir.exists():
                        return {"success": True, "data": {"results": [], "total": 0}}

                    entries = []
                    for f in sorted(search_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                        try:
                            raw = f.read_text(encoding="utf-8")
                            data = json.loads(raw)
                            entries.append({
                                "filename": f.name,
                                "query": data.get("query", ""),
                                "method": data.get("method", ""),
                                "total_passages": data.get("total_passages", 0),
                                "max_score": data.get("max_score", 0.0),
                                "elapsed_ms": data.get("elapsed_ms", 0),
                                "fallback_used": data.get("fallback_used", False),
                                "saved_at": data.get("_saved_at", ""),
                            })
                        except Exception:
                            continue
                    return {"success": True, "data": {"results": entries, "total": len(entries)}}

                return await asyncio.get_event_loop().run_in_executor(None, _blocking)
            except Exception as e:
                darwin.error(f"get_search_results failed: {type(e).__name__}: {e}")
                return {"success": False, "error": str(e)}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Search Result By ID",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def get_search_result_by_id(
        ctx: Context,
        filename: str,
        results_dir: Optional[str] = None,
        filter: Optional[bool] = None,
    ) -> dict:
        """Load a saved search result by its filename.

        Pass the filename from get_search_results (e.g. "machine-learning_20260718_143022.json").
        Returns the full RetrievalContext with passages, metadata, and scores.
        If not found, returns an error suggesting you run search_darwin with save=true.
        Pass filter=true to strip bloat (clean, audit, candidates_found, raw_results, etc.).
        """
        async with darwin_bridge(ctx):
            try:
                import asyncio

                def _blocking():
                    search_dir = Path(results_dir) if results_dir else _get_results_dir()
                    # Sanitize filename: strip directories, reject traversal
                    safe_name = Path(filename).name
                    if safe_name != filename or ".." in filename or "/" in filename or "\\" in filename:
                        return {"success": False, "error": f"Invalid filename: {filename!r}"}
                    filepath = search_dir / safe_name
                    if not filepath.exists() or not filepath.is_file():
                        return {
                            "success": False,
                            "error": (
                                f"Result '{filename}' not found in {search_dir}. "
                                "Run search_darwin with save=true to create one."
                            ),
                        }
                    try:
                        data = json.loads(filepath.read_text(encoding="utf-8"))
                        if filter is True:
                            data = _filter_context(data)
                        return {"success": True, "data": data}
                    except Exception as e:
                        return {"success": False, "error": f"Failed to read {filename}: {e}"}

                return await asyncio.get_event_loop().run_in_executor(None, _blocking)
            except Exception as e:
                darwin.error(f"get_search_result_by_id failed: {type(e).__name__}: {e}")
                return {"success": False, "error": str(e)}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Search Structured Lists",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def search_lists(
        ctx: Context,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = "auto",
        store: Optional[str] = None,
        files: Optional[List[str]] = None,
        top_k: Optional[int] = 5,
        rerank: Optional[bool] = False,
        filter_boost_factor: Optional[float] = 0.2,
        save: Optional[bool] = False,
    ) -> dict:
        """Search structured data (JSON arrays, CSVs) by field values with hybrid text fallback.

        This tool routes through RetrievalOrchestrator.search_lists(), following the same
        orchestration pattern as search_darwin. It handles store resolution, logging,
        session management, and result formatting.

        Parameters:
          query: Text query for hybrid fallback (optional if filters provided).
          filters: Dict of field-value pairs to filter by (e.g. {"status": "failed"}).
          mode: "auto" (try structured, fallback to text), "structured" (filter only),
                or "text" (hybrid only).
          store: Store name/path/ID (optional, uses default if not provided).
          files: Limit to specific files (optional).
          top_k: Max results to return (default 5).
          rerank: Enable reranking on results (default false).
          filter_boost_factor: Score boost for filter matches (default 0.2).
          save: Save results to disk (default false).

        Returns:
          When structured match found:
            {success: true, mode: "structured", records: [{fields, score, source, ...}], total, warning: null}
          When falling back to text:
            {success: true, mode: "text", passages: [...], total, warning: "No structured match..."}
          When no results:
            {success: true, mode: "structured", records: [], total: 0, warning: "No records matched..."}
        """
        async with darwin_bridge(ctx):
            try:
                from core.Orchestrators.lib import StructuredSearchParams

                def _blocking():
                    # Resolve store
                    if store:
                        from core.Orchestrators.lib import resolve_store
                        resolved_store = resolve_store(store)
                        resolved_store.bootstrap(quiet=True)
                    else:
                        resolved_store = retrieval_orch.store

                    # ── Pre-validate filters ────────────────────────────
                    validation_warnings = []
                    validated_filters = filters
                    if filters:
                        report = resolved_store.validate_filters(filters, files)
                        validated_filters = report.get("valid", filters)
                        validation_warnings = report.get("warnings", [])

                        if not validated_filters:
                            return {
                                "success": True,
                                "mode": "structured",
                                "records": [],
                                "passages": [],
                                "total": 0,
                                "warning": (
                                    "No valid filters after validation. "
                                    + "; ".join(validation_warnings)
                                ),
                                "validation": report,
                            }

                    from core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
                    orch = RetrievalOrchestrator(store=resolved_store)

                    params = StructuredSearchParams(
                        query=query or "",
                        filters=validated_filters,
                        mode=mode or "auto",
                        top_k=top_k or 5,
                        files=files,
                        filter_boost_factor=filter_boost_factor or 0.2,
                        rerank=rerank or False,
                        save=save or False,
                    )

                    ctx_result = orch.search_lists(params)

                    if ctx_result is None or ctx_result.is_empty():
                        warning = None
                        if validated_filters:
                            if mode == "structured":
                                warning = "No records matched the specified filters"
                            else:
                                warning = "No structured match found — no text results either"
                        if validation_warnings:
                            warning = (warning + "; " if warning else "") + "; ".join(validation_warnings)
                        return {
                            "success": True,
                            "mode": "structured" if validated_filters and mode != "text" else "text",
                            "records": [],
                            "passages": [],
                            "total": 0,
                            "warning": warning,
                            "validation": {"warnings": validation_warnings} if validation_warnings else None,
                        }

                    # Determine mode from context
                    ctx_mode = "structured" if "structured" in (ctx_result.method or "") else "text"

                    if ctx_mode == "structured":
                        # Extract fields from structured passages
                        records = []
                        for p in ctx_result.passages:
                            meta = p.metadata
                            fields = getattr(meta, "_structured_fields", {})
                            records.append({
                                "fields": fields,
                                "score": meta.avg_score,
                                "source": meta.source,
                                "chunk_index": meta.chunk_indices[0] if meta.chunk_indices else 0,
                                "heading": meta.heading or "",
                            })
                        result = {
                            "success": True,
                            "mode": "structured",
                            "records": records,
                            "passages": [],
                            "total": ctx_result.total_passages,
                            "warning": ctx_result.fallback_reason,
                        }
                    else:
                        # Text fallback — return passages
                        passages = [p.to_dict() for p in ctx_result.passages]
                        result = {
                            "success": True,
                            "mode": "text",
                            "records": [],
                            "passages": passages,
                            "total": ctx_result.total_passages,
                            "warning": ctx_result.fallback_reason,
                        }

                    if validation_warnings:
                        result["validation"] = {"warnings": validation_warnings}
                    return result

                import asyncio
                return await asyncio.get_event_loop().run_in_executor(None, _blocking)
            except Exception as e:
                darwin.error(f"search_lists failed: {type(e).__name__}: {e}")
                return {"success": False, "error": str(e)}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Structured Schema",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def get_schema(
        ctx: Context,
        file: Optional[str] = None,
        files: Optional[List[str]] = None,
        list_all: Optional[bool] = None,
        validate_filters: Optional[Dict[str, Any]] = None,
        store: Optional[str] = None,
    ) -> dict:
        """Get schema information for structured files (JSON arrays, CSVs).

        Three modes of operation:

        1. Single file: file="tickets.csv"
           Returns schema for one file (keys, types, record_count).

        2. Multiple files: files=["tickets.csv", "orders.json"]
           Returns schemas mapped to each file. Also detects overlapping keys
           (keys that appear in multiple files with potentially different meanings).

        3. List all: list_all=true
           Returns schemas for ALL indexed structured files in the store.

        4. Validate filters: validate_filters={"status": "open", "priority": "high"}
           Validates filter keys/types against schemas before searching.
           Returns valid filters, warnings about missing keys or type mismatches,
           and overlapping key detection.

        If the file is not structured or not found, returns an error.
        """
        async with darwin_bridge(ctx):
            try:
                def _blocking():
                    resolved_store = None
                    if store:
                        resolved_store = resolve_store(store)
                        resolved_store.bootstrap(quiet=True)
                    else:
                        resolved_store = retrieval_orch.store

                    # ── Validate filters mode ──────────────────────────
                    if validate_filters is not None:
                        report = resolved_store.validate_filters(
                            validate_filters, files
                        )
                        return {
                            "success": True,
                            "data": report,
                        }

                    # ── List all mode ──────────────────────────────────
                    if list_all:
                        all_schemas = resolved_store.get_all_schemas()
                        overlapping = resolved_store.find_overlapping_keys()
                        return {
                            "success": True,
                            "data": {
                                "schemas": all_schemas,
                                "total_files": len(all_schemas),
                                "overlapping_keys": overlapping,
                            },
                        }

                    # ── Multi-file mode ────────────────────────────────
                    if files:
                        schemas = {}
                        missing = []
                        for f in files:
                            s = resolved_store.load_schema(f)
                            if s is None:
                                missing.append(f)
                            else:
                                schemas[f] = s

                        overlapping = resolved_store.find_overlapping_keys()
                        # Only report overlapping for keys in these files
                        all_keys = set()
                        for s in schemas.values():
                            all_keys.update(s.get("keys", []))
                        relevant_overlapping = {
                            k: v for k, v in overlapping.items()
                            if k in all_keys
                        }

                        result = {
                            "schemas": schemas,
                            "found": len(schemas),
                            "missing": missing,
                            "overlapping_keys": relevant_overlapping,
                        }
                        if missing:
                            result["warning"] = (
                                f"No schemas found for: {missing}. "
                                "These files may not be indexed in structured mode."
                            )
                        return {"success": True, "data": result}

                    # ── Single file mode (default) ─────────────────────
                    if not file:
                        return {
                            "success": False,
                            "error": "Provide file, files, list_all=true, or validate_filters",
                        }

                    schema = resolved_store.load_schema(file)
                    if schema is None:
                        return {
                            "success": False,
                            "error": (
                                f"No schema found for '{file}'. "
                                "File may not be indexed in structured mode."
                            ),
                        }
                    return {"success": True, "data": schema}

                import asyncio
                return await asyncio.get_event_loop().run_in_executor(None, _blocking)
            except Exception as e:
                darwin.error(f"get_schema failed: {type(e).__name__}: {e}")
                return {"success": False, "error": str(e)}
