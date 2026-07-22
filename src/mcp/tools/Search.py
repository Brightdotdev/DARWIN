import json
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations

from src.core.Orchestrators.lib import RetrievalParameters, resolve_store
from src.core.Logger.DarwinLogger import darwin
from src.mcp.logging_bridge import darwin_bridge

_MAX_QUERY_LEN = 4096

_orch_cache: dict = {}  # store_key -> RetrievalOrchestrator


def _validate_query(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if len(query) > _MAX_QUERY_LEN:
        raise ValueError(f"query exceeds {_MAX_QUERY_LEN} chars")
    return query


def _get_results_dir() -> Path:
    from src.lib.AppPaths import set_darwin_base_dir
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
    ) -> dict:
        """Search the DARWIN knowledge base for relevant passages.

        Defaults (applied by RetrievalParameters if not provided):
        - method: "hybrid"
        - top_k: 5
        - max_passages: 5
        - max_chars: 6000
        - min_passage_score: 0.15
        - merge_strategy: "adjacent"
        - fill_gaps: false
        - max_per_source: 2
        - filter_structural: true
        - structural_penalty_factor: 0.5
        - diversity_rerank: true
        - include_audit: false
        - fallback_strategy: "both"
        - rerank: false

        Usage tips:
        - Check 'summary' in the result for a quick overview before reading passages.
        - method='hybrid' (default) combines keyword + semantic; best for most queries.
        - method='semantic' is better for conceptual/fuzzy queries.
        - method='keyword' is better for exact terms or code references.
        - If results are sparse, try lowering min_passage_score (e.g. 0.15).
        - fallback_used=true means hybrid search broadened — check fallback_reason.
        - Pass store name/path/ID to search a specific store.
        - Use prefer_sources/exclude_sources to target or skip specific files.
        - save=true persists results as JSON for later retrieval via get_search_results.
        - include_audit=true adds per-passage scoring breakdown (raw_score, penalty_factor, etc.).
        - diversity_rerank=false disables source diversity penalty — useful for deep dives into one file.
        - structural_penalty_factor controls how aggressively non-body content (tables, captions) is penalized. Lower = harsher.
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

                def _blocking():
                    if store:
                        if store not in _orch_cache:
                            from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
                            resolved = resolve_store(store)
                            resolved.bootstrap(quiet=True)
                            _orch_cache[store] = RetrievalOrchestrator(store=resolved)
                        return _orch_cache[store].retrieve_and_resolve(params)
                    return retrieval_orch.retrieve_and_resolve(params)

                import asyncio
                ctx_result = await asyncio.get_event_loop().run_in_executor(None, _blocking)
                if ctx_result is None or ctx_result.is_empty():
                    return {"success": True, "data": {"passages": [], "total": 0}}

                return {
                    "success": True,
                    "data": {
                        "passages": [p.to_dict() for p in ctx_result.passages],
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
        passage count, max score, timestamp). Use get_search_result_by_id
        to load a specific result.
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
    ) -> dict:
        """Load a saved search result by its filename.

        Pass the filename from get_search_results (e.g. "machine-learning_20260718_143022.json").
        Returns the full RetrievalContext with passages, metadata, and scores.
        If not found, returns an error suggesting you run search_darwin with save=true.
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
                        return {"success": True, "data": data}
                    except Exception as e:
                        return {"success": False, "error": f"Failed to read {filename}: {e}"}

                return await asyncio.get_event_loop().run_in_executor(None, _blocking)
            except Exception as e:
                darwin.error(f"get_search_result_by_id failed: {type(e).__name__}: {e}")
                return {"success": False, "error": str(e)}
