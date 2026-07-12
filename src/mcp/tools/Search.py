import traceback
from typing import Optional

from mcp.types import ToolAnnotations

from src.core.Orchestrators.lib import RetrievalParameters, resolve_store
from src.core.Orchestrators.lib import Method, MergeStrategy, FallbackStrategy

_MAX_QUERY_LEN = 4096


def _validate_query(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if len(query) > _MAX_QUERY_LEN:
        raise ValueError(f"query exceeds {_MAX_QUERY_LEN} chars")
    return query


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
    def search_darwin(
        query: str,
        method: Method = "hybrid",
        store: Optional[str] = None,
        files: Optional[str] = None,
        top_k: int = 5,
        max_passages: int = 5,
        max_chars: int = 6000,
        min_passage_score: float = 0.3,
        merge_strategy: MergeStrategy = "adjacent",
        fill_gaps: bool = False,
        max_per_source: int = 2,
        prefer_sources: Optional[str] = None,
        exclude_sources: Optional[str] = None,
        filter_structural: bool = True,
        fallback_strategy: FallbackStrategy = "both",
        rerank: bool = False,
    ) -> dict:
        try:
            query = _validate_query(query)
            params = RetrievalParameters(
                query=query,
                method=method,
                store=store,
                files=[files] if files else None,
                top_k=top_k,
                max_passages=max_passages,
                max_chars=max_chars,
                min_passage_score=min_passage_score,
                merge_strategy=merge_strategy,
                fill_gaps=fill_gaps,
                max_per_source=max_per_source,
                prefer_sources=[prefer_sources] if prefer_sources else None,
                exclude_sources=[exclude_sources] if exclude_sources else None,
                filter_structural=filter_structural,
                fallback_strategy=fallback_strategy,
                rerank=rerank,
            )
            if store:
                from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
                resolved = resolve_store(store)
                resolved.bootstrap(quiet=True)
                ctx = RetrievalOrchestrator(store=resolved).retrieve_and_resolve(params)
            else:
                ctx = retrieval_orch.retrieve_and_resolve(params)
            if ctx is None or ctx.is_empty():
                return {"success": True, "data": {"passages": [], "total": 0}}

            return {
                "success": True,
                "data": {
                    "passages": [p.to_dict() for p in ctx.passages],
                    "total": ctx.total_passages,
                    "total_chars": ctx.total_chars,
                    "sources": ctx.sources_used,
                    "max_score": ctx.max_score,
                    "avg_score": ctx.avg_score,
                    "elapsed_ms": ctx.elapsed_ms,
                    "fallback_used": ctx.fallback_used,
                    "fallback_reason": ctx.fallback_reason,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e), "detail": traceback.format_exc()}
