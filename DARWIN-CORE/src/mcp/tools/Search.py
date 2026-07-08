import json
import traceback
from typing import Optional

from src.core.Retrival.OrchestratorTypes import (
    OrchestratorConfig,
    MergeStrategy,
    FallbackStrategy,
)


def register(mcp, retrieval_orch):
    @mcp.tool()
    def search_darwin(
        query: str,
        method: str = "hybrid",
        files: Optional[str] = None,
        top_k: int = 5,
        max_passages: int = 5,
        max_chars: int = 6000,
        min_passage_score: float = 0.3,
        merge_strategy: str = "adjacent",
        fill_gaps: bool = False,
        max_per_source: int = 2,
        prefer_sources: Optional[str] = None,
        exclude_sources: Optional[str] = None,
        filter_structural: bool = True,
        fallback_strategy: str = "both",
        rerank: bool = False,
    ) -> dict:
        try:
            cfg = OrchestratorConfig(
                query=query,
                method=method,
                files=[files] if files else None,
                top_k=top_k,
                max_passages=max_passages,
                max_chars=max_chars,
                min_passage_score=min_passage_score,
                merge_strategy=MergeStrategy(merge_strategy),
                fill_gaps=fill_gaps,
                max_per_source=max_per_source,
                prefer_sources=[prefer_sources] if prefer_sources else None,
                exclude_sources=[exclude_sources] if exclude_sources else None,
                filter_structural=filter_structural,
                fallback_strategy=FallbackStrategy(fallback_strategy),
                rerank=rerank,
            )
            ctx = retrieval_orch.retrieve_and_resolve(cfg)
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
