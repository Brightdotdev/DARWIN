"""
MCP Resources — expose Darwin data as readable resources.

Resources are read-only data sources that MCP clients can list and fetch.
Unlike tools (which execute actions), resources provide context for the agent.
"""

import json


def register(mcp, store, models_orchestrator=None):
    """Register MCP resources on a FastMCP instance."""

    @mcp.resource("darwin://store/files")
    def store_files() -> str:
        """List all files tracked by the current store."""
        try:
            report = store.get_pipeline_report()
            return json.dumps({
                "files": report,
                "total": len(report) if isinstance(report, list) else 0,
            }, default=str, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("darwin://store/report")
    def store_report() -> str:
        """Full pipeline report for the current store."""
        try:
            report = store.get_pipeline_report()
            return json.dumps(report, default=str, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("darwin://store/stats")
    def store_stats() -> str:
        """Store statistics: file count, sizes, stage status."""
        try:
            report = store.get_pipeline_report()
            files = report if isinstance(report, list) else []
            total_files = len(files)
            stages_seen = set()
            for f in files:
                if isinstance(f, dict):
                    for key in f:
                        if key not in ("filename", "file", "name"):
                            stages_seen.add(key)
            return json.dumps({
                "total_files": total_files,
                "stages": sorted(stages_seen),
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    if models_orchestrator is not None:
        @mcp.resource("darwin://models/status")
        def models_status() -> str:
            """Current model status across all categories."""
            try:
                status = models_orchestrator.status()
                return json.dumps({
                    "all_ready": status.all_ready,
                    "summary": status.summary,
                    "categories": {
                        name: {
                            "available": cs.available,
                            "active": cs.active_model_id,
                            "error": cs.error,
                        }
                        for name, cs in status.categories.items()
                    },
                }, indent=2)
            except Exception as e:
                return json.dumps({"error": str(e)})

        @mcp.resource("darwin://models/recommendations")
        def models_recommendations() -> str:
            """Recommended models for each category."""
            try:
                recs = models_orchestrator.get_recommendations()
                return json.dumps({
                    "embedding": recs.embedding,
                    "nlp": recs.nlp,
                    "ocr": recs.ocr,
                    "reranker": recs.reranker,
                    "vision": recs.vision,
                }, indent=2)
            except Exception as e:
                return json.dumps({"error": str(e)})
