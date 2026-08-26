"""
MCP Resources — expose Darwin data as readable resources.

Resources are read-only data sources that MCP clients can list and fetch.
Unlike tools (which execute actions), resources provide context for the agent.

Endpoints:
  darwin://store/report  — pipeline report for all files (merged: files + stats)
  darwin://search/results — saved search results index
"""

import json


def _get_results_dir():
    from lib.AppPaths import set_darwin_base_dir
    return set_darwin_base_dir() / "search-results"


def register(mcp, store):
    """Register MCP resources on a FastMCP instance."""

    @mcp.resource("darwin://store/report")
    def store_report() -> str:
        """Pipeline report — all files with completed stages and aggregate stats.

        Returns JSON with:
          - files: list of per-file pipeline status dicts
          - total_files: count of tracked files
          - stages: list of stage keys found across all files
        """
        try:
            report = store.get_pipeline_report()
            files = report if isinstance(report, list) else []
            stages_seen = set()
            for f in files:
                if isinstance(f, dict):
                    for key in f:
                        if key not in ("filename", "file", "name"):
                            stages_seen.add(key)
            return json.dumps({
                "files": files,
                "total_files": len(files),
                "stages": sorted(stages_seen),
            }, default=str, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("darwin://search/results")
    def search_results() -> str:
        """Saved search results index — lists all previously saved searches.

        Returns JSON with:
          - results: list of saved result summaries (query, method, score, timestamp)
          - total: count of saved results
        """
        try:
            search_dir = _get_results_dir()
            if not search_dir.exists():
                return json.dumps({"results": [], "total": 0})

            entries = []
            from pathlib import Path
            for f in sorted(search_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
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
            return json.dumps({"results": entries, "total": len(entries)}, default=str, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
