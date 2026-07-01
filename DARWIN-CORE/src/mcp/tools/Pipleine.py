# src/mcp/tools/pipeline.py
"""
Pipeline MCP tool registration.
All tools return JSON: {"success": bool, "data": ...} or {"success": false, "error": ..., "detail": ...}
"""

import json
import traceback
from typing import Any, Dict, List, Optional

from src.core.Orchestrators.PipelineOrchestrator import (
    PipelineOrchestrator,
    ParsingArgs,
    ChunkingArgs,
    IndexingArgs,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import LogType


# ── Singleton orchestrator (inject your real store instance) ──
_pipeline_orch = PipelineOrchestrator(store=DarwinStore())


def _ok(data: Any) -> str:
    return json.dumps({"success": True, "data": data}, default=str)


def _err(message: str, detail: Optional[str] = None) -> str:
    return json.dumps({
        "success": False,
        "error": message,
        "detail": detail,
    }, default=str)


def _dict_to_typed_params(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert JSON-style dict parameters into typed dataclasses.
    Unknown keys are passed through as-is for future stages.
    """
    out = dict(parameters)
    if "parsing" in out:
        out["parsing"] = ParsingArgs(**out["parsing"])
    if "chunking" in out:
        out["chunking"] = ChunkingArgs(**out["chunking"])
    if "indexing" in out:
        out["indexing"] = IndexingArgs(**out["indexing"])
    return out


def register_pipeline_tools(mcp):
    """Register all pipeline tools on the shared FastMCP instance."""

    @mcp.tool()
    def build_pipeline(
        parameters: Dict[str, Any],
        stages: List[str],
        strict: bool = True,
    ) -> str:
        """
        Execute Darwin pipeline stages.
        Parameters are JSON dicts auto-converted to typed args.
        """
        try:
            typed = _dict_to_typed_params(parameters)
            result = _pipeline_orch.build_pipeline(typed, stages, strict=strict)

            return _ok({
                "success": result.success,
                "parameters_validated": result.parameters_validated,
                "stages_completed": result.stages_completed,
                "stages_failed": result.stages_failed,
                "summary": result.summary,
                "logs": result.logs,
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def validate_pipeline_parameters(
        parameters: Dict[str, Any],
        stages: List[str],
    ) -> str:
        """Dry-run validation without executing any compute."""
        try:
            typed = _dict_to_typed_params(parameters)
            error = _pipeline_orch.validate_parameters(typed, stages)

            if error:
                return _ok({"valid": False, "error": error})
            return _ok({
                "valid": True,
                "message": "All parameters valid for requested stages",
                "stages_requested": stages,
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def get_pipeline_schema() -> str:
        """Return available stages and their required parameter keys."""
        try:
            return _ok({
                "stage_order": PipelineOrchestrator.STAGE_ORDER,
                "required_parameters": {
                    stage: PipelineOrchestrator.REQUIRED_PARAMS.get(stage, [])
                    for stage in PipelineOrchestrator.STAGE_ORDER
                },
                "parameter_schemas": {
                    "parsing": {
                        "target": "str (required)",
                        "files": "list[str]",
                        "images": "bool",
                        "caption": "bool",
                        "ocr": "bool",
                        "recursive": "bool",
                        "rerun": "bool",
                        "precise": "bool",
                        "save": "bool",
                        "cleanup": "bool",
                        "skip_if_markdown": "bool",
                    },
                    "chunking": {
                        "files": "list[str]",
                        "inspect": "bool",
                        "rerun": "bool",
                    },
                    "indexing": {
                        "threshold": "float (required)",
                        "idf_smoothing": "float (required)",
                        "only_kw": "bool",
                        "path": "str",
                        "only_semantic": "bool",
                        "files": "list[str]",
                        "inspect": "bool",
                        "rerun": "bool",
                    },
                },
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def get_pipeline_logs(
        levels: Optional[List[str]] = None,
    ) -> str:
        """Fetch recent logs from the Darwin logger."""
        try:
            from src.core.Logger.DarwinLogger import darwin
            level_map = {
                "info": LogType.INFO,
                "success": LogType.SUCCESS,
                "warning": LogType.WARNING,
                "error": LogType.ERROR,
            }
            if levels:
                log_levels = {level_map[l.lower()] for l in levels if l.lower() in level_map}
            else:
                log_levels = {LogType.INFO, LogType.SUCCESS, LogType.WARNING, LogType.ERROR}

            logs = darwin.get_logs(levels=log_levels)
            return _ok({"logs": logs, "count": len(logs)})
        except Exception as e:
            return _err(str(e), traceback.format_exc())