"""
MCP tools for PipelineOrchestrator.
Exposes run_pipeline, purge_artifacts, and pipeline_report via FastMCP.
"""

import json
import re
import traceback
from typing import Optional

from mcp.types import ToolAnnotations

from src.core.Orchestrators.lib import PipelineParameters as PipelineInput
from src.core.Orchestrators.lib import PipelineMode, resolve_store
from src.core.Orchestrators.PipelineOrchestrator import PipelineOrchestrator

_STAGE_MAP = {
    "ingest": "ingestion", "extract": "ingestion", "parse": "ingestion",
    "chunk": "ingestion", "ocr": "ingestion", "clean": "ingestion",
    "index": "indexing", "embed": "indexing",
}
_VALID_STAGES = set(_STAGE_MAP.keys()) | {"ingestion", "indexing"}
_MAX_TARGET_LEN = 2048


def _validate_stages(stages: list[str]) -> list[str]:
    clean = [s.strip().lower() for s in stages if s.strip()]
    if not clean:
        raise ValueError("stages must not be empty")
    for s in clean:
        if s not in _VALID_STAGES:
            raise ValueError(f"unknown stage '{s}'; valid: {sorted(_VALID_STAGES)}")
    return [_STAGE_MAP.get(s, s) for s in clean]


def _validate_target(target: str) -> str:
    target = target.strip()
    if not target:
        raise ValueError("target must not be empty")
    if len(target) > _MAX_TARGET_LEN:
        raise ValueError(f"target exceeds {_MAX_TARGET_LEN} chars")
    if re.search(r'[<>"|;`]', target):
        raise ValueError("target contains invalid characters")
    return target


def _validate_filenames(filenames: list[str]) -> list[str]:
    if not filenames:
        raise ValueError("filenames must not be empty")
    clean = []
    for f in filenames:
        f = f.strip()
        if not f:
            raise ValueError("filename must not be empty")
        if len(f) > 2048:
            raise ValueError("filename exceeds 2048 chars")
        if re.search(r'[<>"|;`\0]', f) or ".." in f:
            raise ValueError(f"filename contains invalid characters: {f}")
        clean.append(f)
    return clean


def register(mcp, pipeline):
    @mcp.tool(
        annotations=ToolAnnotations(
            title="Run Pipeline",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
    )
    def run_pipeline(
        stages: list[str],
        target: str = "",
        store: Optional[str] = None,
        files: Optional[str] = None,
        ocr: bool = False,
        recursive: bool = False,
        precise: bool = False,
        rerun: bool = False,
        threshold: float = 0.5,
        idf_smoothing: float = 1.5,
        only_kw: bool = False,
        only_semantic: bool = False,
        strict: bool = True,
        log_mode: PipelineMode = "quiet",
    ) -> dict:
        """Run ingestion/indexing pipeline stages."""
        try:
            stages = _validate_stages(stages)
            target = _validate_target(target)
            params = PipelineInput(
                target=target,
                store=store,
                stages=stages,
                files=[files] if files else None,
                ocr=ocr,
                recursive=recursive,
                precise=precise,
                rerun=rerun,
                threshold=threshold,
                idf_smoothing=idf_smoothing,
                only_kw=only_kw,
                only_semantic=only_semantic,
                strict=strict,
                log_mode=log_mode,
            )
            if store:
                resolved = resolve_store(store)
                resolved.bootstrap(quiet=True)
                result = PipelineOrchestrator(store=resolved).build_pipeline(params)
            else:
                result = pipeline.build_pipeline(params)
            return {
                "success": result.success,
                "data": {
                    "stages_completed": [s.value for s in result.stages_completed],
                    "stages_failed": result.stages_failed,
                    "summary": result.summary,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e), "detail": traceback.format_exc()}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Purge Artifacts",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def purge_artifacts(stages: list[str], filenames: list[str]) -> dict:
        """Delete pipeline artifacts for specific files."""
        try:
            stages = _validate_stages(stages)
            filenames = _validate_filenames(filenames)
            pipeline.purge(stages, filenames)
            return {"success": True, "data": {"message": "Purge completed"}}
        except Exception as e:
            return {"success": False, "error": str(e), "detail": traceback.format_exc()}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Pipeline Report",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def pipeline_report(filename: Optional[str] = None) -> dict:
        """Get pipeline report for files."""
        try:
            report = pipeline.store.get_pipeline_report(filename)
            return {"success": True, "data": report}
        except Exception as e:
            return {"success": False, "error": str(e), "detail": traceback.format_exc()}
