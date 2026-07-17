"""
MCP tools for PipelineOrchestrator.
Exposes run_pipeline, purge_artifacts, and pipeline_report via FastMCP.
"""

import re
from typing import Optional

from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations

from src.core.Orchestrators.lib import PipelineParameters as PipelineInput
from src.core.Orchestrators.lib import resolve_store
from src.core.Orchestrators.PipelineOrchestrator import PipelineOrchestrator
from src.core.Logger.DarwinLogger import darwin
from src.mcp.logging_bridge import darwin_bridge

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
    async def run_pipeline(
        ctx: Context,
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
    ) -> dict:
        """Run ingestion/indexing pipeline stages.

        Stages: 'ingestion' (or aliases: ingest, extract, parse, chunk, ocr, clean)
                'indexing'  (or aliases: index, embed)

        Usage tips:
        - Always pass a 'target' path or URL for ingestion.
        - Use 'rerun=true' to re-process files that already have artifacts.
        - 'strict=true' (default) halts on first failure; set false to continue.
        - Logs returned are filtered to WARNING/ERROR/SUCCESS only.
        - Pass store name/path/ID to run against a specific store.
        """
        async with darwin_bridge(ctx):
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
                )

                def _blocking():
                    if store:
                        resolved = resolve_store(store)
                        resolved.bootstrap(quiet=True)
                        return PipelineOrchestrator(store=resolved).build_pipeline(params)
                    return pipeline.build_pipeline(params)

                import asyncio
                result = await asyncio.get_event_loop().run_in_executor(None, _blocking)
                return {
                    "success": result.success,
                    "data": {
                        "stages_completed": [s.value for s in result.stages_completed],
                        "stages_failed": result.stages_failed,
                        "summary": result.summary,
                    },
                }
            except Exception as e:
                darwin.error(f"run_pipeline failed: {type(e).__name__}: {e}")
                return {"success": False, "error": str(e)}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Purge Artifacts",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def purge_artifacts(ctx: Context, stages: list[str], filenames: list[str], dry_run: bool = False) -> dict:
        """Delete pipeline artifacts for specific files.

        Usage tips:
        - Always pass dry_run=true first to preview what will be deleted.
        - Stages: 'ingestion' removes markdown/records; 'indexing' removes embeddings/vector/bm25.
        - filenames must be exact matches to the original filenames used during ingestion.
        """
        async with darwin_bridge(ctx):
            try:
                stages = _validate_stages(stages)
                filenames = _validate_filenames(filenames)
                if dry_run:
                    return {"success": True, "data": {
                        "message": "Dry run — no changes made",
                        "stages": stages,
                        "filenames": filenames,
                    }}

                import asyncio
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: pipeline.purge(stages, filenames)
                )
                return {"success": True, "data": {"message": "Purge completed"}}
            except Exception as e:
                darwin.error(f"purge_artifacts failed: {type(e).__name__}: {e}")
                return {"success": False, "error": str(e)}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Pipeline Report",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def pipeline_report(ctx: Context, filename: Optional[str] = None) -> dict:
        """Get pipeline report for files — shows which stages are complete per file.

        Usage tips:
        - Call without arguments to see all files.
        - Pass a specific filename to check that file's status only.
        - Use this before running pipeline to decide if rerun is needed.
        """
        async with darwin_bridge(ctx):
            try:
                import asyncio
                report = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: pipeline.store.get_pipeline_report(filename)
                )
                return {"success": True, "data": report}
            except Exception as e:
                darwin.error(f"pipeline_report failed: {type(e).__name__}: {e}")
                return {"success": False, "error": str(e)}
