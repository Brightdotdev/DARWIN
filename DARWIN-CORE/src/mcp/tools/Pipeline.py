"""
MCP tools for PipelineOrchestrator.
Exposes run_pipeline, purge_artifacts, and pipeline_report via FastMCP.
"""

import json
from typing import Optional

from src.core.Orchestrators.lib import PipelineParameters as PipelineInput
from src.core.Orchestrators.lib import PipelineMode, resolve_store


def register(mcp, pipeline):
    """Register pipeline tools on a FastMCP instance."""

    @mcp.tool()
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
    ) -> str:
        """Run ingestion/indexing pipeline stages."""
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
            from src.core.Orchestrators.PipelineOrchestrator import PipelineOrchestrator
            resolved = resolve_store(store)
            resolved.bootstrap()
            result = PipelineOrchestrator(store=resolved).build_pipeline(params)
        else:
            result = pipeline.build_pipeline(params)
        return json.dumps(result.__dict__, default=str)

    @mcp.tool()
    def purge_artifacts(stages: list[str], filenames: list[str]) -> str:
        """Delete pipeline artifacts for specific files."""
        pipeline.purge(stages, filenames)
        return "Purge completed"

    @mcp.tool()
    def pipeline_report(filename: Optional[str] = None) -> str:
        """Get pipeline report for files."""
        report = pipeline.store.get_pipeline_report(filename)
        return json.dumps(report, default=str)
