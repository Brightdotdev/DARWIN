#!/usr/bin/env python3
"""
MCP server for PipelineOrchestrator.
Exposes tools to run ingestion/indexing pipelines and purge artifacts.
"""

import argparse
import json
from typing import List, Optional, Any, Dict
from pathlib import Path

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.types as types

from src.core.Orchestrators.PipelineOrchestrator import (
    PipelineOrchestrator,
    PipelineParameters,
    PipelineStage,
    PipelineResult,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Storage.StoreRegistry import DarwinStoreRegistry
from src.mcp.clibase import configure, OutputMode, LogFilter

# ── Server instance ────────────────────────────────────────────────
app = Server("mcp-pipeline")

# ── Store resolution ──────────────────────────────────────────────
_store: Optional[DarwinStore] = None


def get_store(store_id: Optional[str] = None) -> DarwinStore:
    global _store
    if _store is None:
        if store_id:
            _store = DarwinStore.load_store(store_id)
        else:
            ds = DarwinStore.get_default_store()
            if ds is None:
                raise ValueError("No default store found. Provide store_id.")
            _store = ds
    return _store


# ── Tools ──────────────────────────────────────────────────────────


@app.list_tools()
async def list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name="run_pipeline",
            description="Run an ingestion/indexing pipeline with given parameters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "store_id": {"type": "string", "description": "Optional store ID"},
                    "parsing": {"type": "object", "description": "ParsingArgs as dict"},
                    "chunking": {
                        "type": "object",
                        "description": "ChunkingArgs as dict",
                    },
                    "indexing": {
                        "type": "object",
                        "description": "IndexingArgs as dict",
                    },
                    "stages": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["ingestion", "indexing"]},
                        "description": "Stages to run",
                    },
                    "strict": {"type": "boolean", "default": True},
                    "log_mode": {
                        "type": "string",
                        "enum": ["quiet", "normal"],
                        "default": "quiet",
                    },
                },
                "required": ["parsing", "chunking", "indexing", "stages"],
            },
        ),
        types.Tool(
            name="purge_artifacts",
            description="Delete artifacts for specific files in given stages.",
            inputSchema={
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "stages": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["ingestion", "indexing"]},
                    },
                    "filenames": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["stages", "filenames"],
            },
        ),
        types.Tool(
            name="pipeline_report",
            description="Get pipeline report for a file (via StoreOrchestrator convenience).",
            inputSchema={
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "filename": {"type": "string", "description": "Optional filename"},
                },
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[types.TextContent]:
    # Configure logging based on tool call (could pass log_mode)
    configure(mode=OutputMode.TERMINAL, log_filter=LogFilter.ALL)

    store_id = arguments.get("store_id")
    store = get_store(store_id)
    orchestrator = PipelineOrchestrator(store)

    if name == "run_pipeline":
        params = PipelineParameters(
            parsing=arguments.get("parsing"),
            chunking=arguments.get("chunking"),
            indexing=arguments.get("indexing"),
        )
        stages = arguments.get("stages", [])
        strict = arguments.get("strict", True)
        log_mode = arguments.get("log_mode", "quiet")
        result: PipelineResult = orchestrator.build_pipeline(
            params, stages, strict=strict, log_mode=log_mode
        )
        return [
            types.TextContent(
                type="text", text=json.dumps(result.__dict__, default=str)
            )
        ]

    elif name == "purge_artifacts":
        stages = arguments["stages"]
        filenames = arguments["filenames"]
        orchestrator.purge(stages, filenames)
        return [types.TextContent(type="text", text="Purge completed")]

    elif name == "pipeline_report":
        filename = arguments.get("filename")
        report = store.get_pipeline_report(filename)
        return [types.TextContent(type="text", text=json.dumps(report, default=str))]

    raise ValueError(f"Unknown tool: {name}")


# ── FastMCP registration (used by src.mcp.Server) ─────────────────


def register(mcp, pipeline):
    """Register pipeline tools on a FastMCP instance."""

    @mcp.tool()
    def run_pipeline(
        stages: List[str],
        store_id: Optional[str] = None,
        target: Optional[str] = None,
        files: Optional[str] = None,
        ocr: bool = False,
        recursive: bool = False,
        precise: bool = False,
        rerun: bool = False,
        threshold: float = 0.5,
        idf_smoothing: float = 1.5,
        only_kw: bool = False,
        only_semantic: bool = False,
    ) -> str:
        """Run ingestion/indexing pipeline stages."""
        params = PipelineParameters()
        if "ingestion" in stages:
            from src.core.Orchestrators.PipelineOrchestrator import (
                ParsingArgs,
                ChunkingArgs,
            )

            file_list = [files] if files else []
            params.parsing = ParsingArgs(
                target=target,
                files=file_list,
                ocr=ocr,
                recursive=recursive,
                precise=precise,
                rerun=rerun,
            )
            params.chunking = ChunkingArgs(files=file_list, rerun=rerun)
        if "indexing" in stages:
            from src.core.Orchestrators.PipelineOrchestrator import IndexingArgs

            file_list = [files] if files else []
            params.indexing = IndexingArgs(
                threshold=threshold,
                idf_smoothing=idf_smoothing,
                only_kw=only_kw,
                only_semantic=only_semantic,
                files=file_list,
                rerun=rerun,
            )
        result = pipeline.build_pipeline(params, stages)
        return json.dumps(result.__dict__, default=str)

    @mcp.tool()
    def purge_artifacts(stages: List[str], filenames: List[str]) -> str:
        """Delete pipeline artifacts for specific files."""
        pipeline.purge(stages, filenames)
        return "Purge completed"

    @mcp.tool()
    def pipeline_report(filename: Optional[str] = None) -> str:
        """Get pipeline report for files."""
        report = pipeline.store.get_pipeline_report(filename)
        return json.dumps(report, default=str)


# ── Entry point ────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766, help="SSE port")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--store-id", help="Default store ID")
    args = parser.parse_args()

    # Set store globally if provided
    if args.store_id:
        global _store
        _store = DarwinStore.load_store(args.store_id)

    configure(mode=OutputMode.TERMINAL, show_banner=False)
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await app.run(streams[0], streams[1], app.create_initialization_options())

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ]
    )
    uvicorn.run(starlette_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
