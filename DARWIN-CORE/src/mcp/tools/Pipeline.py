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
    StageParameters,
    PipelineStage,
    PipelineResult,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Storage.StoreRegistry import DarwinStoreRegistry
from src.core.Logger.DarwinLogger import darwin
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
                darwin.error("No default store found. Provide store_id.")
                return None
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
        from src.core.Orchestrators.lib import PipelineParameters as PipelineInput
        parsing = arguments.get("parsing") or {}
        stages = arguments.get("stages", ["ingestion", "indexing"])
        params = PipelineInput(
            target=parsing.get("target", ""),
            stages=stages,
            files=parsing.get("files"),
            ocr=parsing.get("ocr", False),
            recursive=parsing.get("recursive", False),
            precise=parsing.get("precise", False),
            rerun=parsing.get("rerun", False),
            threshold=(arguments.get("indexing") or {}).get("threshold", 0.5),
            idf_smoothing=(arguments.get("indexing") or {}).get("idf_smoothing", 1.5),
            only_kw=(arguments.get("indexing") or {}).get("only_kw", False),
            only_semantic=(arguments.get("indexing") or {}).get("only_semantic", False),
            strict=arguments.get("strict", True),
            log_mode=arguments.get("log_mode", "quiet"),
        )
        result: PipelineResult = orchestrator.build_pipeline(params)
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

    return [types.TextContent(type="text", text=json.dumps({"success": False, "error": f"Unknown tool: {name}"}))]


# ── FastMCP registration (used by src.mcp.Server) ─────────────────


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
