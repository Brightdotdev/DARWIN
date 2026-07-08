"""
MCP server definition – tools for pipeline, retrieval, store.
"""

import json
from typing import Optional, List, Any, Dict
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.types as types

from src.core.Orchestrators.PipelineOrchestrator import (
    PipelineOrchestrator,
    PipelineParameters,
    PipelineStage,
)
from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
from src.core.Retrival.OrchestratorTypes import (
    OrchestratorConfig,
    MergeStrategy,
)
from src.core.Orchestrators.StoreOrchestrator import StoreOrchestrator
from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType


def create_mcp_server(store_id: Optional[str] = None) -> Server:
    """Factory to create an MCP server with the given default store."""
    app = Server("darwin-mcp")

    # Resolve store
    def get_store(sid: Optional[str] = None) -> DarwinStore:
        if sid:
            return DarwinStore.load_store(sid)
        elif store_id:
            return DarwinStore.load_store(store_id)
        else:
            ds = DarwinStore.get_default_store()
            if ds is None:
                raise ValueError("No default store found. Provide store_id.")
            return ds

    # Instantiate orchestrators lazily
    _pipeline_orch = None
    _retrieval_orch = None
    _store_orch = None

    def get_pipeline(sid: Optional[str] = None) -> PipelineOrchestrator:
        nonlocal _pipeline_orch
        if _pipeline_orch is None or (sid and _pipeline_orch.store.store_id != sid):
            store = get_store(sid)
            _pipeline_orch = PipelineOrchestrator(store)
        return _pipeline_orch

    def get_retrieval(sid: Optional[str] = None) -> RetrievalOrchestrator:
        nonlocal _retrieval_orch
        if _retrieval_orch is None or (sid and _retrieval_orch.store.store_id != sid):
            store = get_store(sid)
            _retrieval_orch = RetrievalOrchestrator(store)
        return _retrieval_orch

    def get_store_orch(sid: Optional[str] = None) -> StoreOrchestrator:
        nonlocal _store_orch
        if _store_orch is None or (sid and _store_orch._store.store_id != sid):
            if sid:
                _store_orch = StoreOrchestrator.load_store(sid)
            else:
                ds = get_store(sid)
                _store_orch = StoreOrchestrator(ds.darwin_dir, ds.store_name)
        return _store_orch

    # ---------- List tools ----------
    @app.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            # Pipeline tools
            types.Tool(
                name="run_pipeline",
                description="Run an ingestion/indexing pipeline.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "store_id": {"type": "string"},
                        "parsing": {"type": "object"},
                        "chunking": {"type": "object"},
                        "indexing": {"type": "object"},
                        "stages": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["ingestion", "indexing"],
                            },
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
                description="Delete artifacts for specific files.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "store_id": {"type": "string"},
                        "stages": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["ingestion", "indexing"],
                            },
                        },
                        "filenames": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["stages", "filenames"],
                },
            ),
            # Retrieval tool
            types.Tool(
                name="retrieve",
                description="Retrieve passages for a query.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "store_id": {"type": "string"},
                        "query": {"type": "string"},
                        "method": {
                            "type": "string",
                            "enum": ["dense", "sparse", "hybrid"],
                            "default": "hybrid",
                        },
                        "files": {"type": "array", "items": {"type": "string"}},
                        "top_k": {"type": "integer", "default": 10},
                        "max_passages": {"type": "integer", "default": 5},
                        "max_chars": {"type": "integer", "default": 10000},
                        "min_passage_score": {"type": "number", "default": 0.0},
                        "max_per_source": {"type": "integer", "default": 0},
                        "merge_strategy": {
                            "type": "string",
                            "enum": ["none", "adjacent", "all"],
                            "default": "adjacent",
                        },
                        "fill_gaps": {"type": "boolean", "default": True},
                        "rerank": {"type": "boolean", "default": False},
                        "rerank_weight": {"type": "number", "default": 0.5},
                        "filter_structural": {"type": "boolean", "default": True},
                        "structural_penalty_factor": {"type": "number", "default": 0.7},
                        "exclude_sources": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "prefer_sources": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "include_raw_results": {"type": "boolean", "default": False},
                        "include_audit": {"type": "boolean", "default": False},
                        "include_logs": {"type": "boolean", "default": False},
                    },
                    "required": ["query"],
                },
            ),
            # Store tools (subset)
            types.Tool(
                name="list_stores",
                description="List all available stores.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="store_info",
                description="Get current store info.",
                inputSchema={
                    "type": "object",
                    "properties": {"store_id": {"type": "string"}},
                },
            ),
            types.Tool(
                name="file_status",
                description="Get processing status for a file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "store_id": {"type": "string"},
                        "filename": {"type": "string"},
                    },
                    "required": ["filename"],
                },
            ),
            types.Tool(
                name="store_status",
                description="Get overall store health.",
                inputSchema={
                    "type": "object",
                    "properties": {"store_id": {"type": "string"}},
                },
            ),
            types.Tool(
                name="pipeline_report",
                description="Get pipeline report for a file.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "store_id": {"type": "string"},
                        "filename": {"type": "string"},
                    },
                },
            ),
            types.Tool(
                name="search_readiness",
                description="Check if a file is ready for search.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "store_id": {"type": "string"},
                        "filename": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["dense", "sparse", "hybrid"],
                            "default": "hybrid",
                        },
                    },
                    "required": ["filename"],
                },
            ),
            types.Tool(
                name="get_logs",
                description="Retrieve historical logs from the logger.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session": {"type": "string"},
                        "levels": {
                            "type": "string",
                            "description": "Comma-separated: info,warning,error,success",
                        },
                        "limit": {"type": "integer", "default": 100},
                    },
                },
            ),
            # Models tools – only admin tools, no installation (can add later)
            types.Tool(
                name="list_models",
                description="List registered models (read-only).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["embedding", "nlp", "ocr", "reranker", "vision"],
                        }
                    },
                },
            ),
        ]

    # ---------- Call tool ----------
    @app.call_tool()
    async def call_tool(name: str, arguments: Any) -> List[types.TextContent]:
        # Common helper: get store_id from arguments
        store_id = arguments.get("store_id")
        darwin.info(f"MCP tool called: {name} with args: {arguments}")

        try:
            if name == "run_pipeline":
                orch = get_pipeline(store_id)
                params = PipelineParameters(
                    parsing=arguments.get("parsing"),
                    chunking=arguments.get("chunking"),
                    indexing=arguments.get("indexing"),
                )
                stages = [PipelineStage(s) for s in arguments.get("stages", [])]
                strict = arguments.get("strict", True)
                log_mode = arguments.get("log_mode", "quiet")
                result = orch.build_pipeline(
                    params, stages, strict=strict, log_mode=log_mode
                )
                return [
                    types.TextContent(
                        type="text", text=json.dumps(result.__dict__, default=str)
                    )
                ]

            elif name == "purge_artifacts":
                orch = get_pipeline(store_id)
                stages = [PipelineStage(s) for s in arguments.get("stages", [])]
                filenames = arguments.get("filenames", [])
                orch.purge(stages, filenames)
                return [types.TextContent(type="text", text="Purge completed")]

            elif name == "retrieve":
                orch = get_retrieval(store_id)
                cfg = OrchestratorConfig(
                    query=arguments["query"],
                    method=arguments.get("method", "hybrid"),
                    files=arguments.get("files", []),
                    top_k=arguments.get("top_k", 10),
                    max_passages=arguments.get("max_passages", 5),
                    max_chars=arguments.get("max_chars", 10000),
                    min_passage_score=arguments.get("min_passage_score", 0.0),
                    max_per_source=arguments.get("max_per_source", 0),
                    merge_strategy=MergeStrategy(
                        arguments.get("merge_strategy", "adjacent")
                    ),
                    fill_gaps=arguments.get("fill_gaps", True),
                    rerank=arguments.get("rerank", False),
                    rerank_weight=arguments.get("rerank_weight", 0.5),
                    filter_structural=arguments.get("filter_structural", True),
                    structural_penalty_factor=arguments.get(
                        "structural_penalty_factor", 0.7
                    ),
                    exclude_sources=arguments.get("exclude_sources", []),
                    prefer_sources=arguments.get("prefer_sources", []),
                    include_raw_results=arguments.get("include_raw_results", False),
                    include_audit=arguments.get("include_audit", False),
                )
                include_logs = arguments.get("include_logs", False)
                context = orch.retrieve_and_resolve(cfg)
                if include_logs and context.passages:
                    darwin.loglist(context.passages, LogType.INFO)
                # Return context as dict
                return [
                    types.TextContent(
                        type="text", text=json.dumps(context.__dict__, default=str)
                    )
                ]

            elif name == "list_stores":
                stores = StoreOrchestrator.list_stores()
                return [
                    types.TextContent(type="text", text=json.dumps(stores, default=str))
                ]

            elif name == "store_info":
                orch = get_store_orch(store_id)
                info = orch.info()
                return [
                    types.TextContent(type="text", text=json.dumps(info, default=str))
                ]

            elif name == "file_status":
                orch = get_store_orch(store_id)
                filename = arguments["filename"]
                status = orch.file_status(filename)
                return [
                    types.TextContent(type="text", text=json.dumps(status, default=str))
                ]

            elif name == "store_status":
                orch = get_store_orch(store_id)
                status = orch.store_status()
                return [
                    types.TextContent(type="text", text=json.dumps(status, default=str))
                ]

            elif name == "pipeline_report":
                orch = get_store_orch(store_id)
                filename = arguments.get("filename")
                report = orch.pipeline_report(filename)
                return [
                    types.TextContent(type="text", text=json.dumps(report, default=str))
                ]

            elif name == "search_readiness":
                orch = get_store_orch(store_id)
                filename = arguments["filename"]
                mode = arguments.get("mode", "hybrid")
                readiness = orch.search_readiness(filename, mode)
                return [
                    types.TextContent(
                        type="text", text=json.dumps(readiness, default=str)
                    )
                ]

            elif name == "get_logs":
                session = arguments.get("session")
                levels = arguments.get("levels")
                limit = arguments.get("limit", 100)
                logs = LoggerOrchestrator.get_logs(
                    session=session, levels=levels, limit=limit
                )
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(logs, default=str),
                    )
                ]

            elif name == "list_models":
                from src.core.Orchestrators.ModelOrchestrator import modelsOrchestrator

                category = arguments.get("category")
                if category:
                    from src.core.Models.Lib.ModelRegistry import ModelCategory

                    cat_enum = ModelCategory(category)
                    models = modelsOrchestrator.list_registered(cat_enum)
                else:
                    models = modelsOrchestrator.list_registered()
                return [
                    types.TextContent(type="text", text=json.dumps(models, default=str))
                ]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            darwin.error(f"MCP tool {name} failed: {e}")
            return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return app
