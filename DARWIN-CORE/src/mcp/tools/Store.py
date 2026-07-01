# src/mcp/tools/store.py
"""
StoreOrchestrator MCP tool registration.
Refactored to the unified registration pattern.
"""

import json
import traceback
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.core.Orchestrators.StoreOrchestrator import (
    StoreOrchestrator,
    storeOrchestrator,
)


def _ok(data: Any) -> str:
    return json.dumps({"success": True, "data": data}, default=str)


def _err(message: str, detail: Optional[str] = None) -> str:
    return json.dumps({
        "success": False,
        "error": message,
        "detail": detail,
    }, default=str)


def register_store_tools(mcp):
    """Register all storage / pipeline tools on the shared FastMCP instance."""
    orch = storeOrchestrator

    # ── Store Lifecycle ──

    @mcp.tool()
    def get_store_info() -> str:
        try:
            return _ok(orch.info())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def switch_store(darwin_dir: str, name: Optional[str] = None) -> str:
        try:
            return _ok(orch.switch(Path(darwin_dir), name))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def list_stores() -> str:
        try:
            return _ok(StoreOrchestrator.list_stores())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def get_default_store() -> str:
        try:
            ds = StoreOrchestrator.get_default()
            return _ok(ds.info() if ds else None)
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_store_by_id(store_id: str) -> str:
        try:
            global storeOrchestrator
            storeOrchestrator = StoreOrchestrator.load_store(store_id)
            return _ok(storeOrchestrator.info())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_store_by_name(name: str) -> str:
        try:
            global storeOrchestrator
            storeOrchestrator = StoreOrchestrator.load_by_name(name)
            return _ok(storeOrchestrator.info())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Source Management ──

    @mcp.tool()
    def register_file(file_path: str, page_count: int = 0, file_size: int = 0) -> str:
        try:
            return _ok(orch.register(file_path, page_count, file_size))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def register_url(slug: str, url: str, content_hash: str, page_count: int = 0) -> str:
        try:
            return _ok(orch.register_url(slug, url, content_hash, page_count))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def remove_file(filename: str) -> str:
        try:
            return _ok(orch.remove(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def list_files() -> str:
        try:
            return _ok(orch.list_files())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def get_file_info(filename: str) -> str:
        try:
            return _ok(orch.get_file(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Artifacts: Markdown ──

    @mcp.tool()
    def save_markdown(filename: str, display_text: str, embedding_text: str = "",
                      extras: Optional[str] = None) -> str:
        try:
            parsed_extras = json.loads(extras) if extras else None
            return _ok(orch.save_markdown(filename, display_text, embedding_text, parsed_extras))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_markdown(filename: str) -> str:
        try:
            return _ok(orch.load_markdown(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_markdown(filename: str) -> str:
        try:
            return _ok(orch.delete_markdown(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Artifacts: Chunks ──

    @mcp.tool()
    def save_chunks(filename: str, records: str) -> str:
        try:
            parsed = json.loads(records)
            return _ok(orch.save_chunks(filename, parsed))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_chunks(filename: str, columns: Optional[str] = None) -> str:
        try:
            cols = json.loads(columns) if columns else None
            return _ok(orch.load_chunks(filename, cols))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_chunks(filename: str) -> str:
        try:
            return _ok(orch.delete_chunks(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Artifacts: Embeddings ──

    @mcp.tool()
    def save_embeddings(filename: str, vectors: str, indices: Optional[str] = None) -> str:
        try:
            parsed_vectors = json.loads(vectors)
            parsed_indices = json.loads(indices) if indices else None
            return _ok(orch.save_embeddings(filename, parsed_vectors, parsed_indices))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_embeddings(filename: str) -> str:
        try:
            return _ok(orch.load_embeddings(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_embeddings(filename: str) -> str:
        try:
            return _ok(orch.delete_embeddings(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Artifacts: Vector Index ──

    @mcp.tool()
    def save_vector_index(filename: str, vectors: str) -> str:
        try:
            parsed = json.loads(vectors)
            return _ok(orch.save_vector(filename, parsed))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_vector_index(filename: str) -> str:
        try:
            return _ok(orch.delete_vector(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Artifacts: BM25 ──

    @mcp.tool()
    def save_bm25(filename: str, data: str) -> str:
        try:
            parsed = json.loads(data)
            return _ok(orch.save_bm25(filename, parsed))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_bm25(filename: str) -> str:
        try:
            return _ok(orch.load_bm25(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_bm25(filename: str) -> str:
        try:
            return _ok(orch.delete_bm25(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Artifacts: Graph ──

    @mcp.tool()
    def save_graph(filename: str, nodes: str, edges: str) -> str:
        try:
            parsed_nodes = json.loads(nodes)
            parsed_edges = json.loads(edges)
            return _ok(orch.save_graph(filename, parsed_nodes, parsed_edges))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def save_graph_html(filename: str, html: str) -> str:
        try:
            return _ok(orch.save_graph_html(filename, html))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_graph(filename: str) -> str:
        try:
            return _ok(orch.load_graph(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_graph(filename: str) -> str:
        try:
            return _ok(orch.delete_graph(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_graph_html(filename: str) -> str:
        try:
            return _ok(orch.delete_graph_html(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Artifacts: Wiki ──

    @mcp.tool()
    def save_wiki(filename: str, wiki: str) -> str:
        try:
            parsed = json.loads(wiki)
            return _ok(orch.save_wiki(filename, parsed))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_wiki(filename: str) -> str:
        try:
            return _ok(orch.load_wiki(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_wiki(filename: str) -> str:
        try:
            return _ok(orch.delete_wiki(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Search ──

    @mcp.tool()
    def search_file(filename: str, query_vec: str, k: int = 10,
                    min_score: float = 0.0, columns: Optional[str] = None,
                    force_numpy: bool = False) -> str:
        try:
            vec = json.loads(query_vec)
            cols = json.loads(columns) if columns else None
            return _ok(orch.search(filename, vec, k=k, min_score=min_score,
                                   columns=cols, force_numpy=force_numpy))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def search_files(filenames: str, query_vec: str, top_k: int = 10,
                     min_score: float = 0.0, columns: Optional[str] = None,
                     force_numpy: bool = False) -> str:
        try:
            files = json.loads(filenames)
            vec = json.loads(query_vec)
            cols = json.loads(columns) if columns else None
            return _ok(orch.search_multi(files, vec, top_k=top_k, min_score=min_score,
                                          columns=cols, force_numpy=force_numpy))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Status & Health ──

    @mcp.tool()
    def file_status(filename: str) -> str:
        try:
            return _ok(orch.file_status(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def store_status() -> str:
        try:
            return _ok(orch.store_status())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def pipeline_report(filename: Optional[str] = None) -> str:
        try:
            return _ok(orch.pipeline_report(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def search_readiness(filename: str, mode: str = "hybrid") -> str:
        try:
            return _ok(orch.search_readiness(filename, mode))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def ready_files(filenames: Optional[str] = None, mode: str = "hybrid") -> str:
        try:
            files = json.loads(filenames) if filenames else None
            return _ok(orch.ready_files(files, mode))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Integrity ──

    @mcp.tool()
    def check_integrity(content_hash: str) -> str:
        try:
            return _ok(orch.check_integrity(content_hash))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def repair_store(content_hash: Optional[str] = None, dry_run: bool = False) -> str:
        try:
            return _ok(orch.repair(content_hash, dry_run))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def audit_store() -> str:
        try:
            return _ok(orch.audit())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def embedding_health(model_id: str) -> str:
        try:
            return _ok(orch.embedding_health(model_id))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def reembed_check(filenames: Optional[str] = None) -> str:
        try:
            files = json.loads(filenames) if filenames else None
            return _ok(orch.reembed_check(files))
        except Exception as e:
            return _err(str(e), traceback.format_exc())