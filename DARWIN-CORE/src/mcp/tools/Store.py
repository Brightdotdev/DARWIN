# src/mcp/tools/store.py
"""
Store MCP tool registration.
Artifact I/O -> DarwinStore directly.
Status/health -> StoreInspector.
"""

import json
import traceback
from typing import Any, Dict, List, Optional

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Storage.StoreRegistry import DarwinStoreRegistry
from src.core.Storage.StoreInspector import StoreInspector


def _ok(data: Any) -> str:
    return json.dumps({"success": True, "data": data}, default=str)


def _err(message: str, detail: Optional[str] = None) -> str:
    return json.dumps({
        "success": False,
        "error": message,
        "detail": detail,
    }, default=str)


def _source_to_dict(src) -> dict:
    return {
        "uuid": src.uuid,
        "display_name": src.display_name,
        "content_hash": src.content_hash,
        "current_path": src.current_path,
        "file_size": src.file_size,
        "created_at": src.created_at,
        "updated_at": src.updated_at,
    }


def register_store_tools(mcp):
    """Register all storage / pipeline tools on the shared FastMCP instance."""

    # Bootstrap default store + inspector
    store = DarwinStore()
    DarwinStoreRegistry.register("main", store, default=True)
    inspector = StoreInspector(store)

    def _set_active(new_store: DarwinStore, key: str = "main"):
        nonlocal store, inspector
        store = new_store
        inspector = StoreInspector(store)
        DarwinStoreRegistry.register(key, store, default=True)

    # ── Store Lifecycle / Overview ──

    @mcp.tool()
    def get_store_info() -> str:
        try:
            return _ok(inspector.info())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def switch_store(darwin_dir: str, name: Optional[str] = None) -> str:
        try:
            from pathlib import Path
            new_store = DarwinStore(Path(darwin_dir), name)
            _set_active(new_store, name or "main")
            return _ok(inspector.info())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def list_stores() -> str:
        try:
            return _ok(StoreInspector.list_stores())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def get_default_store() -> str:
        try:
            ins = StoreInspector.get_default()
            return _ok(ins.info() if ins else None)
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_store_by_id(store_id: str) -> str:
        try:
            new_store = DarwinStore.load_store(store_id)
            _set_active(new_store, store_id)
            return _ok(inspector.info())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_store_by_name(name: str) -> str:
        try:
            new_store = DarwinStore.load_store_by_name(name)
            _set_active(new_store, name)
            return _ok(inspector.info())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Source Management (direct to DarwinStore) ──

    @mcp.tool()
    def register_file(file_path: str, page_count: int = 0, file_size: int = 0) -> str:
        try:
            src = store.register(file_path, page_count, file_size)
            return _ok(_source_to_dict(src))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def register_url(slug: str, url: str, content_hash: str, page_count: int = 0) -> str:
        try:
            src = store.register_url(slug, url, content_hash, page_count)
            return _ok(_source_to_dict(src))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def remove_file(filename: str) -> str:
        try:
            ok = store.remove_file(filename)
            return _ok({"removed": ok, "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def list_files() -> str:
        try:
            status = store.status()
            return _ok([{"name": k, **v} for k, v in status.get("files", {}).items()])
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def get_file_info(filename: str) -> str:
        try:
            src = store.get_source(filename)
            return _ok(_source_to_dict(src) if src else None)
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Artifacts: Markdown (direct to DarwinStore) ──

    @mcp.tool()
    def save_markdown(filename: str, display_text: str, embedding_text: str = "",
                      extras: Optional[str] = None) -> str:
        try:
            parsed_extras = json.loads(extras) if extras else None
            doc = {
                "display_text": display_text,
                "embedding_text": embedding_text or display_text,
                **(parsed_extras or {}),
            }
            store.save_markdown(filename, doc)
            return _ok({"saved": True, "stage": "markdown", "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_markdown(filename: str) -> str:
        try:
            return _ok(store.load_markdown(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_markdown(filename: str) -> str:
        try:
            return _ok({"deleted": store.delete_markdown(filename),
                        "stage": "markdown", "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Artifacts: Chunks (direct to DarwinStore) ──

    @mcp.tool()
    def save_chunks(filename: str, records: str) -> str:
        try:
            parsed = json.loads(records)
            store.save_records(filename, parsed)
            return _ok({"saved": True, "stage": "chunks", "count": len(parsed), "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_chunks(filename: str, columns: Optional[str] = None) -> str:
        try:
            cols = json.loads(columns) if columns else None
            table = store.load_records(filename, cols)
            if table.num_rows == 0:
                return _ok([])
            return _ok(table.to_pylist())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_chunks(filename: str) -> str:
        try:
            return _ok({"deleted": store.delete_records(filename),
                        "stage": "chunks", "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Artifacts: Embeddings (direct to DarwinStore) ──

    @mcp.tool()
    def save_embeddings(filename: str, vectors: str, indices: Optional[str] = None) -> str:
        try:
            import numpy as np
            parsed_vectors = json.loads(vectors)
            parsed_indices = json.loads(indices) if indices else None
            arr = np.array(parsed_vectors, dtype=np.float32)
            store.save_embeddings(filename, arr, parsed_indices)
            return _ok({
                "saved": True, "stage": "embeddings",
                "count": len(parsed_vectors), "dimensions": arr.shape[1],
                "filename": filename,
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_embeddings(filename: str) -> str:
        try:
            arr = store.load_embeddings(filename)
            return _ok(arr.astype(np.float32).tolist() if arr is not None else None)
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_embeddings(filename: str) -> str:
        try:
            return _ok({"deleted": store.delete_embeddings(filename),
                        "stage": "embeddings", "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Artifacts: Vector Index (direct to DarwinStore) ──

    @mcp.tool()
    def save_vector_index(filename: str, vectors: str) -> str:
        try:
            import numpy as np
            parsed = json.loads(vectors)
            arr = np.array(parsed, dtype=np.float32)
            store.save_vector(filename, arr)
            return _ok({"saved": True, "stage": "vector", "count": len(parsed), "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_vector_index(filename: str) -> str:
        try:
            return _ok({"deleted": store.delete_vector(filename),
                        "stage": "vector", "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Artifacts: BM25 (direct to DarwinStore) ──

    @mcp.tool()
    def save_bm25(filename: str, data: str) -> str:
        try:
            parsed = json.loads(data)
            store.save_bm25(filename, parsed)
            return _ok({"saved": True, "stage": "bm25", "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_bm25(filename: str) -> str:
        try:
            return _ok(store.load_bm25(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_bm25(filename: str) -> str:
        try:
            return _ok({"deleted": store.delete_bm25(filename),
                        "stage": "bm25", "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Artifacts: Graph (direct to DarwinStore) ──

    @mcp.tool()
    def save_graph(filename: str, nodes: str, edges: str) -> str:
        try:
            parsed_nodes = json.loads(nodes)
            parsed_edges = json.loads(edges)
            store.save_graph(filename, parsed_nodes, parsed_edges)
            return _ok({
                "saved": True, "stage": "graph",
                "node_count": len(parsed_nodes), "edge_count": len(parsed_edges),
                "filename": filename,
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def save_graph_html(filename: str, html: str) -> str:
        try:
            store.save_graph_html(filename, html)
            return _ok({"saved": True, "stage": "graph_vis", "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_graph(filename: str) -> str:
        try:
            return _ok(store.load_graph(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_graph(filename: str) -> str:
        try:
            return _ok({"deleted": store.delete_graph(filename),
                        "stage": "graph", "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_graph_html(filename: str) -> str:
        try:
            return _ok({"deleted": store.delete_graph_html(filename),
                        "stage": "graph_vis", "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Artifacts: Wiki (direct to DarwinStore) ──

    @mcp.tool()
    def save_wiki(filename: str, wiki: str) -> str:
        try:
            parsed = json.loads(wiki)
            store.save_wiki(filename, parsed)
            return _ok({"saved": True, "stage": "wiki", "page_count": len(parsed), "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def load_wiki(filename: str) -> str:
        try:
            return _ok(store.load_wiki(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def delete_wiki(filename: str) -> str:
        try:
            return _ok({"deleted": store.delete_wiki(filename),
                        "stage": "wiki", "filename": filename})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Search (direct to DarwinStore) ──

    @mcp.tool()
    def search_file(filename: str, query_vec: str, k: int = 10,
                    min_score: float = 0.0, columns: Optional[str] = None,
                    force_numpy: bool = False) -> str:
        try:
            import numpy as np
            vec = np.array(json.loads(query_vec), dtype=np.float32)
            cols = json.loads(columns) if columns else None
            return _ok(store.search_single(vec, filename, k=k, min_score=min_score,
                                           columns=cols, force_numpy=force_numpy))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def search_files(filenames: str, query_vec: str, top_k: int = 10,
                     min_score: float = 0.0, columns: Optional[str] = None,
                     force_numpy: bool = False) -> str:
        try:
            import numpy as np
            files = json.loads(filenames)
            vec = np.array(json.loads(query_vec), dtype=np.float32)
            cols = json.loads(columns) if columns else None
            return _ok(store.search_multi(vec, files, top_k=top_k, min_score=min_score,
                                          columns=cols, force_numpy=force_numpy))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Status & Health (via StoreInspector) ──

    @mcp.tool()
    def file_status(filename: str) -> str:
        try:
            return _ok(inspector.file_status(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def store_status() -> str:
        try:
            return _ok(inspector.store_status())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def pipeline_report(filename: Optional[str] = None) -> str:
        try:
            return _ok(inspector.pipeline_report(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def search_readiness(filename: str, mode: str = "hybrid") -> str:
        try:
            return _ok(inspector.search_readiness(filename))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def ready_files(filenames: Optional[str] = None, mode: str = "hybrid") -> str:
        try:
            files = json.loads(filenames) if filenames else None
            return _ok(inspector.ready_files(files, mode))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Integrity (via StoreInspector) ──

    @mcp.tool()
    def check_integrity(content_hash: str) -> str:
        try:
            return _ok(inspector.check_integrity(content_hash))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def repair_store(content_hash: Optional[str] = None, dry_run: bool = False) -> str:
        try:
            return _ok(inspector.repair(content_hash, dry_run))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def audit_store() -> str:
        try:
            return _ok(inspector.audit())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def embedding_health(model_id: str) -> str:
        try:
            return _ok(inspector.embedding_health(model_id))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def reembed_check(filenames: Optional[str] = None) -> str:
        try:
            files = json.loads(filenames) if filenames else None
            return _ok(inspector.reembed_check(files))
        except Exception as e:
            return _err(str(e), traceback.format_exc())
