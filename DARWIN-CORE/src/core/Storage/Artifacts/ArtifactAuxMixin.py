from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.core.Storage.StoreUtils import normalize_record, read_json, write_json, now
from src.core.Storage.Transanctions import write_text_atomic
from src.core.Logger.DarwinLogger import darwin, LogType


class ArtifactAuxMixin:
    """Stopwords, Graph, and Wiki artifacts — I/O and deletion."""

    # ── Stopwords ───────────────────────────────────────────────────────

    def has_stopwords(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return False
        return self._artifacts.has_artifact(h, "stopwords")
    def save_stopwords(self, filename: str, stopwords: set[str]) -> None:
        """Persist stopwords. Empty sets are allowed (clears saved stopwords)."""
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return
        path = self.paths.stopwords_path(h)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"stopwords": sorted(stopwords)})
        self._meta.update(h, "stopwords")
        self._contents.set_stage(h, "stopwords")
        self._cache.invalidate(f"stopwords:{h}")

    def load_stopwords(self, filename: str) -> set[str]:
        """Load persisted stopwords. Always returns a set, never None."""
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return set()  # FIX: was returning None, which breaks set unions
        path = self.paths.stopwords_path(h)
        if not path.exists():
            return set()
        data = read_json(path) or {}
        return set(data.get("stopwords", []))

    def delete_stopwords(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return False
        path = self.paths.stopwords_path(h)
        removed = False
        if path.exists():
            path.unlink()
            removed = True
        self._contents.set_stage(h, "stopwords", complete=False)
        self._cache.invalidate(f"stopwords:{h}")
        return removed

    # ── Graph ───────────────────────────────────────────────────────────

    def has_graph(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return False
        return self._artifacts.has_artifact(h, "graph") and self._artifacts.has_artifact(h, "graph_vis")

    def save_graph(self, filename: str, nodes: list[Any], edges: list[Any]) -> None:
        if not nodes and not edges:
            darwin.log(f"No graph data to save for {filename}", LogType.ERROR)
            return

        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return
        self._artifacts.ensure_dir(h)

        nodes_path = self.paths.graph_nodes_path(h)
        edges_path = self.paths.graph_edges_path(h)

        nodes_path.parent.mkdir(parents=True, exist_ok=True)

        if nodes:
            nodes_table = pa.Table.from_pylist([normalize_record(n) for n in nodes])
            pq.write_table(nodes_table, nodes_path, compression="zstd")

        if edges:
            edges_table = pa.Table.from_pylist([normalize_record(e) for e in edges])
            pq.write_table(edges_table, edges_path, compression="zstd")

        self._meta.update(h, "graph", extras={
            "graph": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "built_at": now(),
            }
        })
        self._contents.set_stage(h, "graph")
        self._cache.invalidate(f"graph:{h}")

        darwin.log(f"Graph saved: {len(nodes)} nodes, {len(edges)} edges", LogType.SUCCESS)

    def save_graph_html(self, filename: str, html: str) -> None:
        if html is None:
            darwin.log(f"No graph HTML to save for {filename}", LogType.ERROR)
            return
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return
        path = self.paths.graph_html_path(h)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, html)
        self._meta.update(h, "graph_vis")
        self._contents.set_stage(h, "graph_vis")
        self._cache.invalidate(f"graph_vis:{h}")

    def load_graph(self, filename: str) -> dict:
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return
        cache_key = f"graph:{h}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        nodes_path = self.paths.graph_nodes_path(h)
        edges_path = self.paths.graph_edges_path(h)

        result: dict[str, list] = {"nodes": [], "edges": []}
        if nodes_path.exists():
            result["nodes"] = pq.read_table(nodes_path).to_pydict()
        if edges_path.exists():
            result["edges"] = pq.read_table(edges_path).to_pydict()

        self._cache.put(cache_key, result)
        return result

    def delete_graph(self, filename: str) -> bool:
        """Remove graph nodes/edges and clear the registry flag."""
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return False
        removed = False
        for p in (self.paths.graph_nodes_path(h), self.paths.graph_edges_path(h)):
            if p.exists():
                p.unlink()
                removed = True
        self._contents.set_stage(h, "graph", complete=False)
        self._cache.invalidate(f"graph:{h}")
        return removed

    def delete_graph_html(self, filename: str) -> bool:
        """Remove graph HTML and clear the registry flag."""
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return False
        path = self.paths.graph_html_path(h)
        removed = False
        if path.exists():
            path.unlink()
            removed = True
        self._contents.set_stage(h, "graph_vis", complete=False)
        self._cache.invalidate(f"graph_vis:{h}")
        return removed

    # ── Wiki ────────────────────────────────────────────────────────────

    def has_wiki(self, filename: str) -> bool:
        h = self._resolve_hash(filename)
        if h is None:
            return False
        # Keep old artifact check, but also validate the file
        if not self._artifacts.has_artifact(h, "wiki"):
            return False
        return self.paths.wiki_path(h).exists()
    def save_wiki(self, filename: str, wiki: dict) -> None:
        if not wiki:
            darwin.log(f"No wiki pages to save for {filename}", LogType.ERROR)
            return

        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return
        self._artifacts.ensure_dir(h)

        wiki_path = self.paths.wiki_path(h)
        wiki_path.parent.mkdir(parents=True, exist_ok=True)

        write_json(wiki_path, wiki)
        self._meta.update(h, "wiki")
        self._contents.set_stage(h, "wiki")
        self._cache.invalidate(f"wiki:{h}")

        darwin.log(f"Wiki saved for {filename}", LogType.SUCCESS)

    def load_wiki(self, filename: str) -> dict | None:
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return None
        path = self.paths.wiki_path(h)
        if not path.exists():
            return None
        return read_json(path)

    def delete_wiki(self, filename: str) -> bool:
        """Remove wiki parquet and clear the registry flag."""
        h = self._resolve_hash(filename)
        if h is None:
            darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
            return False
        path = self.paths.wiki_path(h)
        removed = False
        if path.exists():
            path.unlink()
            removed = True
        self._contents.set_stage(h, "wiki", complete=False)
        self._cache.invalidate(f"wiki:{h}")
        return removed