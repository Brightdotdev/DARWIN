from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

from src.core.Storage.DiskUsage import build_tree, disk_node_to_dict, find_node, print_tree
from src.core.Models.ModelStore import ModelStore
from src.core.Logger.DarwinLogger import darwin


class DashboardService:
    """UI-oriented dashboard and disk usage tree, separate from storage core."""

    def __init__(self, store):
        self.store = store
        self._cached_tree = None

    def _build_disk_tree(self):
        if self._cached_tree is None:
            def hash_label_fn(h: str) -> str:
                sources = self.store._sources.get_by_hash(h)
                if sources:
                    names = ", ".join(s.display_name for s in sources)
                    return f"{names} ({h[:8]}…)"
                return ""

            model_store = ModelStore()
            self._cached_tree = build_tree(
                self.store._darwin_dir,
                self.store._sources,
                model_store=model_store,
                hash_label_fn=hash_label_fn,
            )
        return self._cached_tree

    def get_treemap(self, subpath: str | None = None) -> dict:
        tree = self._build_disk_tree()
        if subpath:
            parts = subpath.strip("/").split("/")
            node = find_node(tree, parts)
            if node is None:
                darwin.error(f"Subpath not found: {subpath}")
                return {}
            return disk_node_to_dict(node)
        return disk_node_to_dict(tree)

    def print_treemap(self, subpath: str | None = None) -> None:
        tree = self._build_disk_tree()
        if subpath:
            parts = subpath.strip("/").split("/")
            node = find_node(tree, parts)
            if node is None:
                print(f"Path not found: {subpath}")
                return
        else:
            node = tree
        print_tree(node)

    def get_disk_breakdown(self, filenames: str | list[str] | None = None) -> dict:
        def _file_breakdown(filename: str) -> dict:
            h = self.store._resolve_hash(filename)
            if h is None:
                return {}
            breakdown: dict = {}
            total = 0
            path_map = {
                "markdown": ["markdown", "source", "display"],
                "chunks": ["chunks", "file"],
                "embeddings": ["embeddings", "file", "index_map"],
                "vector": ["vector", "index", "meta"],
                "bm25": ["bm25", "index", "meta", "tokens"],
                "stopwords": ["stopwords", "file"],
                "graph": ["graph", "nodes", "edges"],
                "graph_vis": ["graph_vis", "file"],
                "wiki": ["wiki", "file"],
                "images": ["images", "dir"],
                "meta": ["meta", "file"],
            }
            for name, (stage, *keys) in path_map.items():
                size = 0
                if stage == "images" and keys[0] == "dir":
                    p = self.store.paths.dir_for_hash(h) / "images"
                    if p.exists():
                        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                else:
                    for key in keys:
                        p = self.store.paths.artifact_path(h, stage, key)
                        if p.exists():
                            if p.is_file():
                                size += p.stat().st_size
                            elif p.is_dir():
                                size += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                breakdown[name] = {"bytes": size, "human": self._human_bytes(size)}
                total += size
            breakdown["total"] = {"bytes": total, "human": self._human_bytes(total)}
            return breakdown

        if filenames is None:
            all_sources = self.store._sources.list_all()
            files = {}
            grand_total = 0
            for src in all_sources:
                bd = _file_breakdown(src.display_name)
                files[src.display_name] = bd
                grand_total += bd["total"]["bytes"]
            return {
                "scope": "all",
                "files": files,
                "grand_total": {"bytes": grand_total, "human": self._human_bytes(grand_total)},
            }

        if isinstance(filenames, str):
            filenames = [filenames]

        files = {}
        total = 0
        for name in filenames:
            bd = _file_breakdown(name)
            files[name] = bd
            total += bd["total"]["bytes"]
        return {
            "scope": "selected",
            "files": files,
            "total": {"bytes": total, "human": self._human_bytes(total)},
        }

    @staticmethod
    def _human_bytes(size: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"
