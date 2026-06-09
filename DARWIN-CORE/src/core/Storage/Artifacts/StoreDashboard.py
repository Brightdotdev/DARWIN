from __future__ import annotations

from typing import Any
from pathlib import Path
from src.core.Logger.DarwinLogger import darwin, LogType


class StoreDashboard:
    """Dashboard printing, treemap, and per-file disk accounting."""

    def print_dashboard(self) -> None:
        stats = self.status()["stats"]
        needing = [
            s for s in self._sources.list_all()
            if not self._contents.get_status(s.content_hash).is_core_complete
        ]
        health = self.full_registry_health_check()

        darwin.log(f"[DARWIN DIRERCTORY] {Path(self._darwin_dir).resolve()}")
        darwin.log(
            f"Files Registered: {stats['total_files']} | "
            f"Searchable: {stats['keyword_searchable_files']} | "
            f"Needs work: {stats['files_needing_work']} | "
            f"Total chunks: {stats['total_chunks']} | "
            f"Total pages: {stats['total_pages']}"
        )
        
        if health["summary"]["broken"] > 0:
            darwin.log(
                f"⚠️  Registry integrity issue: {health['summary']['broken']} hash(es) "
                f"have missing artifacts but the registry thinks they're done.", LogType.WARNING
            )
            for broken in health["broken_hashes"]:
                names = [s.display_name for s in self._sources.get_by_hash(broken["hash"])]
                darwin.log(
                    f"   📄 {', '.join(names)} — missing: {', '.join(broken['missing'])}", LogType.WARNING
                )
            darwin.log("Try this:", "Run darwin_store.repair_registry() to fix automatically.")
          

        for source in needing:
            st = self._contents.get_status(source.content_hash)
            darwin.log(f"[FILE] {source.display_name}")
            darwin.log(f"[NEEDS] {st.missing_stages}")
          


    def _build_disk_tree(self):
        if not hasattr(self, "_cached_disk_tree"):
            from src.core.Storage.DiskUsage import build_tree
            from src.core.Models.ModelStore import get_model_store

            def hash_to_label(h: str) -> str:
                sources = self._sources.get_by_hash(h)
                if sources:
                    names = ", ".join(s.display_name for s in sources)
                    return f"{names} ({h[:8]}…)"
                return ""

            self._cached_disk_tree = build_tree(
                self._darwin_dir,
                self._sources,
                get_model_store(),
                hash_label_fn=hash_to_label,
            )
        return self._cached_disk_tree

    def get_treemap(self, subpath: str | None = None) -> dict:
        from src.core.Storage.DiskUsage import disk_node_to_dict, find_node
        tree = self._build_disk_tree()
        if subpath:
            parts = subpath.strip("/").split("/")
            node = find_node(tree, parts)
            if node is None:
                raise ValueError(f"Subpath not found: {subpath}")
            return disk_node_to_dict(node)
        return disk_node_to_dict(tree)

    def print_treemap(self, subpath: str | None = None) -> None:
        from src.core.Storage.DiskUsage import print_tree, find_node
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

    def get_disk_breakdown(
        self,
        filenames: str | list[str] | None = None,
    ) -> dict:
        def _file_breakdown(filename: str) -> dict:
            h = self._resolve_hash(filename)
            breakdown: dict[str, Any] = {}
            total = 0

            path_map = {
                "markdown": [self.paths.markdown_path(h), self.paths.display_markdown_path(h)],
                "chunks": [self.paths.chunks_path(h)],
                "embeddings": [self.paths.embedding_path(h), self.paths.embedding_index_map_path(h)],
                "vector": [self.paths.vector_index_path(h), self.paths.vector_json_path(h)],
                "bm25": [self.paths.bm25_path(h), self.paths.bm25_meta_path(h), self.paths.bm25_tokens_path(h)],
                "stopwords": [self.paths.stopwords_path(h)],
                "graph": [self.paths.graph_nodes_path(h), self.paths.graph_edges_path(h)],
                "graph_vis": [self.paths.graph_html_path(h)],
                "wiki": [self.paths.wiki_path(h)],
                "images": [self.paths.ensure_image_dir(h)],
                "meta": [self.paths.meta_path(h)],
            }

            for name, paths in path_map.items():
                size = 0
                for p in paths:
                    if p.exists():
                        if p.is_dir():
                            size += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                        else:
                            size += p.stat().st_size
                breakdown[name] = {"bytes": size, "human": self._human_bytes(size)}
                total += size

            breakdown["total"] = {"bytes": total, "human": self._human_bytes(total)}
            return breakdown

        if filenames is None:
            all_sources = self._sources.list_all()
            files: dict[str, Any] = {}
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

        files: dict[str, Any] = {}
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