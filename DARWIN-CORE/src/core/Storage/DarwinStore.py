from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.Storage.StorePaths import StorePaths, ARTIFACT_MANIFEST
from src.core.Storage.FileRegistry import FileRegistry, FileRecord
from src.lib.Terminal import TerminalUI
from src.lib.AppPaths import get_darwin_dir

from src.core.Storage.MetaMixn import MetaMixin
from src.core.Storage.RetrievalArtifacts import (
    VectorMixin,
    EmbeddingsMixin,
    SearchMixin,
)
from src.core.Storage.KnowledgeArtifacts import (
    GraphMixin,
    WikiMixin,
)
from src.core.Storage.StoreArtifacts import (
    MarkdownMixin,
    RecordsMixin,
    BM25Mixin,
    StopwordsMixin,
)
from src.core.Storage.StoreUtils import now


def _get_current_embedding_model() -> str:
    from src.core.Models import ModelManager, ModelCategory
    loader = ModelManager.get_loader()
    return loader.get_model_key(ModelCategory.EMBEDDING)



class DarwinStore(
    MetaMixin,
    MarkdownMixin,
    VectorMixin,
    RecordsMixin,
    EmbeddingsMixin,
    BM25Mixin,
    StopwordsMixin,
    GraphMixin,
    WikiMixin,
    SearchMixin,
):
    def __init__(self, darwin_dir: Path | None = None):
        self._darwin_dir = darwin_dir or get_darwin_dir()
        self.paths = StorePaths(self._darwin_dir)
        self.registry = FileRegistry(self.paths)
        self._artifact_cache: dict[str, Any] = {}
        self._init_setup()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _hash_for(self, filename: str) -> str:
        """Resolve a registered filename to its content hash."""
        record = self.registry.find_by_name(filename)
        if not record:
            raise KeyError(f"File not registered: {filename}")
        return record.file_hash

    def _invalidate_disk_tree(self) -> None:
        """Invalidate the cached disk tree so it rebuilds on next access."""
        if hasattr(self, "_cached_disk_tree"):
            delattr(self, "_cached_disk_tree")

    # ── Setup ─────────────────────────────────────────────────────────────

    def _init_setup(self) -> None:
        from src.features.Hardware.SystemDetector import detect_system
        from src.core.Models import ModelManager
        from src.features.Hardware.DeviceManager import DeviceManager

        if DeviceManager._config is None:
            DeviceManager.init(detect_system())

        ModelManager.setup()

    # ── Cache ─────────────────────────────────────────────────────────────

    def invalidate(self, filename: str) -> None:
        """Clear all cached artifacts for a file after save."""
        keys_to_remove = [k for k in self._artifact_cache if k.endswith(f":{filename}")]
        for k in keys_to_remove:
            del self._artifact_cache[k]

   
   

    def get_embedding_health(self, model_id: str) -> dict:
        from src.core.Models.ModelRegistry import REGISTRY
        from src.core.Models.ModelStore import get_model_store

        model_store = get_model_store()
        health = {
            "active_model": model_id,
            "up_to_date": [],
            "needs_reembed": [],
            "model_missing": [],
            "never_embedded": [],
        }

        id_to_key = {spec.model_id: key for key, spec in REGISTRY.items()}

        for record in self.registry.list_all():
            name = record.original_name

            if not record.embedding_model:
                health["never_embedded"].append(name)
                continue

            if record.embedding_model == model_id:
                health["up_to_date"].append(name)
                continue

            old_key = id_to_key.get(record.embedding_model)
            if old_key and model_store.is_downloaded(old_key):
                health["needs_reembed"].append(name)
            else:
                health["model_missing"].append(name)

        return health

    # ── Registry delegation ───────────────────────────────────────────────

    def register(
        self,
        file_path: str | Path,
        page_count: int = 0,
        file_size: int = 0,
    ) -> FileRecord:
        record = self.registry.register(
            file_path, page_count=page_count, file_size=file_size
        )
        self._invalidate_disk_tree()
        return record



    def remove_file(self, filename: str) -> bool:
        """Remove everything known about a file by its original name."""
        removed_count = self.registry.remove_by_name(filename)
        if removed_count:
            self._invalidate_disk_tree()
        return removed_count > 0

    def ensure_image_dir(self, filename: str) -> Path:
        """Create and return the image subdirectory for a registered file."""
        return self.paths.ensure_image_dir(self._hash_for(filename))

    # ── File status ───────────────────────────────────────────────────────

    def file_status(self, file_path: str | Path) -> dict:
        file_path = Path(file_path).resolve()
        record = self.registry.get_or_register(file_path)

        if file_path.exists():
            # Refresh metadata for an existing file
            record = self.registry.register(file_path)

        return {
            "file_hash": record.file_hash,
            "original_name": record.original_name,
            "last_seen_path": record.last_seen_path,
            "file_size_mb": round(record.file_size_mb, 1),
            "page_count": record.page_count,
            "is_parsed": record.is_parsed,
            "is_chunked": record.is_chunked,
            "is_embedded": record.is_embedded,
            "is_kw_searchable": record.is_kw_searchable,
            "is_embedding_searchable": record.is_embedding_searchable,
            "is_hybrid_searchable": record.is_hybrid_searchable,
            "has_graph": record.has_graph,
            "has_wiki": record.has_wiki,
            "progress_percent": record.progress_percent(),
            "status_summary": record.status_summary(),
            "missing_stages": record.missing_stages(),
            "embedding_model": record.embedding_model,
            "chunk_count": record.chunk_count,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    def status(self) -> dict:
        stats = self.registry.stats()
        files: dict[str, dict] = {}

        for record in self.registry.list_all():
            files[record.original_name] = {
                "file_hash": record.file_hash,
                "original_name": record.original_name,
                "last_seen_path": record.last_seen_path,
                "file_size_mb": round(record.file_size_mb, 1),
                "page_count": record.page_count,
                "is_parsed": record.is_parsed,
                "is_chunked": record.is_chunked,
                "is_embedded": record.is_embedded,
                "is_kw_searchable": record.is_kw_searchable,
                "is_embedding_searchable": record.is_embedding_searchable,
                "is_hybrid_searchable": record.is_hybrid_searchable,
                "has_graph": record.has_graph,
                "has_wiki": record.has_wiki,
                "progress_percent": record.progress_percent(),
                "status_summary": record.status_summary(),
                "missing_stages": record.missing_stages(),
                "embedding_model": record.embedding_model,
                "chunk_count": record.chunk_count,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            }

        return {
            "darwin_dir": str(self._darwin_dir),
            "stats": stats,
            "files": files,
        }
    def get_or_register(self, file_path: str | Path) -> FileRecord:
        return self.registry.get_or_register(file_path)

    def update_pipeline(self, file_path: str | Path, stage: str) -> FileRecord:
        record = self.registry.update_pipeline(file_path, stage)
        self.invalidate(Path(file_path).name)
        return record

        
        # ── Registry repair / health ──────────────────────────────────────────

    def repair_registry(self, filename: str | None = None, dry_run: bool = False) -> dict:
        if filename:
            record = self.registry.find_by_name(filename)
            if not record:
                return {"error": f"File not registered: {filename}"}
            records = [record]
        else:
            records = self.registry.list_all()

        result = {
            "files_checked": len(records),
            "files_healthy": 0,
            "files_repaired": 0,
            "files_removed": 0,
            "repairs": [],
        }

        all_stages = list(ARTIFACT_MANIFEST.keys()) if ARTIFACT_MANIFEST else []

        for record in records:
            name = record.original_name
            report = self.validate_artifact_integrity(name)
            if "error" in report:
                # Skip files that failed integrity check
                continue

            if report["is_healthy"]:
                result["files_healthy"] += 1
                continue

            broken_stages = report["actually_missing"]
            orphan_stages = report["actually_present"]

            # CASE 1: ALL artifacts missing → entire index deleted
            if all_stages and broken_stages and set(broken_stages) == set(all_stages):
                if not dry_run:
                    data = self.registry.load_data()
                    fh = record.file_hash
                    if fh in data.get("files", {}):
                        del data["files"][fh]
                        self.registry.save_data(data)
                        TerminalUI.bold(
                            f"Removed '{name}' from registry — "
                            f"entire index directory was deleted"
                        )
                        self._invalidate_disk_tree()

                result["files_removed"] += 1
                result["repairs"].append({
                    "filename": name,
                    "action": "removed",
                    "reason": "entire index directory deleted",
                })
                continue

            # CASE 2: Some stages broken — clear timestamps
            if not dry_run and broken_stages:
                data = self.registry.load_data()
                fh = record.file_hash
                updates = data["files"][fh].get("updates", {})
                for stage in broken_stages:
                    if stage in updates:
                        del updates[stage]
                        TerminalUI.bold(
                            f"Repaired registry for '{name}': cleared '{stage}' "
                            f"(artifacts missing, registry was incorrect)"
                        )
                data["files"][fh]["updated_at"] = now()
                self.registry.save_data(data)
                self._invalidate_disk_tree()

                result["files_repaired"] += 1
                result["repairs"].append({
                    "filename": name,
                    "cleared_stages": broken_stages,
                })

            # CASE 3: Orphan artifacts — files exist but registry doesn't know
            if orphan_stages and not dry_run:
                TerminalUI.bold(
                    f"Found orphan artifacts for '{name}': {orphan_stages} "
                    f"(files exist but registry doesn't know — auto-registering)"
                )
                for stage in orphan_stages:
                    self.registry.update_pipeline(
                        file_hash=record.file_hash, stage=stage
                    )

        return result

    def full_registry_health_check(self) -> dict:
        """Run a complete health check without repairing."""
        report = self.repair_registry(dry_run=True)

        file_details = {}
        for record in self.registry.list_all():
            integrity = self.validate_artifact_integrity(record.original_name)
            file_details[record.original_name] = integrity

        return {
            "summary": {
                "total_files": report["files_checked"],
                "healthy": report["files_healthy"],
                "broken": report["files_checked"] - report["files_healthy"],
                "would_repair": len(report["repairs"]),
            },
            "broken_files": [
                {
                    "filename": r["filename"],
                    "missing_artifacts": r.get("cleared_stages", []),
                }
                for r in report["repairs"]
            ],
            "all_files": file_details,
        }

    # ── Search capabilities ───────────────────────────────────────────────

    def get_searchables(self, filename: str) -> dict:
        KW_REQUIRED = ["markdown", "chunks", "bm25", "stopwords"]
        SEMANTIC_REQUIRED = ["markdown", "chunks", "embeddings", "vector"]

        record = self.registry.find_by_name(filename)
        if record is None:
            return {
                "filename": filename,
                "kw_searchable": False,
                "semantic_searchable": False,
                "hybrid_searchable": False,
                "missing_stages": ["not_registered"],
            }

        report = self.validate_artifact_integrity(filename)
        missing = report.get("actually_missing", [])

        kw_searchable = not any(stage in missing for stage in KW_REQUIRED)
        semantic_searchable = not any(stage in missing for stage in SEMANTIC_REQUIRED)
        hybrid_searchable = kw_searchable and semantic_searchable

        return {
            "filename": filename,
            "kw_searchable": kw_searchable,
            "semantic_searchable": semantic_searchable,
            "hybrid_searchable": hybrid_searchable,
            "missing_stages": missing,
        }

    # ── Dashboard ─────────────────────────────────────────────────────────

    def print_dashboard(self) -> None:
        stats = self.registry.stats()
        needing = self.registry.list_needing_work()
        health = self.full_registry_health_check()

        TerminalUI.print_darwin()
        TerminalUI.blank()
        TerminalUI.bold("DARWIN DASHBOARD")
        TerminalUI.dim(
            f"Files indexed: {stats['total_files']} | "
            f"Searchable: {stats['keyword_searchable_files']} | "
            f"Needs work: {stats['files_needing_work']} | "
            f"Total chunks: {stats['total_chunks']} | "
            f"Total pages: {stats['total_pages']}"
        )
        TerminalUI.blank()

        if health["summary"]["broken"] > 0:
            TerminalUI.warn(
                f"⚠️  Registry integrity issue: {health['summary']['broken']} file(s) "
                f"have missing artifacts but the registry thinks they're done."
            )
            for broken in health["broken_files"]:
                TerminalUI.warn(
                    f"   📄 {broken['filename']} — missing: {', '.join(broken['missing_artifacts'])}"
                )
            TerminalUI.info(
                "Try this:",
            "   Run darwin_store.repair_registry() to fix automatically."
            )
            TerminalUI.blank()

        for record in needing:
            TerminalUI.info("file", record.original_name)
            TerminalUI.info("needs", record.missing_stages())
            TerminalUI.blank()

    # ── Disk treemap ─────────────────────────────────────────────────────

    def _build_disk_tree(self):
        """Return (and cache) the full disk tree of the Darwin directory."""
        if not hasattr(self, "_cached_disk_tree"):
            from src.core.Storage.DiskUsage import build_tree
            from src.core.Models.ModelStore import get_model_store
            self._cached_disk_tree = build_tree(
                self._darwin_dir, self.registry, get_model_store()
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