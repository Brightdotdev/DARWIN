import shutil
from pathlib import Path

from dataclasses import dataclass
from typing import Any, List, Optional
from src.core.Models.ModelStore import get_model_store
from src.core.Storage.StorePaths import StorePaths, default_darwin_dir
from src.core.Storage.FileRegistry import FileRegistry, FileRecord
from src.core.Storage.StoreArtifacts import (
    MetaMixin,
    ConfigMixin,
    MarkdownMixin,
    VectorMixin,
    RecordsMixin,
    EmbeddingsMixin,
    BM25Mixin,
    StopwordsMixin,
    GraphMixin,
    SearchMixin,
    WikiMixin)
from src.core.Storage.StoreUtils import DEFAULT_CONFIG, now, read_json, write_json, invalidate_model_cache
from src.lib.Terminal import TerminalUI


class DarwinStore(
    MetaMixin,
    ConfigMixin,
    MarkdownMixin,
    VectorMixin,
    RecordsMixin,
    EmbeddingsMixin,
    BM25Mixin,
    StopwordsMixin,
    GraphMixin,
    WikiMixin,
    SearchMixin
):
    def __init__(self, darwin_dir: Path = None):
        self._darwin_dir = darwin_dir or default_darwin_dir()
        self.paths = StorePaths(self._darwin_dir)
        self.registry = FileRegistry(self.paths)
        self._artifact_cache: dict = {}

        from src.core.Models import ModelManager
        prefs = self.get_model_preferences() if self.paths.config_path.exists() else None
        ModelManager.setup(preferences=prefs)

        self._init_config()

        from src.core.Models.Main import ModelManager
        self.models = ModelManager(darwin_store=self)

    def invalidate(self, filename: str) -> None:
        """Clear all cached artifacts for a file after save."""
        keys_to_remove = [k for k in self._artifact_cache if k.endswith(f":{filename}")]
        for k in keys_to_remove:
            del self._artifact_cache[k]

    def _init_config(self) -> None:
        if self.paths.config_path.exists():
            return

        from src.core.Models import ModelManager, ModelCategory

        ModelManager.setup()
        loader = ModelManager.get_loader()

        if loader is None:
            TerminalUI.warn("ModelLoader not available during config init")
            config = {
                **DEFAULT_CONFIG,
                "created_at": now(),
            }
        else:
            config = {
                **DEFAULT_CONFIG,
                "created_at": now(),
                "embedding_model": loader.get_model_key(ModelCategory.EMBEDDING),
                "vision_model":    loader.get_model_key(ModelCategory.VISION),
                "reranker_model":  loader.get_model_key(ModelCategory.RERANKER),
                "tokenizer_model": loader.get_model_key(ModelCategory.TOKENIZER),
            }

        write_json(self.paths.config_path, config)
        TerminalUI.dim("Created config.json with detected models")

    # ─────────────────────────────────────────────

    def update_model_preference(self, category: str, key: str) -> None:
        config = read_json(self.paths.config_path)
        config[f"{category}_model"] = key
        write_json(self.paths.config_path, config)
        invalidate_model_cache()
        TerminalUI.dim(f"Config updated: {category}_model → {key}")

    def get_model_preferences(self) -> dict:
        config = read_json(self.paths.config_path)
        return {
            "embedding": config.get("embedding_model"),
            "reranker": config.get("reranker_model"),
            "vision": config.get("vision_model"),
            "tokenizer": config.get("tokenizer_model"),
        }

    # ─────────────────────────────────────────────

    def get_embedded_files_by_model(self) -> dict:
        from src.core.Models.ModelRegistry import REGISTRY
        from src.core.Models.ModelStore import ModelStore

        prefs = self.get_model_preferences()
        model_store = ModelStore()

        id_to_key = {spec.model_id: key for key, spec in REGISTRY.items()}

        active_ids = {}
        for category, key in prefs.items():
            if key and key in REGISTRY:
                active_ids[category] = REGISTRY[key].model_id

        result = {"embedding": {}}

        for record in self.registry.list_all():
            model_id = record.embedding_model
            if not model_id:
                continue

            if model_id not in result["embedding"]:
                registry_key = id_to_key.get(model_id)
                is_installed = (
                    model_store.is_downloaded(registry_key)
                    if registry_key else False
                )

                result["embedding"][model_id] = {
                    "model_id": model_id,
                    "registry_key": registry_key,
                    "model_installed": is_installed,
                    "is_active": model_id == active_ids.get("embedding"),
                    "files": [],
                    "file_count": 0,
                }

            result["embedding"][model_id]["files"].append(record.original_name)
            result["embedding"][model_id]["file_count"] += 1

        return result

    # ─────────────────────────────────────────────

    def get_embedding_health(self) -> dict:
        from src.core.Models.ModelRegistry import REGISTRY
        from src.core.Models.ModelStore import ModelStore

        prefs = self.get_model_preferences()
        active_key = prefs.get("embedding")

        active_model_id = (
            REGISTRY[active_key].model_id
            if active_key in REGISTRY else None
        )

        health = {
            "active_model": active_model_id,
            "up_to_date": [],
            "needs_reembed": [],
            "model_missing": [],
            "never_embedded": [],
        }

        id_to_key = {spec.model_id: key for key, spec in REGISTRY.items()}
        model_store = ModelStore()

        for record in self.registry.list_all():
            name = record.original_name

            if not record.embedding_model:
                health["never_embedded"].append(name)
                continue

            if record.embedding_model == active_model_id:
                health["up_to_date"].append(name)
                continue

            old_key = id_to_key.get(record.embedding_model)

            if old_key and model_store.is_downloaded(old_key):
                health["needs_reembed"].append(name)
            else:
                health["model_missing"].append(name)

        return health

    # ─────────────────────────────────────────────

    def register(
        self,
        file_path: str | Path,
        page_count: int = 0,
        file_size: int = 0,
    ) -> FileRecord:
        return self.registry.register(file_path, page_count=page_count, file_size=file_size)

    def get_or_register(self, file_path: str | Path) -> FileRecord:
        return self.registry.get_or_register(file_path)

    def update_pipeline(self, file_path: str | Path, stage: str, **extras) -> FileRecord:
        return self.registry.update_pipeline(file_path, stage, **extras)

    # ─────────────────────────────────────────────
    def remove_file(self, filename: str) -> bool:
        """
        Remove everything known about a file, using its original name.
        Returns True if at least one record was removed.
        """
        removed_count = self.registry.remove_by_name(filename)
        return removed_count > 0


    # ─────────────────────────────────────────────

    def image_dir(self, filename: str) -> Path:
        fh = self._hash_for(filename)
        return self.paths.dir_for_hash(fh) / "images"

    def ensure_image_dir(self, filename: str) -> Path:
        d = self.image_dir(filename)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def image_dir_for_path(self, file_path: str | Path) -> Path:
        fh = self.paths.file_hash(file_path)
        d = self.paths.dir_for_hash(fh) / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ─────────────────────────────────────────────

    def file_status(self, file_path: str | Path) -> dict:
        file_path = Path(file_path).resolve()
        record = self.registry.get_or_register(file_path)

        if file_path.exists():
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
        files = {}

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
                    
                "has_graph": record.has_graph,
                "has_wiki": record.has_wiki,
                    "is_kw_searchable": record.is_kw_searchable,
                "is_embedding_searchable": record.is_embedding_searchable,
                "is_hybrid_searchable": record.is_hybrid_searchable,
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



    def repair_registry(self, filename: str = None, dry_run: bool = False) -> dict:
        """
        Check and repair registry integrity.

        Args:
            filename: If None, check/repair ALL files. If provided, check/repair one.
            dry_run: If True, report what WOULD be fixed without changing anything.

        Returns:
            {
                "files_checked": 5,
                "files_healthy": 3,
                "files_repaired": 2,
                "repairs": [
                    {"filename": "bio.pdf", "cleared_stages": ["chunks", "bm25"]},
                    ...
                ]
            }
        """
        from src.core.Storage.StoreArtifacts import ARTIFACT_MANIFEST

        if filename:
            records = [self.registry._find_by_name(filename)]
            if records[0] is None:
                return {"error": f"File not registered: {filename}"}
        else:
            records = self.registry.list_all()

        result = {
            "files_checked": len(records),
            "files_healthy": 0,
            "files_repaired": 0,
            "repairs": [],
        }

        for record in records:
            name = record.original_name
            report = self.validate_artifact_integrity(name)

            if report["is_healthy"]:
                result["files_healthy"] += 1
                continue

            # Found broken stages — the registry claims they're done but files don't exist
            broken_stages = report["actually_missing"]

            if not dry_run and broken_stages:
                # Clear the broken stage timestamps from the registry
                # This effectively "un-does" the pipeline progress for those stages
                data = self.registry._load()
                fh = record.file_hash

                for stage in broken_stages:
                    if stage in data["files"][fh].get("updates", {}):
                        del data["files"][fh]["updates"][stage]
                        TerminalUI.bold(
                            f"Repaired registry for '{name}': cleared '{stage}' "
                            f"(artifacts missing, registry was incorrect)"  )

                data["files"][fh]["updated_at"] = now()
                self.registry._save(data)

                result["files_repaired"] += 1
                result["repairs"].append({
                    "filename": name,
                    "cleared_stages": broken_stages,
                })

            # Also handle the inverse: files exist but registry doesn't know
            orphan_stages = report["actually_present"]
            if orphan_stages and not dry_run:
                TerminalUI.bold(
                    f"Found orphan artifacts for '{name}': {orphan_stages} "
                    f"(files exist but registry doesn't know — auto-registering)"                )
                for stage in orphan_stages:
                    self.registry.update_pipeline(name, stage)

        return result


    def full_registry_health_check(self) -> dict:
        """
        Run a complete health check on the entire registry.
        Returns a dashboard-friendly report.
        """
        report = self.repair_registry(dry_run=True)  # Check without modifying

        # Enrich with per-file details
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
                    "missing_artifacts": r["cleared_stages"],
                }
                for r in report["repairs"]
            ],
            "all_files": file_details,
        }
    # ─────────────────────────────────────────────

    def print_dashboard(self):
        stats = self.registry.stats()
        needing = self.registry.list_needing_work()

        # Quick health check (non-repairing)
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

        # ═══ Health warning ═══
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
                "   Run darwin_store.repair_registry() to fix automatically."
            )
            TerminalUI.blank()

        for record in needing:
            TerminalUI.info("file", record.original_name)
            TerminalUI.info("needs", record.missing_stages())
            TerminalUI.blank()

    
        # ── Disk Treemap API ─────────────────────────────────────────────

    def _build_disk_tree(self):
        """Return (and cache) the full disk tree of the Darwin directory."""
        if not hasattr(self, '_cached_disk_tree'):

            from src.core.Storage.DiskUsage import build_tree

            self._cached_disk_tree = build_tree(
                self._darwin_dir, self.registry, get_model_store()
            )
        return self._cached_disk_tree

    def get_treemap(self, subpath: str = None) -> dict:
        """
        Return a nested dict representation of disk usage.

        Parameters
        ----------
        subpath : str, optional
            A forward‑slash‑separated path relative to the Darwin directory.
            Examples: "indexes/abc123def", "/models", "conversations"

        Returns
        -------
        dict with keys 'name', 'size', and optionally 'children' and 'label'.
        Suitable for feeding a React treemap component.
        """
        from src.core.Storage.DiskUsage import disk_node_to_dict, find_node


        tree = self._build_disk_tree()
        if subpath:
            parts = subpath.strip("/").split("/")
            node = find_node(tree, parts)
            if node is None:
                raise ValueError(f"Subpath not found: {subpath}")
            return disk_node_to_dict(node)
        return disk_node_to_dict(tree)

    def print_treemap(self, subpath: str = None):
        """
        Print an ASCII tree of disk usage to the terminal.

        Parameters
        ----------
        subpath : str, optional
            Same format as `get_treemap`. If omitted, prints the whole Darwin directory.
        """
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