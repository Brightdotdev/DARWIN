import shutil
from pathlib import Path

from src.core.Storage.StorePaths import StorePaths, default_darwin_dir
from src.core.Storage.FileRegistry import FileRegistry, FileRecord
from src.core.Storage.StoreArtifacts import (
    MetaMixin,
    ConfigMixin,
    MarkdownMixin,
    VectorMixin,
    ChunksMixin,
    EmbeddingsMixin,
    BM25Mixin,
    StopwordsMixin,
    GraphMixin,
    WikiMixin,
)
from src.core.Storage.StoreUtils import DEFAULT_CONFIG, now, read_json, write_json
from src.lib.Terminal import TerminalUI


class DarwinStore(
    MetaMixin,
    ConfigMixin,
    MarkdownMixin,
    VectorMixin,
    ChunksMixin,
    EmbeddingsMixin,
    BM25Mixin,
    StopwordsMixin,
    GraphMixin,
    WikiMixin,
):
    def __init__(self):
        self._darwin_dir = default_darwin_dir()
        self.paths = StorePaths(self._darwin_dir)
        self.registry = FileRegistry(self.paths)

        from src.core.Models import setup, get_loader, ModelCategory

        prefs = self.get_model_preferences() if self.paths.config_path.exists() else None
        setup(preferences=prefs)

        self._init_config()

        from src.core.Models.Main import ModelManager
        self.models = ModelManager(darwin_store=self)

    # ─────────────────────────────────────────────

    def _init_config(self) -> None:
        if self.paths.config_path.exists():
            return

        from src.core.Models import setup, get_loader, ModelCategory

        setup()
        loader = get_loader()

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

    def remove_file(self, file_path: str | Path) -> bool:
        file_path = Path(file_path).resolve()
        removed = self.registry.remove(file_path)

        if file_path.exists():
            fh = self.paths.file_hash(file_path)
            index_dir = self.paths.dir_for_hash(fh)
            if index_dir.exists():
                shutil.rmtree(index_dir)
                return True

        return removed

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
            ref = Path(record.last_seen_path) if record.last_seen_path else record.original_name
            files[record.original_name] = self.file_status(ref)

        return {
            "darwin_dir": str(self._darwin_dir),
            "stats": stats,
            "files": files,
        }

    # ─────────────────────────────────────────────

    def print_dashboard(self):
        stats = self.registry.stats()
        needing = self.registry.list_needing_work()

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

        for record in needing:
            TerminalUI.info("file", record.original_name)
            TerminalUI.info("needs", record.missing_stages())
            TerminalUI.blank()
