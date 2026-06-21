from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src.core.Storage.ArtifactStore import ArtifactStore
from src.core.Storage.CacheManager import CacheManager
from src.core.Storage.HealthMonitor import HealthMonitor
from src.core.Storage.MetaManager import MetaManager
from src.core.Storage.StoreRegistry import ContentIndex, Registry, SourceManager
from src.core.Storage.StorePaths import StorePaths
from src.lib.AppPaths import get_darwin_dir
from src.core.Logger.DarwinLogger import darwin,LogType


def get_current_embedding_model() -> str:
    from src.core.Models.ModelConfig import get_active_model
    from src.core.Models.Lib.ModelSpec import  ModelCategory
    return get_active_model(ModelCategory.EMBEDDING.name.lower()) or ""


def get_current_model_dimensions() -> int:
    try:
        from src.core.Models.EmbeddingConfig import get_preset
        key = get_current_embedding_model()
        presets = get_preset(key)
        if presets:
            return presets.dimensions
        _DIMENSIONS = {
            "intfloat/e5-small-v2": 384,
            "BAAI/bge-m3": 1024,
            "GritLM/GritLM-7B": 4096,
        }
        return _DIMENSIONS.get(key, 0)
    except Exception:
        return 0


class StoreCore:
    """Infrastructure, init, cache, and internal helpers."""

    def __init__(self, darwin_dir: Path | None = None):
        self._darwin_dir = darwin_dir or get_darwin_dir()
        self.paths = StorePaths(self._darwin_dir)
        self.paths.indexes_dir.mkdir(parents=True, exist_ok=True)

        self._registry = Registry(self.paths.registry_db_path)
        self._sources = SourceManager(self._registry)
        self._contents = ContentIndex(self._registry)
        self._artifacts = ArtifactStore(self.paths)
        self._cache = CacheManager(maxsize=256)
        self._meta = MetaManager(self.paths)
        self._health = HealthMonitor(
            self._registry, self._sources, self._contents, self._artifacts, self.paths
        )

        self._init_setup()

    def _init_setup(self) -> None:
        from src.core.Models.ModelManager import modelManager
        modelManager.initialize()
        self.print_dashboard()

    def _resolve_hash(self, filename: str) -> str | None:
        source = self._sources.find_by_name(filename)
        if not source:
            darwin.log(f"File not registered: {filename}", LogType.ERROR)
            darwin.log(f"All operations will be aborted.", LogType.ERROR)
            darwin.log(f"[TIP] Try parsing the file first.")
            return None
        return source.content_hash

    def _hash_for(self, filename: str) -> str:
        return self._resolve_hash(filename)



    def invalidate(self, filename: str) -> None:
        prefix = f"*:{self._resolve_hash(filename)}"
        self._cache.invalidate(prefix)

    @staticmethod
    def _human_bytes(size: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"