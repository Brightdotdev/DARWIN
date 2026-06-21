from __future__ import annotations

from pathlib import Path

from src.core.Storage.DataModels import Source

class StoreSources:
    """Registration, removal, and path helpers for user-facing files."""

    def register(
        self,
        file_path: str | Path,
        page_count: int = 0,
        file_size: int = 0,
    ) -> Source:
        source, _ = self._sources.register(file_path, page_count, file_size)
        return source

    def get_or_register(self, file_path: str | Path) -> Source:
        path = Path(file_path)
        source = self._sources.find_by_name(path.name)
        if source:
            return source
        return self.register(file_path)

    def remove_file(self, filename: str) -> bool:
        source = self._sources.find_by_name(filename)
        if not source:
            return False
        self._sources.remove_by_name(filename)
        if self._sources.count_for_hash(source.content_hash) == 0:
            self._artifacts.delete_hash_dir(source.content_hash)
            self._cache.invalidate(f"*:{source.content_hash}")
        return True

    def ensure_image_dir(self, filename: str) -> Path:
        return self.paths.ensure_image_dir(self._resolve_hash(filename))

    def image_dir(self, filename: str) -> Path:
        """Alias for ingestion code that expects image_dir()."""
        return self.ensure_image_dir(filename)