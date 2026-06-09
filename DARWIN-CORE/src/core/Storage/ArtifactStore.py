from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.Storage.Constants import ARTIFACT_MANIFEST, OPTIONAL_ARTIFACT_KEYS
from src.core.Storage.StorePaths import StorePaths


class ArtifactStore:
    """Dumb, format-agnostic artifact I/O. Knows paths, not serialization."""

    def __init__(self, paths: StorePaths):
        self.paths = paths

    def _artifact_paths(self, content_hash: str, stage: str) -> list[tuple[str, Path]]:
        spec = ARTIFACT_MANIFEST.get(stage)
        if not spec:
            return []
        dir_path = self.paths.dir_for_hash(content_hash)
        return [(key, dir_path / rel_path) for key, rel_path in spec.items()]

    def has_artifact(self, content_hash: str, stage: str) -> bool:
        entries = self._artifact_paths(content_hash, stage)
        if not entries:
            return False
        optional = OPTIONAL_ARTIFACT_KEYS.get(stage, set())
        for key, p in entries:
            if key in optional:
                continue
            if key == "dir":
                if not p.exists() or not p.is_dir():
                    return False
            else:
                if not p.exists():
                    return False
        return True

    def ensure_dir(self, content_hash: str) -> Path:
        return self.paths.ensure_dir_for_hash(content_hash)

    def delete_hash_dir(self, content_hash: str) -> None:
        from src.core.Storage.Transanctions import delete_tree
        dir_path = self.paths.dir_for_hash(content_hash)
        if dir_path.exists():
            delete_tree(dir_path)

    def list_hashes_on_disk(self) -> list[str]:
        indexes = self.paths.indexes_dir
        if not indexes.exists():
            return []
        return [d.name for d in indexes.iterdir() if d.is_dir()]