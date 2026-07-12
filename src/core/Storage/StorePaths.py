from __future__ import annotations

import hashlib
from functools import cached_property
from pathlib import Path

from src.core.Logger.DarwinLogger import darwin
from src.core.Storage.Constants import ARTIFACT_MANIFEST, OPTIONAL_ARTIFACT_KEYS
from src.core.Storage.Transanctions import delete_tree


def hash_file(file_path: str | Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def is_file_hash(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


class StorePaths:
    def __init__(self, darwin_dir: str | Path):
        if darwin_dir is None:
            darwin.error("darwin_dir cannot be None")
            self.root = Path()
            return
        darwin_dir = Path(darwin_dir)
        if darwin_dir.exists() and not darwin_dir.is_dir():
            darwin.error(f"{darwin_dir} exists and is not a directory")
            self.root = Path()
            return
        darwin_dir.mkdir(parents=True, exist_ok=True)
        self.root = darwin_dir.resolve()

    @cached_property
    def indexes_dir(self) -> Path:
        return self.root / "indexes"

    @property
    def file_registry_path(self) -> Path:
     return self.indexes_dir / "file_registry.json"

    @property
    def registry_db_path(self) -> Path:
     return self.indexes_dir / "registry.db"

    @cached_property
    def checkpoints_dir(self) -> Path:
     return self.indexes_dir / "checkpoints"

    @cached_property
    def conversations_dir(self) -> Path:
     return self.indexes_dir / "conversations"

    def dir_for_hash(self, file_hash: str) -> Path:
        if not is_file_hash(file_hash):
            darwin.error(f"Invalid file hash: {file_hash}")
            return Path()
        return self.indexes_dir / file_hash

    def ensure_dir_for_hash(self, file_hash: str) -> Path:
        path = self.dir_for_hash(file_hash)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_image_dir(self, file_hash: str) -> Path:
        path = self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["images"]["dir"]
        path.mkdir(parents=True, exist_ok=True)
        return path


    # --- Unified artifact path resolution ---
    def artifact_path(self, content_hash: str, stage: str, key: str) -> Path:
        """Return the absolute path for a specific artifact key under a stage."""
        rel = ARTIFACT_MANIFEST[stage][key]
        return self.dir_for_hash(content_hash) / rel

    # Convenience alias for meta.json
    def meta_path(self, content_hash: str) -> Path:
        return self.artifact_path(content_hash, "meta", "file")

    # --- ArtifactStore merged methods ---
    def _artifact_entries(self, content_hash: str, stage: str) -> list[tuple[str, Path]]:
        spec = ARTIFACT_MANIFEST.get(stage)
        if not spec:
            return []
        dir_path = self.dir_for_hash(content_hash)
        return [(key, dir_path / rel_path) for key, rel_path in spec.items()]

    def has_artifact(self, content_hash: str, stage: str) -> bool:
        entries = self._artifact_entries(content_hash, stage)
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
        return self.ensure_dir_for_hash(content_hash)

    def delete_hash_dir(self, content_hash: str) -> None:
        dir_path = self.dir_for_hash(content_hash)
        if dir_path.exists():
            delete_tree(dir_path)

    def list_hashes_on_disk(self) -> list[str]:
        if not self.indexes_dir.exists():
            return []
        return [d.name for d in self.indexes_dir.iterdir() if d.is_dir()]