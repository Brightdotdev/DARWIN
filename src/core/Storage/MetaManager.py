from __future__ import annotations

from pathlib import Path
from typing import Any

from core.Storage.StorePaths import StorePaths
from core.Storage.StoreUtils import now, read_json, write_json


class MetaManager:
    """Handles per-hash meta.json read/write."""

    def __init__(self, paths: StorePaths):
        self.paths = paths

    def load(self, content_hash: str) -> dict:
        path = self.paths.meta_path(content_hash)
        return read_json(path) or {}

    def update(
        self,
        content_hash: str,
        component: str,
        extras: dict[str, Any] | None = None,
    ) -> None:
        meta = self.load(content_hash)
        meta.setdefault("updates", {})
        ts = now()
        meta["updates"][component] = ts
        meta["updated_at"] = ts
        if extras:
            for key, value in extras.items():
                if isinstance(value, dict) and isinstance(meta.get(key), dict):
                    meta[key].update(value)
                else:
                    meta[key] = value
        write_json(self.paths.meta_path(content_hash), meta)

    def batch_update(self, content_hash: str, updates: dict[str, dict]) -> None:
        meta = self.load(content_hash)
        meta.setdefault("updates", {})
        ts = now()
        for component, extras in updates.items():
            meta["updates"][component] = ts
            if extras:
                for key, value in extras.items():
                    if isinstance(value, dict) and isinstance(meta.get(key), dict):
                        meta[key].update(value)
                    else:
                        meta[key] = value
        meta["updated_at"] = ts
        write_json(self.paths.meta_path(content_hash), meta)