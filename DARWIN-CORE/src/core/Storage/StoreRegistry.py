from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Optional

from src.lib.AppPaths import default_darwin_config_dir,get_darwin_dir


SCHEMA = """
CREATE TABLE IF NOT EXISTS stores (
    id TEXT PRIMARY KEY,
    slug TEXT UNIQUE,
    name TEXT,
    darwin_dir TEXT UNIQUE,
    created_at TEXT,
    updated_at TEXT,
    last_opened TEXT,
    is_default INTEGER DEFAULT 0,
    cache_json TEXT
);
"""


class StoreRegistry:
    """Thread-safe SQLite registry for tracking DarwinStore instances (central)."""

    def __init__(self, registry_path: Optional[Path] = None):
        if registry_path is None:
            registry_path = default_darwin_config_dir() / "stores_registry.db"
        self.db_path = Path(registry_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.commit()

    def _execute(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def _commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def register(self, darwin_dir: Path, store_id: Optional[str] = None, name: Optional[str] = None) -> str:
        """Register a store. Returns its UUID. If store_id is None, generate one."""
        darwin_dir = Path(darwin_dir).resolve()
        if store_id is None:
            store_id = str(uuid.uuid4())
        # Generate slug from directory name if name not provided
        slug = name if name else darwin_dir.name.lower().replace(" ", "_")
        # Ensure slug is unique by appending a suffix if needed
        base_slug = slug
        counter = 1
        while True:
            existing = self._execute("SELECT id FROM stores WHERE slug = ?", (slug,)).fetchone()
            if not existing or existing["id"] == store_id:
                break
            slug = f"{base_slug}_{counter}"
            counter += 1

        now = self._now()
        # Check if store already exists by ID or path
        existing = self._execute("SELECT * FROM stores WHERE darwin_dir = ? OR id = ?", (str(darwin_dir), store_id)).fetchone()
        if existing:
            # Update existing record
            self._execute(
                "UPDATE stores SET slug = ?, name = ?, darwin_dir = ?, updated_at = ?, last_opened = ? WHERE id = ?",
                (slug, name, str(darwin_dir), now, now, existing["id"])
            )
            store_id = existing["id"]
        else:
            # Insert new
            self._execute(
                "INSERT INTO stores (id, slug, name, darwin_dir, created_at, updated_at, last_opened) VALUES (?,?,?,?,?,?,?)",
                (store_id, slug, name, str(darwin_dir), now, now, now)
            )

        # If this is the default darwin directory, set as default
        if darwin_dir == get_darwin_dir():
            self._execute("UPDATE stores SET is_default = 0 WHERE is_default = 1 AND id != ?", (store_id,))
            self._execute("UPDATE stores SET is_default = 1 WHERE id = ?", (store_id,))

        self._commit()
        return store_id

    def get_store(self, store_id: str) -> Optional[dict]:
        row = self._execute("SELECT * FROM stores WHERE id = ?", (store_id,)).fetchone()
        return dict(row) if row else None

    def get_store_by_path(self, darwin_dir: Path) -> Optional[dict]:
        row = self._execute("SELECT * FROM stores WHERE darwin_dir = ?", (str(darwin_dir),)).fetchone()
        return dict(row) if row else None

    def get_store_by_slug(self, slug: str) -> Optional[dict]:
        row = self._execute("SELECT * FROM stores WHERE slug = ?", (slug,)).fetchone()
        return dict(row) if row else None

    def list_stores(self) -> list[dict]:
        rows = self._execute("SELECT * FROM stores ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def get_default_store(self) -> Optional[dict]:
        row = self._execute("SELECT * FROM stores WHERE is_default = 1").fetchone()
        return dict(row) if row else None

    def set_default_store(self, store_id: str) -> None:
        self._execute("UPDATE stores SET is_default = 0 WHERE is_default = 1")
        self._execute("UPDATE stores SET is_default = 1 WHERE id = ?", (store_id,))
        self._commit()

    def update_cache(self, store_id: str, cache_data: dict) -> None:
        self._execute(
            "UPDATE stores SET cache_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(cache_data), self._now(), store_id)
        )
        self._commit()

    def delete_store(self, store_id: str) -> None:
        self._execute("DELETE FROM stores WHERE id = ?", (store_id,))
        self._commit()

    @staticmethod
    def _now() -> str:
        from src.core.Storage.StoreUtils import now
        return now()

    def close(self) -> None:
        with self._lock:
            self._conn.close()