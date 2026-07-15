from __future__ import annotations

import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.Logger.DarwinLogger import darwin
from src.core.Storage.StoreUtils import now


# ── Ambient in-memory registry for active store instances ───────────────

class DarwinStoreRegistry:
    """Ambient context registry for active DarwinStore instances.
    Thin, global, and in-memory. Complements explicit injection;
    it does NOT replace it.
    """

    _stores: Dict[str, Any] = {}
    _default_key: Optional[str] = None

    @classmethod
    def register(cls, key: str, store: Any, *, default: bool = False) -> None:
        cls._stores[key] = store
        if default or cls._default_key is None:
            cls._default_key = key

    @classmethod
    def get(cls, key: Optional[str] = None) -> Any:
        target = key or cls._default_key
        if target is None:
            darwin.error("No DarwinStore registered")
            return None
        if target not in cls._stores:
            darwin.error(
                f"Store '{target}' not found. Registered: {list(cls._stores)}"
            )
            return None
        return cls._stores[target]

    @classmethod
    def list_stores(cls) -> Dict[str, dict]:
        """Health/status snapshot of all registered stores."""
        return {
            key: {
                "path": str(store.darwin_dir),
                "exists": store.darwin_dir.exists(),
                "size_mb": cls._dir_size(store.darwin_dir),
                "name": store.store_name,
                "id": store.store_id,
            }
            for key, store in cls._stores.items()
        }

    @classmethod
    def create_store(cls, key: str, path: Path) -> Any:
        # Lazy import to avoid circular dependency
        from src.core.Storage.DarwinStore import DarwinStore
        store = DarwinStore(path, name=key)
        cls.register(key, store)
        return store

    @classmethod
    def delete_store(cls, key: str, *, nuke_disk: bool = False) -> None:
        store = cls._stores.pop(key, None)
        if store is None:
            darwin.error(f"Store '{key}' not registered")
            return None
        if nuke_disk and store.darwin_dir.exists():
            shutil.rmtree(store.darwin_dir)

    @staticmethod
    def _dir_size(path: Path) -> float:
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return round(total / (1024 * 1024), 2)


# ── Persistent SQLite registry for store metadata ──────────────────────

class StoreRegistry:
    """Persistent central registry for DarwinStore metadata.

    Uses a SQLite database at ~/.darwin/stores_registry.db.
    Singleton pattern — multiple instantiations share the same connection.
    """

    _instance: Optional[StoreRegistry] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._db_path = Path.home() / ".darwin" / "stores_registry.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS stores (
                id TEXT PRIMARY KEY,
                slug TEXT UNIQUE,
                name TEXT,
                darwin_dir TEXT NOT NULL,
                created_at TEXT,
                updated_at TEXT,
                last_opened TEXT,
                is_default INTEGER DEFAULT 0,
                cache_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_stores_name ON stores(name);
            CREATE INDEX IF NOT EXISTS idx_stores_dir ON stores(darwin_dir);
        """)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.commit()

    def register(self, darwin_dir: Path, store_id: str, name: Optional[str] = None) -> str:
        ts = now()
        slug = name or Path(darwin_dir).name

        # Try update first, then insert if no row exists
        cursor = self._conn.execute(
            """
            INSERT INTO stores (id, slug, name, darwin_dir, created_at, updated_at, last_opened, is_default)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                name = excluded.name,
                darwin_dir = excluded.darwin_dir,
                updated_at = excluded.updated_at,
                last_opened = excluded.last_opened
            RETURNING id
            """,
            (store_id, slug, name, str(darwin_dir), ts, ts, ts, 0),
        )
        row = cursor.fetchone()
        self._conn.commit()
        return row[0]  # returns the actual id (either newly inserted or existing)

    def list_stores(self) -> list[dict]:
        """Return all registered stores as a list of dicts."""
        rows = self._conn.execute(
            "SELECT * FROM stores ORDER BY name"
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["is_default"] = bool(d.get("is_default", 0))
            result.append(d)
        return result

    def get_store(self, store_id: str) -> Optional[dict]:
        """Fetch a store by its UUID."""
        row = self._conn.execute(
            "SELECT * FROM stores WHERE id = ?", (store_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["is_default"] = bool(d.get("is_default", 0))
        return d

    def get_default_store(self) -> Optional[dict]:
        """Return the default store, or the most recently opened one."""
        row = self._conn.execute(
            "SELECT * FROM stores WHERE is_default = 1 LIMIT 1"
        ).fetchone()
        if row:
            d = dict(row)
            d["is_default"] = bool(d.get("is_default", 0))
            return d
        # Fallback to most recently opened
        row = self._conn.execute(
            "SELECT * FROM stores ORDER BY last_opened DESC LIMIT 1"
        ).fetchone()
        if row:
            d = dict(row)
            d["is_default"] = bool(d.get("is_default", 0))
            return d
        return None

    def set_default(self, store_id: str) -> None:
        """Mark a store as the default."""
        self._conn.execute("BEGIN")
        try:
            self._conn.execute("UPDATE stores SET is_default = 0")
            self._conn.execute(
                "UPDATE stores SET is_default = 1 WHERE id = ?",
                (store_id,),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
