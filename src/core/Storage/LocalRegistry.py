from __future__ import annotations

import hashlib
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Optional, Any

from src.core.Storage.DataModels import Content, Source
from src.core.Storage.StoreUtils import file_hash, now


SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    uuid TEXT PRIMARY KEY,
    display_name TEXT NOT NULL UNIQUE,
    current_path TEXT,
    content_hash TEXT NOT NULL,
    file_size INTEGER DEFAULT 0,
    mtime REAL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sources_hash ON sources(content_hash);
CREATE INDEX IF NOT EXISTS idx_sources_name ON sources(display_name);

CREATE TABLE IF NOT EXISTS contents (
    hash TEXT PRIMARY KEY,
    stages TEXT NOT NULL DEFAULT '',
    embedding_model TEXT,
    chunk_count INTEGER DEFAULT 0,
    page_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class LocalRegistry:
    """Thread-safe SQLite registry for a single DarwinStore (per-store)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
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

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def executemany(self, sql: str, params: list[tuple]) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.executemany(sql, params)

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class SourceManager:
    """Manages the sources table: user-facing file entries."""

    def __init__(self, registry: LocalRegistry):
        self._reg = registry

    def _row_to_source(self, row: sqlite3.Row) -> Source:
        return Source(
            uuid=row["uuid"],
            display_name=row["display_name"],
            current_path=row["current_path"],
            content_hash=row["content_hash"],
            file_size=row["file_size"],
            mtime=row["mtime"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def register_url(self, slug: str, url: str, content_hash: str, page_count: int = 0) -> tuple[Source, bool]:
        """Register a URL with a pre-computed content hash (hash of fetched body)."""
        ts = now()
        existing = self.find_by_name(slug)
        if existing:
            if existing.content_hash == content_hash:
                self._reg.execute(
                    "UPDATE sources SET updated_at=? WHERE display_name=?",
                    (ts, slug)
                )
                self._reg.commit()
                return self.find_by_name(slug), False
            else:
                old_hash = existing.content_hash
                self._reg.execute(
                    "UPDATE sources SET content_hash=?, updated_at=? WHERE display_name=?",
                    (content_hash, ts, slug)
                )
                self._reg.commit()
                self._maybe_gc_content(old_hash)
                is_new = self._ensure_content(content_hash, page_count, ts)
                return self.find_by_name(slug), is_new

        source_uuid = str(uuid.uuid4())
        self._reg.execute(
            "INSERT INTO sources (uuid, display_name, content_hash, file_size, mtime, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (source_uuid, slug, content_hash, 0, 0, ts, ts)
        )
        self._reg.commit()
        is_new = self._ensure_content(content_hash, page_count, ts)
        return self.find_by_name(slug), is_new

    def register(
        self,
        file_path: Path,
        page_count: int = 0,
        file_size: int = 0,
    ) -> tuple[Source, bool]:
        """
        Register a file. Returns (Source, is_new_content).
        is_new_content=True when the content hash has never been seen.
        """
        path = Path(file_path).resolve()
        name = path.name
        fh = file_hash(path)
        size = file_size or path.stat().st_size
        mtime = path.stat().st_mtime
        ts = now()

        existing = self.find_by_name(name)

        if existing:
            if existing.content_hash == fh:
                self._reg.execute(
                    "UPDATE sources SET current_path=?, mtime=?, updated_at=? WHERE uuid=?",
                    (str(path), mtime, ts, existing.uuid),
                )
                self._reg.commit()
                return self.find_by_name(name), False
            else:
                old_hash = existing.content_hash
                self._reg.execute(
                    "UPDATE sources SET content_hash=?, current_path=?, file_size=?, mtime=?, updated_at=? WHERE uuid=?",
                    (fh, str(path), size, mtime, ts, existing.uuid),
                )
                self._reg.commit()
                self._maybe_gc_content(old_hash)
                is_new = self._ensure_content(fh, page_count, ts)
                return self.find_by_name(name), is_new

        source_uuid = str(uuid.uuid4())
        self._reg.execute(
            "INSERT INTO sources (uuid, display_name, current_path, content_hash, file_size, mtime, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (source_uuid, name, str(path), fh, size, mtime, ts, ts),
        )
        self._reg.commit()
        is_new = self._ensure_content(fh, page_count, ts)
        return self.find_by_name(name), is_new

    def find_by_name(self, name: str) -> Optional[Source]:
        row = self._reg.execute(
            "SELECT * FROM sources WHERE display_name = ?", (name,)
        ).fetchone()
        return self._row_to_source(row) if row else None

    def get_by_hash(self, content_hash: str) -> list[Source]:
        rows = self._reg.execute(
            "SELECT * FROM sources WHERE content_hash = ?", (content_hash,)
        ).fetchall()
        return [self._row_to_source(r) for r in rows]

    def list_all(self) -> list[Source]:
        rows = self._reg.execute("SELECT * FROM sources ORDER BY display_name").fetchall()
        return [self._row_to_source(r) for r in rows]

    def remove_by_name(self, name: str) -> bool:
        source = self.find_by_name(name)
        if not source:
            return False
        self._reg.execute("DELETE FROM sources WHERE uuid = ?", (source.uuid,))
        self._reg.commit()
        self._maybe_gc_content(source.content_hash)
        return True

    def update_path(self, name: str, new_path: str, mtime: float) -> None:
        ts = now()
        self._reg.execute(
            "UPDATE sources SET current_path=?, mtime=?, updated_at=? WHERE display_name=?",
            (str(new_path), mtime, ts, name),
        )
        self._reg.commit()

    def count(self) -> int:
        row = self._reg.execute("SELECT COUNT(*) FROM sources").fetchone()
        return row[0] if row else 0

    def count_for_hash(self, content_hash: str) -> int:
        row = self._reg.execute(
            "SELECT COUNT(*) FROM sources WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return row[0] if row else 0

    def _ensure_content(self, content_hash: str, page_count: int, ts: str) -> bool:
        """Insert content row if missing. Returns True if inserted (brand new hash)."""
        row = self._reg.execute(
            "SELECT hash FROM contents WHERE hash = ?", (content_hash,)
        ).fetchone()
        if row:
            return False
        self._reg.execute(
            "INSERT INTO contents (hash, page_count, created_at, updated_at) VALUES (?,?,?,?)",
            (content_hash, page_count, ts, ts),
        )
        self._reg.commit()
        return True

    def _maybe_gc_content(self, content_hash: str) -> None:
        """Delete content row and artifacts if no sources reference it."""
        count = self.count_for_hash(content_hash)
        if count == 0:
            self._reg.execute("DELETE FROM contents WHERE hash = ?", (content_hash,))
            self._reg.commit()


class ContentIndex:
    """Manages the contents table: immutable blob pipeline state."""

    def __init__(self, registry: LocalRegistry):
        self._reg = registry

    def _row_to_content(self, row: sqlite3.Row) -> Content:
        return Content(
            hash=row["hash"],
            stages=set(row["stages"].split(",")) if row["stages"] else set(),
            embedding_model=row["embedding_model"],
            chunk_count=row["chunk_count"],
            page_count=row["page_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get(self, content_hash: str) -> Optional[Content]:
        row = self._reg.execute(
            "SELECT * FROM contents WHERE hash = ?", (content_hash,)
        ).fetchone()
        return self._row_to_content(row) if row else None

    def list_all(self) -> list[Content]:
        rows = self._reg.execute("SELECT * FROM contents").fetchall()
        return [self._row_to_content(r) for r in rows]

    def set_stage(self, content_hash: str, stage: str, complete: bool = True) -> None:
        content = self.get(content_hash)
        if not content:
            return
        stages = content.stages.copy()
        if complete:
            stages.add(stage)
        else:
            stages.discard(stage)
        ts = now()
        self._reg.execute(
            "UPDATE contents SET stages=?, updated_at=? WHERE hash=?",
            (",".join(sorted(stages)), ts, content_hash),
        )
        self._reg.commit()

    def update_meta(
        self,
        content_hash: str,
        chunk_count: Optional[int] = None,
        page_count: Optional[int] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        ts = now()
        parts = ["updated_at = ?"]
        params: list[Any] = [ts]
        if chunk_count is not None:
            parts.append("chunk_count = ?")
            params.append(chunk_count)
        if page_count is not None:
            parts.append("page_count = ?")
            params.append(page_count)
        if embedding_model is not None:
            parts.append("embedding_model = ?")
            params.append(embedding_model)
        params.append(content_hash)
        sql = f"UPDATE contents SET {', '.join(parts)} WHERE hash = ?"
        self._reg.execute(sql, tuple(params))
        self._reg.commit()