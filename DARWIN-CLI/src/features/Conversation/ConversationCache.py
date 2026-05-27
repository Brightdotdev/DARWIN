"""
ConversationCache.py
General-purpose conversation-scoped cache for tool results, web retrieval,
query computations, and expensive intermediate results.
SQLite-backed with TTL, hit tracking, and deterministic key hashing.
"""

import sqlite3
import json
import hashlib
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import threading


@dataclass
class CacheEntry:
    cache_id: str
    conv_id: Optional[str]       # None = global cache
    cache_key: str               # deterministic hash
    tool_name: Optional[str]     # e.g. "web_search", "calculator", "query_retrieval"
    cache_value: Dict[str, Any]  # structured payload
    created_at: str
    expires_at: Optional[str]    # ISO timestamp; None = never
    hit_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


class ConversationCache:
    """
    Two-tier cache:
      - Conversation-scoped: tied to a specific conv_id (contextual memory)
      - Global-scoped: conv_id is NULL (shared across all conversations)

    Cache keys are SHA256 hashes of normalized inputs to ensure deterministic
    lookup regardless of formatting differences.
    """

    SCHEMA = """
    PRAGMA journal_mode=WAL;
    PRAGMA foreign_keys=ON;

    CREATE TABLE IF NOT EXISTS cache_entries (
        cache_id TEXT PRIMARY KEY,
        conv_id TEXT,                      -- NULL = global
        cache_key TEXT NOT NULL,
        tool_name TEXT,
        cache_value TEXT NOT NULL,         -- JSON payload
        created_at TEXT NOT NULL,
        expires_at TEXT,                   -- NULL = no expiry
        hit_count INTEGER DEFAULT 0,
        metadata TEXT,
        UNIQUE(conv_id, cache_key) ON CONFLICT REPLACE
    );

    CREATE INDEX IF NOT EXISTS idx_cache_lookup 
        ON cache_entries(conv_id, cache_key, expires_at);
    CREATE INDEX IF NOT EXISTS idx_cache_tool 
        ON cache_entries(tool_name, created_at);
    CREATE INDEX IF NOT EXISTS idx_cache_expiry 
        ON cache_entries(expires_at);
    """

    def __init__(self, db_path: str = "darwin_cache.db"):
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        with self._write_lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.executescript(self.SCHEMA)
            conn.commit()
            conn.close()

    @contextmanager
    def _transaction(self):
        with self._write_lock:
            conn = self._connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # ── Key Generation ──────────────────────────────────────

    @staticmethod
    def make_key(*args, **kwargs) -> str:
        """
        Deterministic SHA256 key from normalized inputs.
        Handles dicts, lists, strings, numbers. Ignores key ordering in dicts.
        """
        def normalize(obj):
            if isinstance(obj, dict):
                return {k: normalize(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, (list, tuple)):
                return [normalize(v) for v in obj]
            elif isinstance(obj, str):
                return obj.strip().lower()
            return obj

        payload = json.dumps({"args": normalize(args), "kwargs": normalize(kwargs)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    # ── Core Operations ─────────────────────────────────────

    def get(self, cache_key: str, 
            conv_id: Optional[str] = None,
            tool_name: Optional[str] = None) -> Optional[CacheEntry]:
        """
        Lookup cache entry. Returns None if expired or missing.
        Increments hit_count on successful read.
        """
        conn = self._connection()
        now = datetime.utcnow().isoformat()

        # Try conversation-scoped first, then global fallback
        candidates = []
        if conv_id:
            row = conn.execute(
                """SELECT * FROM cache_entries 
                   WHERE conv_id = ? AND cache_key = ?
                   AND (expires_at IS NULL OR expires_at > ?)
                   LIMIT 1""",
                (conv_id, cache_key, now)
            ).fetchone()
            if row:
                candidates.append(row)

        # Global fallback
        if not candidates:
            row = conn.execute(
                """SELECT * FROM cache_entries 
                   WHERE conv_id IS NULL AND cache_key = ?
                   AND (expires_at IS NULL OR expires_at > ?)
                   LIMIT 1""",
                (cache_key, now)
            ).fetchone()
            if row:
                candidates.append(row)

        if not candidates:
            return None

        row = candidates[0]

        # Increment hit count
        with self._transaction() as txn:
            txn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1 WHERE cache_id = ?",
                (row["cache_id"],)
            )

        return CacheEntry(
            cache_id=row["cache_id"],
            conv_id=row["conv_id"],
            cache_key=row["cache_key"],
            tool_name=row["tool_name"],
            cache_value=json.loads(row["cache_value"]),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            hit_count=row["hit_count"] + 1,
            metadata=json.loads(row["metadata"]) if row["metadata"] else None
        )

    def set(self, cache_key: str,
            value: Dict[str, Any],
            conv_id: Optional[str] = None,
            tool_name: Optional[str] = None,
            ttl_seconds: Optional[int] = None,
            metadata: Optional[Dict[str, Any]] = None) -> CacheEntry:
        """
        Store value in cache. ttl_seconds=None means no expiry.
        """
        now = datetime.utcnow()
        expires = (now + timedelta(seconds=ttl_seconds)).isoformat() if ttl_seconds else None
        cache_id = f"cache_{uuid.uuid4().hex[:12]}"

        with self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cache_entries
                   (cache_id, conv_id, cache_key, tool_name, cache_value, 
                    created_at, expires_at, hit_count, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cache_id, conv_id, cache_key, tool_name,
                 json.dumps(value),
                 now.isoformat(), expires, 0,
                 json.dumps(metadata) if metadata else None)
            )

        return CacheEntry(
            cache_id=cache_id, conv_id=conv_id, cache_key=cache_key,
            tool_name=tool_name, cache_value=value,
            created_at=now.isoformat(), expires_at=expires,
            hit_count=0, metadata=metadata
        )

    def invalidate(self, cache_key: str, conv_id: Optional[str] = None) -> bool:
        """Remove specific entry."""
        with self._transaction() as conn:
            if conv_id:
                cur = conn.execute(
                    "DELETE FROM cache_entries WHERE conv_id = ? AND cache_key = ?",
                    (conv_id, cache_key)
                )
            else:
                cur = conn.execute(
                    "DELETE FROM cache_entries WHERE cache_key = ?",
                    (cache_key,)
                )
            return cur.rowcount > 0

    def invalidate_by_tool(self, tool_name: str, conv_id: Optional[str] = None):
        """Invalidate all entries for a specific tool."""
        with self._transaction() as conn:
            if conv_id:
                conn.execute(
                    "DELETE FROM cache_entries WHERE conv_id = ? AND tool_name = ?",
                    (conv_id, tool_name)
                )
            else:
                conn.execute(
                    "DELETE FROM cache_entries WHERE tool_name = ?",
                    (tool_name,)
                )

    def clear_conversation_cache(self, conv_id: str):
        """Wipe all conversation-scoped entries."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM cache_entries WHERE conv_id = ?", (conv_id,))

    def clear_global_cache(self):
        """Wipe all global entries."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM cache_entries WHERE conv_id IS NULL")

    def clear_expired(self) -> int:
        """Delete expired entries. Returns count deleted."""
        now = datetime.utcnow().isoformat()
        with self._transaction() as conn:
            cur = conn.execute(
                "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,)
            )
            return cur.rowcount

    # ── Inspection ──────────────────────────────────────────

    def list_entries(self, conv_id: Optional[str] = None,
                     tool_name: Optional[str] = None,
                     limit: int = 100) -> List[CacheEntry]:
        conn = self._connection()
        now = datetime.utcnow().isoformat()

        conditions = ["(expires_at IS NULL OR expires_at > ?)"]
        params = [now]

        if conv_id is not None:
            conditions.append("conv_id = ?")
            params.append(conv_id)
        elif conv_id is None and tool_name is None:
            # When explicitly asking for global with no tool filter
            conditions.append("conv_id IS NULL")

        if tool_name:
            conditions.append("tool_name = ?")
            params.append(tool_name)

        where_clause = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM cache_entries WHERE {where_clause} ORDER BY created_at DESC LIMIT ?",
            (*params, limit)
        ).fetchall()

        return [
            CacheEntry(
                cache_id=r["cache_id"], conv_id=r["conv_id"], cache_key=r["cache_key"],
                tool_name=r["tool_name"],
                cache_value=json.loads(r["cache_value"]),
                created_at=r["created_at"], expires_at=r["expires_at"],
                hit_count=r["hit_count"],
                metadata=json.loads(r["metadata"]) if r["metadata"] else None
            )
            for r in rows
        ]

    def get_stats(self) -> Dict[str, Any]:
        conn = self._connection()
        total = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
        global_count = conn.execute(
            "SELECT COUNT(*) FROM cache_entries WHERE conv_id IS NULL"
        ).fetchone()[0]
        conv_count = conn.execute(
            "SELECT COUNT(DISTINCT conv_id) FROM cache_entries WHERE conv_id IS NOT NULL"
        ).fetchone()[0]
        expired = conn.execute(
            """SELECT COUNT(*) FROM cache_entries 
               WHERE expires_at IS NOT NULL AND expires_at <= ?""",
            (datetime.utcnow().isoformat(),)
        ).fetchone()[0]
        top_tools = conn.execute(
            """SELECT tool_name, COUNT(*) as c, SUM(hit_count) as hits
               FROM cache_entries GROUP BY tool_name ORDER BY c DESC LIMIT 5"""
        ).fetchall()

        return {
            "total_entries": total,
            "global_entries": global_count,
            "conversations_with_cache": conv_count,
            "expired_entries": expired,
            "top_tools": [{"tool": r["tool_name"], "entries": r["c"], "hits": r["hits"]} 
                         for r in top_tools],
            "db_path": str(self.db_path)
        }

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None