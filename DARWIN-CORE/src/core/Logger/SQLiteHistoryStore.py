"""
SQLiteHistoryStore — Ring buffer + SQLite‑backed persistence for Darwin Logger.

- Hot ring buffer for all event types (real-time display)
- SQLite persists only LogEvents (progress/spinner/session are ephemeral)
- WAL mode for concurrent read/write
- Auto‑retention: deletes logs older than 14 days by default
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.core.Logger.DarwinLogger import (
    DarwinEvent,
    LogEvent,
    LogType,
    ProgressStartEvent,
    ProgressUpdateEvent,
    ProgressEndEvent,
    SpinnerStartEvent,
    SpinnerUpdateEvent,
    SpinnerEndEvent,
    SessionStartEvent,
    SessionEndEvent,
)

_RETENTION_DAYS = 14
_BATCH_SIZE = 50


class SQLiteHistoryStore:
    def __init__(
        self,
        max_memory: int = 10_000,
        persist_dir: Optional[Path] = None,
        retention_days: int = _RETENTION_DAYS,
    ) -> None:
        self._events: deque[DarwinEvent] = deque(maxlen=max_memory)
        self._lock = threading.RLock()
        self._persist_dir = persist_dir
        self._retention_days = retention_days
        self._pending: List[LogEvent] = []
        self._db: Optional[sqlite3.Connection] = None
        self._last_flush = 0.0
        self._flushed_total = 0

        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            db_path = persist_dir / "logs.db"
            self._db = sqlite3.connect(str(db_path), check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()
            self._apply_retention()
            self._heat_ring_buffer()

    def _init_schema(self) -> None:
        if not self._db:
            return
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS log_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session TEXT NOT NULL DEFAULT '',
                log_type TEXT NOT NULL,
                severity INTEGER NOT NULL DEFAULT 20,
                message TEXT NOT NULL DEFAULT '',
                payload TEXT,
                format TEXT NOT NULL DEFAULT '',
                timestamp REAL NOT NULL,
                datetime TEXT NOT NULL
            )
        """)
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_le_session ON log_events(session)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_le_ts ON log_events(timestamp)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_le_severity ON log_events(severity)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_le_sess_sev ON log_events(session, severity)"
        )
        self._db.commit()

    def _heat_ring_buffer(self, count: int = 1000) -> None:
        if not self._db:
            return
        try:
            cursor = self._db.execute(
                "SELECT session, log_type, severity, message, payload, format, timestamp, datetime "
                "FROM log_events ORDER BY id DESC LIMIT ?",
                (count,),
            )
            for row in cursor.fetchall():
                session, log_type_str, severity, message, payload_str, fmt, ts, dt = row
                lt = (
                    LogType(log_type_str)
                    if log_type_str in {e.value for e in LogType}
                    else LogType.INFO
                )
                payload = json.loads(payload_str) if payload_str else None
                ev = LogEvent(
                    message=message,
                    log_type=lt,
                    session=session or "",
                    timestamp=ts,
                    payload=payload,
                    format=fmt or "",
                )
                self._events.appendleft(ev)
        except Exception:
            pass

    def _apply_retention(self) -> None:
        if not self._db:
            return
        try:
            cutoff = time.time() - self._retention_days * 86400
            self._db.execute("DELETE FROM log_events WHERE timestamp < ?", (cutoff,))
            self._db.commit()
        except Exception:
            pass

    def append(self, event: DarwinEvent) -> None:
        with self._lock:
            self._events.append(event)
            if isinstance(event, LogEvent) and self._db is not None:
                self._pending.append(event)
                if len(self._pending) >= _BATCH_SIZE:
                    self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        if not self._pending or not self._db:
            return
        try:
            rows = []
            now_str = datetime.utcnow().isoformat()
            for e in self._pending:
                rows.append(
                    (
                        e.session,
                        e.log_type.value,
                        e.log_type.severity,
                        e.message,
                        json.dumps(e.payload, default=str)
                        if e.payload is not None
                        else None,
                        e.format,
                        e.timestamp,
                        now_str,
                    )
                )
            self._db.executemany(
                "INSERT INTO log_events (session, log_type, severity, message, payload, format, timestamp, datetime) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._db.commit()
            self._flushed_total += len(self._pending)
            self._pending.clear()
            self._last_flush = time.time()
        except Exception:
            pass

    def flush(self) -> None:
        with self._lock:
            self._flush_unlocked()

    def get_logs(
        self,
        session: Optional[str] = None,
        levels: Optional[Set[LogType]] = None,
        after: Optional[float] = None,
        before: Optional[float] = None,
        limit: int = 1000,
    ) -> List[LogEvent]:
        with self._lock:
            entries = list(self._events)
        result: List[LogEvent] = []
        for e in reversed(entries):
            if not isinstance(e, LogEvent):
                continue
            if levels and e.log_type not in levels:
                continue
            if session is not None and e.session != session:
                continue
            if after is not None and e.timestamp < after:
                continue
            if before is not None and e.timestamp > before:
                continue
            result.append(e)
            if len(result) >= limit:
                return result
        if self._db is not None and len(result) < limit:
            remaining = limit - len(result)
            try:
                query = "SELECT session, log_type, severity, message, payload, format, timestamp, datetime FROM log_events"
                conditions: List[str] = []
                params: List[Any] = []
                if session is not None:
                    conditions.append("session = ?")
                    params.append(session)
                if levels is not None:
                    placeholders = ",".join("?" for _ in levels)
                    conditions.append(f"log_type IN ({placeholders})")
                    params.extend(lt.value for lt in levels)
                if after is not None:
                    conditions.append("timestamp >= ?")
                    params.append(after)
                if before is not None:
                    conditions.append("timestamp <= ?")
                    params.append(before)
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                query += " ORDER BY id DESC LIMIT ?"
                params.append(remaining)
                cursor = self._db.execute(query, params)
                for row in cursor.fetchall():
                    log_type_str, severity, message, payload_str, fmt, ts = (
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[6],
                    )
                    lt = (
                        LogType(log_type_str)
                        if log_type_str in {e.value for e in LogType}
                        else LogType.INFO
                    )
                    payload = json.loads(payload_str) if payload_str else None
                    result.append(
                        LogEvent(
                            message=message,
                            log_type=lt,
                            session=row[0] or "",
                            timestamp=ts,
                            payload=payload,
                            format=fmt or "",
                        )
                    )
            except Exception:
                pass
        return result

    def get_events(
        self, session: Optional[str] = None, limit: int = 500
    ) -> List[DarwinEvent]:
        with self._lock:
            entries = list(self._events)
        result = []
        for e in reversed(entries):
            if session is not None and getattr(e, "session", "") != session:
                continue
            result.append(e)
            if len(result) >= limit:
                break
        return result

    def sessions(self) -> List[str]:
        with self._lock:
            seen: List[str] = []
            for e in self._events:
                s = getattr(e, "session", "")
                if s and s not in seen:
                    seen.append(s)
            return seen

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            log_count = sum(1 for e in self._events if isinstance(e, LogEvent))
            by_level: Dict[str, int] = {}
            for e in self._events:
                if isinstance(e, LogEvent):
                    by_level[e.log_type.value] = by_level.get(e.log_type.value, 0) + 1
            db_size = 0
            if self._db:
                try:
                    row = self._db.execute("SELECT COUNT(*) FROM log_events").fetchone()
                    db_size = row[0] if row else 0
                except Exception:
                    pass
            return {
                "total_events": len(self._events),
                "log_events": log_count,
                "by_level": by_level,
                "sessions": self.sessions(),
                "persist_file": str(self._persist_dir / "logs.db")
                if self._persist_dir
                else None,
                "persisted_count": db_size,
                "flushed_total": self._flushed_total,
            }

    def purge_before(self, timestamp: float) -> int:
        if not self._db:
            return 0
        try:
            cursor = self._db.execute(
                "DELETE FROM log_events WHERE timestamp < ?", (timestamp,)
            )
            self._db.commit()
            deleted = cursor.rowcount
            return deleted
        except Exception:
            return 0

    def purge_session(self, session_name: str) -> int:
        if not self._db:
            return 0
        try:
            cursor = self._db.execute(
                "DELETE FROM log_events WHERE session = ?", (session_name,)
            )
            self._db.commit()
            with self._lock:
                self._events = deque(
                    (
                        e
                        for e in self._events
                        if getattr(e, "session", "") != session_name
                    ),
                    maxlen=self._events.maxlen,
                )
            return cursor.rowcount
        except Exception:
            return 0

    def export_jsonl(self, path: Path, session: Optional[str] = None) -> int:
        count = 0
        try:
            with open(path, "w", encoding="utf-8") as f:
                for e in self.get_logs(session=session, limit=100000):
                    f.write(json.dumps(e.to_dict(), default=str) + "\n")
                    count += 1
            return count
        except Exception:
            return count

    def close(self) -> None:
        self.flush()
        if self._db:
            self._db.close()
            self._db = None
