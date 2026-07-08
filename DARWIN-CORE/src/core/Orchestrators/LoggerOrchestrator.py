"""
LoggerOrchestrator — Query, manage, and export Darwin log history.

Public methods   → exposed as MCP tools (get_logs, sessions, stats)
Admin methods    → CLI-only (purge, export, retention, compact)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from src.core.Logger.DarwinLogger import darwin, LogType, LogFilter, OutputMode


class LoggerOrchestrator:
    """Facade over DarwinLogger's history for MCP and admin use."""

    # ──────────────────────────────────────────────
    # Configure — admin‑driven logger setup
    # ──────────────────────────────────────────────

    @staticmethod
    def configure(
        mode: str = "terminal",
        persist: bool = True,
        show_banner: bool = False,
        log_filter: str = "all",
        sse_port: int = 8765,
        sse_host: str = "127.0.0.1",
        persist_config: bool = True,
        override: bool = True,
    ) -> None:
        mode_map = {
            "terminal": OutputMode.TERMINAL,
            "sse": OutputMode.SSE,
            "both": OutputMode.BOTH,
            "none": OutputMode.NONE,
        }
        filter_map = {
            "all": LogFilter.ALL,
            "quiet": LogFilter.QUIET,
            "silent": LogFilter.SILENT,
        }
        darwin.configure(
            mode=mode_map.get(mode, OutputMode.TERMINAL),
            persist=persist,
            show_banner=show_banner,
            log_filter=filter_map.get(log_filter, LogFilter.ALL),
            sse_port=sse_port,
            sse_host=sse_host,
            persist_config=persist_config,
            override=override,
        )

    # ──────────────────────────────────────────────
    # Public (MCP-accessible)
    # ──────────────────────────────────────────────

    @staticmethod
    def get_logs(
        session: Optional[str] = None,
        levels: Optional[str] = None,
        after: Optional[float] = None,
        before: Optional[float] = None,
        limit: int = 100,
    ) -> List[dict]:
        level_set = None
        if levels:
            level_set = set()
            for l in levels.split(","):
                l = l.strip()
                if l in {e.value for e in LogType}:
                    level_set.add(LogType(l))
        logs = darwin.get_logs(
            session=session, levels=level_set, after=after, before=before, limit=limit
        )
        return [l.to_dict() for l in logs]

    @staticmethod
    def get_sessions() -> List[str]:
        return darwin.get_sessions()

    @staticmethod
    def get_session_info(session_name: str) -> dict:
        logs = darwin.get_logs(session=session_name)
        errors = [l for l in logs if l.log_type == LogType.ERROR]
        warnings_list = [l for l in logs if l.log_type == LogType.WARNING]
        infos = [l for l in logs if l.log_type == LogType.INFO]
        successes = [l for l in logs if l.log_type == LogType.SUCCESS]
        duration = None
        if logs:
            ts_min = min(l.timestamp for l in logs)
            ts_max = max(l.timestamp for l in logs)
            duration = ts_max - ts_min
        return {
            "session": session_name,
            "total_logs": len(logs),
            "errors": len(errors),
            "warnings": len(warnings_list),
            "info": len(infos),
            "successes": len(successes),
            "duration_seconds": round(duration, 3) if duration else None,
        }

    @staticmethod
    def history_stats() -> dict:
        return darwin.history_stats()

    # ──────────────────────────────────────────────
    # Admin (CLI-only)
    # ──────────────────────────────────────────────

    @staticmethod
    def purge_logs(
        before_timestamp: Optional[float] = None, days: Optional[int] = None
    ) -> int:
        if days is not None:
            before_timestamp = time.time() - days * 86400
        if before_timestamp is None:
            return 0
        store = darwin._history
        return store.purge_before(before_timestamp)

    @staticmethod
    def purge_session(session_name: str) -> int:
        store = darwin._history
        return store.purge_session(session_name)

    @staticmethod
    def set_retention(days: int = 14) -> None:
        store = darwin._history
        store._retention_days = days
        store._apply_retention()

    @staticmethod
    def force_flush() -> None:
        darwin._history.flush()

    @staticmethod
    def export_logs(path: str, session: Optional[str] = None) -> int:
        from src.core.Logger.SQLiteHistoryStore import SQLiteHistoryStore

        store = darwin._history
        if isinstance(store, SQLiteHistoryStore):
            return store.export_jsonl(Path(path), session=session)
        return 0

    @staticmethod
    def compact() -> dict:
        store = darwin._history
        if hasattr(store, "_db") and store._db:
            try:
                before = store._db.execute(
                    "SELECT COUNT(*) FROM log_events"
                ).fetchone()[0]
                store._db.execute("VACUUM")
                after = store._db.execute("SELECT COUNT(*) FROM log_events").fetchone()[
                    0
                ]
                return {"before": before, "after": after, "status": "ok"}
            except Exception as e:
                return {"error": str(e)}
        return {"status": "no_db"}

    @staticmethod
    def set_persistence(enabled: bool = True) -> None:
        store = darwin._history
        store._persist_dir = (
            Path(store._persist_dir)
            if enabled and store._persist_dir
            else (None if not enabled else store._persist_dir)
        )

    @staticmethod
    def apply_retention() -> int:
        store = darwin._history
        if hasattr(store, "_apply_retention"):
            store._apply_retention()
            return 0
        return -1
