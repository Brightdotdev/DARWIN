"""
Darwin Logger — Session-aware, queryable structured logging.

Restored features:
  • Progress + Spinner events (terminal + frontend)
  • logjson/loglist with format discrimination
  • __darwin_type__ markers for structured data
  • Default history persistence (~/.darwin/logs)
  • Full event protocol (not just logs)
  • Configurable log filtering (ALL, QUIET, SILENT)
  • Output modes: TERMINAL, SSE, BOTH, NONE
"""
from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class LogType(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class OutputMode(Enum):
    TERMINAL = "terminal"
    SSE = "sse"
    BOTH = "both"          # ← NEW: simultaneous terminal + SSE
    NONE = "none"


class LogFilter(Enum):
    """Filter presets for LogEvents."""
    ALL = "all"            # everything (default)
    QUIET = "quiet"        # errors + warnings + success only (no info)
    SILENT = "silent"      # errors only


# ---------------------------------------------------------------------------
# Events (lightweight frozen dataclasses)
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True, slots=True)
class LogEvent:
    message: str
    log_type: LogType
    session: str
    timestamp: float
    payload: Optional[Any] = None
    format: str = ""           # "json", "list", "" for plain

    def to_dict(self) -> dict:
        d = {
            "message": self.message,
            "log_type": self.log_type.value,
            "session": self.session,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
        }
        if self.payload is not None:
            d["payload"] = self.payload
            d["format"] = self.format
        return d


@dataclasses.dataclass(frozen=True, slots=True)
class ProgressStartEvent:
    id: str
    total: int
    label: str
    session: str
    timestamp: float = dataclasses.field(default_factory=time.time)


@dataclasses.dataclass(frozen=True, slots=True)
class ProgressUpdateEvent:
    id: str
    current: int
    status: str
    session: str
    timestamp: float = dataclasses.field(default_factory=time.time)


@dataclasses.dataclass(frozen=True, slots=True)
class ProgressEndEvent:
    id: str
    session: str
    timestamp: float = dataclasses.field(default_factory=time.time)


@dataclasses.dataclass(frozen=True, slots=True)
class SpinnerStartEvent:
    id: str
    message: str
    session: str
    timestamp: float = dataclasses.field(default_factory=time.time)


@dataclasses.dataclass(frozen=True, slots=True)
class SpinnerUpdateEvent:
    id: str
    message: str
    session: str
    timestamp: float = dataclasses.field(default_factory=time.time)


@dataclasses.dataclass(frozen=True, slots=True)
class SpinnerEndEvent:
    id: str
    success: bool
    session: str
    timestamp: float = dataclasses.field(default_factory=time.time)


@dataclasses.dataclass(frozen=True, slots=True)
class SessionStartEvent:
    name: str
    timestamp: float = dataclasses.field(default_factory=time.time)


@dataclasses.dataclass(frozen=True, slots=True)
class SessionEndEvent:
    name: str
    timestamp: float = dataclasses.field(default_factory=time.time)


# Union type for all events
DarwinEvent = LogEvent | ProgressStartEvent | ProgressUpdateEvent | ProgressEndEvent | SpinnerStartEvent | SpinnerUpdateEvent | SpinnerEndEvent | SessionStartEvent | SessionEndEvent


# ---------------------------------------------------------------------------
# History store — stores ALL events (not just logs)
# ---------------------------------------------------------------------------
class HistoryStore:
    """
    In-memory ring buffer with optional disk persistence.
    Stores ALL event types (logs, progress, spinner, session).
    """

    def __init__(self, max_memory: int = 10_000,
                 persist_dir: Optional[Path] = None,
                 persist_every: float = 1.0) -> None:
        self._events: deque[DarwinEvent] = deque(maxlen=max_memory)
        self._lock = threading.RLock()
        self._persist_dir = persist_dir
        self._persist_every = persist_every
        self._last_flush = 0.0
        self._pending: List[DarwinEvent] = []

        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._log_file = persist_dir / f"darwin_{datetime.now().strftime('%Y%m%d')}.jsonl"
        else:
            self._log_file = None

    def append(self, event: DarwinEvent) -> None:
        with self._lock:
            self._events.append(event)
            if self._log_file:
                self._pending.append(event)
                now = time.time()
                if now - self._last_flush >= self._persist_every:
                    self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        if not self._pending or not self._log_file:
            return
        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                for e in self._pending:
                    d = {k: v for k, v in e.__dict__.items() if not k.startswith("_")}
                    if hasattr(d.get("log_type"), "value"):
                        d["log_type"] = d["log_type"].value
                    f.write(json.dumps(d, default=str) + "\n")
            self._pending.clear()
            self._last_flush = time.time()
        except Exception:
            pass

    def flush(self) -> None:
        with self._lock:
            self._flush_unlocked()

    def get_logs(self,
                 session: Optional[str] = None,
                 levels: Optional[Set[LogType]] = None,
                 after: Optional[float] = None,
                 before: Optional[float] = None,
                 limit: int = 1000) -> List[LogEvent]:
        """Query LogEvents only."""
        with self._lock:
            entries = list(self._events)
        result = []
        for e in reversed(entries):  # newest first
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
                break
        return result

    def get_events(self, session: Optional[str] = None, limit: int = 500) -> List[DarwinEvent]:
        """Query ALL event types."""
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
            return list(dict.fromkeys(
                getattr(e, "session", "") for e in self._events
                if getattr(e, "session", "")
            ))

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            log_count = sum(1 for e in self._events if isinstance(e, LogEvent))
            by_level = {}
            for e in self._events:
                if isinstance(e, LogEvent):
                    by_level[e.log_type.value] = by_level.get(e.log_type.value, 0) + 1
            return {
                "total_events": len(self._events),
                "log_events": log_count,
                "by_level": by_level,
                "sessions": self.sessions(),
                "persist_file": str(self._log_file) if self._log_file else None,
            }


# ---------------------------------------------------------------------------
# Subscriber protocol — all event types, but with default no-ops
# ---------------------------------------------------------------------------
class EventSubscriber:
    """Base subscriber. Override only what you care about."""

    def on_log(self, event: LogEvent) -> None: pass
    def on_progress_start(self, event: ProgressStartEvent) -> None: pass
    def on_progress_update(self, event: ProgressUpdateEvent) -> None: pass
    def on_progress_end(self, event: ProgressEndEvent) -> None: pass
    def on_spinner_start(self, event: SpinnerStartEvent) -> None: pass
    def on_spinner_update(self, event: SpinnerUpdateEvent) -> None: pass
    def on_spinner_end(self, event: SpinnerEndEvent) -> None: pass
    def on_session_start(self, event: SessionStartEvent) -> None: pass
    def on_session_end(self, event: SessionEndEvent) -> None: pass


# ---------------------------------------------------------------------------
# Structured serializer — with __darwin_type__ markers
# ---------------------------------------------------------------------------
_seen_sentinel = object()


def serialize_structured(obj: Any, _seen: Optional[Set[int]] = None) -> Any:
    """
    Convert to JSON-safe with __darwin_type__ markers.
    Preserves type info so receivers know what they're rendering.
    """
    if _seen is None:
        _seen = set()

    obj_id = id(obj)
    if obj_id in _seen:
        return {"__darwin_type__": "cycle", "ref": str(obj_id)}
    _seen.add(obj_id)

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = obj.__class__
        result: Dict[str, Any] = {"__darwin_type__": cls.__name__}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if f.metadata.get("sensitive"):
                value = "[REDACTED]"
            else:
                value = serialize_structured(value, _seen)
            result[f.name] = value
        return result

    if isinstance(obj, list):
        return {
            "__darwin_type__": "list",
            "items": [serialize_structured(item, _seen) for item in obj],
        }

    if isinstance(obj, dict):
        return {k: serialize_structured(v, _seen) for k, v in obj.items()}

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    return str(obj)


# ---------------------------------------------------------------------------
# Session handle — queryable
# ---------------------------------------------------------------------------
class SessionHandle:
    def __init__(self, name: str, store: HistoryStore,
                 start_time: float, logger: DarwinLogger) -> None:
        self.name = name
        self._store = store
        self._start_time = start_time
        self._end_time: Optional[float] = None
        self._logger = logger
        self._token: Optional[contextvars.Token] = None

    def logs(self,
             levels: Optional[Set[LogType]] = None,
             limit: int = 1000) -> List[LogEvent]:
        return self._store.get_logs(session=self.name, levels=levels, limit=limit)

    def errors(self, limit: int = 1000) -> List[LogEvent]:
        return self.logs(levels={LogType.ERROR}, limit=limit)

    def warnings(self, limit: int = 1000) -> List[LogEvent]:
        return self.logs(levels={LogType.WARNING}, limit=limit)

    def success_logs(self, limit: int = 1000) -> List[LogEvent]:
        return self.logs(levels={LogType.SUCCESS}, limit=limit)

    def all(self, limit: int = 1000) -> List[LogEvent]:
        return self.logs(limit=limit)

    def duration(self) -> Optional[float]:
        if self._end_time is None:
            return None
        return self._end_time - self._start_time

    def __enter__(self) -> SessionHandle:
        self._token = self._logger._session_var.set(self.name)
        self._logger._emit(SessionStartEvent(name=self.name))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._end_time = time.time()
        self._logger._emit(SessionEndEvent(name=self.name))
        if self._token is not None:
            self._logger._session_var.reset(self._token)


# ---------------------------------------------------------------------------
# Main logger
# ---------------------------------------------------------------------------
class DarwinLogger:
    def __init__(self) -> None:
        self._subscribers: List[EventSubscriber] = []
        self._history = HistoryStore()
        self._session_var: contextvars.ContextVar[str] = contextvars.ContextVar("darwin_session", default="")
        self._configured = False
        self._mode = OutputMode.NONE
        self._lock = threading.Lock()
        self._log_filter: Callable[[LogEvent], bool] = lambda e: True  # default: all pass
        self._sse_server = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def configure(self,
                  mode: OutputMode = OutputMode.TERMINAL,
                  persist: bool = False,
                  persist_dir: Optional[Path] = None,
                  max_memory: int = 10_000,
                  sse_port: int = 8765,
                  sse_host: str = "127.0.0.1",
                  show_banner: bool = True,
                  log_filter: LogFilter = LogFilter.ALL) -> None:   # ← NEW
        from src.lib.AppPaths import default_darwin_logs
        if self._configured:
            return
        self._configured = True
        self._mode = mode

        # Set filter
        if log_filter == LogFilter.QUIET:
            self._log_filter = lambda e: e.log_type in (LogType.ERROR, LogType.WARNING, LogType.SUCCESS)
        elif log_filter == LogFilter.SILENT:
            self._log_filter = lambda e: e.log_type == LogType.ERROR
        else:
            self._log_filter = lambda e: True

        # Default persist dir
        if persist and persist_dir is None:
            persist_dir = default_darwin_logs()
        elif not persist:
            persist_dir = None

        self._history = HistoryStore(max_memory=max_memory, persist_dir=persist_dir)

        # BOTH terminal + SSE if requested
        if mode in (OutputMode.TERMINAL, OutputMode.BOTH):
            self._attach_terminal(show_banner)
        if mode in (OutputMode.SSE, OutputMode.BOTH):
            self._start_sse(sse_host, sse_port, show_banner)

    def _attach_terminal(self, show_banner: bool) -> None:
        try:
            from src.core.Logger.TerminalUi import TerminalUI
            tui = TerminalUI()
            if show_banner:
                tui.show_banner()
            self.subscribe(tui)
        except ImportError:
            pass

    def _start_sse(self, host: str, port: int, show_banner: bool) -> None:
        try:
            from src.core.Logger.SSEServer import SSEServer
            from src.core.Logger.lib import ASCII_BANNER  
            server = SSEServer(host=host, port=port, logger=self)
            server.start()
            self._sse_server = server
            
            # Emit banner as structured log for SSE clients
            if show_banner:
                self.logjson(                               
                    {"banner": ASCII_BANNER},
                    LogType.INFO
                )

        except Exception as e:
            self._emit(LogEvent(
                message=f"SSE server failed to start: {e}",
                log_type=LogType.ERROR,
                session="",
                timestamp=time.time(),
            ))

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------
    def subscribe(self, subscriber: EventSubscriber) -> None:
        with self._lock:
            if subscriber not in self._subscribers:
                self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: EventSubscriber) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    # ------------------------------------------------------------------
    # Core emit
    # ------------------------------------------------------------------
    def _emit(self, event: DarwinEvent) -> None:
        # Apply filter ONLY to LogEvents before storing/forwarding
        if isinstance(event, LogEvent) and not self._log_filter(event):
            return  # silently drop filtered logs

        self._history.append(event)
        with self._lock:
            subs = self._subscribers[:]
        for sub in subs:
            try:
                self._dispatch(sub, event)
            except Exception:
                pass

    def _dispatch(self, sub: EventSubscriber, event: DarwinEvent) -> None:
        if isinstance(event, LogEvent):
            sub.on_log(event)
        elif isinstance(event, ProgressStartEvent):
            sub.on_progress_start(event)
        elif isinstance(event, ProgressUpdateEvent):
            sub.on_progress_update(event)
        elif isinstance(event, ProgressEndEvent):
            sub.on_progress_end(event)
        elif isinstance(event, SpinnerStartEvent):
            sub.on_spinner_start(event)
        elif isinstance(event, SpinnerUpdateEvent):
            sub.on_spinner_update(event)
        elif isinstance(event, SpinnerEndEvent):
            sub.on_spinner_end(event)
        elif isinstance(event, SessionStartEvent):
            sub.on_session_start(event)
        elif isinstance(event, SessionEndEvent):
            sub.on_session_end(event)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log(self, message: str, log_type: LogType = LogType.INFO) -> None:
        self._emit(LogEvent(
            message=message,
            log_type=log_type,
            session=self._session_var.get(),
            timestamp=time.time(),
        ))

    def info(self, message: str) -> None:
        self.log(message, LogType.INFO)

    def warning(self, message: str) -> None:
        self.log(message, LogType.WARNING)

    def error(self, message: str) -> None:
        self.log(message, LogType.ERROR)

    def success(self, message: str) -> None:
        self.log(message, LogType.SUCCESS)

    def logjson(self, payload: Any, log_type: LogType = LogType.INFO) -> None:
        """Log structured data (dict, dataclass, etc). Preserves type info."""
        structured = serialize_structured(payload)
        self._emit(LogEvent(
            message=json.dumps(structured, indent=2, ensure_ascii=False),
            log_type=log_type,
            session=self._session_var.get(),
            timestamp=time.time(),
            payload=structured,
            format="json",
        ))

    def loglist(self, payload: Any, log_type: LogType = LogType.INFO) -> None:
        """Log a list with type preservation."""
        if not isinstance(payload, list):
            self.warning(f"loglist expected list, got {type(payload).__name__}")
            payload = [payload]
        structured = serialize_structured(payload)
        self._emit(LogEvent(
            message=json.dumps(structured, indent=2, ensure_ascii=False),
            log_type=log_type,
            session=self._session_var.get(),
            timestamp=time.time(),
            payload=structured,
            format="list",
        ))

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------
    def progress_start(self, id: str, total: int, label: str) -> None:
        self._emit(ProgressStartEvent(
            id=id, total=total, label=label,
            session=self._session_var.get(),
        ))

    def progress_update(self, id: str, current: int, status: str = "") -> None:
        self._emit(ProgressUpdateEvent(
            id=id, current=current, status=status,
            session=self._session_var.get(),
        ))

    def progress_end(self, id: str) -> None:
        self._emit(ProgressEndEvent(
            id=id, session=self._session_var.get(),
        ))

    def progress(self, id: str, total: int, label: str = ""):
        """Context manager for progress."""
        return _ProgressContext(self, id, total, label)

    # ------------------------------------------------------------------
    # Spinner
    # ------------------------------------------------------------------
    def spinner_start(self, id: str, message: str = "") -> None:
        self._emit(SpinnerStartEvent(
            id=id, message=message,
            session=self._session_var.get(),
        ))

    def spinner_update(self, id: str, message: str = "") -> None:
        self._emit(SpinnerUpdateEvent(
            id=id, message=message,
            session=self._session_var.get(),
        ))

    def spinner_stop(self, id: str, success: bool = True) -> None:
        self._emit(SpinnerEndEvent(
            id=id, success=success,
            session=self._session_var.get(),
        ))

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def session(self, name: str) -> SessionHandle:
        return SessionHandle(name, self._history, time.time(), self)

    # ------------------------------------------------------------------
    # History queries
    # ------------------------------------------------------------------
    def get_logs(self,
                 session: Optional[str] = None,
                 levels: Optional[Set[LogType]] = None,
                 after: Optional[float] = None,
                 before: Optional[float] = None,
                 limit: int = 1000) -> List[LogEvent]:
        return self._history.get_logs(
            session=session, levels=levels, after=after, before=before, limit=limit
        )

    def get_events(self, session: Optional[str] = None, limit: int = 500) -> List[DarwinEvent]:
        return self._history.get_events(session=session, limit=limit)

    def get_sessions(self) -> List[str]:
        return self._history.sessions()

    def history_stats(self) -> Dict[str, Any]:
        return self._history.stats()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        self._history.flush()
        with self._lock:
            self._subscribers.clear()


# ---------------------------------------------------------------------------
# Progress context manager
# ---------------------------------------------------------------------------
class _ProgressContext:
    def __init__(self, logger: DarwinLogger, id: str, total: int, label: str):
        self.logger = logger
        self.id = id
        self.total = total
        self.label = label

    def __enter__(self):
        self.logger.progress_start(self.id, self.total, self.label)
        return self

    def update(self, current: int, status: str = ""):
        self.logger.progress_update(self.id, current, status)

    def __exit__(self, *args):
        self.logger.progress_end(self.id)


# Global instance
_darwin_instance: Optional[DarwinLogger] = None
_darwin_lock = threading.Lock()


def get_logger() -> DarwinLogger:
    global _darwin_instance
    if _darwin_instance is None:
        with _darwin_lock:
            if _darwin_instance is None:
                _darwin_instance = DarwinLogger()
    return _darwin_instance


darwin = get_logger()