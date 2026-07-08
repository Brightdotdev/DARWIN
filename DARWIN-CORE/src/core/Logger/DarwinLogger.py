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

import contextvars
import dataclasses
import json
import threading
import time
import warnings
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class LogType(Enum):
    DEBUG = ("debug", 10)
    INFO = ("info", 20)
    SUCCESS = ("success", 25)
    WARNING = ("warning", 30)
    ERROR = ("error", 40)

    def __new__(cls, label: str, severity: int):
        obj = object.__new__(cls)
        obj._value_ = label
        obj.severity = severity
        return obj


class OutputMode(Enum):
    TERMINAL = "terminal"
    SSE = "sse"
    BOTH = "both"  # ← NEW: simultaneous terminal + SSE
    NONE = "none"


class LogFilter(Enum):
    """Filter presets for LogEvents."""

    ALL = "all"  # everything (default)
    QUIET = "quiet"  # errors + warnings + success only (no info)
    SILENT = "silent"  # errors only


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
    format: str = ""  # "json", "list", "" for plain

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
DarwinEvent = (
    LogEvent
    | ProgressStartEvent
    | ProgressUpdateEvent
    | ProgressEndEvent
    | SpinnerStartEvent
    | SpinnerUpdateEvent
    | SpinnerEndEvent
    | SessionStartEvent
    | SessionEndEvent
)


# ---------------------------------------------------------------------------
# History store — SQLite‑backed ring buffer (imported lazily to avoid circular)
# ---------------------------------------------------------------------------
def _create_history_store(max_memory: int, persist_dir: Optional[Path]) -> Any:
    from src.core.Logger.SQLiteHistoryStore import SQLiteHistoryStore

    return SQLiteHistoryStore(max_memory=max_memory, persist_dir=persist_dir)


# ---------------------------------------------------------------------------
# Subscriber protocol — all event types, but with default no-ops
# ---------------------------------------------------------------------------
class EventSubscriber:
    """Base subscriber. Override only what you care about."""

    def on_log(self, event: LogEvent) -> None:
        pass

    def on_progress_start(self, event: ProgressStartEvent) -> None:
        pass

    def on_progress_update(self, event: ProgressUpdateEvent) -> None:
        pass

    def on_progress_end(self, event: ProgressEndEvent) -> None:
        pass

    def on_spinner_start(self, event: SpinnerStartEvent) -> None:
        pass

    def on_spinner_update(self, event: SpinnerUpdateEvent) -> None:
        pass

    def on_spinner_end(self, event: SpinnerEndEvent) -> None:
        pass

    def on_session_start(self, event: SessionStartEvent) -> None:
        pass

    def on_session_end(self, event: SessionEndEvent) -> None:
        pass


# ---------------------------------------------------------------------------
# Structured serializer — with __darwin_type__ markers
# ---------------------------------------------------------------------------
_MAX_DEPTH = 20


def serialize_structured(obj: Any, _depth: int = 0) -> Any:
    """
    Convert to JSON-safe with __darwin_type__ markers.
    Preserves type info so receivers know what they're rendering.
    """
    if _depth > _MAX_DEPTH:
        return {"__darwin_type__": "max_depth", "ref": str(type(obj).__name__)}

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = obj.__class__
        result: Dict[str, Any] = {"__darwin_type__": cls.__name__}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if f.metadata.get("sensitive"):
                value = "[REDACTED]"
            else:
                value = serialize_structured(value, _depth + 1)
            result[f.name] = value
        return result

    if isinstance(obj, list):
        return {
            "__darwin_type__": "list",
            "items": [serialize_structured(item, _depth + 1) for item in obj],
        }

    if isinstance(obj, dict):
        return {k: serialize_structured(v, _depth + 1) for k, v in obj.items()}

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    return str(obj)


# ---------------------------------------------------------------------------
# Session handle — queryable
# ---------------------------------------------------------------------------
class SessionHandle:
    def __init__(
        self, name: str, store: Any, start_time: float, logger: DarwinLogger
    ) -> None:
        self.name = name
        self._store = store
        self._start_time = start_time
        self._end_time: Optional[float] = None
        self._logger = logger
        self._token: Optional[contextvars.Token] = None

    def logs(
        self, levels: Optional[Set[LogType]] = None, limit: int = 1000
    ) -> List[LogEvent]:
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
        self._history = _create_history_store(10_000, None)
        self._session_var: contextvars.ContextVar[str] = contextvars.ContextVar(
            "darwin_session", default=""
        )
        self._configured = False
        self._mode = OutputMode.NONE
        self._lock = threading.Lock()
        self._log_filter: Callable[[LogEvent], bool] = lambda e: True
        self._sse_server = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    _CONFIG_FILE = "logging.json"

    @staticmethod
    def _config_path() -> Path:
        from src.lib.AppPaths import default_config_dir

        return default_config_dir() / DarwinLogger._CONFIG_FILE

    @staticmethod
    def _save_config(config: dict) -> None:
        import json

        path = DarwinLogger._config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _load_config() -> Optional[dict]:
        import json

        path = DarwinLogger._config_path()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def configure(
        self,
        mode: OutputMode = OutputMode.TERMINAL,
        persist: bool = True,
        persist_dir: Optional[Path] = None,
        max_memory: int = 10_000,
        sse_port: int = 8765,
        sse_host: str = "127.0.0.1",
        show_banner: bool = True,
        log_filter: LogFilter = LogFilter.ALL,
        persist_config: bool = True,
        override: bool = False,
    ) -> None:
        if self._configured and not override:
            warnings.warn("configure() already called — pass override=True to reapply")
            return

        from src.lib.AppPaths import default_logs_dir

        # ── Load persisted config (unless override discards it) ──
        saved = self._load_config() if persist_config and not override else None

        if saved and not override:
            _mode = OutputMode(saved.get("mode", mode.value))
            _persist = saved.get("persist", persist)
            _persist_dir = (
                Path(saved["persist_dir"]) if saved.get("persist_dir") else persist_dir
            )
            _max_memory = saved.get("max_memory", max_memory)
            _sse_port = saved.get("sse_port", sse_port)
            _sse_host = saved.get("sse_host", sse_host)
            _show_banner = saved.get("show_banner", show_banner)
            _filter_str = saved.get("log_filter", log_filter.value)
            _log_filter = next(
                (f for f in LogFilter if f.value == _filter_str), log_filter
            )
        else:
            _mode = mode
            _persist = persist
            _persist_dir = persist_dir
            _max_memory = max_memory
            _sse_port = sse_port
            _sse_host = sse_host
            _show_banner = show_banner
            _log_filter = log_filter

            # ── Persist the new config ──
            if persist_config:
                self._save_config(
                    {
                        "mode": _mode.value,
                        "persist": _persist,
                        "persist_dir": str(_persist_dir) if _persist_dir else None,
                        "max_memory": _max_memory,
                        "sse_port": _sse_port,
                        "sse_host": _sse_host,
                        "show_banner": _show_banner,
                        "log_filter": _log_filter.value,
                    }
                )

        self._configured = True
        self._mode = _mode

        if _mode == OutputMode.NONE:
            warnings.warn("OutputMode.NONE: all log output is suppressed")

        if _log_filter == LogFilter.QUIET:
            self._log_filter = lambda e: e.log_type.severity >= 25
        elif _log_filter == LogFilter.SILENT:
            self._log_filter = lambda e: e.log_type == LogType.ERROR
        else:
            self._log_filter = lambda e: True

        if _persist and _persist_dir is None:
            _persist_dir = default_logs_dir()
        elif not _persist:
            _persist_dir = None

        self._history = _create_history_store(
            max_memory=_max_memory, persist_dir=_persist_dir
        )

        if _mode in (OutputMode.TERMINAL, OutputMode.BOTH):
            self._attach_terminal(_show_banner)
        if _mode in (OutputMode.SSE, OutputMode.BOTH):
            self._start_sse(_sse_host, _sse_port, _show_banner)

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
            from src.core.Logger.UnifiedSSEServer import UnifiedSSEServer
            from src.core.Logger.lib import ASCII_BANNER

            server = UnifiedSSEServer(host=host, port=port)
            server.start()
            self._sse_server = server

            if show_banner:
                self.logjson({"banner": ASCII_BANNER}, LogType.INFO)

        except Exception as e:
            self._emit(
                LogEvent(
                    message=f"SSE server failed to start: {e}",
                    log_type=LogType.ERROR,
                    session="",
                    timestamp=time.time(),
                )
            )

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
        if isinstance(event, LogEvent) and not self._log_filter(event):
            return

        self._history.append(event)
        with self._lock:
            for sub in self._subscribers:
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
        self._emit(
            LogEvent(
                message=message,
                log_type=log_type,
                session=self._session_var.get(),
                timestamp=time.time(),
            )
        )

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
        summary = str(type(payload).__name__) if not isinstance(payload, dict) else ""
        self._emit(
            LogEvent(
                message=summary or "json",
                log_type=log_type,
                session=self._session_var.get(),
                timestamp=time.time(),
                payload=structured,
                format="json",
            )
        )

    def loglist(self, payload: Any, log_type: LogType = LogType.INFO) -> None:
        """Log a list with type preservation."""
        if not isinstance(payload, list):
            self.warning(f"loglist expected list, got {type(payload).__name__}")
            payload = [payload]
        structured = serialize_structured(payload)
        self._emit(
            LogEvent(
                message=f"list ({len(payload)} items)",
                log_type=log_type,
                session=self._session_var.get(),
                timestamp=time.time(),
                payload=structured,
                format="list",
            )
        )

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------
    def progress_start(self, id: str, total: int, label: str) -> None:
        self._emit(
            ProgressStartEvent(
                id=id,
                total=total,
                label=label,
                session=self._session_var.get(),
            )
        )

    def progress_update(self, id: str, current: int, status: str = "") -> None:
        self._emit(
            ProgressUpdateEvent(
                id=id,
                current=current,
                status=status,
                session=self._session_var.get(),
            )
        )

    def progress_end(self, id: str) -> None:
        self._emit(
            ProgressEndEvent(
                id=id,
                session=self._session_var.get(),
            )
        )

    def progress(self, id: str, total: int, label: str = ""):
        """Context manager for progress."""
        return _ProgressContext(self, id, total, label)

    # ------------------------------------------------------------------
    # Spinner
    # ------------------------------------------------------------------
    def spinner_start(self, id: str, message: str = "") -> None:
        self._emit(
            SpinnerStartEvent(
                id=id,
                message=message,
                session=self._session_var.get(),
            )
        )

    def spinner_update(self, id: str, message: str = "") -> None:
        self._emit(
            SpinnerUpdateEvent(
                id=id,
                message=message,
                session=self._session_var.get(),
            )
        )

    def spinner_stop(self, id: str, success: bool = True) -> None:
        self._emit(
            SpinnerEndEvent(
                id=id,
                success=success,
                session=self._session_var.get(),
            )
        )

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def session(self, name: str) -> SessionHandle:
        return SessionHandle(name, self._history, time.time(), self)

    # ------------------------------------------------------------------
    # History queries
    # ------------------------------------------------------------------
    def get_logs(
        self,
        session: Optional[str] = None,
        levels: Optional[Set[LogType]] = None,
        after: Optional[float] = None,
        before: Optional[float] = None,
        limit: int = 1000,
    ) -> List[LogEvent]:
        return self._history.get_logs(
            session=session, levels=levels, after=after, before=before, limit=limit
        )

    def get_events(
        self, session: Optional[str] = None, limit: int = 500
    ) -> List[DarwinEvent]:
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
