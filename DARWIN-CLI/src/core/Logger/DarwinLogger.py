from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import contextvars
import logging
import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from datetime import datetime

# Internal error logger for subscriber exceptions
_log = logging.getLogger("darwin.logger")


class OutputMode(Enum):
    TERMINAL = "terminal"
    WEBSOCKET = "websocket"
    NONE = "none"
    AUTO = "auto"  # auto-detect


class Verbosity(Enum):
    NORMAL = "normal"   # all logs
    QUIET = "quiet"     # only ERROR, WARNING, SUCCESS


class LogType(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass(frozen=True)
class LogEvent:
    type: str = "log"
    message: str = ""
    log_type: LogType = LogType.INFO
    style: str = ""
    session: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ProgressStartEvent:
    type: str = "progress_start"
    id: str = ""
    total: int = 0
    label: str = ""
    session: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ProgressUpdateEvent:
    type: str = "progress_update"
    id: str = ""
    current: int = 0
    status: str = ""
    session: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ProgressEndEvent:
    type: str = "progress_end"
    id: str = ""
    session: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SpinnerStartEvent:
    type: str = "spinner_start"
    id: str = ""
    message: str = ""
    session: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SpinnerUpdateEvent:
    type: str = "spinner_update"
    id: str = ""
    message: str = ""
    session: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SpinnerEndEvent:
    type: str = "spinner_end"
    id: str = ""
    success: bool = True
    session: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SessionStartEvent:
    type: str = "session_start"
    name: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SessionEndEvent:
    type: str = "session_end"
    name: str = ""
    timestamp: float = field(default_factory=time.time)


class EventSubscriber(ABC):
    @abstractmethod
    def on_log(self, event: LogEvent) -> None: ...

    @abstractmethod
    def on_progress_start(self, event: ProgressStartEvent) -> None: ...

    @abstractmethod
    def on_progress_update(self, event: ProgressUpdateEvent) -> None: ...

    @abstractmethod
    def on_progress_end(self, event: ProgressEndEvent) -> None: ...

    @abstractmethod
    def on_spinner_start(self, event: SpinnerStartEvent) -> None: ...

    @abstractmethod
    def on_spinner_update(self, event: SpinnerUpdateEvent) -> None: ...

    @abstractmethod
    def on_spinner_end(self, event: SpinnerEndEvent) -> None: ...

    @abstractmethod
    def on_session_start(self, event: SessionStartEvent) -> None: ...

    @abstractmethod
    def on_session_end(self, event: SessionEndEvent) -> None: ...


class DarwinLogger:
    _instance: Optional[DarwinLogger] = None
    _lock = threading.Lock()

    def __new__(cls) -> DarwinLogger:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._subscribers: List[EventSubscriber] = []
        self._subscriber_verbosity: Dict[EventSubscriber, Verbosity] = {}
        self._actions: Dict[str, Callable[..., Any]] = {}
        self._lock = threading.Lock()

        self._mode: OutputMode = OutputMode.NONE
        self._configured: bool = False
        self._tui: Any = None
        self._ws_server: Any = None
        self._history_recorder: Any = None
        self._default_verbosity: Verbosity = Verbosity.NORMAL

        # Session context variable
        self._session_var: contextvars.ContextVar[str] = contextvars.ContextVar("darwin_session", default="")

        # CHANGED: multiprocessing support for worker processes
        self._mp_queue: Optional[multiprocessing.Queue] = None
        self._mp_listener: Optional[threading.Thread] = None
        self._mp_shutdown: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Configuration (centralised)
    # ------------------------------------------------------------------
    def configure(
        self,
        mode: Optional[OutputMode] = None,
        enable_websocket: bool = False,
        ws_host: str = "localhost",
        ws_port: int = 8765,
        enable_history: bool = True,
        history_max_age: int = 10800,  # 3 hours
        verbosity_default: Verbosity = Verbosity.NORMAL,
        show_banner: bool = True,
    ) -> None:
        if self._configured:
            return
        self._configured = True
        self._default_verbosity = verbosity_default

        # Determine mode if not given
        if mode is None:
            if os.environ.get("DARWIN_HEADLESS") == "1":
                mode = OutputMode.WEBSOCKET
            elif sys.stdout.isatty():
                mode = OutputMode.TERMINAL
            else:
                mode = OutputMode.WEBSOCKET
        self._mode = mode

        # Attach terminal UI if in terminal mode
        if self._mode == OutputMode.TERMINAL:
            self._attach_terminal(show_banner)

        # Start WebSocket server if enabled
        if enable_websocket:
            self._start_websocket_server(ws_host, ws_port)

        # Start history recorder if enabled
        if enable_history:
            self._start_history_recorder(history_max_age)

    def _attach_terminal(self, show_banner: bool) -> None:
        try:
            from src.core.Logger.TerminalUi import TerminalUI
            self._tui = TerminalUI(logger=self)
            if show_banner:
                self._tui.show_banner()
            self._tui.attach()
        except ImportError:
            pass

    def _start_websocket_server(self, host: str, port: int) -> None:
        try:
            from src.core.Logger.WebsocketHelpers import WebSocketServer
            self._ws_server = WebSocketServer(host=host, port=port)
            self._ws_server.start()
            # Bridge will be created per connection inside the server
        except Exception as e:
            self.log(f"Failed to start WebSocket server: {e}", LogType.ERROR)

    def _start_history_recorder(self, max_age: int) -> None:
        try:
            from src.core.Logger.History import HistoryRecorder
            self._history_recorder = HistoryRecorder(max_age_seconds=max_age)
            self.subscribe(self._history_recorder)
        except Exception as e:
            self.log(f"Failed to start history recorder: {e}", LogType.ERROR)

    # ------------------------------------------------------------------
    # Multiprocessing support — NEW
    # ------------------------------------------------------------------
    def _is_worker_process(self) -> bool:
        """True if we're inside a ProcessPoolExecutor worker."""
        return multiprocessing.current_process().name != "MainProcess"

    def _ensure_mp_queue(self) -> multiprocessing.Queue:
        """Get or create the multiprocessing queue for worker log forwarding."""
        if self._mp_queue is None:
            self._mp_queue = multiprocessing.Queue()
        return self._mp_queue

    def start_mp_listener(self) -> None:
        """
        Start a background thread that drains worker log messages from the
        multiprocessing queue and replays them through the real logger.
        Call this once in the main process before spawning workers.
        """
        if self._mp_listener is not None:
            return  # Already running

        queue = self._ensure_mp_queue()
        self._mp_shutdown.clear()

        def _drain():
            while not self._mp_shutdown.is_set():
                try:
                    # Non-blocking with timeout so we can check shutdown
                    msg = queue.get(timeout=0.1)
                    if msg is None:
                        break
                    level_name, message = msg
                    log_type = LogType(level_name) if level_name in {lt.value for lt in LogType} else LogType.INFO
                    self.log(message, log_type)
                except Exception:
                    pass

        self._mp_listener = threading.Thread(target=_drain, daemon=True)
        self._mp_listener.start()

    def stop_mp_listener(self) -> None:
        """Stop the background listener thread."""
        self._mp_shutdown.set()
        if self._mp_queue is not None:
            try:
                self._mp_queue.put(None, timeout=1.0)
            except Exception:
                pass
        if self._mp_listener is not None:
            self._mp_listener.join(timeout=2.0)
            self._mp_listener = None

    # ------------------------------------------------------------------
    # Subscriber management with verbosity
    # ------------------------------------------------------------------
    def subscribe(self, subscriber: EventSubscriber, verbosity: Optional[Verbosity] = None) -> None:
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)
            self._subscriber_verbosity[subscriber] = verbosity if verbosity is not None else self._default_verbosity

    def unsubscribe(self, subscriber: EventSubscriber) -> None:
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)
            self._subscriber_verbosity.pop(subscriber, None)

    # ------------------------------------------------------------------
    # Action registration (kept for any future actions, no prompts)
    # ------------------------------------------------------------------
    def register_action(self, name: str, func: Callable[..., Any]) -> None:
        self._actions[name] = func

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------
    def _emit(self, event) -> None:
        # CHANGED: in worker processes, forward to main process via queue
        if self._is_worker_process():
            if isinstance(event, LogEvent):
                queue = self._ensure_mp_queue()
                try:
                    queue.put((event.log_type.value, event.message), timeout=0.5)
                except Exception:
                    pass  # Queue full or closed — drop silently
            return

        # Main process: normal subscriber dispatch
        session = self._session_var.get()
        if session and hasattr(event, "session"):
            event = event.__class__(**{**event.__dict__, "session": session})

        for sub in self._subscribers[:]:
            verbosity = self._subscriber_verbosity.get(sub, Verbosity.NORMAL)
            if isinstance(event, LogEvent):
                if verbosity == Verbosity.QUIET and event.log_type not in (LogType.ERROR, LogType.WARNING, LogType.SUCCESS):
                    continue
            try:
                self._dispatch(sub, event)
            except Exception:
                _log.exception("Subscriber failed to handle event %s", event.type)

    def _dispatch(self, sub: EventSubscriber, event) -> None:
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
    # Public API – logging, progress, spinner, sessions
    # ------------------------------------------------------------------
    def log(self, message: str, log_type: LogType = LogType.INFO, style: str = "") -> None:
        self._emit(LogEvent(message=message, log_type=log_type, style=style))

    def progress_start(self, id: str, total: int, label: str) -> None:
        self._emit(ProgressStartEvent(id=id, total=total, label=label))

    def progress_update(self, id: str, current: int, status: str = "") -> None:
        self._emit(ProgressUpdateEvent(id=id, current=current, status=status))

    def progress_end(self, id: str) -> None:
        self._emit(ProgressEndEvent(id=id))

    def spinner_start(self, id: str, message: str = "") -> None:
        self._emit(SpinnerStartEvent(id=id, message=message))

    def spinner_update(self, id: str, message: str = "") -> None:
        self._emit(SpinnerUpdateEvent(id=id, message=message))

    def spinner_stop(self, id: str, success: bool = True) -> None:
        self._emit(SpinnerEndEvent(id=id, success=success))

    def session(self, name: str):
        return _Session(self, name)

    # ------------------------------------------------------------------
    # Transcripts (formatted logs with filtering)
    # ------------------------------------------------------------------
    def get_transcripts(
        self,
        session: Optional[str] = None,
        log_types: Optional[Set[LogType]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 1000,
        format: str = "human",  # 'human' or 'json'
    ) -> List[str]:
        """
        Return formatted log strings.
        - session: filter by session name (empty string for no session)
        - log_types: set of LogType to include (None = all)
        - start_time / end_time: Unix timestamps or ISO format strings
        - limit: max number of events to return (most recent)
        - format: 'human' -> "[YYYY-MM-DD HH:MM:SS] LEVEL: message"
                  'json' -> JSON string per line
        """
        if self._history_recorder is None:
            return []

        def to_timestamp(t):
            if t is None:
                return None
            if isinstance(t, (int, float)):
                return float(t)
            if isinstance(t, str):
                return datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp()
            raise TypeError("start_time/end_time must be float, int, or ISO string")

        start_ts = to_timestamp(start_time)
        end_ts = to_timestamp(end_time)

        events = self._history_recorder.get_events(session=session, limit=limit * 2)

        filtered = []
        for ev in events:
            if not isinstance(ev, LogEvent):
                continue
            if log_types is not None and ev.log_type not in log_types:
                continue
            if start_ts is not None and ev.timestamp < start_ts:
                continue
            if end_ts is not None and ev.timestamp > end_ts:
                continue
            filtered.append(ev)

        if limit and len(filtered) > limit:
            filtered = filtered[-limit:]

        result = []
        for ev in filtered:
            dt = datetime.fromtimestamp(ev.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            level = ev.log_type.value.upper()
            if format == "human":
                result.append(f"[{dt}] {level}: {ev.message}")
            elif format == "json":
                import json
                result.append(json.dumps({
                    "timestamp": ev.timestamp,
                    "datetime": dt,
                    "level": level,
                    "message": ev.message,
                    "session": ev.session,
                }))
            else:
                raise ValueError("format must be 'human' or 'json'")
        return result

    # ------------------------------------------------------------------
    # Bridge factory (used by WebSocket server)
    # ------------------------------------------------------------------
    def _get_bridge(self, ws=None, loop=None):
        try:
            from src.core.Logger.FrontendUi import FrontendBridge
            return FrontendBridge(websocket=ws, fallback_stdout=False, logger=self, loop=loop)
        except ImportError:
            return _FallbackBridge(ws, logger=self)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        self.stop_mp_listener()
        if self._ws_server:
            self._ws_server.shutdown()
        self._subscribers.clear()
        self._subscriber_verbosity.clear()

    # ------------------------------------------------------------------
    # Progress context manager
    # ------------------------------------------------------------------
    class Progress:
        def __init__(self, logger: DarwinLogger, id: str, total: int, label: str):
            self.logger = logger
            self.id = id
            self.total = total
            self.current = 0
            self.label = label

        def __enter__(self):
            self.logger.progress_start(self.id, self.total, self.label)
            return self

        def update(self, current: int, status: str = ""):
            self.current = current
            self.logger.progress_update(self.id, current, status)

        def __exit__(self, *args):
            self.logger.progress_end(self.id)

    def progress(self, id: str, total: int, label: str = "") -> Progress:
        return self.Progress(self, id, total, label)


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------
class _Session:
    def __init__(self, log: DarwinLogger, name: str):
        self.logger = log
        self.name = name
        self._token = None

    def __enter__(self):
        self._token = self.logger._session_var.set(self.name)
        self.logger._emit(SessionStartEvent(name=self.name))
        return self

    def __exit__(self, *args):
        self.logger._emit(SessionEndEvent(name=self.name))
        self.logger._session_var.reset(self._token)


# ---------------------------------------------------------------------------
# Fallback no-op bridge
# ---------------------------------------------------------------------------
class _FallbackBridge(EventSubscriber):
    def __init__(self, ws=None, logger=None):
        self._ws = ws
        self._logger = logger

    def on_log(self, event): pass
    def on_progress_start(self, event): pass
    def on_progress_update(self, event): pass
    def on_progress_end(self, event): pass
    def on_spinner_start(self, event): pass
    def on_spinner_update(self, event): pass
    def on_spinner_end(self, event): pass
    def on_session_start(self, event): pass
    def on_session_end(self, event): pass


# Global instance
darwin = DarwinLogger()