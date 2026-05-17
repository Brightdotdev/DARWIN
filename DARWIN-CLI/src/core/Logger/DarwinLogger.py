# darwin_logger.py — Unified Event Hub (no WebSocket, pure events)
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import contextvars
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from cachetools import TTLCache

# Internal error logger for subscriber exceptions
_log = logging.getLogger("darwin.logger")


class OutputMode(Enum):
    TERMINAL = "terminal"
    WEBSOCKET = "websocket"
    NONE = "none"


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
class PromptEvent:
    type: str = "prompt"
    id: str = ""
    message: str = ""
    options: List[str] = field(default_factory=list)
    action: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
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
    def on_prompt(self, event: PromptEvent) -> None: ...

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
        self._subscriber_filters: Dict[EventSubscriber, Optional[Callable[[Any], bool]]] = {}
        self._actions: Dict[str, Callable[..., Any]] = {}
        self._pending_prompts: Dict[str, asyncio.Future] = {}
        self._resolved_prompts: TTLCache = TTLCache(maxsize=1000, ttl=3600)  # 1h TTL
        self._lock = threading.Lock()

        self._mode: OutputMode = OutputMode.NONE
        self._ws_port: int = 8765
        self._ws_host: str = "localhost"
        self._configured: bool = False
        self._tui: Any = None

        # Session context variable
        self._session_var: contextvars.ContextVar[str] = contextvars.ContextVar("darwin_session", default="")

        # Optional history recorder reference (set externally)
        self._history_recorder: Optional[Any] = None

    # ------------------------------------------------------------------
    # Configuration (simplified, no WebSocket start)
    # ------------------------------------------------------------------
    def configure(
        self,
        mode: Optional[OutputMode] = None,
        ws_port: int = 8765,
        ws_host: str = "localhost",
        show_banner: bool = True,
    ) -> None:
        if self._configured:
            return
        self._configured = True

        if mode is None:
            if os.environ.get("DARWIN_HEADLESS") == "1":
                mode = OutputMode.WEBSOCKET
            elif sys.stdout.isatty():
                mode = OutputMode.TERMINAL  # terminal also starts WS via ws_helpers
            else:
                mode = OutputMode.WEBSOCKET

        self._mode = mode
        self._ws_port = ws_port
        self._ws_host = ws_host

        # Attach terminal if we have a TTY
        if self._mode == OutputMode.TERMINAL:
            self._attach_terminal(show_banner)

    def _attach_terminal(self, show_banner: bool) -> None:
        try:
            from src.core.Logger.TerminalUi import TerminalUI
            self._tui = TerminalUI(logger=self)
            if show_banner:
                self._tui.show_banner()
            self._tui.attach()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Subscriber management (with filtering)
    # ------------------------------------------------------------------
    def subscribe(self, subscriber: EventSubscriber, filter: Optional[Callable[[Any], bool]] = None) -> None:
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)
            self._subscriber_filters[subscriber] = filter

    def unsubscribe(self, subscriber: EventSubscriber) -> None:
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)
            self._subscriber_filters.pop(subscriber, None)

    # ------------------------------------------------------------------
    # Action registration
    # ------------------------------------------------------------------
    def register_action(self, name: str, func: Callable[..., Any]) -> None:
        self._actions[name] = func

    # ------------------------------------------------------------------
    # Emit helpers
    # ------------------------------------------------------------------
    def _emit(self, event) -> None:
        # Attach session if inside a context manager
        session = self._session_var.get()
        if session:
            event = event.__class__(**{**event.__dict__, "session": session})

        for sub in self._subscribers:
            filter_fn = self._subscriber_filters.get(sub)
            if filter_fn and not filter_fn(event):
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
        elif isinstance(event, PromptEvent):
            sub.on_prompt(event)
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
    # Public API – unchanged
    # ------------------------------------------------------------------
    def log(self, message: str, log_type: LogType = LogType.INFO, style: str = "") -> None:
        self._emit(LogEvent(message=message, log_type=log_type, style=style))

    def progress_start(self, id: str, total: int, label: str) -> None:
        self._emit(ProgressStartEvent(id=id, total=total, label=label))

    def progress_update(self, id: str, current: int, status: str = "") -> None:
        self._emit(ProgressUpdateEvent(id=id, current=current, status=status))

    def progress_end(self, id: str) -> None:
        self._emit(ProgressEndEvent(id=id))

    async def prompt(
        self,
        id: str,
        message: str,
        options: List[str],
        action: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        params = params or {}
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._lock:
            self._pending_prompts[id] = future
        self._emit(PromptEvent(id=id, message=message, options=options, action=action, params=params))
        choice = await future
        func = self._actions.get(action)
        if func is None:
            raise RuntimeError(f"Action '{action}' not registered")
        if asyncio.iscoroutinefunction(func):
            return await func(choice=choice, **params)
        return func(choice=choice, **params)

    def resolve_prompt(self, id: str, choice: str) -> None:
        with self._lock:
            future = self._pending_prompts.pop(id, None)
            if future is not None:
                self._resolved_prompts[id] = True
        if future and not future.done():
            loop = future.get_loop()
            loop.call_soon_threadsafe(future.set_result, choice)

    def is_prompt_resolved(self, id: str) -> bool:
        return id in self._resolved_prompts

    # ------------------------------------------------------------------
    # New: Spinner
    # ------------------------------------------------------------------
    def spinner_start(self, id: str, message: str = "") -> None:
        self._emit(SpinnerStartEvent(id=id, message=message))

    def spinner_update(self, id: str, message: str = "") -> None:
        self._emit(SpinnerUpdateEvent(id=id, message=message))

    def spinner_stop(self, id: str, success: bool = True) -> None:
        self._emit(SpinnerEndEvent(id=id, success=success))

    # ------------------------------------------------------------------
    # New: Session context manager
    # ------------------------------------------------------------------
    def session(self, name: str):
        return _Session(self, name)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        # Cancel all pending prompts
        for fut in self._pending_prompts.values():
            if not fut.done():
                fut.cancel()
        self._pending_prompts.clear()
        self._subscribers.clear()
        self._subscriber_filters.clear()

    # ------------------------------------------------------------------
    # Bridge factory (always FrontendBridge)
    # ------------------------------------------------------------------
    def _get_bridge(self, ws=None):
        try:
            from src.core.Logger.FrontendUi import FrontendBridge
            return FrontendBridge(websocket=ws, fallback_stdout=False, logger=self)
        except ImportError:
            # Fallback no-op bridge (extremely rare)
            return _FallbackBridge(ws, logger=self)

    # ------------------------------------------------------------------
    # Inner context manager for progress (unchanged)
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
    def __init__(self, logger: DarwinLogger, name: str):
        self.logger = logger
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
# Fallback no-op bridge (used only if FrontendUi.py is missing)
# ---------------------------------------------------------------------------
class _FallbackBridge(EventSubscriber):
    def __init__(self, ws=None, logger=None):
        self._ws = ws
        self._logger = logger

    def on_log(self, event): pass
    def on_progress_start(self, event): pass
    def on_progress_update(self, event): pass
    def on_progress_end(self, event): pass
    def on_prompt(self, event): pass
    def on_spinner_start(self, event): pass
    def on_spinner_update(self, event): pass
    def on_spinner_end(self, event): pass
    def on_session_start(self, event): pass
    def on_session_end(self, event): pass


# Global instance
darwin = DarwinLogger()