# =============================================================================
# darwin_logger.py — Unified Core Event System with Auto-Detection
# =============================================================================
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class OutputMode(Enum):
    TERMINAL = "terminal"
    WEBSOCKET = "websocket"
    BOTH = "both"
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
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ProgressStartEvent:
    type: str = "progress_start"
    id: str = ""
    total: int = 0
    label: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ProgressUpdateEvent:
    type: str = "progress_update"
    id: str = ""
    current: int = 0
    status: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ProgressEndEvent:
    type: str = "progress_end"
    id: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class PromptEvent:
    type: str = "prompt"
    id: str = ""
    message: str = ""
    options: List[str] = field(default_factory=list)
    action: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventSubscriber:
    def on_log(self, event: LogEvent) -> None: ...
    def on_progress_start(self, event: ProgressStartEvent) -> None: ...
    def on_progress_update(self, event: ProgressUpdateEvent) -> None: ...
    def on_progress_end(self, event: ProgressEndEvent) -> None: ...
    def on_prompt(self, event: PromptEvent) -> None: ...


class DarwinLogger:
    _instance: Optional[DarwinLogger] = None

    def __new__(cls) -> DarwinLogger:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        
        self._subscribers: List[EventSubscriber] = []
        self._actions: Dict[str, Callable[..., Any]] = {}
        self._pending_prompts: Dict[str, asyncio.Future] = {}
        self._resolved_prompts: set = set()
        self._lock = threading.Lock()
        
        self._mode: OutputMode = OutputMode.NONE
        self._ws_port: int = 8765
        self._ws_host: str = "localhost"
        self._configured: bool = False
        self._ws_server_started: bool = False  # Prevent double-start
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None  # Track dedicated WS loop
        self._tui: Any = None

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
            if os.environ.get("DARWIN_MODE"):
                mode = OutputMode(os.environ["DARWIN_MODE"])
            elif os.environ.get("DARWIN_HEADLESS") == "1":
                mode = OutputMode.WEBSOCKET
            elif sys.stdout.isatty():
                mode = OutputMode.BOTH
            else:
                mode = OutputMode.WEBSOCKET
        
        self._mode = mode
        self._ws_port = ws_port
        self._ws_host = ws_host
        
        # Only attach terminal — WS is started externally
        if mode in (OutputMode.TERMINAL, OutputMode.BOTH):
            self._attach_terminal(show_banner)

    async def configure_async(
        self,
        mode: Optional[OutputMode] = None,
        ws_port: int = 8765,
        ws_host: str = "localhost",
        show_banner: bool = True,
    ) -> None:
        """Async configuration — only use if WS should auto-start."""
        self.configure(mode=mode, ws_port=ws_port, ws_host=ws_host, show_banner=show_banner)
        if self._mode in (OutputMode.WEBSOCKET, OutputMode.BOTH) and not self._ws_server_started:
            await self.start_ws_server()

    async def start_ws_server(self) -> None:
        """Start WebSocket server if not already running."""
        if self._ws_server_started:
            return
        self._ws_server_started = True
        await self._run_ws_server()

    def set_ws_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the dedicated event loop for WebSocket operations."""
        self._ws_loop = loop

    def _attach_terminal(self, show_banner: bool) -> None:
        try:
            from src.core.Logger.TerminalUi import TerminalUI
            self._tui = TerminalUI(logger=self)
            if show_banner:
                self._tui.show_banner()
            self._tui.attach()
        except ImportError:
            pass

    async def _run_ws_server(self) -> None:
        try:
            import websockets
        except ImportError:
            return

        async def handler(ws):
            bridge = self._get_bridge(ws)
            self.subscribe(bridge)
            await bridge.send_banner()
            
            
            try:
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                        if data.get("type") == "prompt_response":
                            payload = data["payload"]
                            self.resolve_prompt(payload["id"], payload["choice"])
                    except (json.JSONDecodeError, KeyError):
                        pass
            except Exception:
                pass
            finally:
                self.unsubscribe(bridge)

        try:
            async with websockets.serve(
                handler, 
                self._ws_host, 
                self._ws_port,
                reuse_port=False,  # Prevent double-bind on Windows
            ):
                self.log(f"WebSocket ready on ws://{self._ws_host}:{self._ws_port}", LogType.SUCCESS)
                await asyncio.Future()  # Run forever
        except asyncio.CancelledError:
            pass

    def _get_bridge(self, ws=None):
        try:
            from src.core.Logger.FrontendUi import FrontendBridge
            return FrontendBridge(websocket=ws, fallback_stdout=False, logger=self)
        except ImportError:
            return _MinimalBridge(ws, logger=self)

    def subscribe(self, subscriber: EventSubscriber) -> None:
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: EventSubscriber) -> None:
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)

    def register_action(self, name: str, func: Callable[..., Any]) -> None:
        self._actions[name] = func

    def _get_ws_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Get the loop to use for scheduling async callbacks."""
        if self._ws_loop and self._ws_loop.is_running():
            return self._ws_loop
        try:
            loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            return None

    def _emit(self, event) -> None:
        for sub in self._subscribers:
            try:
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
            except Exception:
                pass

    def log(
        self,
        message: str,
        log_type: LogType = LogType.INFO,
        style: str = "",
    ) -> None:
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
        
        self._emit(PromptEvent(
            id=id, message=message, options=options,
            action=action, params=params,
        ))
        
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
                self._resolved_prompts.add(id)
        if future and not future.done():
            loop = future.get_loop()
            loop.call_soon_threadsafe(future.set_result, choice)

    def is_prompt_resolved(self, id: str) -> bool:
        return id in self._resolved_prompts

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


class _MinimalBridge(EventSubscriber):
    def __init__(self, ws=None, logger: Optional[DarwinLogger] = None):
        self._ws = ws
        self._logger = logger

    def _schedule(self, coro):
        """Schedule coroutine in the WebSocket loop."""
        if self._logger:
            loop = self._logger._get_ws_loop()
            if loop:
                asyncio.run_coroutine_threadsafe(coro, loop)
                return
        # Fallback: try current loop
        try:
            loop = asyncio.get_running_loop()
            asyncio.ensure_future(coro)
        except RuntimeError:
            pass

    async def send_banner(self):
        banner = ["DARWIN SYSTEM", "Event-Driven Core"]
        await self._send({
            "type": "log",
            "payload": {"message": "\n".join(banner), "log_type": "info", "style": "banner"}
        })

    async def _send(self, data: dict):
        if self._ws:
            try:
                await self._ws.send(json.dumps(data))
            except Exception:
                pass

    def on_log(self, event: LogEvent):
        self._schedule(self._send({
            "type": "log",
            "payload": {"message": event.message, "log_type": event.log_type.value, "style": event.style}
        }))

    def on_progress_start(self, event: ProgressStartEvent):
        self._schedule(self._send({
            "type": "progress",
            "payload": {"event": "start", "id": event.id, "total": event.total, "label": event.label}
        }))

    def on_progress_update(self, event: ProgressUpdateEvent):
        self._schedule(self._send({
            "type": "progress",
            "payload": {"event": "update", "id": event.id, "current": event.current, "status": event.status}
        }))

    def on_progress_end(self, event: ProgressEndEvent):
        self._schedule(self._send({
            "type": "progress", "payload": {"event": "end", "id": event.id}
        }))

    def on_prompt(self, event: PromptEvent):
        self._schedule(self._send({
            "type": "prompt",
            "payload": {"id": event.id, "message": event.message, "options": event.options,
                        "action": event.action, "params": event.params}
        }))


darwin = DarwinLogger()