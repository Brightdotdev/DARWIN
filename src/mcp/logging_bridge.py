"""
MCP Logging Bridge — forwards DarwinLogger events as MCP log notifications.

Two ways to use:

1. Context manager in tool handlers (preferred):
       from src.mcp.logging_bridge import darwin_bridge
       async with darwin_bridge(ctx):
           # ... tool logic ...

2. Manual subscribe/unsubscribe:
       bridge = MCPLogBridge(ctx)
       darwin.subscribe(bridge)
       try:
           # ... tool logic ...
       finally:
           darwin.unsubscribe(bridge)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

from src.core.Logger.DarwinLogger import (
    darwin,
    EventSubscriber,
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

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context


class MCPLogBridge(EventSubscriber):
    """Subscribes to DarwinLogger and forwards events as MCP log notifications."""

    _LEVEL_MAP = {
        LogType.DEBUG: "debug",
        LogType.INFO: "info",
        LogType.SUCCESS: "info",
        LogType.WARNING: "warning",
        LogType.ERROR: "error",
    }

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self._closed = False

    def on_log(self, event: LogEvent) -> None:
        """Called by DarwinLogger for each log event. Schedules async MCP send."""
        if self._closed:
            return
        import asyncio

        level = self._LEVEL_MAP.get(event.log_type, "info")
        message = event.message or ""
        logger = f"darwin.{event.session}" if event.session else "darwin"

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send(level, message, logger))
        except RuntimeError:
            pass

    def on_progress_start(self, event: ProgressStartEvent) -> None:
        if self._closed:
            return
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send("info", f"Progress started: {event.total} steps", "darwin.progress"))
        except RuntimeError:
            pass

    def on_progress_update(self, event: ProgressUpdateEvent) -> None:
        if self._closed:
            return
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send("info", f"Progress: {event.current}/{event.total}", "darwin.progress"))
        except RuntimeError:
            pass

    def on_progress_end(self, event: ProgressEndEvent) -> None:
        if self._closed:
            return
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send("info", "Progress complete", "darwin.progress"))
        except RuntimeError:
            pass

    def on_spinner_start(self, event: SpinnerStartEvent) -> None:
        if self._closed:
            return
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send("info", event.message or "Working...", "darwin.spinner"))
        except RuntimeError:
            pass

    def on_spinner_update(self, event: SpinnerUpdateEvent) -> None:
        pass  # spinner frames are cosmetic, skip

    def on_spinner_end(self, event: SpinnerEndEvent) -> None:
        pass  # no notification needed

    def on_session_start(self, event: SessionStartEvent) -> None:
        if self._closed:
            return
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send("info", f"Session '{event.session}' started", "darwin.session"))
        except RuntimeError:
            pass

    def on_session_end(self, event: SessionEndEvent) -> None:
        if self._closed:
            return
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send("info", f"Session '{event.session}' ended", "darwin.session"))
        except RuntimeError:
            pass

    async def _send(self, level: str, message: str, logger: str) -> None:
        if self._closed:
            return
        try:
            await self._ctx.log(level, message, logger_name=logger)
        except Exception:
            pass

    def close(self) -> None:
        self._closed = True


@asynccontextmanager
async def darwin_bridge(ctx: Context) -> AsyncIterator[None]:
    """Subscribe DarwinLogger → MCP for the duration of a tool call."""
    bridge = MCPLogBridge(ctx)
    darwin.subscribe(bridge)
    try:
        yield
    finally:
        bridge.close()
        darwin.unsubscribe(bridge)
