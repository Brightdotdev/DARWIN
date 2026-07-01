"""
SSE (Server-Sent Events) server for one-way log streaming.
Replaces WebSocket. Supports ALL event types: logs, progress, spinner, sessions.

Endpoints:
    GET /stream   — SSE event stream (all events as JSON)
    GET /history  — Query params: session, levels, limit
    GET /health   — Health check
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Optional

from aiohttp import web
from aiohttp_sse import sse_response

from src.core.Logger.DarwinLogger import (
    EventSubscriber, DarwinEvent, LogEvent, LogType,
    ProgressStartEvent, ProgressUpdateEvent, ProgressEndEvent,
    SpinnerStartEvent, SpinnerUpdateEvent, SpinnerEndEvent,
    SessionStartEvent, SessionEndEvent,
)


class SSESubscriber(EventSubscriber):
    """Buffers ALL events for a single SSE client."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._queue: asyncio.Queue[DarwinEvent] = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    def _put(self, event: DarwinEvent) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    def on_log(self, event: LogEvent) -> None: self._put(event)
    def on_progress_start(self, event: ProgressStartEvent) -> None: self._put(event)
    def on_progress_update(self, event: ProgressUpdateEvent) -> None: self._put(event)
    def on_progress_end(self, event: ProgressEndEvent) -> None: self._put(event)
    def on_spinner_start(self, event: SpinnerStartEvent) -> None: self._put(event)
    def on_spinner_update(self, event: SpinnerUpdateEvent) -> None: self._put(event)
    def on_spinner_end(self, event: SpinnerEndEvent) -> None: self._put(event)
    def on_session_start(self, event: SessionStartEvent) -> None: self._put(event)
    def on_session_end(self, event: SessionEndEvent) -> None: self._put(event)

    async def iter_events(self):
        """Async generator yielding SSE data lines."""
        while not self._closed:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=15.0)
                yield json.dumps(_event_to_dict(event), default=str)
            except asyncio.TimeoutError:
                yield ""  # keepalive

    def close(self) -> None:
        self._closed = True


def _event_to_dict(event: DarwinEvent) -> dict:
    """Serialize any event to a dict with type discriminator."""
    base = {"type": event.__class__.__name__}
    base.update({k: v for k, v in event.__dict__.items() if not k.startswith("_")})
    if hasattr(base.get("log_type"), "value"):
        base["log_type"] = base["log_type"].value
    return base


class SSEServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765,
                 logger=None) -> None:
        self._host = host
        self._port = port
        self._logger = logger
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    def start(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        async def stream(request):
            sub = SSESubscriber()
            self._logger.subscribe(sub)
            try:
                async with sse_response(request) as resp:
                    async for data in sub.iter_events():
                        if data:
                            await resp.send(data)
                        if self._shutdown.is_set():
                            break
                return resp
            finally:
                sub.close()
                self._logger.unsubscribe(sub)

        async def health(request):
            return web.Response(text="ok")

        async def history(request):
            session = request.query.get("session")
            levels_raw = request.query.get("levels", "")
            limit = int(request.query.get("limit", 100))
            levels = None
            if levels_raw:
                levels = {LogType(l.strip()) for l in levels_raw.split(",") if l.strip()}
            logs = self._logger.get_logs(session=session, levels=levels, limit=limit)
            return web.json_response([l.to_dict() for l in logs])

        app = web.Application()
        app.router.add_get("/stream", stream)
        app.router.add_get("/health", health)
        app.router.add_get("/history", history)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()

        self._logger.log(
            f"SSE ready on http://{self._host}:{self._port}/stream",
            LogType.SUCCESS
        )

        while not self._shutdown.is_set():
            await asyncio.sleep(0.5)

        await runner.cleanup()

    def shutdown(self, timeout: float = 3.0) -> None:
        self._shutdown.set()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=timeout)