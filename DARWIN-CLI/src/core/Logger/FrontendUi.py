# FrontendUi.py — WebSocket / JSON emitter (the real bridge)
from __future__ import annotations

import asyncio
import json
import sys
from typing import Optional

from src.core.Logger.DarwinLogger import (
    EventSubscriber,
    LogEvent,
    ProgressStartEvent,
    ProgressUpdateEvent,
    ProgressEndEvent,
    SpinnerStartEvent,
    SpinnerUpdateEvent,
    SpinnerEndEvent,
    SessionStartEvent,
    SessionEndEvent,
    DarwinLogger,
)
from src.core.Logger.lib import ASCII_BANNER


class FrontendBridge(EventSubscriber):
    def __init__(
        self,
        websocket=None,
        fallback_stdout: bool = False,
        logger: Optional[DarwinLogger] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._ws = websocket
        self._fallback = fallback_stdout
        self._logger = logger
        self._loop = loop

    def _schedule(self, coro):
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        else:
            try:
                loop = asyncio.get_running_loop()
                asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError:
                pass

    async def send_banner(self) -> None:
        await self._send({
            "type": "log",
            "payload": {
                "message": "\n".join(ASCII_BANNER),
                "log_type": "info",
                "style": "banner",
            }
        })

    async def _send(self, data: dict) -> None:
        if self._ws:
            try:
                await self._ws.send(json.dumps(data))
            except Exception:
                pass
        elif self._fallback:
            sys.stdout.write(json.dumps(data) + "\n")
            sys.stdout.flush()

    # ---------- Event handlers ----------
    def on_log(self, event: LogEvent) -> None:
        self._schedule(self._send({
            "type": "log",
            "payload": {
                "message": event.message,
                "log_type": event.log_type.value,
                "style": event.style,
                "session": event.session,
            }
        }))

    def on_progress_start(self, event: ProgressStartEvent) -> None:
        self._schedule(self._send({
            "type": "progress",
            "payload": {"event": "start", "id": event.id, "total": event.total, "label": event.label, "session": event.session}
        }))

    def on_progress_update(self, event: ProgressUpdateEvent) -> None:
        self._schedule(self._send({
            "type": "progress",
            "payload": {"event": "update", "id": event.id, "current": event.current, "status": event.status, "session": event.session}
        }))

    def on_progress_end(self, event: ProgressEndEvent) -> None:
        self._schedule(self._send({
            "type": "progress", "payload": {"event": "end", "id": event.id, "session": event.session}
        }))

    def on_spinner_start(self, event: SpinnerStartEvent) -> None:
        self._schedule(self._send({
            "type": "spinner",
            "payload": {"event": "start", "id": event.id, "message": event.message, "session": event.session}
        }))

    def on_spinner_update(self, event: SpinnerUpdateEvent) -> None:
        self._schedule(self._send({
            "type": "spinner",
            "payload": {"event": "update", "id": event.id, "message": event.message, "session": event.session}
        }))

    def on_spinner_end(self, event: SpinnerEndEvent) -> None:
        self._schedule(self._send({
            "type": "spinner",
            "payload": {"event": "end", "id": event.id, "success": event.success, "session": event.session}
        }))

    def on_session_start(self, event: SessionStartEvent) -> None:
        self._schedule(self._send({
            "type": "session",
            "payload": {"event": "start", "name": event.name}
        }))

    def on_session_end(self, event: SessionEndEvent) -> None:
        self._schedule(self._send({
            "type": "session",
            "payload": {"event": "end", "name": event.name}
        }))