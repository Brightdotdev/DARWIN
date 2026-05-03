# =============================================================================
# frontend_bridge.py — WebSocket / JSON emitter
# =============================================================================
from __future__ import annotations

import asyncio
import json
import sys

from src.core.Logger.DarwinLogger import (
    EventSubscriber,
    LogEvent,
    ProgressStartEvent,
    ProgressUpdateEvent,
    ProgressEndEvent,
    PromptEvent,
    DarwinLogger,
)
from typing import Optional

ASCII_BANNER = [
    "██████╗  █████╗ ██████╗ ██╗    ██╗██╗███╗   ██╗",
    "██╔══██╗██╔══██╗██╔══██╗██║    ██║██║████╗  ██║",
    "██║  ██║███████║██████╔╝██║ █╗ ██║██║██╔██╗ ██║",
    "██║  ██║██╔══██║██╔══██╗██║███╗██║██║██║╚██╗██║",
    "██████╔╝██║  ██║██║  ██║╚███╔███╔╝██║██║ ╚████║",
    "╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝",
]


class FrontendBridge(EventSubscriber):
    def __init__(
        self, 
        websocket=None, 
        fallback_stdout: bool = True,
        logger: Optional[DarwinLogger] = None,
    ) -> None:
        self._ws = websocket
        self._fallback = fallback_stdout
        self._logger = logger

    def _schedule(self, coro):
        """Schedule coroutine in the WebSocket loop, not the main loop."""
        if self._logger:
            loop = self._logger._get_ws_loop()
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(coro, loop)
                return
        # Fallback for when no dedicated WS loop exists
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

    def on_log(self, event: LogEvent) -> None:
        self._schedule(self._send({
            "type": "log",
            "payload": {
                "message": event.message,
                "log_type": event.log_type.value,
                "style": event.style,
            }
        }))

    def on_progress_start(self, event: ProgressStartEvent) -> None:
        self._schedule(self._send({
            "type": "progress",
            "payload": {"event": "start", "id": event.id, "total": event.total, "label": event.label}
        }))

    def on_progress_update(self, event: ProgressUpdateEvent) -> None:
        self._schedule(self._send({
            "type": "progress",
            "payload": {"event": "update", "id": event.id, "current": event.current, "status": event.status}
        }))

    def on_progress_end(self, event: ProgressEndEvent) -> None:
        self._schedule(self._send({
            "type": "progress",
            "payload": {"event": "end", "id": event.id}
        }))

    def on_prompt(self, event: PromptEvent) -> None:
        self._schedule(self._send({
            "type": "prompt",
            "payload": {
                "id": event.id,
                "message": event.message,
                "options": event.options,
                "action": event.action,
                "params": event.params,
            }
        }))