"""
UnifiedSSEServer — single server for log streaming + MCP protocol.
Mounts:
  - /stream          → log SSE (via sse-starlette)
  - /mcp/sse         → MCP SSE endpoint
  - /mcp/messages/   → MCP message handler
  - /history         → query historical logs
  - /health          → health check
"""

from __future__ import annotations

import asyncio
import json
import threading
import queue
import time
from typing import Optional, Any, Dict, List
from pathlib import Path

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from sse_starlette.sse import EventSourceResponse

# MCP server
from mcp.server import Server as MCPServer
from mcp.server.sse import SseServerTransport

# Darwin logger
from src.core.Logger.DarwinLogger import (
    darwin,
    LogType,
    LogEvent,
    DarwinEvent,
    ProgressStartEvent,
    ProgressUpdateEvent,
    ProgressEndEvent,
    SpinnerStartEvent,
    SpinnerUpdateEvent,
    SpinnerEndEvent,
    SessionStartEvent,
    SessionEndEvent,
)

# MCP tools definition
from src.core.Logger.McpServer import create_mcp_server

# -------------------------------------------------------------------
# Shared SSE subscriber (thread‑safe queue)
# -------------------------------------------------------------------


class SSEQueueSubscriber:
    """Thread‑safe subscriber that puts events into a queue."""

    def __init__(self, maxsize: int = 1000):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._closed = False

    def _put(self, event: DarwinEvent) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            pass

    def on_log(self, event: LogEvent) -> None:
        self._put(event)

    def on_progress_start(self, event: ProgressStartEvent) -> None:
        self._put(event)

    def on_progress_update(self, event: ProgressUpdateEvent) -> None:
        self._put(event)

    def on_progress_end(self, event: ProgressEndEvent) -> None:
        self._put(event)

    def on_spinner_start(self, event: SpinnerStartEvent) -> None:
        self._put(event)

    def on_spinner_update(self, event: SpinnerUpdateEvent) -> None:
        self._put(event)

    def on_spinner_end(self, event: SpinnerEndEvent) -> None:
        self._put(event)

    def on_session_start(self, event: SessionStartEvent) -> None:
        self._put(event)

    def on_session_end(self, event: SessionEndEvent) -> None:
        self._put(event)

    async def iter_events(self, loop: asyncio.AbstractEventLoop):
        """Async generator yielding JSON‑encoded events."""
        while not self._closed:
            try:
                event = await loop.run_in_executor(None, self._queue.get, True, 1.0)
                yield json.dumps(_event_to_dict(event), default=str)
            except queue.Empty:
                yield ""  # keepalive
            except Exception:
                continue

    def close(self):
        self._closed = True


def _event_to_dict(event: DarwinEvent) -> dict:
    """Serialize any event to a dict with type discriminator."""
    base = {"type": event.__class__.__name__}
    base.update({k: v for k, v in event.__dict__.items() if not k.startswith("_")})
    if hasattr(base.get("log_type"), "value"):
        base["log_type"] = base["log_type"].value
    return base


# -------------------------------------------------------------------
# Unified server class
# -------------------------------------------------------------------


class UnifiedSSEServer:
    """
    Single server providing:
      - Log SSE at /stream
      - MCP transport at /mcp/sse and /mcp/messages/
      - Log history at /history
      - Health at /health
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        store_id: Optional[str] = None,
        log_filter: str = "all",
        persist_logs: bool = True,
        persist_dir: Optional[Path] = None,
    ):
        self.host = host
        self.port = port
        self.store_id = store_id
        self.log_filter = log_filter
        self.persist_logs = persist_logs
        self.persist_dir = persist_dir

        self._mcp_server: Optional[MCPServer] = None
        self._sse_transport: Optional[SseServerTransport] = None
        self._app: Optional[Starlette] = None
        self._uvicorn_server = None
        self._serve_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    def start(self) -> None:
        """Start the server in a background thread."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            darwin.error(f"Unified server crashed: {e}")
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        from src.core.Logger.DarwinLogger import LogFilter, OutputMode

        filter_map = {
            "all": LogFilter.ALL,
            "quiet": LogFilter.QUIET,
            "silent": LogFilter.SILENT,
        }
        darwin.configure(
            mode=OutputMode.BOTH,
            persist=self.persist_logs,
            persist_dir=self.persist_dir,
            log_filter=filter_map.get(self.log_filter, LogFilter.ALL),
            show_banner=True,
            override=True,
            persist_config=True,
        )
        darwin.info(f"Starting unified server on {self.host}:{self.port}")

        self._mcp_server = create_mcp_server(store_id=self.store_id)
        self._sse_transport = SseServerTransport("/mcp/messages/")

        app = Starlette(
            routes=[
                Route("/stream", self._handle_log_sse),
                Route("/mcp/sse", self._handle_mcp_sse, methods=["GET"]),
                Mount("/mcp/messages/", app=self._sse_transport.handle_post_message),
                Route("/history", self._handle_history),
                Route("/health", self._handle_health),
            ]
        )
        self._app = app

        import uvicorn

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        config.install_signal_handlers = False
        server = uvicorn.Server(config)
        self._uvicorn_server = server

        self._serve_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.3)

        darwin.success(f"Unified server ready on http://{self.host}:{self.port}")
        darwin.info("  - Log SSE: /stream")
        darwin.info("  - MCP SSE: /mcp/sse")
        darwin.info("  - History: /history")
        darwin.info("  - Health: /health")

        while not self._shutdown.is_set():
            await asyncio.sleep(0.5)

        server.should_exit = True
        if self._serve_task:
            await self._serve_task
        darwin.info("Unified server stopped")

    # ---- Route handlers ----

    async def _handle_log_sse(self, request: Request) -> EventSourceResponse:
        """Endpoint for log SSE via sse-starlette."""
        subscriber = SSEQueueSubscriber()
        darwin.subscribe(subscriber)

        async def event_stream():
            try:
                async for data in subscriber.iter_events(asyncio.get_event_loop()):
                    if self._shutdown.is_set():
                        break
                    if data:
                        yield {"event": "log", "data": data}
            finally:
                subscriber.close()
                darwin.unsubscribe(subscriber)

        return EventSourceResponse(event_stream())

    async def _handle_mcp_sse(self, request: Request) -> Response:
        """MCP SSE endpoint – delegates to SseServerTransport."""
        async with self._sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await self._mcp_server.run(
                streams[0],
                streams[1],
                self._mcp_server.create_initialization_options(),
            )
        return Response(status_code=200)

    async def _handle_history(self, request: Request) -> JSONResponse:
        """Return historical logs from the logger."""
        session = request.query_params.get("session")
        levels_raw = request.query_params.get("levels", "")
        limit = int(request.query_params.get("limit", 100))
        levels = None
        if levels_raw:
            levels = {LogType(l.strip()) for l in levels_raw.split(",") if l.strip()}
        logs = darwin.get_logs(session=session, levels=levels, limit=limit)
        return JSONResponse([l.to_dict() for l in logs])

    async def _handle_health(self, request: Request) -> Response:
        return Response(text="ok", status_code=200)

    def shutdown(self, timeout: float = 3.0) -> None:
        """Shut down the server."""
        self._shutdown.set()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=timeout)
