"""
Single server providing MCP tools + log streaming + history + health.

Transports:
  SSE  (legacy)     /sse + /messages/
  HTTP (modern)     /mcp (Streamable HTTP)

Endpoints:
  /stream             Log SSE — real-time darwin events
  /history            Query historical logs
  /health             Health check (store + models)
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import queue
import threading
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

from starlette.requests import Request
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from src.core.Logger.DarwinLogger import (
    darwin, LogType, LogEvent, DarwinEvent, EventSubscriber,
)


# ---------------------------------------------------------------------------
# SSE subscriber — thread-safe queue fed by darwin events
# ---------------------------------------------------------------------------


class SSEQueueSubscriber(EventSubscriber):
    """Thread-safe subscriber that buffers DarwinEvents for SSE consumers."""

    def __init__(self, maxsize: int = 1000):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._closed = False

    def on_log(self, event: LogEvent) -> None:
        self._put(event)

    def on_progress_start(self, event) -> None:
        self._put(event)

    def on_progress_update(self, event) -> None:
        self._put(event)

    def on_progress_end(self, event) -> None:
        self._put(event)

    def on_spinner_start(self, event) -> None:
        self._put(event)

    def on_spinner_update(self, event) -> None:
        self._put(event)

    def on_spinner_end(self, event) -> None:
        self._put(event)

    def on_session_start(self, event) -> None:
        self._put(event)

    def on_session_end(self, event) -> None:
        self._put(event)

    def _put(self, event) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            import warnings
            warnings.warn("SSE event queue full — event dropped")

    async def iter_events(self, loop: asyncio.AbstractEventLoop):
        """Async generator yielding JSON-encoded event strings."""
        while not self._closed:
            try:
                event = await loop.run_in_executor(None, self._queue.get, True, 1.0)
                yield json.dumps(_event_to_dict(event), default=str)
            except queue.Empty:
                continue
            except Exception as e:
                darwin.debug(f"SSE event iteration error: {type(e).__name__}: {e}")
                continue

    def close(self) -> None:
        self._closed = True


def _event_to_dict(event) -> dict:
    """Serialize any DarwinEvent to a dict with a type discriminator."""
    if dataclasses.is_dataclass(event) and not isinstance(event, type):
        base = dataclasses.asdict(event)
    else:
        base = dict(event.__dict__) if hasattr(event, "__dict__") else {}
    base["type"] = event.__class__.__name__
    if hasattr(base.get("log_type"), "value"):
        base["log_type"] = base["log_type"].value
    return base


# ---------------------------------------------------------------------------
# Unified server
# ---------------------------------------------------------------------------


class UnifiedServer:
    """
    Provides log SSE + MCP tools + history + health in one process.

    Two usage modes:
      * **Background thread** — created by ``DarwinLogger._start_sse()``
        when ``OutputMode.SSE`` or ``OutputMode.BOTH`` is selected.
      * **Foreground (CLI)** — used by ``python -m src.mcp --sse``.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_port_retries: int = 10,
        transport: str = "sse",
    ):
        self.host = host
        self._requested_port = port
        self.port = port
        self.max_port_retries = max_port_retries
        self.transport = transport
        self._mcp: Optional[FastMCP] = None
        self._uvicorn_server = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown = threading.Event()

    # ---- app construction ----

    def _resolve_port(self) -> int:
        """Check port availability, scan for a free one if occupied."""
        from src.lib.NetUtils import is_port_available, find_available_port

        if is_port_available(self.host, self._requested_port):
            self.port = self._requested_port
            return self.port

        darwin.warning(
            f"Port {self._requested_port} in use — scanning for free port..."
        )
        found = find_available_port(
            self.host, self._requested_port + 1, self.max_port_retries
        )
        if found is None:
            raise OSError(
                f"No available port found in range "
                f"{self._requested_port}-{self._requested_port + self.max_port_retries - 1}"
            )
        self.port = found
        darwin.info(f"Using port {self.port} instead", style="green")
        self._save_port_to_config(self.port)
        return self.port

    @staticmethod
    def _save_port_to_config(port: int) -> None:
        """Persist the resolved port to logging.json so MCP and logger share it."""
        try:
            from src.lib.AppPaths import default_config_dir
            path = default_config_dir() / "logging.json"
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("sse_port") != port:
                data["sse_port"] = port
                path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
                darwin.info(f"Saved port {port} to logging.json")
        except Exception as e:
            darwin.debug(f"Failed to save port to logging.json: {type(e).__name__}: {e}")

    def _build_app(self) -> FastMCP:
        """Register all MCP tools + custom HTTP routes and return the FastMCP instance."""
        from src.mcp.app_factory import create_mcp_app
        from src.core.Orchestrators.ModelOrchestrator import modelsOrchestrator

        mcp, orch = create_mcp_app(host=self.host, port=self.port)

        # ---- custom HTTP routes ----

        @mcp.custom_route("/stream", methods=["GET"])
        async def log_stream(request: Request) -> EventSourceResponse:
            """SSE endpoint: real-time darwin log events."""
            subscriber = SSEQueueSubscriber()
            darwin.subscribe(subscriber)

            async def event_stream():
                try:
                    loop = asyncio.get_event_loop()
                    async for data in subscriber.iter_events(loop):
                        if self._shutdown.is_set():
                            break
                        if data:
                            yield {"event": "log", "data": data}
                finally:
                    subscriber.close()
                    darwin.unsubscribe(subscriber)

            return EventSourceResponse(event_stream())

        @mcp.custom_route("/history", methods=["GET"])
        async def log_history(request: Request) -> JSONResponse:
            """Query historical logs. Supports ?before=<timestamp> for lazy loading."""
            session = request.query_params.get("session")
            levels_raw = request.query_params.get("levels", "")
            try:
                limit = min(int(request.query_params.get("limit", 100)), 10000)
            except (ValueError, TypeError):
                limit = 100
            before_raw = request.query_params.get("before")
            before_ts = float(before_raw) if before_raw else None
            levels = None
            if levels_raw:
                try:
                    levels = {LogType(l.strip()) for l in levels_raw.split(",") if l.strip()}
                except ValueError:
                    pass
            logs = darwin.get_logs(session=session, levels=levels, before=before_ts, limit=limit)
            return JSONResponse([l.to_dict() for l in logs])

        @mcp.custom_route("/health", methods=["GET"])
        async def health_check(request: Request) -> JSONResponse:
            checks = {}
            healthy = True

            try:
                report = orch.store.get_pipeline_report()
                checks["store"] = {
                    "status": "ok",
                    "files": len(report) if isinstance(report, list) else 0,
                }
            except Exception as e:
                checks["store"] = {"status": "error", "error": str(e)}
                healthy = False

            try:
                status = modelsOrchestrator.status()
                checks["models"] = {
                    "status": "ok" if status.all_ready else "degraded",
                    "all_ready": status.all_ready,
                    "summary": status.summary,
                }
                if not status.all_ready:
                    healthy = False
            except Exception as e:
                checks["models"] = {"status": "error", "error": str(e)}
                healthy = False

            return JSONResponse(
                {"status": "ok" if healthy else "degraded", "checks": checks},
                status_code=200 if healthy else 503,
            )

        @mcp.custom_route("/dashboard", methods=["GET"])
        async def dashboard(request: Request):
            """Serve the log dashboard HTML."""
            from starlette.responses import HTMLResponse
            html_path = Path(__file__).parent / "dashboard.html"
            return HTMLResponse(html_path.read_text(encoding="utf-8"))

        @mcp.custom_route("/banner", methods=["GET"])
        async def banner(request: Request) -> JSONResponse:
            """Return the Darwin ASCII banner for frontend display."""
            from src.core.Logger.lib import ASCII_BANNER
            return JSONResponse({"banner": ASCII_BANNER})

        self._mcp = mcp
        return mcp

    # ---- lifecycle: background thread (for DarwinLogger) ----

    def start(self) -> None:
        """Start the uvicorn server in a background daemon thread."""
        self._resolve_port()
        self._build_app()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            import uvicorn

            if self.transport == "http":
                app = self._mcp.streamable_http_app()
            else:
                app = self._mcp.sse_app()
            config = uvicorn.Config(
                app,
                host=self.host,
                port=self.port,
                log_level="info",
            )
            config.install_signal_handlers = False
            server = uvicorn.Server(config)
            self._uvicorn_server = server
            self._loop.run_until_complete(server.serve())
        except Exception as e:
            darwin.error(f"Unified server failed: {e}")
        finally:
            self._loop.close()

    # ---- lifecycle: foreground (for CLI) ----

    def run_foreground(self, transport: str = "sse") -> None:
        """Block and serve.  Used by ``python -m src.mcp --sse`` or ``--http``."""
        self._resolve_port()
        self._build_app()
        import uvicorn

        if transport == "http":
            darwin.info(f"Starting Streamable HTTP on {self.host}:{self.port}")
            app = self._mcp.streamable_http_app()
        else:
            darwin.info(f"Starting SSE on {self.host}:{self.port}")
            app = self._mcp.sse_app()
        darwin.info(f"Dashboard: http://{self.host}:{self.port}/dashboard")
        uvicorn.run(app, host=self.host, port=self.port, log_level="info")

    # ---- shutdown ----

    def shutdown(self, timeout: float = 3.0) -> None:
        """Gracefully stop the server (used in background-thread mode)."""
        self._shutdown.set()
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=timeout)
