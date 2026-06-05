# ws_helpers.py — WebSocket background server (single source of truth)
import asyncio
import json
import threading
from typing import Optional
from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity


class WebSocketServer:
    """Manages a WebSocket server in a background thread with its own event loop."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self._host = host
        self._port = port   
        self._actual_port: int = port
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server_started = threading.Event()
        self._shutdown_requested = threading.Event()

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    @property
    def actual_port(self) -> int:
        return self._actual_port

    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._server_started.wait(timeout=10):
            self.shutdown()
            # Don't raise an exception here since we don't want to crash the whole application if the WebSocket server fails, but we should log it.
            darwin.log("WebSocket server failed to start within timeout", LogType.ERROR)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        server_task = self._loop.create_task(self._start_server())
        try:
            self._loop.run_forever()
        finally:
            if not server_task.done():
                server_task.cancel()
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()

    async def _start_server(self) -> None:
        try:
            import websockets
        except ImportError:
            self._server_started.set()
            return

        async def handler(ws):
            bridge = darwin._get_bridge(ws, loop=self._loop)
            darwin.subscribe(bridge, verbosity=Verbosity.NORMAL)  # WebSocket can have its own verbosity
            await bridge.send_banner()
            try:
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                        if data.get("type") == "get_history":
                            limit = data.get("payload", {}).get("limit", 200)
                            session = data.get("payload", {}).get("session")
                            log_types = data.get("payload", {}).get("log_types")
                            start_time = data.get("payload", {}).get("start_time")
                            end_time = data.get("payload", {}).get("end_time")
                            # Convert log_types from strings to LogType enums
                            lt_set = None
                            if log_types:
                                lt_set = {LogType(lt) for lt in log_types}
                            transcripts = darwin.get_transcripts(
                                session=session,
                                log_types=lt_set,
                                start_time=start_time,
                                end_time=end_time,
                                limit=limit,
                                format="json"
                            )
                            await ws.send(json.dumps({"type": "history", "payload": transcripts}))
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                darwin.unsubscribe(bridge)

        port = self._port
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                async with websockets.serve(handler, self._host, port):
                    self._actual_port = port
                    darwin.log(f"WebSocket ready on ws://{self._host}:{port}", LogType.SUCCESS)
                    self._server_started.set()
                    while not self._shutdown_requested.is_set():
                        await asyncio.sleep(0.5)
                    return
            except OSError as e:
                if attempt < max_attempts - 1:
                    old_port = port
                    port += 1
                    darwin.log(f"Port {old_port} in use, trying {port}...", LogType.INFO)
                else:
                    darwin.log(f"All ports {self._port}–{port} are in use", LogType.ERROR)
                    self._server_started.set()
                    darwin.log("Failed to start WebSocket server", LogType.ERROR)

    def shutdown(self, timeout: float = 3.0) -> None:
        if self._loop is None or self._thread is None:
            return
        self._shutdown_requested.set()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            darwin.log("[WebSocket] Server thread did not terminate in time", LogType.ERROR)