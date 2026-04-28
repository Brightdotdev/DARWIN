# src/core/Logger/ws_helpers.py — shared WebSocket setup
import asyncio
import threading
from typing import Optional
from src.core.Logger.DarwinLogger import darwin, LogType, OutputMode


# ── WebSocket background thread management ─────────────────────────────────────

class WebSocketServer:
    """Manages a WebSocket server in a background thread with its own event loop."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self._host = host
        self._port = port
        self._actual_port: int = port  # The port we actually got
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server_started = threading.Event()
        self._shutdown_requested = threading.Event()
    
    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop
    
    @property
    def actual_port(self) -> int:
        """The port the server actually bound to (may differ from requested)."""
        return self._actual_port
    
    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
        )
        self._thread.start()
        
        if not self._server_started.wait(timeout=10):
            raise RuntimeError("WebSocket server failed to start within timeout")
    
    def _run_loop(self) -> None:
        """Run the event loop. This runs in the background thread."""
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
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()
    
    async def _start_server(self) -> None:
        """Start the WebSocket server and block until shutdown. Auto-increments port if taken."""
        try:
            import websockets
        except ImportError:
            self._server_started.set()
            return
        
        async def handler(ws):
            bridge = darwin._get_bridge(ws)
            darwin.subscribe(bridge)
            await bridge.send_banner()
            darwin.log("Frontend client connected", LogType.SUCCESS)
            try:
                async for raw in ws:
                    try:
                        import json
                        data = json.loads(raw)
                        if data.get("type") == "prompt_response":
                            payload = data["payload"]
                            darwin.resolve_prompt(payload["id"], payload["choice"])
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                darwin.unsubscribe(bridge)
        
        # Try ports, incrementing if taken
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
                    return  # Success — exit the loop
            except OSError:
                if attempt < max_attempts - 1:
                    old_port = port
                    port += 1
                    darwin.log(f"Port {old_port} in use, trying {port}...", LogType.INFO)
                else:
                    darwin.log(f"All ports {self._port}–{port} are in use", LogType.ERROR)
                    self._server_started.set()
                    raise
    
    def shutdown(self, timeout: float = 3.0) -> None:
        """Shutdown the WebSocket server and its event loop."""
        if self._loop is None or self._thread is None:
            return
        
        self._shutdown_requested.set()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=timeout)
        
        if self._thread.is_alive():
            print("[WebSocket] Server thread did not terminate in time")


# ── Module-level helper functions ──────────────────────────────────────────────

def setup_websocket(ws_port: int = 8765, no_ws: bool = False, show_banner: bool = True) -> Optional[WebSocketServer]:
    """Shared WebSocket setup for all CLI entry points."""
    if no_ws:
        mode = OutputMode.TERMINAL
    else:
        mode = OutputMode.BOTH
    
    
    darwin.configure(mode=mode, ws_port=ws_port, show_banner=show_banner)
    
    if no_ws or mode not in (OutputMode.WEBSOCKET, OutputMode.BOTH):
        return None
    
    ws_server = WebSocketServer(port=ws_port)
    try:
        ws_server.start()
        darwin.set_ws_loop(ws_server.loop)
        return ws_server
    except RuntimeError as e:
        darwin.log(f"WebSocket server failed: {e}", LogType.WARNING)
        return None


def shutdown_websocket(ws_server: Optional[WebSocketServer]) -> None:
    """Clean shutdown for WebSocket server."""
    if ws_server:
        ws_server.shutdown()