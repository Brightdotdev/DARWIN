"""
darwin_mp.py — Multiprocessing bridge for Darwin Logger.

Workers send log dicts through a multiprocessing.Queue.
A background thread in the main process reads them and emits via Darwin.
"""
import multiprocessing
import threading
from typing import Optional

from core.Logger.DarwinLogger import darwin, LogType


class DarwinMPBridge:
    """
    Usage:
        with DarwinMPBridge() as bridge:
            with ProcessPoolExecutor(
                max_workers=n,
                initializer=DarwinMPBridge.worker_init,
                initargs=(bridge.get_queue(),),
            ) as executor:
                ...
    """

    def __init__(self) -> None:
        self._queue: Optional[multiprocessing.Queue] = None
        self._listener: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    def get_queue(self) -> multiprocessing.Queue:
        if self._queue is None:
            self._queue = multiprocessing.Queue()
        return self._queue

    @staticmethod
    def worker_init(queue: multiprocessing.Queue) -> None:
        """Called in each worker process."""
        global _worker_darwin_queue
        _worker_darwin_queue = queue

    def _drain(self) -> None:
        """Background thread in main process."""
        queue = self._queue
        if queue is None:
            return
        while not self._shutdown.is_set():
            try:
                data = queue.get(timeout=0.1)
                if data is None:
                    break
                darwin.log(data["message"], LogType(data["log_type"]))
            except Exception:
                pass

    def __enter__(self):
        self._shutdown.clear()
        self.get_queue()
        self._listener = threading.Thread(target=self._drain, daemon=True)
        self._listener.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._shutdown.set()
        if self._queue is not None:
            try:
                self._queue.put(None, timeout=1.0)
            except Exception:
                pass
        if self._listener is not None:
            self._listener.join(timeout=2.0)
            self._listener = None


# Worker-side global — set by worker_init()
_worker_darwin_queue: Optional[multiprocessing.Queue] = None


def worker_log(message: str, log_type: LogType = LogType.INFO) -> None:
    """Log from inside a worker process."""
    if _worker_darwin_queue is None:
        return
    _worker_darwin_queue.put({
        "message": message,
        "log_type": log_type.value,
    })