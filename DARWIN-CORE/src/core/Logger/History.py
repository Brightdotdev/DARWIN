# history.py — Event history recorder with file persistence (auto-cleanup)
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

from src.core.Logger.DarwinLogger import (
    EventSubscriber, LogEvent, ProgressStartEvent, ProgressUpdateEvent, ProgressEndEvent,
    SpinnerStartEvent, SpinnerUpdateEvent, SpinnerEndEvent,
    SessionStartEvent, SessionEndEvent, darwin,
)

_HISTORY_DIR = Path(".darwin_logs")
_CLEANUP_INTERVAL = 300  # seconds


class HistoryRecorder(EventSubscriber):
    """Records all events in memory, and writes session events to file."""

    def __init__(self, max_age_seconds: int = 10800) -> None:
        self._events: List[object] = []
        self._lock = threading.Lock()
        self._active_session: Optional[str] = None
        self._session_events: List[object] = []
        self._max_age = max_age_seconds
        _HISTORY_DIR.mkdir(exist_ok=True)
        self._start_cleanup_thread()

    def on_log(self, event: LogEvent) -> None:
        self._store(event)

    def on_progress_start(self, event: ProgressStartEvent) -> None:
        self._store(event)

    def on_progress_update(self, event: ProgressUpdateEvent) -> None:
        self._store(event)

    def on_progress_end(self, event: ProgressEndEvent) -> None:
        self._store(event)

    def on_spinner_start(self, event: SpinnerStartEvent) -> None:
        self._store(event)

    def on_spinner_update(self, event: SpinnerUpdateEvent) -> None:
        self._store(event)

    def on_spinner_end(self, event: SpinnerEndEvent) -> None:
        self._store(event)

    def on_session_start(self, event: SessionStartEvent) -> None:
        with self._lock:
            self._active_session = event.name
            self._session_events = [event]
        self._store(event)

    def on_session_end(self, event: SessionEndEvent) -> None:
        self._store(event)
        with self._lock:
            if self._active_session:
                self._dump_session(event.name, self._session_events + [event])
                self._active_session = None
                self._session_events = []

    def _store(self, event) -> None:
        with self._lock:
            self._events.append(event)
            if self._active_session:
                self._session_events.append(event)

    def _dump_session(self, session_name: str, events: List[object]) -> None:
        filename = f"{session_name}_{time.time():.0f}.jsonl"
        filepath = _HISTORY_DIR / filename
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for ev in events:
                    d = {k: v for k, v in ev.__dict__.items() if not k.startswith("_")}
                    if hasattr(d.get("log_type"), "value"):
                        d["log_type"] = d["log_type"].value
                    f.write(json.dumps(d) + "\n")
        except Exception as e:
            print(f"[History] Failed to write session log: {e}")

    def _start_cleanup_thread(self) -> None:
        def _cleanup_loop():
            while True:
                self._remove_old_files()
                time.sleep(_CLEANUP_INTERVAL)
        t = threading.Thread(target=_cleanup_loop, daemon=True)
        t.start()

    def _remove_old_files(self) -> None:
        now = time.time()
        try:
            for f in _HISTORY_DIR.iterdir():
                if f.is_file() and f.suffix == ".jsonl":
                    age = now - f.stat().st_mtime
                    if age > self._max_age:
                        f.unlink()
        except Exception:
            pass

    def get_events(self, session: Optional[str] = None, limit: int = 500) -> List[object]:
        with self._lock:
            if session is not None:
                # session == "" means events without a session
                return [e for e in self._events if getattr(e, "session", "") == session][-limit:]
            return self._events[-limit:]