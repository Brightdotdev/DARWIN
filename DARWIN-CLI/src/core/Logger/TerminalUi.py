# TerminalUi.py — Rich CLI Renderer with spinner and session prefix
from __future__ import annotations

import asyncio
import sys
import threading
import time
from typing import Dict, Optional

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn,
    TaskProgressColumn,
)

from src.core.Logger.DarwinLogger import (
    DarwinLogger, EventSubscriber, LogEvent, LogType,
    ProgressStartEvent, ProgressUpdateEvent, ProgressEndEvent,
    PromptEvent, SpinnerStartEvent, SpinnerUpdateEvent, SpinnerEndEvent,
    SessionStartEvent, SessionEndEvent, darwin,
)
from src.core.Logger.lib import ASCII_BANNER  # You must have this file

# Platform-specific non-blocking input (unchanged)
if sys.platform == "win32":
    import msvcrt
    def _char_available(): return msvcrt.kbhit()
    def _read_char(): return msvcrt.getwch()
else:
    import select, tty, termios
    def _char_available(): return bool(select.select([sys.stdin], [], [], 0.0)[0])
    def _read_char():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

LOG_ICONS = {
    LogType.INFO:    ("[cyan]ℹ[/cyan]", "cyan"),
    LogType.WARNING: ("[yellow]⚠[/yellow]", "yellow"),
    LogType.ERROR:   ("[red]✗[/red]", "red"),
    LogType.SUCCESS: ("[green]✓[/green]", "green"),
}


class TerminalUI(EventSubscriber):
    def __init__(self, logger: Optional[DarwinLogger] = None) -> None:
        self.console = Console(force_terminal=True)
        self.logger = logger or darwin
        self._progress_trackers: Dict[str, tuple] = {}
        self._spinner_trackers: Dict[str, tuple] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._prompt_cancel_events: Dict[str, threading.Event] = {}
        self._current_session: str = ""

    def show_banner(self) -> None:
        self.console.clear()
        for line in ASCII_BANNER:
            self.console.print(f"[cyan]{line}[/cyan]")
        self.console.print()

    def attach(self) -> None:
        self.logger.subscribe(self)  # No filter needed (or add one)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    def detach(self) -> None:
        self.logger.unsubscribe(self)

    def _session_prefix(self) -> str:
        return f"[dim][{self._current_session}][/dim] " if self._current_session else ""

    # ---------- Session ----------
    def on_session_start(self, event: SessionStartEvent) -> None:
        self._current_session = event.name
        self.console.print(f"  [dim]Session started: {event.name}[/dim]")

    def on_session_end(self, event: SessionEndEvent) -> None:
        self._current_session = ""
        self.console.print(f"  [dim]Session ended: {event.name}[/dim]")

    # ---------- Log ----------
    def on_log(self, event: LogEvent) -> None:
        icon, _ = LOG_ICONS.get(event.log_type, ("", "white"))
        prefix = self._session_prefix()
        self.console.print(f"  {prefix}{icon} {event.message}")

    # ---------- Progress ----------
    def on_progress_start(self, event: ProgressStartEvent) -> None:
        progress = Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )
        progress.start()
        task_id = progress.add_task(f"{self._session_prefix()}{event.label}", total=event.total)
        self._progress_trackers[event.id] = (progress, task_id)

    def on_progress_update(self, event: ProgressUpdateEvent) -> None:
        tracker = self._progress_trackers.get(event.id)
        if tracker is None:
            return
        progress, task_id = tracker
        description = event.status if event.status else None
        if description:
            description = f"{self._session_prefix()}{description}"
        progress.update(task_id, completed=event.current, description=description)

    def on_progress_end(self, event: ProgressEndEvent) -> None:
        tracker = self._progress_trackers.pop(event.id, None)
        if tracker:
            progress, _ = tracker
            progress.stop()

    # ---------- Spinner ----------
    def on_spinner_start(self, event: SpinnerStartEvent) -> None:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            console=self.console,
            transient=True,
        )
        progress.start()
        task_id = progress.add_task(f"{self._session_prefix()}{event.message}", total=None)
        self._spinner_trackers[event.id] = (progress, task_id)

    def on_spinner_update(self, event: SpinnerUpdateEvent) -> None:
        tracker = self._spinner_trackers.get(event.id)
        if tracker is None:
            return
        progress, task_id = tracker
        progress.update(task_id, description=f"{self._session_prefix()}{event.message}")

    def on_spinner_end(self, event: SpinnerEndEvent) -> None:
        tracker = self._spinner_trackers.pop(event.id, None)
        if tracker:
            progress, _ = tracker
            progress.stop()
            icon = "✓" if event.success else "✗"
            self.console.print(f"  {self._session_prefix()}{icon} {event.id} done")

    # ---------- Prompt (kept original, just for completeness) ----------
    def on_prompt(self, event: PromptEvent) -> None:
        if self._loop is None:
            return
        cancel_event = threading.Event()
        self._prompt_cancel_events[event.id] = cancel_event
        _logger = self.logger
        _console = self.console
        _loop = self._loop

        def _prompt_and_resolve():
            cancel_event.wait(1.0)
            if _logger.is_prompt_resolved(event.id):
                _console.print("  [dim](resolved by another client)[/dim]")
                self._prompt_cancel_events.pop(event.id, None)
                return
            _console.print()
            _console.print(f"  [bold yellow]?[/bold yellow] {event.message}")
            for i, opt in enumerate(event.options, 1):
                _console.print(f"    [dim]{i}.[/dim] {opt}")
            _console.print(f"  Enter choice [1-{len(event.options)}]: ", end="")
            sys.stdout.flush()
            buffer = ""
            while not _logger.is_prompt_resolved(event.id):
                if not _char_available():
                    time.sleep(0.1)
                    continue
                char = _read_char()
                if _logger.is_prompt_resolved(event.id):
                    _console.print("\n  [dim](resolved by another client)[/dim]")
                    break
                if char in ('\r', '\n'):
                    _console.print()
                    raw = buffer.strip()
                    buffer = ""
                    if raw.isdigit() and 1 <= int(raw) <= len(event.options):
                        selected = event.options[int(raw) - 1]
                        _loop.call_soon_threadsafe(_logger.resolve_prompt, event.id, selected)
                        break
                    elif raw:
                        _console.print(f"  [red]Invalid choice: {raw}[/red]")
                        _console.print("  Please select one of the available options")
                        _console.print(f"  Enter choice [1-{len(event.options)}]: ", end="")
                        sys.stdout.flush()
                elif char == '\x03':
                    _console.print("^C")
                    break
                elif char == '\x04':
                    break
                elif char in ('\x08', '\x7f'):
                    if buffer:
                        buffer = buffer[:-1]
                        _console.print('\b \b', end="")
                        sys.stdout.flush()
                elif char == '\x1b':
                    pass
                elif len(char) == 1 and ord(char) >= 32:
                    buffer += char
                    _console.print(char, end="")
                    sys.stdout.flush()
            _console.print()
            self._prompt_cancel_events.pop(event.id, None)

        threading.Thread(target=_prompt_and_resolve, daemon=True).start()