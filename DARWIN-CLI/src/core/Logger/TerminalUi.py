# =============================================================================
# terminalui.py — Rich CLI Renderer (Windows + Unix compatible)
# =============================================================================
from __future__ import annotations

import asyncio
import sys
import threading
import time
from typing import Dict, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.core.Logger.lib import ASCII_BANNER

from src.core.Logger.DarwinLogger import (
    DarwinLogger,
    EventSubscriber,
    LogEvent,
    LogType,
    ProgressEndEvent,
    ProgressStartEvent,
    ProgressUpdateEvent,
    PromptEvent,
    darwin,
)

# Platform-specific non-blocking input
if sys.platform == "win32":
    import msvcrt

    def _char_available() -> bool:
        return msvcrt.kbhit()

    def _read_char() -> str:
        return msvcrt.getwch()

else:
    import select

    def _char_available() -> bool:
        return bool(select.select([sys.stdin], [], [], 0.0)[0])

    def _read_char() -> str:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# Inline ASCII banner to avoid circular imports


LOG_ICONS = {
    LogType.INFO:    ("[cyan]ℹ[/cyan]", "cyan"),
    LogType.WARNING: ("[yellow]⚠[/yellow]", "yellow"),
    LogType.ERROR:   ("[red]✗[/red]", "red"),
    LogType.SUCCESS: ("[green]✓[/green]", "green"),
}


class TerminalUI(EventSubscriber):
    """Subscribes to DarwinLogger. Renders events with Rich."""

    def __init__(self, logger: Optional[DarwinLogger] = None) -> None:
        self.console = Console(force_terminal=True)
        self.logger = logger or darwin
        self._progress_trackers: Dict[str, tuple] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._prompt_cancel_events: Dict[str, threading.Event] = {}

    def show_banner(self) -> None:
        self.console.clear()
        self.console.print()
        for line in ASCII_BANNER:
            self.console.print(f"[cyan]{line}[/cyan]")
        self.console.print()
        
    def attach(self) -> None:
        self.logger.subscribe(self)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    def detach(self) -> None:
        self.logger.unsubscribe(self)

    def on_log(self, event: LogEvent) -> None:
        icon, _ = LOG_ICONS.get(event.log_type, ("", "white"))
        self.console.print(f"  {icon} {event.message}")

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
        task_id = progress.add_task(event.label, total=event.total)
        self._progress_trackers[event.id] = (progress, task_id)

    def on_progress_update(self, event: ProgressUpdateEvent) -> None:
        tracker = self._progress_trackers.get(event.id)
        if tracker is None:
            return
        progress, task_id = tracker
        progress.update(task_id, completed=event.current,
                        description=event.status if event.status else None)

    def on_progress_end(self, event: ProgressEndEvent) -> None:
        tracker = self._progress_trackers.pop(event.id, None)
        if tracker is None:
            return
        progress, _ = tracker
        progress.stop()

    def on_prompt(self, event: PromptEvent) -> None:
        """Show prompt in terminal. Exits when resolved by another UI."""
        if self._loop is None:
            return

        cancel_event = threading.Event()
        self._prompt_cancel_events[event.id] = cancel_event

        # Capture references to avoid closure issues
        _logger = self.logger
        _console = self.console
        _loop = self._loop

        def _prompt_and_resolve():
            # Small delay to allow other UIs to potentially resolve first
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

                # Check if resolved during read
                if _logger.is_prompt_resolved(event.id):
                    _console.print("\n  [dim](resolved by another client)[/dim]")
                    break

                if char in ('\r', '\n'):
                    _console.print()
                    raw = buffer.strip()
                    buffer = ""
                    if raw.isdigit() and 1 <= int(raw) <= len(event.options):
                        selected = event.options[int(raw) - 1]
                        _loop.call_soon_threadsafe(
                            _logger.resolve_prompt, event.id, selected
                        )
                        break
                    elif raw:
                        _console.print(f"  [red]Invalid choice: {raw}[/red]")
                        _console.print("  Please select one of the available options")
                        _console.print(f"  Enter choice [1-{len(event.options)}]: ", end="")
                        sys.stdout.flush()
                elif char == '\x03':  # Ctrl+C
                    _console.print("^C")
                    break
                elif char == '\x04':  # Ctrl+D
                    break
                elif char in ('\x08', '\x7f'):  # Backspace
                    if buffer:
                        buffer = buffer[:-1]
                        _console.print('\b \b', end="")
                        sys.stdout.flush()
                elif char == '\x1b':  # ESC — ignore escape sequences
                    pass
                elif len(char) == 1 and ord(char) >= 32:  # Printable characters
                    buffer += char
                    _console.print(char, end="")
                    sys.stdout.flush()

            _console.print()
            self._prompt_cancel_events.pop(event.id, None)

        threading.Thread(target=_prompt_and_resolve, daemon=True).start()