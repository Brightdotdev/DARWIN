"""
Terminal UI subscriber — Rich-based output.
Handles logs, progress bars, spinners, and structured data trees.
"""

from __future__ import annotations

import sys
import threading
from typing import Any, Dict, Optional

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)
from rich.tree import Tree

from src.core.Logger.DarwinLogger import (
    EventSubscriber,
    LogEvent,
    LogType,
    ProgressStartEvent,
    ProgressUpdateEvent,
    ProgressEndEvent,
    SpinnerStartEvent,
    SpinnerUpdateEvent,
    SpinnerEndEvent,
    SessionStartEvent,
    SessionEndEvent,
    resolve_event_style,
)
from src.core.Logger.lib import ASCII_BANNER


_ICONS = {
    LogType.INFO: "[cyan]i[/cyan]",
    LogType.WARNING: "[yellow]![/yellow]",
    LogType.ERROR: "[red]x[/red]",
    LogType.SUCCESS: "[green]+[/green]",
    LogType.DEBUG: "[dim]•[/dim]",
}


_stdout_reconfigured = False


class TerminalUI(EventSubscriber):
    def __init__(self, console: Optional[Console] = None) -> None:
        global _stdout_reconfigured
        if console is None:
            if sys.platform == "win32" and not _stdout_reconfigured:
                try:
                    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                    _stdout_reconfigured = True
                except Exception:
                    _stdout_reconfigured = True
            console = Console(force_terminal=True)
        self.console = console
        self._lock = threading.Lock()
        self._current_session: str = ""
        self._progress = Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )
        self._spinner = Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            console=self.console,
            transient=True,
        )
        self._progress_tasks: Dict[str, int] = {}
        self._spinner_tasks: Dict[str, int] = {}
        self._progress_started = False
        self._spinner_started = False

    def show_banner(self) -> None:
        self.console.clear()
        for line in ASCII_BANNER:
            self.console.print(f"[cyan]{line}[/cyan]")
        self.console.print()

    def on_session_start(self, event: SessionStartEvent) -> None:
        self._current_session = event.name

    def on_session_end(self, event: SessionEndEvent) -> None:
        self._current_session = ""
    

    def on_log(self, event: LogEvent) -> None:
        icon = _ICONS.get(event.log_type, "")
        prefix = f"[dim][{event.session}][/dim] " if event.session else ""
        style = resolve_event_style(event)

        with self._lock:
            if event.payload is not None:
                self.console.print(f"  {prefix}{icon} [bold]{event.message}[/bold]")
                tree = self._build_tree(event.payload)
                self.console.print(tree)
            else:
                msg = f"  {prefix}{icon} {event.message}"
                self.console.print(msg, style=style or None)

    def _build_tree(self, data: Any, label: Optional[str] = None) -> Tree:
        if label is None:
            label = "root"
        tree = Tree(label)

        if isinstance(data, dict):
            dtype = data.get("__darwin_type__")
            if dtype == "list":
                items = data.get("items", [])
                list_tree = tree.add(
                    f"[bold cyan]list[/bold cyan] ({len(items)} items)"
                )
                for i, item in enumerate(items):
                    list_tree.add(self._build_tree(item, f"[{i}]"))
                return tree
            elif dtype in ("cycle", "max_depth"):
                tree.add(f"[italic red]{dtype} → {data.get('ref', '?')}[/italic red]")
                return tree
            else:
                if dtype:
                    tree.label = f"[bold green]{dtype}[/bold green]"
                for key, value in data.items():
                    if key == "__darwin_type__":
                        continue
                    tree.add(self._build_tree(value, f"[bold]{key}[/bold]"))
        elif isinstance(data, list):
            list_tree = tree.add(f"[bold cyan]list[/bold cyan] ({len(data)} items)")
            for i, item in enumerate(data):
                list_tree.add(self._build_tree(item, f"[{i}]"))
        else:
            tree.add(str(data))
        return tree

    def on_progress_start(self, event: ProgressStartEvent) -> None:
        if not self._progress_started:
            self._progress.start()
            self._progress_started = True
        task_id = self._progress.add_task(event.label, total=event.total)
        self._progress_tasks[event.id] = task_id

    def on_progress_update(self, event: ProgressUpdateEvent) -> None:
        task_id = self._progress_tasks.get(event.id)
        if task_id is None:
            return
        desc = event.status if event.status else None
        self._progress.update(task_id, completed=event.current, description=desc)

    def on_progress_end(self, event: ProgressEndEvent) -> None:
        task_id = self._progress_tasks.pop(event.id, None)
        if task_id is not None:
            self._progress.remove_task(task_id)
        if not self._progress_tasks and self._progress_started:
            self._progress.stop()
            self._progress_started = False

    def on_spinner_start(self, event: SpinnerStartEvent) -> None:
        if not self._spinner_started:
            self._spinner.start()
            self._spinner_started = True
        task_id = self._spinner.add_task(event.message, total=None)
        self._spinner_tasks[event.id] = task_id

    def on_spinner_update(self, event: SpinnerUpdateEvent) -> None:
        task_id = self._spinner_tasks.get(event.id)
        if task_id is None:
            return
        self._spinner.update(task_id, description=event.message)

    def on_spinner_end(self, event: SpinnerEndEvent) -> None:
        task_id = self._spinner_tasks.pop(event.id, None)
        if task_id is not None:
            self._spinner.remove_task(task_id)
        if not self._spinner_tasks and self._spinner_started:
            self._spinner.stop()
            self._spinner_started = False
            icon = "✓" if event.success else "✗"
            self.console.print(f"  {icon} {event.id} done")
