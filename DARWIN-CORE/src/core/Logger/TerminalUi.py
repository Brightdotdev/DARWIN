"""
Terminal UI subscriber — Rich-based output.
Handles logs, progress bars, spinners, and structured data trees.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TaskProgressColumn,
)
from rich.tree import Tree

from src.core.Logger.DarwinLogger import (
    EventSubscriber, LogEvent, LogType,
    ProgressStartEvent, ProgressUpdateEvent, ProgressEndEvent,
    SpinnerStartEvent, SpinnerUpdateEvent, SpinnerEndEvent,
    SessionStartEvent, SessionEndEvent,
)
from src.core.Logger.lib import ASCII_BANNER  



_ICONS = {
    LogType.INFO:    ("[cyan]ℹ[/cyan]", "cyan"),
    LogType.WARNING: ("[yellow]⚠[/yellow]", "yellow"),
    LogType.ERROR:   ("[red]✗[/red]", "red"),
    LogType.SUCCESS: ("[green]✓[/green]", "green"),
}



class TerminalUI(EventSubscriber):
    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console(force_terminal=True)
        self._progress_trackers: Dict[str, tuple] = {}
        self._spinner_trackers: Dict[str, tuple] = {}
        self._current_session: str = ""

    def show_banner(self) -> None:
        self.console.clear()
        for line in ASCII_BANNER:
            self.console.print(f"[cyan]{line}[/cyan]")
        self.console.print()

    # ---------- Session ----------
    def on_session_start(self, event: SessionStartEvent) -> None:
        self._current_session = event.name
        self.console.print(f"  [dim]Session started: {event.name}[/dim]")

    def on_session_end(self, event: SessionEndEvent) -> None:
        self._current_session = ""
        self.console.print(f"  [dim]Session ended: {event.name}[/dim]")

    # ---------- Log ----------
    def on_log(self, event: LogEvent) -> None:
        icon, _ = _ICONS.get(event.log_type, ("", "white"))
        prefix = f"[dim][{event.session}][/dim] " if event.session else ""

        if event.payload is not None:
            self.console.print(f"  {prefix}{icon} [bold]{event.message.split(chr(10))[0]}[/bold]")
            tree = self._build_tree(event.payload)
            self.console.print(tree)
        else:
            self.console.print(f"  {prefix}{icon} {event.message}")

    def _build_tree(self, data: Any, label: Optional[str] = None) -> Tree:
        """Recursively build a Rich Tree from the serialized structure."""
        if label is None:
            label = "root"
        tree = Tree(label)

        if isinstance(data, dict):
            dtype = data.get("__darwin_type__")
            if dtype == "list":
                items = data.get("items", [])
                list_tree = tree.add(f"[bold cyan]list[/bold cyan] ({len(items)} items)")
                for i, item in enumerate(items):
                    list_tree.add(self._build_tree(item, f"[{i}]"))
                return tree
            elif dtype == "cycle":
                tree.add(f"[italic red]cycle → {data['ref']}[/italic red]")
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
        task_id = progress.add_task(event.label, total=event.total)
        self._progress_trackers[event.id] = (progress, task_id)

    def on_progress_update(self, event: ProgressUpdateEvent) -> None:
        tracker = self._progress_trackers.get(event.id)
        if tracker is None:
            return
        progress, task_id = tracker
        desc = event.status if event.status else None
        progress.update(task_id, completed=event.current, description=desc)

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
        task_id = progress.add_task(event.message, total=None)
        self._spinner_trackers[event.id] = (progress, task_id)

    def on_spinner_update(self, event: SpinnerUpdateEvent) -> None:
        tracker = self._spinner_trackers.get(event.id)
        if tracker is None:
            return
        progress, task_id = tracker
        progress.update(task_id, description=event.message)

    def on_spinner_end(self, event: SpinnerEndEvent) -> None:
        tracker = self._spinner_trackers.pop(event.id, None)
        if tracker:
            progress, _ = tracker
            progress.stop()
            icon = "✓" if event.success else "✗"
            self.console.print(f"  {icon} {event.id} done")