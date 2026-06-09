from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn,
    TaskProgressColumn,
)
from rich.tree import Tree

from src.core.Logger.DarwinLogger import (
    DarwinLogger, EventSubscriber, LogEvent, LogType,
    ProgressStartEvent, ProgressUpdateEvent, ProgressEndEvent,
    SpinnerStartEvent, SpinnerUpdateEvent, SpinnerEndEvent,
    SessionStartEvent, SessionEndEvent, darwin,
)
from src.core.Logger.lib import ASCII_BANNER

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
        self._current_session: str = ""

    def show_banner(self) -> None:
        self.console.clear()
        self.console.print()
        for line in ASCII_BANNER:
            self.console.print(f"[cyan]{line}[/cyan]")
        self.console.print()

    def attach(self) -> None:
        self.logger.subscribe(self)

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
        if event.payload is not None:
            self.console.print(f"{prefix}{icon} [bold]{event.format.upper()}[/bold]")
            tree = self._build_tree(event.payload)
            self.console.print(tree)
        else:
            self.console.print(f"  {prefix}{icon} {event.message}")

    def _build_tree(self, data: Any, label: Optional[str] = None) -> Tree:
        """Recursively build a Rich Tree from the serialised structure."""
        if label is None:
            label = "root"
        tree = Tree(label)

        if isinstance(data, dict):
            darwin_type = data.get("__darwin_type__")
            if darwin_type == "list":
                items = data.get("items", [])
                list_tree = tree.add(f"[bold cyan]list[/bold cyan] ({len(items)} items)")
                for i, item in enumerate(items):
                    item_tree = self._build_tree(item, f"[{i}]")
                    list_tree.children.append(item_tree)
                return tree
            elif darwin_type == "cycle":
                tree.add(f"[italic red]cycle → {data['ref']}[/italic red]")
                return tree
            else:
                if darwin_type:
                    tree.label = f"[bold green]{darwin_type}[/bold green]"
                for key, value in data.items():
                    if key == "__darwin_type__":
                        continue
                    child = self._build_tree(value, f"[bold]{key}[/bold]")
                    tree.children.append(child)
        elif isinstance(data, list):
            list_tree = tree.add(f"[bold cyan]list[/bold cyan] ({len(data)} items)")
            for i, item in enumerate(data):
                item_tree = self._build_tree(item, f"[{i}]")
                list_tree.children.append(item_tree)
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