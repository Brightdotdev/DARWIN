"""
src/lib/Terminal.py
-------------------
All terminal UI for Darwin.
Pure rich — no raw ANSI codes anywhere.
"""

import time
import sys
import io
from contextlib import contextmanager
from typing import List, Dict, Callable, Optional

from rich.console import Console
from rich.panel   import Panel
from rich.live    import Live
from rich.table   import Table
from rich.syntax  import Syntax
from rich.rule    import Rule
from rich.spinner import Spinner

# 1. Initialize the global console instance
# We force 'utf-8' here as a secondary fallback for Windows environments
_console = Console(force_terminal=True)

class Timer:
    """Stage timer that prints through a specific rich console."""

    def __init__(self, name: str, console: Console):
        self.name = name
        self.console = console
        self._t0 = time.perf_counter()
        self._last = self._t0

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.2f}s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.2f}m ({seconds:.1f}s)"
        return f"{seconds/3600:.2f}h ({minutes:.1f}m)"

    def split(self, label: str = "", msg: str = "") -> float:
        now = time.perf_counter()
        elapsed = now - self._last
        self._last = now
        time_str = self._format_time(elapsed)
        
        if not msg:
            msg = f"Finished {label}" if label else "Step completed"

        self.console.print(
            f"  [dim]⏱[/dim]  [dim cyan][{self.name}][/dim cyan] "
            f"[dim]{msg}[/dim] [dim]·[/dim] "
            f"[dim]elapsed[/dim] : [cyan]{time_str}[/cyan]"
        )
        return elapsed

    def done(self, msg: str = "Stage done") -> float:
        total = time.perf_counter() - self._t0
        time_str = self._format_time(total)

        self.console.print(
            f"  [dim]⏱[/dim]  [dim cyan][{self.name}][/dim cyan] "
            f"[dim]{msg}[/dim] [dim]·[/dim] "
            f"[cyan]total time[/cyan] : [bold cyan]{time_str}[/bold cyan]"
        )
        return total


class TerminalUI:
    # Use the shared console instance as a class attribute
    console = _console

    ASCII = [
        "  ██████╗  █████╗ ██████╗ ██╗    ██╗██╗███╗   ██╗",
        "  ██╔══██╗██╔══██╗██╔══██╗██║    ██║██║████╗  ██║",
        "  ██║  ██║███████║██████╔╝██║ █╗ ██║██║██╔██╗ ██║",
        "  ██║  ██║██╔══██║██╔══██╗██║███╗██║██║██║╚██╗██║",
        "  ██████╔╝██║  ██║██║  ██║╚███╔███╔╝██║██║ ╚████║",
        "  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝",
    ]
    # -- Timer factory --

    @classmethod
    def timer(cls, name: str) -> Timer:
        return Timer(name, console=cls.console)

    # -- Splash Screens --

    @classmethod
    def print_ascii_splash(cls) -> None:
        cls.console.print()
        for line in cls.ASCII:
            cls.console.print(f"[cyan]{line}[/cyan]")
        cls.console.print()
        cls.console.print(Panel(
            "[bold]welcome to darwin[/bold]\n\n"
            "let's get you set up. this takes about 2 minutes.\n"
            "you can change any of these settings later with:\n"
            "[cyan]python main.py config[/cyan]",
            border_style="dim",
            padding=(1, 2),
        ))
        cls.console.print()
    
    @classmethod
    def print_darwin(cls) -> None:
        cls.console.print()
        for line in cls.ASCII:
            cls.console.print(f"[cyan]{line}[/cyan]")
        cls.console.print()

    @classmethod
    def loading_with_ascii(cls):
        spinner = Spinner("dots", text="Starting...")

        with Live(spinner, refresh_per_second=10):
            time.sleep(4)

        cls.console.clear()
        for line in cls.ASCII:
            cls.console.print(line, style="bold cyan")
        cls.console.print()
        

    @classmethod
    def fade_in_ascii(cls):
        styles = ["dim", "cyan", "bold cyan"]

        for style in styles:
            cls.console.clear()
            for line in cls.ASCII:
                cls.console.print(line, style=style)
            time.sleep(0.2)



    @classmethod
    def print_project_status(cls, store, query: str) -> None:
        status = store.status()
        config = status.get("config", {})
        files = status.get("files", [])

        llm_display = config.get("llm_model") or "none"

        cls.console.print()
        for line in cls.ASCII:
            cls.console.print(f"[cyan]{line}[/cyan]")
        cls.console.print()

        cls.console.print(f"  [dim]project[/dim]  →  [white]{status.get('project_dir', 'unknown')}[/white]")
        cls.console.print(f"  [dim]query[/dim]    →  [yellow]{query}[/yellow]")
        cls.console.print(f"  [dim]llm[/dim]      →  [white]{llm_display}[/white]")
        cls.console.print()

        if files:
            for f in files:
                chunks = f.get("chunk_count", 0)
                fresh = "[green]fresh[/green]" if f.get("is_fresh") else "[yellow]stale[/yellow]"
                has_all = all([f.get("has_chunks"), f.get("has_embeddings"), f.get("has_bm25")])
                indexed = "[green]indexed[/green]" if has_all else "[yellow]not indexed[/yellow]"
                name = f.get("filename", "unnamed")[:30]
                cls.console.print(f"  [dim]{name}[/dim] → {fresh} · {indexed} · {chunks} chunks")
        else:
            cls.console.print("  [dim]no files indexed yet[/dim]")

        cls.console.print()
        cls.rule()
        cls.console.print()

    # -- Progress Displays --

    @classmethod
    @contextmanager
    def parse_progress(cls, filename: str):
        state = {"msg": "starting...", "done": False}

        def _make_renderable(frame: str = "⠋"):
            t = Table.grid(padding=(0, 1))
            t.add_column(width=3)
            t.add_column()
            t.add_column()
            t.add_row(f"[cyan]{frame}[/cyan]", f"[bold white]{filename}[/bold white]", f"[dim]{state['msg']}[/dim]")
            return t

        frames = ["⠋", "⠙", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        frame_idx = [0]

        try:
            with Live(_make_renderable(), console=cls.console, refresh_per_second=10, transient=True) as live:
                def update(msg: str) -> None:
                    state["msg"] = msg
                    frame_idx[0] = (frame_idx[0] + 1) % len(frames)
                    live.update(_make_renderable(frames[frame_idx[0]]))
                yield update
            cls.success(f"{filename} [dim]— {state['msg']}[/dim]")
        except Exception as e:
            cls.error(f"{filename} failed: {e}")
            raise

    @classmethod
    @contextmanager
    def multi_parse_progress(cls, filenames: List[str]):
        states: Dict[str, str] = {name: "queued" for name in filenames}
        frames = ["⠋", "⠙", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        ticks: Dict[str, int] = {name: 0 for name in filenames}
        done: set = set()

        def _make_renderable():
            t = Table.grid(padding=(0, 1))
            t.add_column(width=3)
            t.add_column(min_width=30)
            t.add_column()
            for name in filenames:
                if name in done:
                    icon, label = "[green]✓[/green]", f"[dim]{name}[/dim]"
                else:
                    frame = frames[ticks[name] % len(frames)]
                    icon, label = f"[cyan]{frame}[/cyan]", f"[white]{name}[/white]"
                t.add_row(icon, label, f"[dim]{states[name]}[/dim]")
            return t

        with Live(_make_renderable(), console=cls.console, refresh_per_second=10, transient=True) as live:
            def get_callback(filename: str):
                def _update(msg: str) -> None:
                    states[filename] = msg
                    ticks[filename] += 1
                    if any(kw in msg.lower() for kw in ("done", "complete", "saved", "success")):
                        done.add(filename)
                    live.update(_make_renderable())
                return _update
            yield get_callback

        cls.blank()
        for name in filenames:
            icon = "[green]✓[/green]" if name in done else "[red]✗[/red]"
            cls.console.print(f"  {icon} [bold white]{name}[/bold white]  [dim]{states[name]}[/dim]")

    # -- Basic Messaging --

    @classmethod
    def step(cls, label: str) -> None:
        cls.console.print(f"  [cyan]→[/cyan] {label}...", end="")

    @classmethod
    def done(cls, extra: str = "") -> None:
        suffix = f" [dim]{extra}[/dim]" if extra else ""
        cls.console.print(f" [green]done[/green]{suffix}")

    @classmethod
    def dim(cls, text: str) -> None:
        cls.console.print(f"  [dim]{text}[/dim]")

    @classmethod
    def bold(cls, text: str) -> None:
        cls.console.print(f"  [bold]{text}[/bold]")

    @classmethod
    def success(cls, message: str) -> None:
        cls.console.print(f"  [green]✓[/green] {message}")

    @classmethod
    def error(cls, message: str) -> None:
        cls.console.print(f"  [red]✗[/red] {message}")

    @classmethod
    def warn(cls, message: str) -> None:
        cls.console.print(f"  [yellow]⚠[/yellow] {message}")

    @classmethod
    def info(cls, label: str, value: str) -> None:
        cls.console.print(f"  [dim]{label}[/dim]  [white]{value}[/white]")

    @classmethod
    def yellow(cls, text: str) -> None:
        cls.console.print(f"  [yellow]{text}[/yellow]")

    @classmethod
    def blank(cls) -> None:
        cls.console.print()

    @classmethod
    def rule(cls, title: str = "") -> None:
        cls.console.print(Rule(title, style="dim"))

    @classmethod
    def panel(cls, text: str, title: str = "", border_style: str = "blue") -> None:
        cls.console.print(Panel(text, title=title or None, border_style=border_style, padding=(1, 2)))

    @classmethod
    def code_block(cls, text: str, language: str = "text") -> None:
        syntax = Syntax(text, language, theme="monokai", line_numbers=False)
        cls.console.print(syntax)

    
    @classmethod
    def print_project_status(cls, store) -> None:
        status = store.status()
        summary = status.get("summary", {})
        files = status.get("files", {})

        cls.console.clear()
        cls.print_darwin()

        cls.console.print(f"  [dim]project[/dim] → [white]{status.get('project_dir')}[/white]")
        cls.console.print()

        # 🔹 Summary panel
        cls.console.print(Panel(
            f"[bold]Files:[/bold] {summary.get('total_files',0)}    "
            f"[green]Ready:[/green] {summary.get('ready_for_search',0)}    "
            f"[cyan]Complete:[/cyan] {summary.get('fully_processed',0)}    "
            f"[yellow]Fresh:[/yellow] {summary.get('fresh_files',0)}",
            border_style="dim"
        ))

        cls.console.print()

        # 🔹 Table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("File", style="white", no_wrap=True)
        table.add_column("Fresh")
        table.add_column("Chunks")
        table.add_column("Embed")
        table.add_column("BM25")
        table.add_column("Graph")
        table.add_column("Status")

        for filename, f in files.items():
            comp = f["components"]

            def icon(v): return "[green]✓[/green]" if v else "[dim]·[/dim]"



            table.add_row(
                filename[:25],
                "[green]✓[/green]" if f["fresh"] else "[yellow]~[/yellow]",
                icon(comp["chunks"]),
                icon(comp["embeddings"]),
                icon(comp["bm25"]),
                icon(comp["graph"])
            )

        cls.console.print(table)
        cls.console.print()
        cls.rule()
    
    @classmethod
    def print_file_status(cls, store, filename: str) -> None:
        f = store.file_status(filename)

        cls.console.clear()
        cls.print_darwin()

        cls.console.print(f"  [bold white]{filename}[/bold white]")
        cls.console.print()

        # 🔹 Top status
        freshness = "[green]fresh[/green]" if f["fresh"] else "[yellow]stale[/yellow]"

        if f["fully_processed"]:
            state = "[green]complete[/green]"
        elif f["ready_for_search"]:
            state = "[cyan]ready for search[/cyan]"
        else:
            state = "[yellow]partial[/yellow]"

        cls.console.print(Panel(
            f"{freshness} · {state}",
            border_style="dim"
        ))

        cls.console.print()

        # 🔹 Components table
        table = Table(show_header=False)
        table.add_column("Component", style="dim")
        table.add_column("Status")

        comp = f["components"]

        def row(name, val):
            icon = "[green]✓[/green]" if val else "[red]✗[/red]"
            table.add_row(name, icon)

        row("Markdown", comp["markdown"])
        row("Chunks", comp["chunks"])
        row("Embeddings", comp["embeddings"])
        row("BM25", comp["bm25"])
        row("Stopwords", comp["stopwords"])
        row("Graph", comp["graph"])

        cls.console.print(table)
        cls.console.print()

        # 🔹 Meta info
        meta = f["meta"]

        cls.console.print(Panel(
            f"[dim]pages[/dim]      : {meta.get('page_count','-')}\n"
            f"[dim]chunks[/dim]     : {meta.get('chunk_count','-')}\n"
            f"[dim]parser[/dim]     : {meta.get('parser','-')}\n"
            f"[dim]updated[/dim]    : {meta.get('updated_at','-')}",
            title="meta",
            border_style="dim"
        ))

        cls.console.print()
        cls.rule()