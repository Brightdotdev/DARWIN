"""
src/lib/Terminal.py
-------------------
All terminal UI for Darwin.
Pure rich — no raw ANSI codes anywhere.

Design Philosophy:
- One progress pattern: @contextmanager + Progress()
- All trackers build on the same foundation
- Consistent API across all progress displays
"""

import time
from contextlib import contextmanager
from typing import List, Dict, Optional, Callable, Any
from rich.progress import (
    Progress, 
    BarColumn, 
    TextColumn, 
    TimeElapsedColumn,
    SpinnerColumn,
    TaskProgressColumn,
)
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.syntax import Syntax
from rich.rule import Rule
from rich.spinner import Spinner

# Initialize the global console instance
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
    """Unified terminal UI using Rich."""
    
    console = _console

    ASCII = [
        "  ██████╗  █████╗ ██████╗ ██╗    ██╗██╗███╗   ██╗",
        "  ██╔══██╗██╔══██╗██╔══██╗██║    ██║██║████╗  ██║",
        "  ██║  ██║███████║██████╔╝██║ █╗ ██║██║██╔██╗ ██║",
        "  ██║  ██║██╔══██║██╔══██╗██║███╗██║██║██║╚██╗██║",
        "  ██████╔╝██║  ██║██║  ██║╚███╔███╔╝██║██║ ╚████║",
        "  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝",
    ]

    # =========================================================================
    # Timer Factory
    # =========================================================================

    @classmethod
    def timer(cls, name: str) -> Timer:
        """Create a named timer for tracking stage durations."""
        return Timer(name, console=cls.console)

    # =========================================================================
    # Core Progress Pattern (The Foundation)
    # =========================================================================

    @classmethod
    @contextmanager
    def progress_bar(
        cls,
        description: str = "Processing",
        total: Optional[int] = None,
        transient: bool = True,
        show_speed: bool = False,
    ):
        """
        Core progress bar pattern. All other progress methods build on this.
        
        Args:
            description: Task description text
            total: Total steps (None for indeterminate)
            transient: Clear bar when done
            show_speed: Show items/second
            
        Yields:
            progress: Rich Progress instance
            task_id: ID of the created task
        """
        columns = [
            TextColumn("[cyan]{task.description}"),
            BarColumn(bar_width=30),
        ]
        
        if show_speed:
            columns.append(TaskProgressColumn())
        else:
            columns.append(TextColumn("[bold]{task.percentage:>5.1f}%"))
        
        columns.append(TimeElapsedColumn())
        
        progress = Progress(
            *columns,
            console=cls.console,
            transient=transient,
            auto_refresh=True,
        )
        
        with progress:
            task_id = progress.add_task(description, total=total)
            yield progress, task_id

    @classmethod
    @contextmanager
    def spinner_progress(cls, description: str = "Working"):
        """
        Simple spinner with text for indeterminate operations.
        """
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[cyan]{task.description}"),
            console=cls.console,
            transient=True,
        )
        
        with progress:
            task_id = progress.add_task(description, total=None)
            yield progress, task_id

    # =========================================================================
    # Specialized Progress Trackers (Built on Core Pattern)
    # =========================================================================

    @classmethod
    @contextmanager
    def track_embedding(cls, total_batches: int, model_name: str, file_name: str):
        """
        Specialized tracker for embedding progress.
        Shows model info and batch speed.
        """
        state = {
            "start_time": time.perf_counter(),
            "batch_times": [],
        }
        
        progress = Progress(
            TextColumn("[cyan]Embedding {task.fields[file]}"),
            BarColumn(bar_width=30),
            TextColumn("[bold]{task.percentage:>5.1f}%"),
            TextColumn("•"),
            TextColumn("[magenta]{task.fields[speed]:>6}s/batch"),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[model]}[/dim]"),
            console=cls.console,
            transient=False,
        )
        
        with progress:
            task_id = progress.add_task(
                f"[{model_name}]",
                total=total_batches,
                file=file_name[:20],
                model=model_name.split('/')[-1],
                speed="0.00",
            )
            
            def update(step: int, msg: Optional[str] = None):
                """Update progress and recalculate speed."""
                now = time.perf_counter()
                state["batch_times"].append(now)
                
                # Calculate average speed over last 5 batches
                if len(state["batch_times"]) > 1:
                    recent = state["batch_times"][-5:] if len(state["batch_times"]) >= 5 else state["batch_times"]
                    if len(recent) > 1:
                        avg = (recent[-1] - recent[0]) / (len(recent) - 1)
                    else:
                        avg = 0
                else:
                    avg = 0
                
                progress.update(
                    task_id,
                    completed=step,
                    speed=f"{avg:.2f}",
                    description=f"[cyan]Embedding[/cyan] {msg}" if msg else None,
                )
            
            try:
                yield update
            finally:
                # Ensure 100% completion display
                progress.update(task_id, completed=total_batches)

    @classmethod
    @contextmanager
    def parse_progress(cls, filename: str):
        """
        Single file parsing progress with spinner animation.
        """
        with cls.spinner_progress(f"Parsing {filename}") as (progress, task_id):
            state = {"status": "starting..."}
            
            def update(msg: str):
                state["status"] = msg
                progress.update(task_id, description=f"[cyan]Parsing {filename}[/cyan] — [dim]{msg}[/dim]")
            
            try:
                yield update
                progress.update(task_id, description=f"[green]✓ Parsed {filename}[/green] — [dim]{state['status']}[/dim]")
            except Exception:
                progress.update(task_id, description=f"[red]✗ Failed {filename}[/red]")
                raise

    @classmethod
    @contextmanager
    def multi_parse_progress(cls, filenames: List[str]):
        """
        Multiple file parsing progress with per-file status.
        """
        states: Dict[str, Dict] = {
            name: {"status": "queued", "done": False}
            for name in filenames
        }
        
        with cls.progress_bar(
            description="Processing files",
            total=len(filenames),
            transient=False,
        ) as (progress, task_id):
            
            def get_callback(filename: str):
                def _update(msg: str):
                    states[filename]["status"] = msg
                    if any(kw in msg.lower() for kw in ("done", "complete", "saved", "success")):
                        states[filename]["done"] = True
                    
                    # Count completed files
                    completed = sum(1 for s in states.values() if s["done"])
                    progress.update(task_id, completed=completed)
                    
                    # Show current file status
                    current = next((n for n in filenames if not states[n]["done"]), "")
                    if current:
                        progress.update(
                            task_id,
                            description=f"[cyan]{current}[/cyan] — [dim]{states[current]['status']}[/dim]"
                        )
                
                return _update
            
            yield get_callback

    @classmethod
    @contextmanager
    def track_chunking(cls, total_chunks: int, filename: str):
        """
        Track chunking progress.
        """
        with cls.progress_bar(
            description=f"Chunking {filename}",
            total=total_chunks,
            transient=False,
        ) as (progress, task_id):
            
            def update(chunks_done: int, msg: str = ""):
                progress.update(
                    task_id,
                    completed=chunks_done,
                    description=f"[cyan]Chunking {filename}[/cyan] — [dim]{msg}[/dim]" if msg else None,
                )
            
            yield update

    # =========================================================================
    # Splash Screens & Headers
    # =========================================================================

    @classmethod
    def print_ascii_splash(cls) -> None:
        """Print welcome splash screen."""
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
        """Clear screen and print ASCII logo."""
        cls.console.clear()
        cls.console.print()
        for line in cls.ASCII:
            cls.console.print(f"[cyan]{line}[/cyan]")
        cls.console.print()



    # =========================================================================
    # Status Displays
    # =========================================================================

    @classmethod
    def print_project_status(cls, store, query: str = "") -> None:
        """Display full project status with all files."""
        status = store.status()
        config = status.get("config", {})
        files = status.get("files", {})

        llm_display = config.get("llm_model") or "none"

        cls.console.print()
        for line in cls.ASCII:
            cls.console.print(f"[cyan]{line}[/cyan]")
        cls.console.print()

        cls.console.print(f"  [dim]project[/dim]  →  [white]{status.get('project_dir', 'unknown')}[/white]")
        if query:
            cls.console.print(f"  [dim]query[/dim]    →  [yellow]{query}[/yellow]")
        cls.console.print(f"  [dim]llm[/dim]      →  [white]{llm_display}[/white]")
        cls.console.print()

        if files:
            # Summary
            total = len(files)
            ready = sum(1 for f in files.values() if f.get("ready_for_search"))
            complete = sum(1 for f in files.values() if f.get("fully_processed"))
            
            cls.console.print(Panel(
                f"[bold]Files:[/bold] {total}    "
                f"[green]Ready:[/green] {ready}    "
                f"[cyan]Complete:[/cyan] {complete}",
                border_style="dim"
            ))
            cls.console.print()
            
            # Table
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("File", style="white")
            table.add_column("Fresh")
            table.add_column("Chunks")
            table.add_column("Embed")
            table.add_column("BM25")
            
            def icon(v): 
                return "[green]✓[/green]" if v else "[dim]·[/dim]"
            
            for filename, f in files.items():
                comp = f.get("components", {})
                table.add_row(
                    filename[:30],
                    "[green]✓[/green]" if f.get("fresh") else "[yellow]~[/yellow]",
                    icon(comp.get("chunks")),
                    icon(comp.get("embeddings")),
                    icon(comp.get("bm25")),
                )
            
            cls.console.print(table)
        else:
            cls.console.print("  [dim]no files indexed yet[/dim]")

        cls.console.print()
        cls.rule()
        cls.console.print()

    @classmethod
    def print_file_status(cls, store, filename: str) -> None:
        """Display detailed status for a single file."""
        f = store.file_status(filename)

        cls.console.clear()
        cls.print_darwin()

        cls.console.print(f"  [bold white]{filename}[/bold white]")
        cls.console.print()

        freshness = "[green]fresh[/green]" if f["fresh"] else "[yellow]stale[/yellow]"

        if f["fully_processed"]:
            state = "[green]complete[/green]"
        elif f["ready_for_search"]:
            state = "[cyan]ready for search[/cyan]"
        else:
            state = "[yellow]partial[/yellow]"

        cls.console.print(Panel(f"{freshness} · {state}", border_style="dim"))
        cls.console.print()

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

    # =========================================================================
    # Basic Messaging
    # =========================================================================

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
    def text(cls, text: str) -> None:
        cls.console.print(f"[white]  {text} [/white]")

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
    def table(cls, headers: List[str], rows: List[List[str]], title: str = "") -> None:
        """Display a formatted table."""
        t = Table(show_header=True, header_style="bold cyan", title=title or None)
        
        for header in headers:
            t.add_column(header)
        
        for row in rows:
            padded = row + [""] * (len(headers) - len(row))
            t.add_row(*padded)
    
        cls.console.print(t)

    @classmethod
    def divider(
        cls, 
        text: str = "", 
        style: str = "dim", 
        char: str = "─",
        width: Optional[int] = None
    ) -> None:
        """Print a horizontal divider."""
        if width is None:
            width = cls.console.width or 80
        
        if text:
            text = f" {text} "
            padding = (width - len(text)) // 2
            line = (char * padding) + text + (char * padding)
            if len(line) < width:
                line += char
        else:
            line = char * width
        
        cls.console.print(f"[{style}]{line}[/{style}]")