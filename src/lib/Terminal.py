"""
src/lib/Terminal.py
-------------------
All terminal UI for Darwin.
Pure rich — no raw ANSI codes anywhere.

Usage:
    from src.lib.Terminal import TerminalUI
    TerminalUI.print_ascii_splash()
    TerminalUI.step("parsing")
    TerminalUI.done("42 chunks in 0.3s")
    TerminalUI.dim("this text is dimmed")
    TerminalUI.bold("this text is bold")
"""

from rich.console import Console
from rich.panel   import Panel

console = Console()


class TerminalUI:

    ASCII = [
        "  ██████╗  █████╗ ██████╗ ██╗    ██╗██╗███╗   ██╗",
        "  ██╔══██╗██╔══██╗██╔══██╗██║    ██║██║████╗  ██║",
        "  ██║  ██║███████║██████╔╝██║ █╗ ██║██║██╔██╗ ██║",
        "  ██║  ██║██╔══██║██╔══██╗██║███╗██║██║██║╚██╗██║",
        "  ██████╔╝██║  ██║██║  ██║╚███╔███╔╝██║██║ ╚████║",
        "  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝",
    ]

    @classmethod
    def print_ascii_splash(cls) -> None:
        """Full onboarding splash with ASCII art and welcome panel."""
        console.print()
        for line in cls.ASCII:
            console.print(f"[cyan]{line}[/cyan]")
        console.print()
        console.print(Panel(
            "[bold]welcome to darwin[/bold]\n\n"
            "let's get you set up. this takes about 2 minutes.\n"
            "you can change any of these settings later with:\n"
            "[cyan]python main.py config[/cyan]",
            border_style = "dim",
            padding      = (1, 2),
        ))
        console.print()

    @classmethod
    def print_mini_splash(cls) -> None:
        """Compact splash for regular command runs — not full onboarding."""
        console.print()
        console.print("[cyan]  ██████╗  █████╗ ██████╗ ██╗    ██╗██╗███╗   ██╗[/cyan]")
        console.print(f"[cyan]  ██╔══██╗[/cyan][dim]  your knowledge. mapped.[/dim]")
        console.print()

    @classmethod
    def print_project_status(cls, store, query: str) -> None:
        """
        Show project status in the search splash.
        Displays current file index state, LLM config, and query.
        """
        status = store.status()
        config = status["config"]
        files  = status["files"]

        llm_display = config.get("llm_model") or "none"

        console.print()
        for line in cls.ASCII:
            console.print(f"[cyan]{line}[/cyan]")
        console.print()

        console.print(f"  [dim]project[/dim]  →  [white]{status['project_dir']}[/white]")
        console.print(f"  [dim]query[/dim]    →  [yellow]{query}[/yellow]")
        console.print(f"  [dim]llm[/dim]      →  [white]{llm_display}[/white]")
        console.print()

        if files:
            for f in files:
                chunks  = f["chunk_count"]
                fresh   = "[green]fresh[/green]"   if f["is_fresh"]  else "[yellow]stale[/yellow]"
                has_all = all([f["has_chunks"], f["has_embeddings"], f["has_bm25"]])
                indexed = "[green]indexed[/green]" if has_all else "[yellow]not indexed[/yellow]"
                name    = f["filename"][:30]
                console.print(
                    f"  [dim]{name}[/dim] → {fresh} · {indexed} · {chunks} chunks"
                )
        else:
            console.print("  [dim]no files indexed yet[/dim]")

        console.print()
        console.print("  " + "─" * 50)
        console.print()

    @classmethod
    def step(cls, label: str) -> None:
        """Print a step in progress — no newline, waits for done()."""
        console.print(f"  [cyan]→[/cyan] {label}...", end="")

    @classmethod
    def done(cls, extra: str = "") -> None:
        """Complete a step started with step()."""
        suffix = f" [dim]{extra}[/dim]" if extra else ""
        console.print(f" [green]done[/green]{suffix}")

    @classmethod
    def dim(cls, text: str) -> None:
        """Print dimmed text (useful for hints, footnotes, metadata)."""
        console.print(f"  [dim]{text}[/dim]")

    @classmethod
    def dim_inline(cls, text: str) -> None:
        """Print dimmed text without newline."""
        console.print(f"[dim]{text}[/dim]", end="")

    @classmethod
    def bold(cls, text: str) -> None:
        """Print bold text (useful for emphasis, titles, important info)."""
        console.print(f"  [bold]{text}[/bold]")

    @classmethod
    def bold_inline(cls, text: str) -> None:
        """Print bold text without newline."""
        console.print(f"[bold]{text}[/bold]", end="")

    @classmethod
    def success_inline(cls, text: str) -> None:
        """Print success text without newline."""
        console.print(f"[green]{text}[/green]", end="")

    @classmethod
    def cyan(cls, text: str) -> None:
        """Print cyan text (useful for commands, filenames)."""
        console.print(f"  [cyan]{text}[/cyan]")

    @classmethod
    def yellow(cls, text: str) -> None:
        """Print yellow text (useful for warnings, queries)."""
        console.print(f"  [yellow]{text}[/yellow]")

    @classmethod
    def green(cls, text: str) -> None:
        """Print green text (useful for success messages)."""
        console.print(f"  [green]{text}[/green]")

    @classmethod
    def error(cls, message: str) -> None:
        """Print an error line."""
        console.print(f"  [red]✗[/red] {message}")

    @classmethod
    def warn(cls, message: str) -> None:
        """Print a warning line."""
        console.print(f"  [yellow]⚠[/yellow] {message}")

    @classmethod
    def success(cls, message: str) -> None:
        """Print a success line."""
        console.print(f"  [green]✓[/green] {message}")

    @classmethod
    def info(cls, label: str, value: str) -> None:
        """Print a labelled info line."""
        console.print(f"  [dim]{label}[/dim]  [white]{value}[/white]")

    @classmethod
    def divider(cls) -> None:
        """Print a visual divider line."""
        console.print(f"  {'─' * 60}")

    @classmethod
    def blank(cls) -> None:
        """Print a blank line."""
        console.print()

    @classmethod
    def header(cls, text: str, level: int = 1) -> None:
        """Print a formatted header with different levels."""
        if level == 1:
            console.print(f"\n[bold cyan]{text}[/bold cyan]")
            console.print(f"[dim]{'─' * len(text)}[/dim]")
        elif level == 2:
            console.print(f"\n[bold]{text}[/bold]")
        else:
            console.print(f"\n[dim]{text}[/dim]")

    @classmethod
    def table(cls, headers: list, rows: list) -> None:
        """Print a simple formatted table."""
        from rich.table import Table
        
        table = Table(show_header=True, header_style="bold cyan")
        for header in headers:
            table.add_column(header)
        
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        
        console.print(table)

    @classmethod
    def progress_bar(cls, current: int, total: int, label: str = "") -> None:
        """Print a simple progress bar (non-refreshing, for logs)."""
        percentage = (current / total) * 100
        bar_length = 30
        filled = int(bar_length * current // total)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        prefix = f"{label} " if label else ""
        console.print(f"  {prefix}[cyan]{bar}[/cyan] [dim]{percentage:.1f}%[/dim]")

    @classmethod
    def code_block(cls, text: str, language: str = "") -> None:
        """Print a code block with syntax highlighting."""
        from rich.syntax import Syntax
        
        syntax = Syntax(text, language or "text", theme="monokai", line_numbers=False)
        console.print(syntax)

    @classmethod
    def panel(cls, text: str, title: str = "", border_style: str = "blue") -> None:
        """Print content in a bordered panel."""
        console.print(Panel(
            text,
            title=title if title else None,
            border_style=border_style,
            padding=(1, 2)
        ))

    @classmethod
    def rule(cls, title: str = "") -> None:
        """Print a horizontal rule with optional title."""
        from rich.rule import Rule
        console.print(Rule(title, style="dim"))

    @classmethod
    def print_onboarding_splash(cls) -> None:
        """Print the full onboarding splash screen with ASCII art and welcome panel."""
        cls.blank()
        for line in cls.ASCII:
            console.print(f"[cyan]{line}[/cyan]")
        cls.blank()
        console.print(Panel(
            "[bold]welcome to darwin[/bold]\n\n"
            "let's get you set up. this takes about 2 minutes.\n"
            "you can change any of these settings later with:\n"
            "[cyan]python main.py config[/cyan]",
            border_style="dim",
            padding=(1, 2),
        ))
        cls.blank()







