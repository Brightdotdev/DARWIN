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

    # Parsing progress:
    with TerminalUI.parse_progress("OpenClaw.pdf") as update:
        parse_pdf(..., progress_callback=update)
"""

from contextlib import contextmanager

from rich.console import Console
from rich.panel   import Panel
from rich.live    import Live
from rich.spinner import Spinner
from rich.text    import Text
from rich.columns import Columns

console = Console()


class TerminalUI:

    ASCII = [
        "  ██████╗  █████╗ ██████╗ ██╗    ██╗██╗███╗   ██╗",
        "  ██╔══██╗██╔══██╗██╔══██╗██║    ██║██║████╗  ██║",
        "  ██║  ██║███████║██████╔╝██║ █╗ ██║██║██╔██╗ ██║",
        "  ██║  ██║██╔══██║██╔══██╗██║███╗██║██║╚██╗██║",
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

    # ── Parsing progress ───────────────────────────────────────────────────────

    @classmethod
    @contextmanager
    def parse_progress(cls, filename: str):
        """
        Context manager that shows a live spinner while a file is being parsed.
        Yields an `update(msg)` callable — pass it as progress_callback.

        Usage:
            with TerminalUI.parse_progress("Bible.pdf") as update:
                doc = parse_pdf(..., progress_callback=update)

        The spinner shows the filename and updates the status line in-place.
        On exit it collapses to a single ✓ / ✗ line.
        """
        # Mutable state shared between the yielded callback and the Live renderer
        state = {"msg": "starting...", "done": False, "error": False}

        def _render():
            spinner = Spinner("dots", style="cyan")
            name_text  = Text(f"  {filename}", style="bold white")
            status_text = Text(f"  {state['msg']}", style="dim")
            # Two-line render: filename on top, status below
            from rich.console import Group
            return Group(
                Text.assemble(("  ", ""), spinner.render(0), (" ", ""), (filename, "bold white")),
                Text(f"     {state['msg']}", style="dim"),
            )

        # Use a simple approach that works reliably across Windows + uv:
        # print spinner frames manually via Live
        from rich.table import Table

        def _make_renderable(frame: str = "⠋"):
            t = Table.grid(padding=(0, 1))
            t.add_column(width=3)
            t.add_column()
            t.add_column()
            t.add_row(
                f"[cyan]{frame}[/cyan]",
                f"[bold white]{filename}[/bold white]",
                f"[dim]{state['msg']}[/dim]",
            )
            return t

        frames = ["⠋", "⠙", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        frame_idx = [0]

        def update(msg: str) -> None:
            """Call this from progress_callback — updates the status line."""
            state["msg"] = msg
            # tick the frame forward
            frame_idx[0] = (frame_idx[0] + 1) % len(frames)

        try:
            with Live(
                _make_renderable(),
                console  = console,
                refresh_per_second = 8,
                transient = True,   # clears the live area on exit
            ) as live:
                # Patch update so it also refreshes the live display
                def update(msg: str) -> None:  # noqa: F811
                    state["msg"] = msg
                    frame_idx[0] = (frame_idx[0] + 1) % len(frames)
                    live.update(_make_renderable(frames[frame_idx[0]]))

                yield update

            # transient=True already cleared the spinner — print final status
            console.print(
                f"  [green]✓[/green] [bold white]{filename}[/bold white]  "
                f"[dim]{state['msg']}[/dim]"
            )

        except Exception:
            state["error"] = True
            console.print(
                f"  [red]✗[/red] [bold white]{filename}[/bold white]  "
                f"[dim]{state['msg']}[/dim]"
            )
            raise

    @classmethod
    @contextmanager
    def multi_parse_progress(cls, filenames: list[str]):
        """
        Progress display for parsing multiple files at once.
        Shows one spinner row per file, all updating live.

        Usage:
            states = {}
            with TerminalUI.multi_parse_progress(file_names) as get_callback:
                for fw in files:
                    cb = get_callback(fw.file.name)
                    # pass cb as progress_callback for that file
        """
        states: dict[str, str] = {name: "queued" for name in filenames}
        frames = ["⠋", "⠙", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        ticks:  dict[str, int] = {name: 0 for name in filenames}
        done:   set[str]       = set()

        def _make_renderable():
            from rich.table import Table
            t = Table.grid(padding=(0, 1))
            t.add_column(width=3)
            t.add_column(min_width=30)
            t.add_column()
            for name in filenames:
                if name in done:
                    icon   = "[green]✓[/green]"
                    status = f"[dim]{states[name]}[/dim]"
                    label  = f"[dim]{name}[/dim]"
                else:
                    frame  = frames[ticks[name] % len(frames)]
                    icon   = f"[cyan]{frame}[/cyan]"
                    status = f"[dim]{states[name]}[/dim]"
                    label  = f"[white]{name}[/white]"
                t.add_row(icon, label, status)
            return t

        def get_callback(filename: str):
            """Returns an update(msg) callback for the given filename."""
            def _update(msg: str) -> None:
                states[filename] = msg
                ticks[filename] += 1
                # Mark done on terminal messages
                if any(kw in msg.lower() for kw in ("done", "complete", "saved")):
                    done.add(filename)
                live.update(_make_renderable())
            return _update

        with Live(
            _make_renderable(),
            console = console,
            refresh_per_second = 8,
            transient = True,
        ) as live:
            yield get_callback

        # Final summary after live area clears
        console.print()
        for name in filenames:
            final_status = states[name]
            icon = "[green]✓[/green]" if name in done else "[dim]·[/dim]"
            console.print(
                f"  {icon} [bold white]{name}[/bold white]  [dim]{final_status}[/dim]"
            )
        console.print()

    # ── All existing methods below — unchanged ─────────────────────────────────

    @classmethod
    def step(cls, label: str) -> None:
        console.print(f"  [cyan]→[/cyan] {label}...", end="")

    @classmethod
    def done(cls, extra: str = "") -> None:
        suffix = f" [dim]{extra}[/dim]" if extra else ""
        console.print(f" [green]done[/green]{suffix}")

    @classmethod
    def dim(cls, text: str) -> None:
        console.print(f"  [dim]{text}[/dim]")

    @classmethod
    def dim_inline(cls, text: str) -> None:
        console.print(f"[dim]{text}[/dim]", end="")

    @classmethod
    def bold(cls, text: str) -> None:
        console.print(f"  [bold]{text}[/bold]")

    @classmethod
    def bold_inline(cls, text: str) -> None:
        console.print(f"[bold]{text}[/bold]", end="")

    @classmethod
    def success_inline(cls, text: str) -> None:
        console.print(f"[green]{text}[/green]", end="")

    @classmethod
    def cyan(cls, text: str) -> None:
        console.print(f"  [cyan]{text}[/cyan]")

    @classmethod
    def yellow(cls, text: str) -> None:
        console.print(f"  [yellow]{text}[/yellow]")

    @classmethod
    def green(cls, text: str) -> None:
        console.print(f"  [green]{text}[/green]")

    @classmethod
    def error(cls, message: str) -> None:
        console.print(f"  [red]✗[/red] {message}")

    @classmethod
    def warn(cls, message: str) -> None:
        console.print(f"  [yellow]⚠[/yellow] {message}")

    @classmethod
    def success(cls, message: str) -> None:
        console.print(f"  [green]✓[/green] {message}")

    @classmethod
    def info(cls, label: str, value: str) -> None:
        console.print(f"  [dim]{label}[/dim]  [white]{value}[/white]")

    @classmethod
    def divider(cls) -> None:
        console.print(f"  {'─' * 60}")

    @classmethod
    def blank(cls) -> None:
        console.print()

    @classmethod
    def header(cls, text: str, level: int = 1) -> None:
        if level == 1:
            console.print(f"\n[bold cyan]{text}[/bold cyan]")
            console.print(f"[dim]{'─' * len(text)}[/dim]")
        elif level == 2:
            console.print(f"\n[bold]{text}[/bold]")
        else:
            console.print(f"\n[dim]{text}[/dim]")

    @classmethod
    def table(cls, headers: list, rows: list) -> None:
        from rich.table import Table
        table = Table(show_header=True, header_style="bold cyan")
        for header in headers:
            table.add_column(header)
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        console.print(table)

    @classmethod
    def progress_bar(cls, current: int, total: int, label: str = "") -> None:
        percentage = (current / total) * 100
        bar_length = 30
        filled     = int(bar_length * current // total)
        bar        = "█" * filled + "░" * (bar_length - filled)
        prefix     = f"{label} " if label else ""
        console.print(f"  {prefix}[cyan]{bar}[/cyan] [dim]{percentage:.1f}%[/dim]")

    @classmethod
    def code_block(cls, text: str, language: str = "") -> None:
        from rich.syntax import Syntax
        syntax = Syntax(text, language or "text", theme="monokai", line_numbers=False)
        console.print(syntax)

    @classmethod
    def panel(cls, text: str, title: str = "", border_style: str = "blue") -> None:
        console.print(Panel(
            text,
            title        = title if title else None,
            border_style = border_style,
            padding      = (1, 2)
        ))

    @classmethod
    def rule(cls, title: str = "") -> None:
        from rich.rule import Rule
        console.print(Rule(title, style="dim"))

    @classmethod
    def print_onboarding_splash(cls) -> None:
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