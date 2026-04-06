# src/lib/Terminal.py
from rich.console import Console
from rich.panel import Panel

# color codes
CYAN  = "\033[96m"
WHITE = "\033[97m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED   = "\033[91m"
DIM   = "\033[2m"
BOLD  = "\033[1m"
RESET = "\033[0m"

console = Console()


class TerminalUI:
    # Make color constants available as class attributes
    CYAN = CYAN
    WHITE = WHITE
    YELLOW = YELLOW
    GREEN = GREEN
    RED = RED
    DIM = DIM
    BOLD = BOLD
    RESET = RESET
    
    ASCII = [
        "  ██████╗  █████╗ ██████╗ ██╗    ██╗██╗███╗   ██╗",
        "  ██╔══██╗██╔══██╗██╔══██╗██║    ██║██║████╗  ██║",
        "  ██║  ██║███████║██████╔╝██║ █╗ ██║██║██╔██╗ ██║",
        "  ██║  ██║██╔══██║██╔══██╗██║███╗██║██║██║╚██╗██║",
        "  ██████╔╝██║  ██║██║  ██║╚███╔███╔╝██║██║ ╚████║",
        "  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝",
    ]

    @classmethod
    def c(cls, color: str, text: str) -> str:
        """Apply color to text"""
        return f"{color}{text}{cls.RESET}"

    @classmethod
    def print_ascii_splash(cls):
        console.print()
        for line in cls.ASCII:
            console.print(f"[cyan]{line}[/cyan]")
        console.print()
        console.print(Panel(
            "[bold]welcome to darwin[/bold]\n\n"
            "let's get you set up. this takes about 2 minutes.\n"
            "you can change any of these settings later with:\n"
            "[cyan]python main.py config[/cyan]",
            border_style="dim",
            padding=(1, 2),
        ))
        console.print()

    @classmethod
    def print_project_status(cls, store, query: str):
        status = store.status()
        config = status["config"]
        files  = status["files"]

        console.print()
        console.print(f"  [dim]project[/dim] → [white]{status['project_dir']}[/white]")
        console.print(f"  [dim]query[/dim]   → [yellow]{query}[/yellow]")
        console.print(f"  [dim]llm[/dim]     → [white]{config.get('llm_model', 'not set')}[/white]\n")

        if files:
            for f in files:
                chunks   = f["chunk_count"]
                fresh    = "[green]fresh[/green]" if f["is_fresh"] else "[yellow]stale[/yellow]"
                has_all  = all([f["has_chunks"], f["has_embeddings"], f["has_bm25"]])
                indexed  = "[green]indexed[/green]" if has_all else "[yellow]not indexed[/yellow]"
                filename = f['filename'][:30]
                console.print(f"  [dim]{filename}[/dim] → {fresh} · {indexed} · {chunks} chunks")
        else:
            console.print("  [dim]no files indexed yet[/dim]")

        console.print()
        console.print("  " + "─" * 50)
        console.print()
        console.print("  [dim]pro tip:[/dim] add files to your project by placing them in the project directory and running:")

    @classmethod
    def step(cls, label: str) -> None:
        """Print a step indicator"""
        print(f"  {cls.c(cls.CYAN, '→')} {label}...", end="", flush=True)

    @classmethod
    def done(cls, extra: str = "") -> None:
        """Mark a step as complete"""
        suffix = f" {cls.c(cls.DIM, extra)}" if extra else ""
        print(f" {cls.c(cls.GREEN, 'done')}{suffix}")

    @classmethod
    def info(cls, label: str, value: str) -> None:
        """Print formatted info line"""
        print(f"  {cls.c(cls.DIM, label):30s} {cls.c(cls.WHITE, value)}")