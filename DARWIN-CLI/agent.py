"""
agent.py
--------
CLI agent for studyagent project.

Commands:
    start  - scan folder and convert supported files to markdown
    help   - show available commands

Creates .agent folder structure with sources/, viz/, config.toml, index.json
"""

import sys
import json
import toml
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import track

console = Console()

AGENT_DIR = Path.cwd() / ".agent"
SOURCES_DIR = AGENT_DIR / "sources"
VIZ_DIR = AGENT_DIR / "viz"
CONFIG_FILE = AGENT_DIR / "config.toml"
INDEX_FILE = AGENT_DIR / "index.json"

SUPPORTED = {".pdf", ".docx", ".pptx", ".txt", ".md"}


def init_agent_dir():
    AGENT_DIR.mkdir(exist_ok=True)
    SOURCES_DIR.mkdir(exist_ok=True)
    VIZ_DIR.mkdir(exist_ok=True)
    if not CONFIG_FILE.exists():
        config = {
            "llm": {"provider": "none", "model": "", "api_key": ""},
            "parser": {"ocr_enabled": False},
        }
        CONFIG_FILE.write_text(toml.dumps(config))
    if not INDEX_FILE.exists():
        INDEX_FILE.write_text(json.dumps({"files": {}}, indent=2))


def load_index():
    return json.loads(INDEX_FILE.read_text())


def save_index(index):
    INDEX_FILE.write_text(json.dumps(index, indent=2))


def scan_folder():
    files = []
    for f in Path.cwd().iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED:
            files.append(f)
    return files


def convert_file(path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pymupdf4llm

        return pymupdf4llm.to_markdown(str(path))
    elif suffix in {".docx", ".pptx", ".txt", ".md"}:
        from markitdown import MarkItDown

        result = MarkItDown().convert(str(path))
        return result.text_content
    return ""


def is_good_enough(text):
    if not text or len(text.strip()) < 100:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) > 0.6


def cmd_start():
    console.print("\n[bold]studyagent[/bold] — starting\n")
    init_agent_dir()
    index = load_index()
    files = scan_folder()

    if not files:
        console.print("[yellow]No supported files found.[/yellow]")
        return

    console.print(f"Found [bold]{len(files)}[/bold] file(s)\n")

    new_files = skip_files = error_files = 0

    for f in track(files, description="Converting files..."):
        file_key = f.name
        if file_key in index["files"]:
            if index["files"][file_key]["modified"] == str(f.stat().st_mtime):
                skip_files += 1
                continue
        try:
            text = convert_file(f)
            if not is_good_enough(text):
                console.print(f"\n[yellow]⚠ {f.name} — empty or garbled[/yellow]")
                error_files += 1
                continue
            out_path = SOURCES_DIR / (f.stem + ".md")
            out_path.write_text(text, encoding="utf-8")
            index["files"][file_key] = {
                "source": str(f),
                "converted": str(out_path),
                "modified": str(f.stat().st_mtime),
                "processed": datetime.now().isoformat(),
                "chars": len(text),
                "words": len(text.split()),
            }
            new_files += 1
        except Exception as e:
            console.print(f"\n[red]✗ {f.name} — {e}[/red]")
            error_files += 1

    save_index(index)
    console.print(f"\n[bold green]✓ Done[/bold green]")
    console.print(f"  Converted : [green]{new_files}[/green]")
    console.print(f"  Skipped   : [dim]{skip_files}[/dim]")
    console.print(
        f"  Errors    : [red]{error_files}[/red]" if error_files else "  Errors    : 0"
    )
    console.print(f"\n  Saved to [dim].agent/sources/[/dim]\n")


def cmd_help():
    console.print("\n[bold]studyagent[/bold]")
    console.print("  [green]start[/green]   — scan folder and convert files")
    console.print("  [green]query[/green]   — ask a question about your documents")
    console.print("  [green]viz[/green]     — open knowledge graph in browser")
    console.print("  [green]status[/green]  — show what has been processed\n")


COMMANDS = {
    "start": cmd_start,
    "help": cmd_help,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd in COMMANDS:
        COMMANDS[cmd]()
    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        cmd_help()



"""
Darwin RAG System
Core built by BrightDotDev
Licensed under MIT with Attribution
"""

__version__ = "0.1.0"
__author__ = "BrightDotDev"
__license__ = "MIT with Attribution"