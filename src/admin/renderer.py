"""
Terminal renderer for darwin-admin CLI.

Zero-dependency ANSI output. Respects NO_COLOR and non-TTY detection.
"""

import os
import re
import sys
from typing import Any, List, Optional, Tuple


class AdminRenderer:
    """
    Clean terminal output. No Rich, no emoji, no heavy deps.
    Uses ANSI colors only when supported. Respects NO_COLOR.
    """

    def __init__(self, plain: bool = False):
        self.plain = plain or os.environ.get("NO_COLOR") or not sys.stdout.isatty()

    def _style(self, text: str, fg: Optional[str] = None, bold: bool = False) -> str:
        if self.plain:
            return text
        codes = []
        if bold:
            codes.append("1")
        if fg == "green":
            codes.append("32")
        elif fg == "red":
            codes.append("31")
        elif fg == "yellow":
            codes.append("33")
        elif fg == "blue":
            codes.append("34")
        elif fg == "cyan":
            codes.append("36")
        elif fg == "bright":
            codes.append("97")
        elif fg == "dim":
            codes.append("90")
        if not codes:
            return text
        return f"\033[{';'.join(codes)}m{text}\033[0m"

    def _dim(self, text: str) -> str:
        return self._style(text, fg="dim")

    def _green(self, text: str) -> str:
        return self._style(text, fg="green")

    def _red(self, text: str) -> str:
        return self._style(text, fg="red")

    def _yellow(self, text: str) -> str:
        return self._style(text, fg="yellow")

    def _cyan(self, text: str) -> str:
        return self._style(text, fg="cyan")

    def _blue(self, text: str) -> str:
        return self._style(text, fg="blue")

    def _bright(self, text: str) -> str:
        return self._style(text, fg="bright", bold=True)

    def _out(self, text: str = "", err: bool = False) -> None:
        stream = sys.stderr if err else sys.stdout
        print(text, file=stream)

    def _divider(self, width: int = 60) -> str:
        return "─" * width

    def header(self, title: str) -> None:
        self._out(self._divider())
        self._out(f"  {self._bright(title)}")
        self._out(self._divider())

    def show_banner(self) -> None:
        """Print the Darwin ASCII banner."""
        from src.core.Logger.lib import ASCII_BANNER
        for line in ASCII_BANNER:
            try:
                self._out(line)
            except UnicodeEncodeError:
                stream = sys.stdout
                stream.buffer.write((line + "\n").encode("utf-8", errors="replace"))

    def section(self, title: str) -> None:
        self._out(f"\n  {self._bright(title)}")

    def ok(self, msg: str) -> None:
        self._out(f"  {self._green('OK')}  {msg}")

    def warn(self, msg: str) -> None:
        self._out(f"  {self._yellow('WARN')}  {msg}", err=True)

    def err(self, msg: str) -> None:
        self._out(f"  {self._red('ERR')}  {msg}", err=True)

    def info(self, msg: str) -> None:
        self._out(f"  {self._dim('>')}  {msg}")

    def _pad(self, text: str, width: int) -> str:
        """Pad text to visual width, accounting for ANSI escape codes."""
        visual_len = len(self._strip_ansi(str(text)))
        padding = max(0, width - visual_len)
        return str(text) + " " * padding

    def table(self, title: str, columns: List[str], rows: List[Tuple]) -> None:
        """Print a clean ASCII table."""
        if not rows:
            self._out(f"  {self._dim('(no data)')}")
            return

        # Calculate column widths
        widths = [len(c) for c in columns]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(self._strip_ansi(str(cell))))

        # Title
        self._out(f"\n  {self._bright(title)}")

        # Header row
        header = "  | ".join(
            self._pad(col, widths[i]) for i, col in enumerate(columns)
        )
        self._out(f"  {header}")
        self._out(f"  {'─' * (len(header) + 2)}")

        # Data rows
        for row in rows:
            line = "  | ".join(
                self._pad(str(cell), widths[i]) for i, cell in enumerate(row)
            )
            self._out(f"  {line}")

        self._out()

    def _strip_ansi(self, text: str) -> str:
        return re.sub(r"\033\[\d+(?:;\d+)*m", "", text)

    def panel(self, title: str, content: str) -> None:
        """Simple bordered panel, no rounded corners."""
        lines = content.split("\n")
        max_width = max(len(self._strip_ansi(l)) for l in lines) if lines else 0
        max_width = max(max_width, len(self._strip_ansi(title)))

        self._out()
        border_width = max_width - len(self._strip_ansi(title)) - 2
        self._out(f"  ┌─ {self._bright(title)} {'─' * border_width}┐")
        for line in lines:
            pad = max_width - len(self._strip_ansi(line))
            self._out(f"  │ {line}{' ' * pad} │")
        self._out(f"  └{'─' * (max_width + 2)}┘")

    def tree(self, label: str, items: List[Tuple[str, Any]]) -> None:
        """Simple tree output."""
        self._out(f"\n  {self._bright(label)}")
        for key, value in items:
            if isinstance(value, list):
                self._out(f"  ├── {key}")
                for v in value:
                    self._out(f"  │   └── {v}")
            else:
                self._out(f"  ├── {key}: {value}")

    def next_steps(self, steps: List[str]) -> None:
        self._out(f"\n  {self._bright('Next steps')}")
        for step in steps:
            self._out(f"    {step}")

    def interactive_select(self, prompt: str, options: List[Tuple[str, str]], default: int = None) -> str:
        """Numbered interactive selector. Returns the selected key."""
        self.section(prompt)
        for i, (key, desc) in enumerate(options, 1):
            self._out(f"  [{i}] {key}  {self._dim(desc)}")
        while True:
            try:
                raw = input(
                    f"  Enter [1-{len(options)}]"
                    + (f" (default: {default})" if default else "")
                    + ": "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                self._out()
                raise KeyboardInterrupt
            if not raw and default is not None:
                return options[default - 1][0]
            try:
                idx = int(raw)
                if 1 <= idx <= len(options):
                    return options[idx - 1][0]
            except ValueError:
                pass
            self.err("Invalid selection.")


# Global renderer instance
_r = AdminRenderer()
