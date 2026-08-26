"""
Darwin Pretty Traceback – rich, contextual error reporting.

Drop this anywhere in your project (e.g. src/core/Utils/pretty_traceback.py)
and call it from your entry point or wrap your main() with it.

Usage:
    from core.Utils.pretty_traceback import install_pretty_traceback
    install_pretty_traceback()   # one-liner, replaces sys.excepthook

Or manually:
    from core.Utils.pretty_traceback import format_exception
    try:
        ...
    except Exception as e:
        print(format_exception(e))
"""

from __future__ import annotations

import sys
import traceback
import linecache
import os
from pathlib import Path
from typing import Optional, List, Dict, Any


# ── ANSI colors ────────────────────────────────────────────────
class _Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"
    BG_RED  = "\033[41m"
    BG_GREY = "\033[48;5;238m"


# Disable colors if NO_COLOR or not a TTY
_USE_COLOR = (
    sys.stderr.isatty()
    and os.environ.get("NO_COLOR") is None
    and os.environ.get("TERM") != "dumb"
)

def _c(code: str, text: str) -> str:
    return f"{_Colors.__dict__[code]}{text}{_Colors.RESET}" if _USE_COLOR else text


# ── Core formatter ─────────────────────────────────────────────

def _get_source_context(filename: str, lineno: int, context: int = 3) -> List[str]:
    """Read ±context lines around lineno (1-based)."""
    lines = []
    for i in range(lineno - context, lineno + context + 1):
        if i < 1:
            continue
        line = linecache.getline(filename, i)
        if not line:
            break
        lines.append((i, line.rstrip("\n")))
    return lines


def _format_frame(
    filename: str,
    lineno: int,
    name: str,
    line: Optional[str],
    colno: Optional[int] = None,
    end_colno: Optional[int] = None,
    is_last: bool = False,
) -> str:
    """Format a single stack frame with source context."""
    parts = []

    # File header
    rel = os.path.relpath(filename) if Path(filename).is_absolute() else filename
    parts.append(f"{_c('GREY', '  File')} {_c('CYAN', rel)}{_c('GREY', ',')} {_c('YELLOW', f'line {lineno}')}{_c('GREY', ',')} in {_c('GREEN', name)}")

    # Source context
    ctx = _get_source_context(filename, lineno, context=2)
    if ctx:
        max_num_width = len(str(ctx[-1][0]))
        for num, src in ctx:
            num_str = f"{num:>{max_num_width}}"
            if num == lineno:
                # Highlight the error line
                parts.append(f"{_c('RED', '→')} {_c('RED', num_str)} {_c('WHITE', src)}")
                # Underline the error column if available
                if colno is not None and line:
                    caret_start = colno
                    caret_end = end_colno if end_colno is not None else colno + 1
                    # Build underline
                    prefix = " " * (len(num_str) + 3 + caret_start)
                    underline_len = max(1, caret_end - caret_start)
                    underline = "^" * underline_len
                    parts.append(f"  {' ' * len(num_str)} {_c('RED', prefix + underline)}")
            else:
                parts.append(f"  {_c('GREY', num_str)} {_c('DIM', src)}")
    elif line:
        parts.append(f"    {_c('WHITE', line.strip())}")

    parts.append("")
    return "\n".join(parts)


def _format_syntax_error(exc: SyntaxError) -> str:
    """Format a SyntaxError with full context — the caret, the line, the file."""
    parts = []
    parts.append("")
    parts.append(_c("BG_RED", "  SYNTAX ERROR  "))
    parts.append("")

    # File location
    filename = exc.filename or "<unknown>"
    lineno = exc.lineno or 1
    rel = os.path.relpath(filename) if Path(filename).is_absolute() else filename
    parts.append(f"{_c('GREY', '  File')} {_c('CYAN', rel)}{_c('GREY', ',')} {_c('YELLOW', f'line {lineno}')}")

    # Source line with caret
    if exc.text:
        text = exc.text.rstrip("\n")
        parts.append(f"    {_c('WHITE', text)}")
        if exc.offset is not None:
            caret_pos = exc.offset - 1  # offset is 1-based
            caret = " " * caret_pos + "^"
            parts.append(f"    {_c('RED', caret)}")

    # Error message
    parts.append("")
    parts.append(f"  {_c('RED', exc.__class__.__name__)}: {_c('BOLD', str(exc))}")
    parts.append("")

    # Suggestion: if it's an f-string issue
    msg = str(exc).lower()
    if "f-string" in msg or "expecting '}'" in msg:
        parts.append(f"  {_c('YELLOW', '💡  Hint:')} Check for nested quotes or curly braces in your f-string.")
        parts.append(f"     {_c('GREY', 'Use double braces {{ }} for literal braces, or escape quotes.')}")
    elif "eol" in msg or "unexpected eof" in msg:
        parts.append(f"  {_c('YELLOW', '💡  Hint:')} Missing closing quote or bracket?")

    return "\n".join(parts)


def format_exception(
    exc: BaseException,
    limit: Optional[int] = None,
    chain: bool = True,
    capture_locals: bool = False,
) -> str:
    """
    Format an exception into a beautiful, contextual traceback string.

    For SyntaxError: shows the exact line, caret position, and file.
    For runtime errors: shows source context (±2 lines) around each frame.
    """
    # Use TracebackException for clean extraction
    tb_exc = traceback.TracebackException.from_exception(
        exc, limit=limit, capture_locals=capture_locals
    )

    parts: List[str] = []

    # ── SyntaxError shortcut ──
    if isinstance(exc, SyntaxError):
        return _format_syntax_error(exc)

    # ── Regular traceback ──
    parts.append("")
    parts.append(_c("BG_RED", "  TRACEBACK  "))
    parts.append("")
    parts.append(f"{_c('GREY', 'Traceback (most recent call last):')}")
    parts.append("")

    for frame in tb_exc.stack:
        parts.append(_format_frame(
            filename=frame.filename,
            lineno=frame.lineno,
            name=frame.name,
            line=frame.line,
            colno=frame.colno,
            end_colno=frame.end_colno,
        ))

    # Exception type + message
    exc_type = tb_exc.exc_type_str if hasattr(tb_exc, "exc_type_str") else exc.__class__.__name__
    exc_msg = str(exc) if str(exc) else "<no message>"
    parts.append(f"  {_c('RED', exc_type)}: {_c('BOLD', exc_msg)}")
    parts.append("")

    return "\n".join(parts)


def print_exception(
    exc: BaseException,
    limit: Optional[int] = None,
    chain: bool = True,
    file: Any = sys.stderr,
) -> None:
    """Print a pretty-formatted exception to *file* (default stderr)."""
    file.write(format_exception(exc, limit=limit, chain=chain))
    file.write("\n")
    file.flush()


# ── Global hook ────────────────────────────────────────────────

def _darwin_excepthook(exc_type, exc_value, exc_traceback):
    """Custom sys.excepthook that prints our pretty tracebacks."""
    if exc_value is None:
        return
    print_exception(exc_value, file=sys.stderr)


def install_pretty_traceback():
    """
    Replace sys.excepthook with Darwin's pretty formatter.
    Call once at startup (e.g. in main.py before anything else).
    """
    sys.excepthook = _darwin_excepthook


def uninstall_pretty_traceback():
    """Restore the default Python excepthook."""
    sys.excepthook = sys.__excepthook__