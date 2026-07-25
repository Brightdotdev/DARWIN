
from enum import Enum

ASCII_BANNER = [
        "                                                   ",
        "  ██████╗  █████╗ ██████╗ ██╗    ██╗██╗███╗   ██╗  ",
        "  ██╔══██╗██╔══██╗██╔══██╗██║    ██║██║████╗  ██║  ",
        "  ██║  ██║███████║██████╔╝██║ █╗ ██║██║██╔██╗ ██║  ",
        "  ██║  ██║██╔══██║██╔══██╗██║███╗██║██║██║╚██╗██║  ",
        "  ██████╔╝██║  ██║██║  ██║╚███╔███╔╝██║██║ ╚████║  ",
        "  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝  ",
        "                                                   "
    ]



class OutputMode(Enum):
    TERMINAL = "terminal"
    SSE = "sse"
    BOTH = "both"  # ← NEW: simultaneous terminal + SSE
    NONE = "none"


class LogFilter(Enum):
    """Filter presets for LogEvents."""

    ALL = "all"  # everything (default)
    QUIET = "quiet"  # errors + warnings + success only (no info)
    SILENT = "silent"  # errors only



# ---------------------------------------------------------------------------
# Style system — CSS-like cascading style tokens for log messages
# ---------------------------------------------------------------------------

# Categories for conflict resolution (colors are mutually exclusive, modifiers are mutually exclusive)
STYLE_COLORS = frozenset({
    "red", "green", "yellow", "orange1", "blue", "magenta", "cyan", "white", "grey",
    "bright_red", "bright_green", "bright_yellow", "bright_blue",
    "bright_magenta", "bright_cyan", "bright_white",
})

STYLE_MODIFIERS = frozenset({"dim", "bold"})


# Semantic aliases — shorthand for common style combos
STYLE_ALIASES = {
    "bright": "bright_white",
    "orange": "orange1",
    "gray": "grey",
    "success": "green",
    "danger": "red bold",
    "warning": "orange1 bold",
    "muted": "dim white",
    "link": "underline cyan",
    "none": "",
}

# Default styles per LogType
DEFAULT_LOG_STYLES = {
    "debug": "grey italic",
    "info": "dim",
    "success": "green",
    "warning": "orange1 bold",
    "error": "red bold",
}
