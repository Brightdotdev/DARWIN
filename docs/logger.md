# Logger — Structured Logging System

**What it is:** A session-based structured logging framework with SQLite persistence, SSE streaming, Rich terminal output, and a CSS-like style system for log messages. All Darwin components log through the global `darwin` singleton.

**Why it exists:** Standard `print`/`logging` doesn't support progress bars, spinners, SSE streaming for real-time dashboards, structured log queries, or styled terminal output. This logger provides all of that with zero external dependencies beyond Rich.

## Files

| File | Purpose |
|---|---|
| `DarwinLogger.py` | Core framework — `DarwinLogger` class, `LogType` enum, style system, `darwin` singleton |
| `lib.py` | `ASCII_BANNER` constant (DARWIN ASCII art logo) |
| `SQLiteHistoryStore.py` | Persistent log storage with retention, export, purge |
| `TerminalUi.py` | Rich-based terminal UI for log display with style support |
| `DarwinMP.py` | Multi-processing log support |

## Usage

```python
from src.core.Logger.DarwinLogger import darwin, LogType

# Auto-styled by LogType (see Default Styles below)
darwin.info("Processing file...")
darwin.success("Done!")
darwin.warning("Low disk space")
darwin.error("Failed to parse PDF")

# Explicit style overrides
darwin.info("Highlighted message", style="bold cyan")
darwin.warning("Urgent warning", style="red bold italic")
darwin.log("Custom styled", LogType.INFO, style="green underline")
darwin.success("Bright success", style="bright_green")

# Style stacking (CSS-like cascade)
darwin.info("Bold overrides dim", style="bold")         # dim → bold
darwin.info("Red overrides dim", style="red")            # dim → red
darwin.info("Full override", style="bold red")           # dim → bold red
darwin.info("Clear style", style="none")                 # no style at all

# Sessions
with darwin.session("pipeline-1") as s:
    darwin.info("Inside session")

# Progress bar
with darwin.progress("Indexing", total=100) as p:
    for i in range(100):
        p.update(i)

# Spinner
with darwin.spinner("Embedding..."):
    time.sleep(2)

# Structured loglist
darwin.loglist([{"chunk_id": 1, "text": "..."}], LogType.INFO)
```

## Style System

Log messages have CSS-like cascading styles. Each `LogType` has a default style, which can be overridden by passing an explicit `style` string.

### Default Styles

| LogType | Default Style | Visual |
|---|---|---|
| `DEBUG` | `grey italic` | Grey, italic |
| `INFO` | `dim` | Dimmed |
| `SUCCESS` | `green` | Green |
| `WARNING` | `orange1 bold` | Orange, bold |
| `ERROR` | `red bold` | Red, bold |

### Style Tokens

| Category | Tokens | Conflict Rule |
|---|---|---|
| **Colors** | `red`, `green`, `yellow`, `orange1`, `blue`, `magenta`, `cyan`, `white`, `grey` | Last wins |
| **Bright colors** | `bright_red`, `bright_green`, `bright_yellow`, `bright_blue`, `bright_magenta`, `bright_cyan`, `bright_white` | Last wins |
| **Modifiers** | `dim`, `bold` | Last wins |
| **Additive** | `italic`, `underline`, `strike`, `blink`, `reverse` | Stack (additive) |
| **Reset** | `none` | Clears all style |

### Semantic Aliases

| Alias | Expands to | Use for |
|---|---|---|
| `success` | `green` | Positive results |
| `danger` | `red bold` | Errors, failures |
| `warning` | `orange1 bold` | Warnings, cautions |
| `muted` | `dim white` | Secondary metadata |
| `link` | `underline cyan` | Clickable-looking text |
| `bright` | `bright_white` | High-visibility text |
| `orange` | `orange1` | Alias for orange |
| `gray` | `grey` | Alias for grey |

### Cascade Rules

Explicit style overrides the default style per-category (CSS-like):

```
Default: "dim"        Explicit: "bold red"       → Result: "red bold"
Default: "grey italic" Explicit: "green"          → Result: "green italic"
Default: "red bold"   Explicit: "italic underline" → Result: "red bold italic underline"
Default: "dim"        Explicit: "none"             → Result: "" (no style)
Default: "dim"        Explicit: "success"          → Result: "green" (alias expanded)
```

Colors replace colors, modifiers replace modifiers, additive tokens stack.

## Key Types

| Class/Enum | Description |
|---|---|
| `LogType` | `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR` |
| `OutputMode` | `TERMINAL`, `SSE`, `BOTH`, `NONE` |
| `LogFilter` | `ALL`, `QUIET`, `SILENT` |
| `DarwinLogger` | Main logger class with sessions, progress, spinners |

## API Reference

### Logging Methods

```python
def log(self, message: str, log_type: LogType = LogType.INFO, style: str = "") -> None
def info(self, message: str, style: str = "") -> None
def success(self, message: str, style: str = "") -> None
def warning(self, message: str, style: str = "") -> None
def error(self, message: str, style: str = "") -> None
def debug(self, message: str, style: str = "") -> None
def logjson(self, payload: Any, log_type: LogType = LogType.INFO, style: str = "") -> None
def loglist(self, payload: Any, log_type: LogType = LogType.INFO, style: str = "") -> None
```

### Style Resolution (programmatic)

```python
from src.core.Logger.DarwinLogger import resolve_event_style, _merge_style

# Given a LogEvent, get its resolved style string
style = resolve_event_style(event)  # "red bold"

# Merge arbitrary default + explicit styles
style = _merge_style("dim", "bold green")  # "bold green"
```

## SSE Server

The log SSE server has moved to `src/mcp/UnifiedServer.py`. See `docs/mcp.md` for details.

```bash
# Start the unified server (log SSE + MCP tools + history + health)
python -m src.mcp --sse --port 8765
```
