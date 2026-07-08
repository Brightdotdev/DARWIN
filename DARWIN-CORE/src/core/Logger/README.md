# Logger — Structured Logging System

**What it is:** A session-based structured logging framework with SQLite persistence, SSE streaming, and Rich terminal output. All Darwin components log through the global `darwin` singleton.

**Why it exists:** Standard `print`/`logging` doesn't support progress bars, spinners, SSE streaming for real-time dashboards, or structured log queries. This logger provides all of that with zero external dependencies beyond Rich.

## Files

| File | Purpose |
|---|---|
| `DarwinLogger.py` | Core framework — `DarwinLogger` class, `LogType` enum, `darwin` singleton (710 lines) |
| `lib.py` | `ASCII_BANNER` constant (DARWIN ASCII art logo) |
| `McpServer.py` | MCP log streaming tools (WebSocket/SSE) |
| `UnifiedSSEServer.py` | Unified SSE server for real-time log streaming (Starlette + uvicorn) |
| `SQLiteHistoryStore.py` | Persistent log storage with retention, export, purge |
| `TerminalUi.py` | Rich-based terminal UI for log display |
| `DarwinMP.py` | Multi-processing log support |
| `app.py` | Demo / usage examples |

## Usage

```python
from src.core.Logger.DarwinLogger import darwin, LogType

darwin.info("Processing file...")
darwin.success("Done!")
darwin.warning("Low disk space")
darwin.error("Failed to parse PDF")

# Sessions
darwin.start_session("pipeline-1")
darwin.info("Inside session")
darwin.end_session()

# Progress bar
with darwin.progress("Indexing", total=100) as p:
    for i in range(100):
        p.update(i)
        # ...

# Spinner
with darwin.spinner("Embedding..."):
    time.sleep(2)

# Structured loglist
darwin.loglist([{"chunk_id": 1, "text": "..."}], LogType.INFO)
```

## Key Types

| Class/Enum | Description |
|---|---|
| `LogType` | `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR` |
| `OutputMode` | `TERMINAL`, `SSE`, `BOTH`, `NONE` |
| `LogFilter` | `ALL`, `QUIET`, `SILENT` |
| `DarwinLogger` | Main logger class with sessions, progress, spinners |

## SSE Server

```bash
python -c "from src.core.Logger.UnifiedSSEServer import run_sse_server; run_sse_server(port=8081)"
```
