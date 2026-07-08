from typing import Optional
from pathlib import Path
from src.core.Logger.DarwinLogger import darwin, LogFilter, OutputMode


def configure(
    mode: OutputMode = OutputMode.TERMINAL,
    persist: bool = True,
    persist_dir: Optional[Path] = None,
    max_memory: int = 10_000,
    log_filter: LogFilter = LogFilter.ALL,
    show_banner: bool = True,
    sse_port: int = 8765,
    sse_host: str = "127.0.0.1",
    persist_config: bool = True,
    override: bool = True,
) -> None:
    """Set up global logging for orchestrator tools."""
    darwin.configure(
        mode=mode,
        persist=persist,
        persist_dir=persist_dir,
        max_memory=max_memory,
        log_filter=log_filter,
        show_banner=show_banner,
        sse_port=sse_port,
        sse_host=sse_host,
        persist_config=persist_config,
        override=override,
    )
