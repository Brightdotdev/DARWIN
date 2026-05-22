# main.py — Minimal entry point. DarwinLogger handles everything.
from __future__ import annotations

import asyncio
import sys

from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity
from src.core.Logger.actions import ACTION_REGISTRY


def register_all_actions():
    """Register all actions from actions.py."""
    for name, func in ACTION_REGISTRY.items():
        darwin.register_action(name, func)


async def demo_workload():
    """Example workload — no prompts, just logs, progress, spinners."""
    darwin.log("App started", LogType.INFO)
    darwin.log("All systems nominal", LogType.SUCCESS)
    darwin.log("Disk usage at 92%", LogType.WARNING)

    # Progress tracking
    darwin.progress_start("load", 80, label="Loading configuration")
    for i in range(80):
        await asyncio.sleep(0.01)
        darwin.progress_update("load", i + 1, status=f"config part {i+1}")
    darwin.progress_end("load")

    # Or use context manager
    with darwin.progress("chunks", 50, "Chunking files") as p:
        for i in range(50):
            await asyncio.sleep(0.02)
            p.update(i + 1, f"chunk_{i+1}.txt")

    darwin.log("Workload complete", LogType.SUCCESS)


async def main():
    # Register actions (optional, no prompts needed)
    register_all_actions()

    # Configure — auto-detect terminal, enable history, default verbosity normal
    if "--quiet" in sys.argv:
        verbosity = Verbosity.QUIET
    else:
        verbosity = Verbosity.NORMAL

    darwin.configure(
        enable_websocket="--no-ws" not in sys.argv,
        ws_port=8765,
        enable_history=True,
        verbosity_default=verbosity,
        show_banner=True,
    )

    await demo_workload()

    # Keep alive if WebSocket is enabled
    if darwin._ws_server is not None:
        darwin.log("Server running — press Ctrl+C to stop", LogType.INFO)
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        darwin.log("Shutting down", LogType.INFO)