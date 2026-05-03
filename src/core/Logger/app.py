# =============================================================================
# main.py — Minimal entry point.  DarwinLogger handles everything.
# =============================================================================
from __future__ import annotations

import asyncio
import sys

from src.core.Logger.DarwinLogger import darwin, LogType, OutputMode
from src.core.Logger.actions import ACTION_REGISTRY


def register_all_actions():
    """Register all actions from actions.py."""
    for name, func in ACTION_REGISTRY.items():
        darwin.register_action(name, func)


async def demo_workload():
    """Example workload — replace with your actual app logic."""
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

    # Prompt — works in terminal AND browser simultaneously
    result = await darwin.prompt(
        id="choose_model",
        message="Select default model",
        options=["gpt-4", "claude-3-opus", "gemini-ultra"],
        action="select_model",
        params={"context": "initial_setup"},
    )
    darwin.log(f"Selected: {result}", LogType.SUCCESS)


async def main():
    # Register actions
    register_all_actions()

    # Configure — auto-detects terminal vs headless
    # Override with CLI args:
    if "--cli" in sys.argv:
        darwin.configure(mode=OutputMode.TERMINAL)
    elif "--headless" in sys.argv:
        darwin.configure(mode=OutputMode.WEBSOCKET)
    elif "--silent" in sys.argv:
        darwin.configure(mode=OutputMode.NONE)
    else:
        darwin.configure()  # auto-detect

    # Run your app
    await demo_workload()

    # Keep alive if WebSocket is running
    if darwin._mode in (OutputMode.WEBSOCKET, OutputMode.BOTH):
        darwin.log("Server running — press Ctrl+C to stop", LogType.INFO)
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        darwin.log("Shutting down", LogType.INFO)