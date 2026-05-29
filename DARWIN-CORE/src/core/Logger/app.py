# main.py — Entry point with structured logging demos
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import List

from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity


# ---------------------------------------------------------------------------
# Sample dataclasses for structured logging demos
# ---------------------------------------------------------------------------
@dataclass
class Address:
    street: str
    city: str
    zip_code: str


@dataclass
class User:
    name: str
    age: int
    email: str
    addresses: List[Address]
    metadata: dict = field(default_factory=dict)


async def demo_workload():
    """Example workload – plain logs, progress, spinners, and structured data."""

    # -- classic logs ----------------------------------------------------------
    darwin.log("App started", LogType.INFO)
    darwin.log("All systems nominal", LogType.SUCCESS)
    darwin.log("Disk usage at 92%", LogType.WARNING)

    # Progress tracking
    darwin.progress_start("load", 80, label="Loading configuration")
    for i in range(80):
        await asyncio.sleep(0.01)
        darwin.progress_update("load", i + 1, status=f"config part {i+1}")
    darwin.progress_end("load")

    with darwin.progress("chunks", 50, "Chunking files") as p:
        for i in range(50):
            await asyncio.sleep(0.02)
            p.update(i + 1, f"chunk_{i+1}.txt")

    # -- structured logs ------------------------------------------------------
    addr1 = Address(street="123 Main St", city="Springfield", zip_code="12345")
    addr2 = Address(street="456 Oak Ave", city="Shelbyville", zip_code="67890")
    user1 = User(
        name="Alice",
        age=30,
        email="alice@example.com",
        addresses=[addr1],
        metadata={"role": "admin", "active": True},
    )
    user2 = User(
        name="Bob",
        age=25,
        email="bob@example.com",
        addresses=[addr1, addr2],
        metadata={"role": "user", "active": False},
    )

    # 1) Single dataclass
    darwin.logjson(user1, LogType.INFO)

    # 2) Dictionary with dataclass list
    darwin.logjson(
        {"users": [user1, user2], "total": 2, "page": 1},
        LogType.SUCCESS,
    )

    # 3) Direct list using loglist
    darwin.loglist([user1, user2], LogType.INFO)

    # 4) Mixed nested structures
    nested = [
        {"meta": "group A", "members": [user1]},
        {"meta": "group B", "members": [user2]},
    ]
    darwin.logjson(nested, LogType.WARNING)

    # 5) Pure nested dict
    darwin.logjson(
        {"server": {"host": "localhost", "ports": [8080, 8443], "tls": True}},
        LogType.INFO,
    )

    darwin.log("Workload complete — structured data examples finished", LogType.SUCCESS)


async def main():
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

    if darwin._ws_server is not None:
        darwin.log("Server running — press Ctrl+C to stop", LogType.INFO)
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        darwin.log("Shutting down", LogType.INFO)