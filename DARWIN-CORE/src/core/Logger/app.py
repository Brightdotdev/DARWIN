"""
Darwin Logger — Demo of the restored API.

Everything works as before, plus:
  • logjson/loglist with __darwin_type__ markers
  • Progress/Spinner events
  • Queryable sessions: s.errors(), s.warnings(), s.all()
  • Default history persistence to ~/.darwin/logs
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.core.Logger.DarwinLogger import darwin, LogType, OutputMode


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


async def demo_session_api():
    """Your requested API — works exactly as you described."""
    with darwin.session("ingest_pipeline") as s:
        darwin.info("Starting ingestion")
        darwin.logjson({"files": ["a.txt", "b.txt"], "total": 2})

        darwin.progress_start("chunk", 50, "Chunking files")
        for i in range(50):
            await asyncio.sleep(0.01)
            darwin.progress_update("chunk", i + 1, f"chunk_{i+1}.txt")
        darwin.progress_end("chunk")

        darwin.warning("Large file detected", payload={"file": "big.bin", "size_mb": 500})
        darwin.error("Parse failed", payload={"file": "bad.json", "line": 42})
        darwin.success("Ingestion complete", payload={"processed": 2, "failed": 1})

    # --- Query the session after it closes ---
    print("\n" + "=" * 60)
    print("SESSION QUERY DEMO")
    print("=" * 60)

    print(f"\n📋 All logs from session:")
    for entry in s.all():
        print(f"   [{entry.log_type.value.upper()}] {entry.message[:50]}...")

    print(f"\n⚠️  Warnings only:")
    for entry in s.warnings():
        print(f"   {entry.message}")

    print(f"\n❌ Errors only:")
    for entry in s.errors():
        print(f"   {entry.message} — payload: {entry.payload}")

    return s.all()


async def demo_progress_spinner():
    """Progress and spinner still work in terminal and SSE."""
    with darwin.session("batch_job"):
        # Context manager style progress
        with darwin.progress("batch", 100, "Processing batch") as p:
            for i in range(100):
                await asyncio.sleep(0.005)
                p.update(i + 1, f"item_{i+1}")

        # Spinner
        darwin.spinner_start("upload", "Uploading to S3...")
        await asyncio.sleep(0.5)
        darwin.spinner_update("upload", "Uploading to S3... 50%")
        await asyncio.sleep(0.5)
        darwin.spinner_stop("upload", success=True)


async def demo_structured():
    """logjson and loglist with __darwin_type__ markers."""
    addr = Address(street="123 Main", city="Springfield", zip_code="12345")
    user = User(name="Alice", age=30, email="alice@example.com",
                addresses=[addr], metadata={"role": "admin"})

    with darwin.session("user_ops"):
        # Single dataclass — __darwin_type__: "User"
        darwin.logjson(user, LogType.INFO)

        # List of dataclasses — __darwin_type__: "list", items have __darwin_type__: "User"
        darwin.loglist([user, user], LogType.SUCCESS)

        # Plain dict (no __darwin_type__ marker)
        darwin.logjson({"server": {"host": "localhost", "ports": [8080, 8443]}})


async def demo_no_session():
    """History works without any session declared."""
    darwin.info("System boot — no session")
    darwin.error("Something broke — no session")

    print("\n" + "=" * 60)
    print("NO-SESSION QUERY DEMO")
    print("=" * 60)

    # All logs ever (includes no-session + named sessions)
    all_logs = darwin.get_logs()
    print(f"\n📋 All logs (any session): {len(all_logs)}")

    # Only no-session logs
    no_session = darwin.get_logs(session="")
    print(f"\n📋 No-session logs: {len(no_session)}")
    for e in no_session:
        print(f"   [{e.log_type.value}] {e.message}")

    # Stats
    print(f"\n📊 {darwin.history_stats()}")


async def main():
    # ═══════════════════════════════════════════════════════════════════
    # YOUR OLD CONFIG:
    #
    #   darwin.configure(
    
    #       
    #       show_banner=True,
    #   )
    #
    # YOUR NEW CONFIG:
    # ═══════════════════════════════════════════════════════════════════

    # Option 1: Terminal mode (default history to ~/.darwin/logs)
    darwin.configure(
        mode=OutputMode.TERMINAL,
        show_banner=True,
    )

    # Option 2: SSE mode (replaces WebSocket)
    # darwin.configure(
    #     mode=OutputMode.SSE,
    #     sse_port=8765,
    #     sse_host="127.0.0.1",
    #     show_banner=True,
    # )

    # Option 3: Custom persist dir
    # darwin.configure(
    #     mode=OutputMode.TERMINAL,
    #     persist_dir=Path("./my_logs"),
    #     max_memory=10_000,
    #     show_banner=True,
    # )

    await demo_session_api()
    await demo_progress_spinner()
    await demo_structured()
    await demo_no_session()

    darwin.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        darwin.shutdown()