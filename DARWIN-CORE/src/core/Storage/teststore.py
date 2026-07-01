# test_darwin_store.py
# ──────────────────────────────────────────────────────────────────────────────
# Full battle test for DarwinStore v2 — SQLite-backed, Source/Content split.
# No pytest, no asserts, no frameworks. Pure Python, prints results.
# Run: python test_darwin_store.py

from __future__ import annotations


import sys

from pathlib import Path

import numpy as np

# ── Bootstrap project root if needed ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Imports ──────────────────────────────────────────────────────────────

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin




# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # test_root = Path(__file__).parent / "teststore_v2"
    darwin.configure(
     show_banner=True,
    )
    store = DarwinStore("../../darwin_store_test")
    store.repair_registry()
    store.print_pipeline_report()

    # print(f"\n{'#' * 60}")
    # print(f"#  DARWIN STORE V2 BATTLE TEST")
    # print(f"#  Target: {test_root}")
    # print(f"{'#' * 60}")

    # try:
    #     run_tests(test_root)
    # except KeyboardInterrupt:
    #     print("\n\nInterrupted by user.")