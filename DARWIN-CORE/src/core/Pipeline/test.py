#!/usr/bin/env python3
"""
Pure Pipeline Test — Darwin v5.0
Runs Ingestion → Indexing via PipelineOrchestrator only.
No retrieval, no query logic. Just pipeline execution and validation.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore", module="litellm")
os.environ["LITELLM_SUPPRESS_DEBUG_INFO"] = "True"

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Storage.DarwinStore import DarwinStore

from src.core.Orchestrators.PipelineOrchestrator import (
    PipelineOrchestrator,
    PipelineParameters,
    PipelineStage,
)

from src.core.Pipeline.Ingestion.Chunking.lib import ChunkingArgs
from src.core.Pipeline.Ingestion.Parsing.lib import ParsingArgs, url_to_slug
from src.core.Pipeline.Indexing.lib import IndexingArgs
from src.core.Pipeline.Indexing.BM25.lib import DOMAIN_STOP_THRESHOLD, DEFAULT_IDF_SMOOTHING


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

DIRECTORY = "./../../pdfs/"

EXTRA_URLS = [
    "https://en.wikipedia.org/wiki/Book_of_Enoch",
    "http://en.wikipedia.org/wiki/Amalthea_(mythology)",
]


def get_files(dir_path: str) -> List[str]:
    from src.core.Pipeline.Ingestion.Parsing.Parser import scan_directory
    return [str(f) for f in scan_directory(dir_path)]


def get_filenames(dir_path: str) -> List[str]:
    from src.core.Pipeline.Ingestion.Parsing.Parser import scan_directory
    return [f.name for f in scan_directory(dir_path)]


LOCAL_FILES = get_files(DIRECTORY)
SLUGS = [url_to_slug(url) for url in EXTRA_URLS]

ALL_SOURCES = LOCAL_FILES + EXTRA_URLS
FILE_NAMES = get_filenames(DIRECTORY) + SLUGS


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Test
# ═══════════════════════════════════════════════════════════════════════════════

async def run_pipeline_test() -> int:
    darwin.configure()
    store = DarwinStore(name="test")

    darwin.info("═" * 60)
    darwin.info("Darwin Pipeline Test: Ingestion → Indexing")
    darwin.info("═" * 60)

    pipeline = PipelineOrchestrator(store)

    params = PipelineParameters(
        parsing=ParsingArgs(
            target=None,
            files=ALL_SOURCES,
            images=True,
            save=True,
        ),
        chunking=ChunkingArgs(FILE_NAMES),
        indexing=IndexingArgs(
            files=FILE_NAMES,
            threshold=DOMAIN_STOP_THRESHOLD,
            idf_smoothing=DEFAULT_IDF_SMOOTHING,
            only_kw=True,
            rerun=True,
        ),
    )

    # ── Run pipeline ───────────────────────────────────────────────────────────
    result = pipeline.build_pipeline(
        params,
        stages=[PipelineStage.INGESTION, PipelineStage.INDEXING],
        strict=True,
        log_mode="normal",
    )

    # ── Dump per-stage logs ──────────────────────────────────────────────────
    for stage_key, logs in result.logs.items():
        darwin.info(f"\n--- Logs for '{stage_key}' ({len(logs)} entries) ---")
        for log in logs[:10]:
            darwin.info(f"  [{log.log_type.value}] {log.message[:120]}")

    # ── Result status ──────────────────────────────────────────────────────────
    if not result.success:
        darwin.error(f"Pipeline FAILED: {result.summary}")
        for stage, err in result.stages_failed.items():
            darwin.error(f"  {stage}: {err}")
        return 1

    darwin.success(f"Pipeline SUCCEEDED: {result.summary}")

    # ── Validate outputs ───────────────────────────────────────────────────────
    processed = [f for f in FILE_NAMES if store.has_markdown(f)]
    missing = [f for f in FILE_NAMES if not store.has_markdown(f)]

    darwin.info(f"\nFiles with markdown: {len(processed)}/{len(FILE_NAMES)}")
    if missing:
        darwin.warning(f"Missing: {missing}")



    # ── Error/warning dump ─────────────────────────────────────────────────────
    report = darwin.get_logs(levels={LogType.ERROR, LogType.WARNING,LogType.SUCCESS})
    if report:
        with open("pipeline_test_issues.log", "w", encoding="utf-8") as f:
            for log in report:
                f.write(f"[{log.log_type.value.upper()}] {log.message}\n")
        darwin.warning(f"Issues logged to pipeline_test_issues.log ({len(report)} items)")

    darwin.shutdown()
    return 0


if __name__ == "__main__":
    asyncio.run(run_pipeline_test())