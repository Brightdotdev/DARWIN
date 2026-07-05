#!/usr/bin/env python3
"""
Retrieval Orchestrator Test — Darwin v5.0
Runs the full retrieval pipeline (keyword, semantic, hybrid) via the orchestrator.
Saves results and logs to JSON/txt for debugging.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore", module="litellm")
os.environ["LITELLM_SUPPRESS_DEBUG_INFO"] = "True"

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Storage.DarwinStore import DarwinStore

from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
from src.core.Retrival.OrchestratorTypes import (
    OrchestratorConfig,
    MergeStrategy,
    FallbackStrategy,
)

from src.core.Pipeline.Ingestion.Parsing.lib import url_to_slug
from src.core.Pipeline.Ingestion.Parsing.Parser import scan_directory


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

DIRECTORY = "./../../pdfs/"

EXTRA_URLS = [
    "https://en.wikipedia.org/wiki/Book_of_Enoch",
    "http://en.wikipedia.org/wiki/Amalthea_(mythology)",
]

def get_files(dir_path: str) -> List[str]:
    return [str(f) for f in scan_directory(dir_path)]

def get_filenames(dir_path: str) -> List[str]:
    return [f.name for f in scan_directory(dir_path)]

LOCAL_FILES = get_files(DIRECTORY)
SLUGS = [url_to_slug(url) for url in EXTRA_URLS]

ALL_SOURCES = LOCAL_FILES + EXTRA_URLS
FILE_NAMES = get_filenames(DIRECTORY) + SLUGS

# Test queries (up to 10)
TEST_QUERIES = [
    "book of enoch",
    "what is graph rag",
    "NVIDIA Nemotron",
    "Bible summary",
    "anthropic paper",
    "traction startup",
    "cinderella",
    "AI diplomacy",
    "giving tree",
    "openclaw ebook",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def result_to_dict(context) -> Dict[str, Any]:
    """Convert a RetrievalContext to a JSON‑serializable dict."""
    if context is None:
        return {"error": "No context"}
    return context.to_dict()


def save_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save results list to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Runner (synchronous, no asyncio needed)
# ═══════════════════════════════════════════════════════════════════════════════

def run_retrieval_test() -> int:
    darwin.configure()
    store = DarwinStore(name="test")
 
 

    darwin.log("═" * 60, LogType.INFO)
    darwin.log("Darwin Retrieval Orchestrator Test", LogType.INFO)
    darwin.log("═" * 60, LogType.INFO)

    # We'll run with hybrid method by default, but test others too.
    methods = ["hybrid", "semantic", "keyword"]
    all_results = []

    for method in methods:
        darwin.log(f"\n--- Testing method: {method.upper()} ---", LogType.INFO)

        # For each method, run a few queries
        queries_to_run = TEST_QUERIES[:5] if method == "hybrid" else TEST_QUERIES[:3]
        for query in queries_to_run:
            darwin.log(f"  Query: '{query}'", LogType.INFO)
       

            # Build config
            cfg = OrchestratorConfig(
                query=query,
                method=method,
                top_k=5,
                max_passages=10,
                max_chars=6000,
                min_passage_score=0.1,
                merge_strategy=MergeStrategy.ADJACENT,
                max_merge_gap=2,
                fill_gaps=True,
                max_per_source=2,          # limit per file
                prefer_diverse_sources=True,
                filter_structural=True,
                structural_penalty_factor=0.5,
                fallback_strategy=FallbackStrategy.BOTH,
                fallback_top_k=20,
                fallback_max_chars=12000,
                fallback_min_score=0.05,
                fallback_min_passages=2,
                include_metadata=True,
                include_audit=True,
                include_raw_results=False,
                return_empty_on_fallback=False,
                rerank=False,              # optional, can set True if reranker is available
                rerank_weight=0.3,
                rerank_top_k=100,
            )

            orchestrator = RetrievalOrchestrator(store)
            context = orchestrator.retrieve_and_resolve(cfg)

            # Convert to dict
            result_entry = {
                "query": query,
                "method": method,
                "context": result_to_dict(context),
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result_entry)

            # Brief logging
            if context and context.passages:
                darwin.log(f"    → {len(context.passages)} passages, {context.total_chars} chars", LogType.SUCCESS)
            else:
                darwin.log(f"    → No passages returned", LogType.WARNING)

    # ── Save results ───────────────────────────────────────────────────────────
    results_dir = Path("retrieval_test_results")
    results_dir.mkdir(exist_ok=True)

    results_path = results_dir / "results.json"
    save_results(all_results, results_path)
    darwin.log(f"\nResults saved to {results_path}", LogType.INFO)

    # ── Save logs (fixed reverse) ─────────────────────────────────────────────
    logs = darwin.get_logs(levels={LogType.INFO, LogType.WARNING, LogType.ERROR, LogType.SUCCESS})
    logs.reverse()   # reverse in-place
    log_path = results_dir / "logs.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(f"[{log.timestamp}] [{log.log_type.value.upper()}] {log.message}\n")
    darwin.log(f"Logs saved to {log_path}", LogType.INFO)

    darwin.log("═" * 60, LogType.INFO)
    darwin.log("Test completed.", LogType.SUCCESS)
    darwin.shutdown()
    return 0


if __name__ == "__main__":
    run_retrieval_test()   # removed asyncio