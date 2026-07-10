#!/usr/bin/env python3
"""
Retrieval Module Test — Direct call to handle_retrieval.
This bypasses the Orchestrator to isolate issues in the retrieval pipeline.
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

from src.core.Retrival.main import handle_retrieval
from src.core.Retrival.RetrievalArgs import RetrievalArgs, RetrievalMethod

from src.core.Pipeline.Ingestion.Parsing.lib import url_to_slug
from src.core.Pipeline.Ingestion.Parsing.Parser import scan_directory


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

DIRECTORY = "./../../../pdfs/"

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


def result_to_dict(results: List) -> List[Dict[str, Any]]:
    """Convert a list of RetrievalResult objects to JSON‑serializable dicts."""
    if results is None:
        return []
    out = []
    for r in results:
        item = {
            "type": r.result_type.value,
            "score": r.score,
            "source": r.source,
            "text": r.text,
            "clean": r.clean,
            "page": r.page,
            "page_end": r.page_end,
            "heading": r.heading,
            "chunk_index": r.chunk_index,
            "overlap_direction": r.overlap_direction.value
            if hasattr(r.overlap_direction, "value")
            else r.overlap_direction,
            "links": [
                {
                    "url": l.url if hasattr(l, "url") else l.get("url", ""),
                    "text": l.text if hasattr(l, "text") else l.get("text", ""),
                }
                for l in (r.links or [])
            ],
            "rrf_score": r.rrf_score,
            "semantic_score": r.semantic_score,
            "keyword_score": r.keyword_score,
            "matched_terms": list(r.matched_terms) if r.matched_terms else [],
            "rerank_score": r.rerank_score,
            "rerank_original_score": r.rerank_original_score,
        }
        out.append(item)
    return out


def save_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save results list to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Runner
# ═══════════════════════════════════════════════════════════════════════════════


def run_retrieval_test() -> int:
    darwin.configure()
    store = DarwinStore(name="test")

    darwin.info("═" * 60)
    darwin.info("Retrieval Module Test (handle_retrieval)")
    darwin.info("═" * 60)

    methods = ["hybrid", "semantic", "keyword"]
    all_results = []

    for method in methods:
        darwin.info(f"\n--- Testing method: {method.upper()} ---")

        # For each method, run a few queries
        queries_to_run = TEST_QUERIES[:5] if method == "hybrid" else TEST_QUERIES[:3]
        for query in queries_to_run:
            darwin.info(f"  Query: '{query}'")

            # Build RetrievalArgs (unified)
            args = RetrievalArgs(
                query=query,
                method=RetrievalMethod(method),
                files=FILE_NAMES,  # use all registered files
                top_k=20,
                pool_size=20,  # for hybrid
                rrf_k=60,
                semantic_weight=1.0,
                keyword_weight=1.0,
                min_score=0.0,
                k1=1.5,
                b=0.75,
                per_doc_top_k=2,  # limit per doc as before
                rerank=False,
                rerank_weight=0.3,
                rerank_top_k=100,
                store_dir=None,
                sse_port=8765,
                no_sse=False,
            )

            # Call the retrieval pipeline directly
            results = handle_retrieval(args, store)

            result_entry = {
                "query": query,
                "method": method,
                "results": result_to_dict(results),
                "count": len(results) if results else 0,
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result_entry)

            # Brief logging
            if results:
                darwin.success(f"    → {len(results)} results")
            else:
                darwin.warning("    → No results returned")

    # ── Save results ───────────────────────────────────────────────────────────
    results_dir = Path(__file__).parent / "retrieval_module_test_results"
    results_dir.mkdir(exist_ok=True)

    results_path = results_dir / "results.json"
    save_results(all_results, results_path)
    darwin.info(f"Results saved to {results_path}")

    # ── Save logs ─────────────────────────────────────────────────────────────
    logs = darwin.get_logs(
        levels={LogType.INFO, LogType.WARNING, LogType.ERROR, LogType.SUCCESS}
    )
    logs.reverse()  # newest last (or reverse to get chronological)
    log_path = results_dir / "logs.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(f"[{log.timestamp}] [{log.log_type.value.upper()}] {log.message}\n")
    darwin.info(f"Logs saved to {log_path}")

    darwin.info("═" * 60)
    darwin.success("Test completed.")
    darwin.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(run_retrieval_test())
