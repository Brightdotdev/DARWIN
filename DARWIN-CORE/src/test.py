#!/usr/bin/env python3
"""
Milestone test — full Darwin pipeline (v5.0).
Uses PipelineOrchestrator with typed stage parameters.
Per-stage session logging captured in result.logs.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore", module="litellm")
os.environ["LITELLM_SUPPRESS_DEBUG_INFO"] = "True"

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Storage.DarwinStore import DarwinStore

# ── Pipeline Orchestrator ────────────────────────────────────────────────────
from src.core.Orchestrators.PipelineOrchestrator import (
    PipelineOrchestrator,
    PipelineParameters,
    PipelineStage,
    KnowledgeArgs,
)

# ── Stage args ───────────────────────────────────────────────────────────────
from src.pipleline.Ingestion.Chunking.lib import ChunkingArgs
from src.pipleline.Ingestion.Parsing.lib import ParsingArgs, url_to_slug
from src.pipleline.Indexing.lib import IndexingArgs
from src.pipleline.Indexing.BM25.lib import DOMAIN_STOP_THRESHOLD, DEFAULT_IDF_SMOOTHING

# ── Retrieval ────────────────────────────────────────────────────────────────
from src.core.Orchestrators.RetrievalOrchestrator import (
    RetrievalOrchestrator,
    OrchestratorConfig,
    MergeStrategy,
    FallbackStrategy,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_files(dir_path: str) -> List[str]:
    from src.pipleline.Ingestion.Parsing.Parser import scan_directory
    return [str(f) for f in scan_directory(dir_path)]


def get_filenames(dir_path: str) -> List[str]:
    from src.pipleline.Ingestion.Parsing.Parser import scan_directory
    return [f.name for f in scan_directory(dir_path)]


def print_context_summary(context, detailed: bool = False) -> None:
    darwin.log("═" * 70, LogType.INFO)
    darwin.log("Retrieval Context Summary", LogType.INFO)
    darwin.log("═" * 70, LogType.INFO)

    darwin.log(f"Query: {context.query}", LogType.INFO)
    darwin.log(f"Method: {context.method}", LogType.INFO)
    darwin.log(f"Passages: {len(context.passages)}", LogType.INFO)
    darwin.log(f"Total chars: {context.total_chars:,}", LogType.INFO)
    darwin.log(f"Sources: {', '.join(context.sources_used) if context.sources_used else 'None'}", LogType.INFO)
    darwin.log(f"Source diversity: {context.source_diversity:.2f}", LogType.INFO)
    darwin.log(f"Max score: {context.max_score:.4f}", LogType.INFO)
    darwin.log(f"Avg score: {context.avg_score:.4f}", LogType.INFO)
    darwin.log(f"Elapsed: {context.elapsed_ms}ms", LogType.INFO)

    if context.fallback_used:
        darwin.log(f"⚠️ Fallback used: {context.fallback_reason}", LogType.WARNING)

    darwin.log(f"Candidates found: {context.candidates_found}", LogType.INFO)
    darwin.log(f"Penalized count: {context.penalized_count}", LogType.INFO)

    if detailed:
        darwin.log("\nPassage Details:", LogType.INFO)
        for i, p in enumerate(context.passages, 1):
            darwin.log(f"\n[{i}] Source: {p.metadata.source}", LogType.INFO)
            darwin.log(f"    Score: {p.metadata.avg_score:.4f}", LogType.INFO)
            darwin.log(f"    Pages: {p.metadata.page_start}-{p.metadata.page_end}", LogType.INFO)
            darwin.log(f"    Chunks: {p.metadata.chunk_count} ({p.metadata.chunk_indices})", LogType.INFO)
            darwin.log(f"    Merged: {p.metadata.is_merged}", LogType.INFO)
            darwin.log(f"    Structural: {p.metadata.structural_type}", LogType.INFO)
            if p.metadata.matched_terms:
                darwin.log(f"    Matched terms: {', '.join(p.metadata.matched_terms[:5])}", LogType.INFO)
            darwin.log(f"    Text preview: {p.text[:150]}...", LogType.INFO)

    darwin.log("═" * 70, LogType.INFO)


def save_retrieval_results(context, output_path: Path, include_raw: bool = False) -> None:
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "query": context.query,
        "method": context.method,
        "total_passages": len(context.passages),
        "total_chars": context.total_chars,
        "elapsed_ms": context.elapsed_ms,
        "candidates_found": context.candidates_found,
        "penalized_count": context.penalized_count,
        "max_score": context.max_score,
        "avg_score": context.avg_score,
        "sources_used": context.sources_used,
        "source_diversity": context.source_diversity,
        "fallback_used": context.fallback_used,
        "fallback_reason": context.fallback_reason,
        "passages": []
    }

    for p in context.passages:
        passage_data = {
            "text": p.text,
            "metadata": {
                "source": p.metadata.source,
                "page_start": p.metadata.page_start,
                "page_end": p.metadata.page_end,
                "chunk_indices": p.metadata.chunk_indices,
                "chunk_count": p.metadata.chunk_count,
                "is_merged": p.metadata.is_merged,
                "gap_filled": p.metadata.gap_filled,
                "structural_type": p.metadata.structural_type,
                "penalty_factor": p.metadata.penalty_factor,
                "was_penalized": p.metadata.was_penalized,
                "avg_score": p.metadata.avg_score,
                "semantic_score": p.metadata.semantic_score,
                "keyword_score": p.metadata.keyword_score,
                "rrf_score": p.metadata.rrf_score,
                "rerank_score": p.metadata.rerank_score,
                "matched_terms": p.metadata.matched_terms,
            }
        }
        if p.audit:
            passage_data["audit"] = p.audit
        output_data["passages"].append(passage_data)

    if include_raw and context.raw_results:
        output_data["raw_results"] = [
            {
                "score": r.score,
                "source": r.source,
                "page": r.page,
                "text": r.text[:200] if r.text else None,
            }
            for r in context.raw_results[:10]
        ]

    output_data["audit_summary"] = context.audit_summary

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

DIRECTORY = "./pdfs/"
LOCAL_FILES = get_files(DIRECTORY)

EXTRA_URLS = [
    "https://en.wikipedia.org/wiki/Book_of_Enoch",
    "http://en.wikipedia.org/wiki/Amalthea_(mythology)",
]
slug_url = [url_to_slug(url) for url in EXTRA_URLS]

ALL_SOURCES = LOCAL_FILES + EXTRA_URLS
file_names = [f for f in get_filenames(DIRECTORY)] + slug_url

TECHNICAL_FILES = [
    "NVIDIA-Nemotron-3-Super-Technical-Report.pdf",
    "FF_ONE_HUNDRED_WAYS_TO_USE_AI.pdf",
    "Essential-GraphRAG.pdf",
]

QUERY = "who are the watchers"


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval Config Presets
# ═══════════════════════════════════════════════════════════════════════════════

def get_retrieval_config(
    mode: str = "balanced",
    exclude_technical: bool = True,
    processed_files: List[str] = None,
) -> OrchestratorConfig:
    common = {
        "query": QUERY,
        "method": "hybrid",
        "files": processed_files or file_names,
        "fill_gaps": False,
        "structural_penalty_factor": 0.5,
        "include_metadata": True,
        "include_audit": True,
        "fallback_strategy": FallbackStrategy.BOTH,
        "fallback_top_k": 15,
        "fallback_max_chars": 10000,
        "fallback_min_score": 0.15,
        "return_empty_on_fallback": False,
        "inspect": False,
    }

    if exclude_technical:
        common["exclude_sources"] = TECHNICAL_FILES
    else:
        common["exclude_sources"] = None

    if mode == "precision":
        return OrchestratorConfig(
            **common,
            merge_strategy=MergeStrategy.ADJACENT,
            top_k=3, max_chars=4000, max_passages=3,
            min_passage_score=0.6, max_per_source=1,
            prefer_diverse_sources=True, require_diverse_sources=False,
            filter_structural=True, include_raw_results=False,
        )
    elif mode == "recall":
        return OrchestratorConfig(
            **common,
            merge_strategy=MergeStrategy.ADJACENT,
            top_k=15, max_chars=12000, max_passages=8,
            min_passage_score=0.15, max_per_source=3,
            prefer_diverse_sources=True, require_diverse_sources=False,
            filter_structural=False, include_raw_results=True,
        )
    elif mode == "experimental":
        return OrchestratorConfig(
            **common,
            merge_strategy=MergeStrategy.NONE,
            top_k=10, max_chars=8000, max_passages=6,
            min_passage_score=0.2, max_per_source=2,
            prefer_diverse_sources=True, require_diverse_sources=False,
            filter_structural=True, include_raw_results=True,
        )
    elif mode == "clean":
        religious_sources = [f for f in (processed_files or file_names) if "bible" in f.lower() or "enoch" in f.lower()]
        return OrchestratorConfig(
            **common,
            merge_strategy=MergeStrategy.ADJACENT,
            top_k=5, max_chars=6000, max_passages=5,
            min_passage_score=0.2, max_per_source=2,
            prefer_diverse_sources=True, require_diverse_sources=True,
            prefer_sources=religious_sources,
            filter_structural=True, include_raw_results=False,
        )
    else:
        return OrchestratorConfig(
            **common,
            merge_strategy=MergeStrategy.ADJACENT,
            top_k=5, max_chars=6000, max_passages=5,
            min_passage_score=0.3, max_per_source=2,
            prefer_diverse_sources=True, require_diverse_sources=False,
            filter_structural=True, include_raw_results=False,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main() -> int:
    darwin.configure()
    store = DarwinStore(Path("./darwin_store_test"))
    store.repair_registry()

    # ── Stage 1–2: PipelineOrchestrator ──────────────────────────────────────
    darwin.log("═" * 60, LogType.INFO)
    darwin.log("Building pipeline via PipelineOrchestrator", LogType.INFO)
    darwin.log("═" * 60, LogType.INFO)

    pipeline = PipelineOrchestrator(store)

    params = PipelineParameters(
        parsing=ParsingArgs(
            target=None,
            files=ALL_SOURCES,
            images=True,
            save=True,
        ),
        chunking=ChunkingArgs(inspect=True),
        indexing=IndexingArgs(
            files=file_names,
            inspect=True,
            threshold=DOMAIN_STOP_THRESHOLD,
            idf_smoothing=DEFAULT_IDF_SMOOTHING,
        ),
        knowledge=KnowledgeArgs(
            files=file_names,
            rerun=False,
            knowledge_stages=["wiki", "graph"],
            min_mentions=1,
            window_size=5,
        ),
    )

    result = pipeline.build_pipeline(
        params,
        [PipelineStage.INGESTION, PipelineStage.INDEXING, PipelineStage.KNOWLEDGE],
        strict=True,
        log_mode="normal",
    )

    # Print per-stage logs for inspection
    for stage_key, logs in result.logs.items():
        darwin.log(f"\n--- Logs for '{stage_key}' ({len(logs)} entries) ---", LogType.INFO)
        for log in logs[:5]:  # cap preview
            darwin.log(f"  [{log.log_type.value}] {log.message[:100]}", LogType.INFO)

    if not result.success:
        darwin.error(f"Pipeline failed: {result.summary}")
        for stage, err in result.stages_failed.items():
            darwin.error(f"  {stage}: {err}")
        return 1

    darwin.success(f"Pipeline OK: {result.summary}")

    # Determine which files actually made it through
    processed_files = [f for f in file_names if store.has_markdown(f)]
    darwin.log(f"Files ready for retrieval: {len(processed_files)}", LogType.INFO)

    if not processed_files:
        darwin.error("No files available after pipeline. Exiting.")
        return 1

    # ── Stage 3: Retrieval (multiple configs) ────────────────────────────────
    darwin.log("═" * 60, LogType.INFO)
    darwin.log("Stage 3: Retrieval with RetrievalOrchestrator", LogType.INFO)
    darwin.log("═" * 60, LogType.INFO)

    orch = RetrievalOrchestrator(store)

    configs_to_test = [
        ("balanced", "Balanced (default)", True),
        ("precision", "High Precision", True),
        ("recall", "High Recall", True),
        ("experimental", "Experimental (no merging)", False),
        ("clean", "Clean (religious only)", True),
    ]

    results_by_mode: Dict[str, Any] = {}

    for mode, description, exclude_tech in configs_to_test:
        darwin.log(f"\n--- {description} Mode ---", LogType.INFO)

        config = get_retrieval_config(mode, exclude_technical=exclude_tech, processed_files=processed_files)
        context = orch.retrieve_and_resolve(config)
        results_by_mode[mode] = context

        print_context_summary(context, detailed=(mode == "balanced"))

        output_path = Path(f"retrieval_results_{mode}.json")
        save_retrieval_results(context, output_path, include_raw=(mode in ["recall", "experimental"]))
        darwin.log(f"Results saved to {output_path}", LogType.SUCCESS)

    # ── Stage 4: Comparison ────────────────────────────────────────────────────
    darwin.log("═" * 60, LogType.INFO)
    darwin.log("Stage 4: Configuration Comparison", LogType.INFO)
    darwin.log("═" * 60, LogType.INFO)

    darwin.log(f"{'Mode':<12} {'Passages':<10} {'Chars':<10} {'Sources':<10} {'Diversity':<10} {'Avg Score':<10}", LogType.INFO)
    darwin.log("-" * 70, LogType.INFO)

    for mode, context in results_by_mode.items():
        darwin.log(
            f"{mode:<12} {len(context.passages):<10} {context.total_chars:<10,} "
            f"{len(context.sources_used):<10} {context.source_diversity:<10.2f} "
            f"{context.avg_score:<10.4f}",
            LogType.INFO,
        )

    # ── Stage 5: Best Result ─────────────────────────────────────────────────
    darwin.log("\n═" * 60, LogType.INFO)
    darwin.log("Stage 5: Best Result Selection", LogType.INFO)
    darwin.log("═" * 60, LogType.INFO)

    best_mode = None
    best_score = 0.0

    for mode, context in results_by_mode.items():
        combined = (context.avg_score * 0.7) + (context.source_diversity * 0.3)
        if combined > best_score and len(context.passages) > 0:
            best_score = combined
            best_mode = mode

    if best_mode:
        darwin.log(f"Best configuration: {best_mode}", LogType.SUCCESS)
        darwin.log(f"Combined score: {best_score:.4f}", LogType.INFO)

        best_context = results_by_mode[best_mode]
        save_retrieval_results(best_context, Path("retrieval_results_best.json"), include_raw=True)

        darwin.log("\nBest Passages:", LogType.INFO)
        for i, p in enumerate(best_context.passages[:3], 1):
            darwin.log(f"\n[{i}] {p.metadata.source} (score={p.metadata.avg_score:.3f})", LogType.INFO)
            darwin.log(f"    {p.text[:200]}...", LogType.INFO)

    # ── Final Summary ────────────────────────────────────────────────────────
    darwin.log("\n═" * 60, LogType.INFO)
    darwin.log("Pipeline Complete", LogType.SUCCESS)
    darwin.log("═" * 60, LogType.INFO)
    darwin.log(f"Total files processed: {len(processed_files)}", LogType.INFO)
    darwin.log(f"Configurations tested: {len(configs_to_test)}", LogType.INFO)

    for mode, context in results_by_mode.items():
        status = "✅" if len(context.passages) > 0 else "❌"
        darwin.log(
            f"{status} {mode}: {len(context.passages)} passages, {context.source_diversity:.2f} diversity",
            LogType.INFO,
        )

    # Dump errors/warnings to file
    errors = darwin.get_logs(levels={LogType.ERROR, LogType.WARNING,LogType.SUCCESS})
    with open("sum.txt", "w", encoding="utf-8") as f:
        for log in errors:
            f.write(f"[{log.session}] [{log.log_type.value.upper()}] {log.message}\n")
            if log.payload:
                f.write(f"  → {log.payload}\n")

    darwin.shutdown()
    return 0


if __name__ == "__main__":
    asyncio.run(main())