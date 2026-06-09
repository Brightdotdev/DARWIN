#!/usr/bin/env python3
"""
Milestone test — full Darwin pipeline:
  Ingestion → Indexing → Knowledge (Wiki + Interactive Graph) → Retrieval (no LLM)

Uses WikiOrchestrator for entity extraction and interactive graph building.
RetrievalOrchestrator for surgical retrieval (context display only, no generation).

Usage:
    uv run test_pipeline_with_knowledge.py
"""

from __future__ import annotations

import sys
import os
from typing import List
import warnings
from pathlib import Path

# ── path bootstrap ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

# ── LiteLLM noise suppression ──────────────────────────────────────────────
warnings.filterwarnings("ignore", module="litellm")
os.environ["LITELLM_SUPPRESS_DEBUG_INFO"] = "True"

# ── env ──────────────────────────────────────────────────────────────────────
os.environ["DARWIN_DISABLE_TORCH"] = "1"

# ── logger ─────────────────────────────────────────────────────────────────
from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity

# ── store ────────────────────────────────────────────────────────────────────
from src.core.Storage.DarwinStore import DarwinStore

# ── pipeline args & handlers ───────────────────────────────────────────────
from src.features.Ingestion.Chunking.lib import ChunkingArgs
from src.features.Ingestion.Parsing.lib import ParsingArgs
from src.features.Ingestion.main import handle_ingestion

from src.features.Indexing.lib import IndexingArgs
from src.features.Indexing.main import handle_indexing
from src.features.Indexing.BM25.lib import DOMAIN_STOP_THRESHOLD, DEFAULT_IDF_SMOOTHING

# ── RETRIEVAL ───────────────────────────────────────────────────────────────
from src.features.Retrival.RetrievalOrchestrator import RetrievalOrchestrator

# ── KNOWLEDGE (NEW: WikiOrchestrator) ──────────────────────────────────────
from src.features.Knowledge.WikiOrchestrator import WikiOrchestrator
from src.features.Knowledge.WikiBuilder import NullWikiIndex

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

def get_files(dir_path: str) -> List[str]:
    from src.features.Ingestion.Parsing.Parser import scan_directory
    files = scan_directory(dir_path)
    return [str(f) for f in files]

def get_parsed(dir_path: str) -> List[str]:
    from src.features.Ingestion.Parsing.Parser import scan_directory
    files = scan_directory(dir_path)
    print(f"Scanned {len(files)} files in {dir_path}")
    for f in files:
        print(f"  Found file: {f.name}")
    return [f.name for f in files]


directory = "./pdfs/"
TO_BE_PROCESSED = get_files(directory)
PARSED_FILES = get_parsed(directory)

FILES = ["book_of_enoch.pdf"]
QUERY = "who are the watchers"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    # ── logger init ──────────────────────────────────────────────────────────
    darwin.configure(
        enable_websocket=False,
        enable_history=True,
        verbosity_default=Verbosity.NORMAL,
        show_banner=True,
    )

    # ── store ──────────────────────────────────────────────────────────────
    store = DarwinStore(Path("./darwin_store_test"))

    with darwin.session("pipeline_wiki_graph_test"):
        # ── 1. ingestion ─────────────────────────────────────────────────────
        darwin.log("═" * 60, LogType.INFO)
        darwin.log("Stage 1: Ingestion", LogType.INFO)
        darwin.log("═" * 60, LogType.INFO)

        parsing_args = ParsingArgs(
            target=None,
            files=TO_BE_PROCESSED,
            images=True,
            precise=False,
            save=True,
        )
        chunking_args = ChunkingArgs(
            inspect=True,
        )
        handle_ingestion(parsing_args, chunking_args, store=store)

        # ── 2. indexing ──────────────────────────────────────────────────────
        darwin.log("═" * 60, LogType.INFO)
        darwin.log("Stage 2: Indexing", LogType.INFO)
        darwin.log("═" * 60, LogType.INFO)

        indexing_args = IndexingArgs(
            files=PARSED_FILES,
            inspect=True,
            threshold=DOMAIN_STOP_THRESHOLD,
            idf_smoothing=DEFAULT_IDF_SMOOTHING,
        )
        handle_indexing(indexing_args, store=store)

        # ── 3. KNOWLEDGE: Wiki + Interactive Graph ──────────────────────────
        darwin.log("═" * 60, LogType.INFO)
        darwin.log("Stage 3: Knowledge (Wiki + Interactive Graph)", LogType.INFO)
        darwin.log("═" * 60, LogType.INFO)

        orchestrator = WikiOrchestrator(store)

        # Process each ingested file through the new wiki pipeline
        for filename in PARSED_FILES:
            darwin.log(f"Building wiki & graph for: {filename}", LogType.INFO)

            # Build/rebuild the entity wiki
            wiki = orchestrator.build_wiki(filename, rerun=True)

            if isinstance(wiki, NullWikiIndex):
                darwin.log(f"  ⚠ No wiki data for {filename} (spaCy may be missing or file empty).", LogType.WARNING)
                continue

            # Display top entities in the terminal for quick inspection
            entities = wiki.list_entities(sort_by="salience", limit=20)
            print(f"  Top entities in {filename}:")
            for e in entities:
                print(f"    {e.canonical_name:<25} {e.type:<12} freq={e.frequency}, salience={e.salience:.3f}")

            # Build and save the interactive graph (with sliders)
            success = orchestrator.build_and_save_interactive_graph(
                filename=filename,
                max_entities=100,
                title=f"Entity Graph: {filename}"
            )
            if success:
                darwin.log(f"  ✓ Interactive graph saved for {filename}", LogType.SUCCESS)
            else:
                darwin.log(f"  ✗ Graph build failed for {filename}", LogType.ERROR)

        # ── 4. RETRIEVAL (without LLM) ─────────────────────────────────────
        darwin.log("═" * 60, LogType.INFO)
        darwin.log("Stage 4: Retrieval (context only, no LLM)", LogType.INFO)
        darwin.log("═" * 60, LogType.INFO)

        orch = RetrievalOrchestrator(store)
        context = orch.retrieve_and_resolve(
            query=QUERY,
            files=FILES,
            method="hybrid",
            top_k=5,
            merge_sources=True,
            fill_gaps=True,
        )
        darwin.log("Retrieved context (first 500 chars):", LogType.INFO)
        print(context[:500] + "..." if len(context) > 500 else context)

        # NOTE: LLM calls have been removed. To enable generation later,
        # uncomment the LLM block and ensure LLMManager is properly initialized.

    darwin.log("\nPipeline complete (no LLM calls).", LogType.SUCCESS)
    darwin.log("Open enoch_interactive.html in your browser to explore the graph.", LogType.INFO)
    darwin.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())