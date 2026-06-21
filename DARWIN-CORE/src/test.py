#!/usr/bin/env python3
"""
Milestone test — full Darwin pipeline:
  Ingestion → Indexing → Knowledge (Wiki + Interactive Graph) → Retrieval (no LLM)

Uses WikiOrchestrator with FULL pipeline support:
  - build_wiki_batch        (parallel wiki builds)
  - build_knowledge         (staged: wiki / graph / both)
  - build_knowledge_batch   (parallel staged builds)
  - list_entities_for_files (multi-file entity overlap detection)

RetrievalOrchestrator for surgical retrieval (context display only, no generation).

NEW: Link and blob support — ephemeral context, not persisted to Darwin store.
     Parsed to markdown for real-time inspection only.

Usage:
    uv run test_pipeline_with_knowledge.py
"""

from __future__ import annotations

import sys
import os
import warnings
from pathlib import Path

# ── path bootstrap ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

# ── LiteLLM noise suppression ──────────────────────────────────────────────
warnings.filterwarnings("ignore", module="litellm")
os.environ["LITELLM_SUPPRESS_DEBUG_INFO"] = "True"


from typing import List
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

# ── KNOWLEDGE (WikiOrchestrator v3.4) ──────────────────────────────────────
from src.features.Knowledge.WikiOrchestrator import WikiOrchestrator
from src.features.Knowledge.WikiIndex import NullWikiIndex

# ── NEW: Blob + Link handling ──────────────────────────────────────────────
from src.features.Ingestion.Parsing.BlobHandler import BlobHandler, BlobInput
from src.features.Ingestion.Parsing.Parser import parse_url, is_url
from src.features.Ingestion.Parsing.lib import ParsedDocument


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

# Query for retrieval stage
QUERY = "who are the watchers"

# ── Ephemeral context (links + blobs) ──────────────────────────────────────
# These are NOT saved to Darwin store. Parsed to markdown for inspection only.

EPHEMERAL_URLS: List[str] = [
    "https://en.wikipedia.org/wiki/Book_of_Enoch",
    "http://en.wikipedia.org/wiki/Amalthea_(mythology)",
]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Progress callback for batch operations
# ═══════════════════════════════════════════════════════════════════════════════

def on_batch_progress(filename: str, success: bool) -> None:
    """Simple progress callback for batch operations."""
    status = "✓" if success else "✗"
    darwin.log(f"  [{status}] {filename}", LogType.INFO if success else LogType.WARNING)


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

    # ── Ephemeral: URLs and blobs (NOT persisted) ──────────────────────────
    if EPHEMERAL_URLS:
        darwin.log("═" * 60, LogType.INFO)
        darwin.log("Ephemeral: Parsing URLs (not saved to store)", LogType.INFO)
        darwin.log("═" * 60, LogType.INFO)
        for url in EPHEMERAL_URLS:
            try:
                parsed = parse_url(url)
                print(f"  URL: {url}")
                print(f"  Title: {parsed.title}")
                print(f"  Content preview: {parsed.content[:200]}...")
                print()
            except Exception as e:
                darwin.log(f"  Failed to parse {url}: {e}", LogType.WARNING)

    # ── 1. INGESTION ─────────────────────────────────────────────────────
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

    # ── 2. INDEXING ──────────────────────────────────────────────────────
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

    # ── 3. KNOWLEDGE: Full WikiOrchestrator Pipeline ────────────────────
    darwin.log("═" * 60, LogType.INFO)
    darwin.log("Stage 3: Knowledge (Full Pipeline)", LogType.INFO)
    darwin.log("═" * 60, LogType.INFO)

    orchestrator = WikiOrchestrator(store, max_workers=4)

    # ── 3a. BATCH WIKI BUILD (parallel) ──────────────────────────────
    darwin.log("  → build_wiki_batch (parallel wiki extraction)", LogType.INFO)
    batch_results = orchestrator.build_wiki_batch(
        filenames=PARSED_FILES,
        rerun=True,
        min_frequency=2,
        use_multiprocess=True,
        max_workers=4,
        on_progress=on_batch_progress,
    )
    for fname, result in batch_results.items():
        status = "✓" if result.success else "✗"
        darwin.log(
            f"    [{status}] {fname}: {result.entity_count} entities "
            f"({result.duration_ms:.0f}ms)",
            LogType.SUCCESS if result.success else LogType.WARNING
        )

    # ── 3b. STAGED KNOWLEDGE BUILD (wiki + graph per file) ───────────
    darwin.log("  → build_knowledge (staged: wiki + graph)", LogType.INFO)
    for filename in PARSED_FILES:
        result = orchestrator.build_knowledge(
            filename=filename,
            stage="both",           # "wiki" | "graph" | "both"
            rerun=False,            # Already built wiki above
            min_frequency=2,
            use_multiprocess=False, # Already extracted
            graph_max_entities=100,
            graph_min_cooccurrence=0.1,
            graph_max_edges_per_node=8,
            graph_title=f"Entity Graph: {filename}",
        )
        if result.get("errors"):
            for err in result["errors"]:
                darwin.log(f"    ✗ {filename}: {err}", LogType.ERROR)
        else:
            wiki_stats = result.get("stats", {}).get("wiki", {})
            graph_stats = result.get("stats", {}).get("graph", {})
            darwin.log(
                f"    ✓ {filename}: wiki={wiki_stats.get('entity_count', 0)} entities, "
                f"graph={graph_stats.get('node_count', 0)} nodes / {graph_stats.get('edge_count', 0)} edges",
                LogType.SUCCESS
            )

    # ── 3c. BATCH KNOWLEDGE BUILD (demonstration) ─────────────────────
    # Uncomment to test parallel staged builds:
    # darwin.log("  → build_knowledge_batch (parallel staged)", LogType.INFO)
    # batch_knowledge = orchestrator.build_knowledge_batch(
    #     filenames=PARSED_FILES,
    #     stage="both",
    #     rerun=False,
    #     max_workers=4,
    #     on_progress=on_batch_progress,
    # )

    # ── 3d. MULTI-FILE ENTITY LISTING WITH OVERLAP ────────────────────
    if len(PARSED_FILES) > 1:
        darwin.log("  → list_entities_for_files (overlap detection)", LogType.INFO)
        multi_result = orchestrator.list_entities_for_files(
            filenames=PARSED_FILES,
            sort_by="salience",
            limit=20,
            min_frequency=1,
            min_files_present=1,
            include_overlap_analysis=True,
        )
        darwin.log(
            f"    Total entities across all files: {multi_result.total_entities_across_all}",
            LogType.INFO
        )
        darwin.log(
            f"    Unique entities: {multi_result.total_unique_entities}",
            LogType.INFO
        )
        darwin.log(
            f"    Overlapping entities (2+ files): {multi_result.overlap_count}",
            LogType.INFO
        )

        # Show overlapping entities
        if multi_result.overlapping:
            darwin.log("    Overlapping entities:", LogType.INFO)
            for eid, data in list(multi_result.overlapping.items())[:10]:
                darwin.log(
                    f"      • {data['canonical_name']} ({data['type']}): "
                    f"appears in {data['source_count']} files, "
                    f"{data['total_mentions']} mentions",
                    LogType.INFO
                )

        # Show unique entities per file
        for fname, unique_ents in multi_result.unique_per_file.items():
            if unique_ents:
                darwin.log(
                    f"    Unique to {fname}: {len(unique_ents)} entities",
                    LogType.INFO
                )

    # ── 3e. PER-FILE ENTITY INSPECTION ────────────────────────────────
    darwin.log("  → Per-file entity inspection", LogType.INFO)
    for filename in PARSED_FILES:
        wiki = orchestrator.get_wiki(filename)
        if isinstance(wiki, NullWikiIndex):
            darwin.log(f"    ⚠ No wiki for {filename}", LogType.WARNING)
            continue

        entities = wiki.list_entities(sort_by="salience", limit=10)
        print(f"    Top entities in {filename}:")
        for e in entities:
            print(f"      {e.canonical_name:<25} {e.type:<12} freq={e.frequency}, salience={e.salience:.3f}")

        # Document profile
        profile = wiki.get_document_profile(filename)
        if profile.get("total_entities", 0) > 0:
            print(f"      Entity density: {profile.get('entity_density', 0):.3f}")
            print(f"      Dominant types: {profile.get('dominant_types', [])}")

    # ── 4. RETRIEVAL (without LLM) ─────────────────────────────────────
    darwin.log("═" * 60, LogType.INFO)
    darwin.log("Stage 4: Retrieval (context only, no LLM)", LogType.INFO)
    darwin.log("═" * 60, LogType.INFO)

    # Use PARSED_FILES for retrieval (FILES was undefined / mismatched)
    target_files = PARSED_FILES if PARSED_FILES else ["book_of_enoch.pdf"]

    orch = RetrievalOrchestrator(store)
    context = orch.retrieve_and_resolve(
        query=QUERY,
        files=target_files,
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