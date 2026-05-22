#!/usr/bin/env python3
"""
Milestone test — full Darwin pipeline:
  Ingestion → Indexing → Knowledge → Retrieval → Generation

Usage:
    uv run test_pipeline.py
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

# ── logger ─────────────────────────────────────────────────────────────────
from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity

# ── store ────────────────────────────────────────────────────────────────────
from src.features.Storage.DarwinStore import DarwinStore

# ── pipeline args & handlers ───────────────────────────────────────────────
from src.features.Ingestion.Chunking.lib import ChunkingArgs
from src.features.Ingestion.Parsing.lib import ParsingArgs
from src.features.Ingestion.main import handle_ingestion

from src.features.Indexing.lib import IndexingArgs
from src.features.Indexing.main import handle_indexing
from src.features.Indexing.BM25.lib import DOMAIN_STOP_THRESHOLD, DEFAULT_IDF_SMOOTHING

from src.features.Knowledge.lib import KnowledgePipelineArgs
from src.features.Knowledge.main import handle_knowledge

# ── orchestrators ────────────────────────────────────────────────────────────
from src.features.Knowledge.WikiOrchestrator import WikiOrchestrator
from src.features.Retrival.RetrievalOrchestrator import RetrievalOrchestrator

# ── LLM ──────────────────────────────────────────────────────────────────────
from src.features.LLM.LLMManager import LLMManager
from src.features.LLM.LLMTypes import LLMResponse


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

FILES = ["book_of_enoch.pdf"]
QUERY = "who are the watchers"


def main() -> int:
    # ── logger init ──────────────────────────────────────────────────────────
    darwin.configure(
        enable_websocket=False,
        enable_history=True,
        verbosity_default=Verbosity.NORMAL,
        show_banner=True,
    )

    # ── store ──────────────────────────────────────────────────────────────
    store = DarwinStore()  # or DarwinStore("/path/to/.darwin")

    with darwin.session("book_of_enoch_pipeline"):
        # ── 1. ingestion ─────────────────────────────────────────────────────
        darwin.log("═" * 60, LogType.INFO)
        darwin.log("Stage 1: Ingestion", LogType.INFO)
        darwin.log("═" * 60, LogType.INFO)

        parsing_args = ParsingArgs(
            target=None,
            files=FILES,
            images=True,
            caption=True,
            rerun=True,
            precise=False,
            save=True,
        )
        chunking_args = ChunkingArgs(
            inspect=True,
            rerun=True,
        )

        # MP listener forwards worker logs from ProcessPoolExecutor
        darwin.start_mp_listener()
        handle_ingestion(parsing_args, chunking_args, store=store)
        darwin.stop_mp_listener()

        # ── 2. indexing ──────────────────────────────────────────────────────
        darwin.log("═" * 60, LogType.INFO)
        darwin.log("Stage 2: Indexing", LogType.INFO)
        darwin.log("═" * 60, LogType.INFO)

        indexing_args = IndexingArgs(
            rerun=True,
            files=FILES,
            inspect=True,
            threshold=DOMAIN_STOP_THRESHOLD,
            idf_smoothing=DEFAULT_IDF_SMOOTHING,
        )
        handle_indexing(indexing_args, store=store)

        # ── 3. knowledge graphs + wikis ──────────────────────────────────────
        darwin.log("═" * 60, LogType.INFO)
        darwin.log("Stage 3: Knowledge", LogType.INFO)
        darwin.log("═" * 60, LogType.INFO)

        knowledge_args = KnowledgePipelineArgs(
            files=FILES,
            visualize=True,
            inspect=True,
            rerun=True,
        )
        handle_knowledge(knowledge_args, store=store)

        # ── 4. retrieval ─────────────────────────────────────────────────────
        darwin.log("═" * 60, LogType.INFO)
        darwin.log("Stage 4: Retrieval", LogType.INFO)
        darwin.log("═" * 60, LogType.INFO)

        # 4a. Wiki-style semantic search
        wiki_orch = WikiOrchestrator(store)
        wiki_context = wiki_orch.search(
            query=QUERY,
            filename=FILES[0],
            top_k=3,
        )
        wiki_prompt = wiki_orch.build_llm_prompt(wiki_context, QUERY)

        darwin.log("\n── Wiki Prompt ──", LogType.INFO)
        darwin.log(wiki_prompt[:500] + "...", LogType.INFO)

        # 4b. Hybrid RAG retrieval
        rag_orch = RetrievalOrchestrator(store)
        rag_context = rag_orch.retrieve_and_resolve(
            query=QUERY,
            filename=FILES[0],
            method="hybrid",
            top_k=5,
            merge_sources=True,
            fill_gaps=True,
        )
        rag_prompt = rag_orch.build_llm_prompt(rag_context)

        darwin.log("\n── RAG Prompt ──", LogType.INFO)
        darwin.log(rag_prompt[:500] + "...", LogType.INFO)

        # ── 5. generation ────────────────────────────────────────────────────
        darwin.log("═" * 60, LogType.INFO)
        darwin.log("Stage 5: Generation", LogType.INFO)
        darwin.log("═" * 60, LogType.INFO)

        LLMManager.initialize()
        llm = LLMManager.get_client()

        # 5a. Generate from RAG context (preferred for factual queries)
        rag_messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise research assistant. Answer using ONLY the provided context. "
                    "Cite sources with [Source: filename, page X]. "
                    "If the context is insufficient, say so explicitly."
                ),
            },
            {"role": "user", "content": rag_prompt},
        ]

        darwin.log("Calling LLM (RAG context)...", LogType.INFO)
        rag_resp: LLMResponse = llm.call_smart(rag_messages)
        if rag_resp.error:
            darwin.log(f"RAG generation error: {rag_resp.error}", LogType.ERROR)
        else:
            darwin.log("\n── RAG Answer ──", LogType.SUCCESS)
            darwin.log(rag_resp.text, LogType.INFO)

        # 5b. Generate from Wiki context (preferred for conceptual / exploratory)
        wiki_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert synthesizer. Use the concept graph and wiki summaries to "
                    "produce a structured, encyclopedic answer with linked concepts."
                ),
            },
            {"role": "user", "content": wiki_prompt},
        ]

        darwin.log("Calling LLM (Wiki context)...", LogType.INFO)
        wiki_resp: LLMResponse = llm.call_smart(wiki_messages)
        if wiki_resp.error:
            darwin.log(f"Wiki generation error: {wiki_resp.error}", LogType.ERROR)
        else:
            darwin.log("\n── Wiki Answer ──", LogType.SUCCESS)
            darwin.log(wiki_resp.text, LogType.INFO)

    darwin.log("\nPipeline complete.", LogType.SUCCESS)
    darwin.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())