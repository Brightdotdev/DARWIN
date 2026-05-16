#!/usr/bin/env python3
"""
test_both_orchestrators.py
--------------------------
Compare RetrievalOrchestrator vs WikiOrchestrator
with source merging and gap filling.

Run:
    uv run python test_both_orchestrators.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.features.Storage.DarwinStore import DarwinStore
from src.features.Retrival.RetrievalOrchestrator import RetrievalOrchestrator, retrieve_for_llm
from src.features.Knowledge.WikiOrchestrator import WikiOrchestrator, wiki_search_for_llm


FILENAME = "book_of_enoch.pdf"  
STORE_DIR = None


def main():
    print("\n" + "🔗" * 25)
    print("  BOTH ORCHESTRATORS — Merge + Gap Fill Test")
    print("🔗" * 25)

    store = DarwinStore()

    queries = [
        "who are the watchers",
        "what is azazel",
        "giants evidence",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"  QUERY: '{query}'")
        print(f"{'=' * 60}")

        # ── DEEP RETRIEVAL ──────────────────────────────────────────
        print(f"\n  📚 DEEP RETRIEVAL (RetrievalOrchestrator)")
        print(f"  {'─' * 50}")

        orch = RetrievalOrchestrator(store)
        context = orch.retrieve_and_resolve(
            query=query,
            filename=FILENAME,
            method="hybrid",
            top_k=5,
            merge_sources=True,
            fill_gaps=True,
        )

        print(f"    Raw results: {len(context.raw_results)}")
        print(f"    Filtered structural: {context.filtered_count}")
        print(f"    Final Sources: {len(context.passages)}")

        for i, p in enumerate(context.passages):
            original_count = len(p.original_results)
            gap_filled = len(p.chunk_indices) - original_count

            print(f"\n    [Source {i+1}] {p.chunk_indices}")
            print(f"      Page {p.page_start}-{p.page_end} | "
                  f"Original chunks: {original_count} | "
                  f"Gap-filled: {gap_filled}")
            print(f"      Text: {p.passage_text[:150]}...")

        # ── SHALLOW WIKI ────────────────────────────────────────────
        print(f"\n  📖 SHALLOW WIKI (WikiOrchestrator)")
        print(f"  {'─' * 50}")

        wiki_orch = WikiOrchestrator(store)
        wiki_context = wiki_orch.search(
            query=query,
            filename=FILENAME,
            top_k=5,
            merge_sources=True,
            fill_gaps=True,
        )

        print(f"    Total concepts: {wiki_context.total_concepts}")
        print(f"    Filtered structural: {wiki_context.filtered_count}")
        print(f"    Related topics: {wiki_context.related_topics[:5]}")

        for i, c in enumerate(wiki_context.concepts):
            print(f"\n    [Concept {i+1}] {c.concept}")
            print(f"      Page {c.page_start}-{c.page_end} | "
                  f"Chunks: {c.chunk_indices} | "
                  f"Merged from: {c.source_count} concepts")
            print(f"      Passage: {c.passage_text[:150]}...")

    print("\n" + "✅" * 25)
    print("  BOTH ORCHESTRATORS TEST COMPLETE")
    print("  Deep + Wiki both merge adjacent and fill gaps")
    print("✅" * 25 + "\n")


if __name__ == "__main__":
    main()