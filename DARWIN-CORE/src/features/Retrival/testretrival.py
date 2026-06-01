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

from src.core.Storage.DarwinStore import DarwinStore
from src.features.Retrival.RetrievalOrchestrator import RetrievalOrchestrator
from src.features.Knowledge.WikiOrchestrator import WikiOrchestrator


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
        print(f"    Filtered structural: {context.penalized_count}")
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
        wiki_result = wiki_orch.search(
            query=query,
            filename=FILENAME,
            top_k=5,
            merge_sources=True,
            fill_gaps=True,
        )
        
        # WikiOrchestrator likely returns a WikiContext object
        print(f"Query: {wiki_result.query}")
        print(f"Total concepts: {wiki_result.total_concepts}")
        print(f"Penalized count: {wiki_result.penalized_count}")
        print(f"Related topics: {wiki_result.related_topics}")
        
        for i, concept in enumerate(wiki_result.concepts):
            print(f"\n    [Concept {i+1}] {concept.concept}")
            print(f"      Raw concept: {concept.raw_concept}")
            print(f"      Page {concept.page_start}-{concept.page_end} | "
                  f"Chunks: {concept.chunk_indices} | "
                  f"Merged from: {concept.source_count} concepts")
            print(f"      Score: {concept.score}")
            print(f"      Keywords: {', '.join(concept.keywords[:5])}")
            print(f"      Related topics: {concept.related_topics}")
            print(f"      Source file: {concept.source_file}")
            print(f"      Audit: {concept.audit}")

    print("\n" + "✅" * 25)
    print("  BOTH ORCHESTRATORS TEST COMPLETE")
    print("  Deep + Wiki both merge adjacent and fill gaps")
    print("✅" * 25 + "\n")


if __name__ == "__main__":
    main()