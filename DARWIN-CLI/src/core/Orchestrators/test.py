#!/usr/bin/env python3
"""
test_query_orchestrator.py
--------------------------
Test QueryOrchestrator with deep, shallow, and combined search.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.features.Storage.DarwinStore import DarwinStore
from src.core.Knowledge.QueryOrchestrator import (
    QueryOrchestrator,
    DeepSearchOptions,
    ShallowSearchOptions,
)


FILENAME = "book_of_enoch.pdf"
STORE_DIR = None


def main():
    print("\n" + "🔗" * 30)
    print("  QUERY ORCHESTRATOR — Deep + Shallow + Combined")
    print("🔗" * 30)

    store = DarwinStore()
    orch = QueryOrchestrator(store)

    query = "who are the watchers"

    # ── DEEP ONLY ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DEEP SEARCH")
    print(f"{'='*60}")

    deep_results = orch.search_deep(
        query=query,
        files=[FILENAME],
        options=DeepSearchOptions(
            method="hybrid",
            top_k=5,
            rerank=False,  # ← DISABLED: reranker produces garbage scores
            max_chars=4000,
        ),
    )

    for r in deep_results:
        print(f"\n  Source: {r.source}")
        print(f"  Modality: {r.modality}")
        print(f"  Items: {r.total_items} | Filtered: {r.filtered_count}")
        print(f"  Scores: {r.scores}")
        for i, p in enumerate(r.passages):
            print(f"\n    [Passage {i+1}] Page {p['page_start']}-{p['page_end']}")
            print(f"    Score: {p.get('score', 0):.3f}")
            print(f"    Text: {p['text'][:120]}...")

    # ── SHALLOW ONLY ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SHALLOW SEARCH")
    print(f"{'='*60}")

    shallow_results = orch.search_shallow(
        query=query,
        files=[FILENAME],
        options=ShallowSearchOptions(
            top_k=3,
            expand_graph=True,
        ),
    )

    for r in shallow_results:
        print(f"\n  Source: {r.source}")
        print(f"  Modality: {r.modality}")
        print(f"  Concepts: {r.total_items} | Related: {r.related_topics[:3]}")
        print(f"  Scores: {r.scores}")
        for i, p in enumerate(r.passages):
            print(f"\n    [Concept {i+1}] {p['concept']}")
            print(f"    Page {p['page_start']}-{p['page_end']} | Score: {p['score']:.3f}")
            print(f"    Text: {p['text'][:120]}...")

    # ── COMBINED ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  COMBINED (Both Modalities)")
    print(f"{'='*60}")

    prompt, all_results = orch.ask(
        query=query,
        files=[FILENAME],
        deep_options=DeepSearchOptions(method="hybrid", top_k=3, rerank=False),
        shallow_options=ShallowSearchOptions(top_k=3),
    )

    print(f"\n  Total results: {len(all_results)}")
    for r in all_results:
        print(f"  - {r.source} [{r.modality}]: {r.total_items} items")

    print(f"\n{'='*60}")
    print("  LLM PROMPT PREVIEW")
    print(f"{'='*60}")
    print(prompt[:800] + "..." if len(prompt) > 800 else prompt)

    # ── MULTI-FILE (simulated) ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("  MULTI-FILE PROMPT STRUCTURE")
    print(f"{'='*60}")
    multi_prompt = orch.build_llm_prompt(all_results, query)
    lines = multi_prompt.split("\n")
    for line in lines:
        if "SOURCE:" in line or "===" in line or "DEEP" in line or "SHALLOW" in line:
            print(f"  {line}")

    print("\n" + "✅" * 30)
    print("  QUERY ORCHESTRATOR TEST COMPLETE")
    print("  Ready for cross-document synthesis:")
    print("  'Book of Enoch says X, but the Bible says Y...'")
    print("✅" * 30 + "\n")


if __name__ == "__main__":
    main()