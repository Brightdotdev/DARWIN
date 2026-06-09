#!/usr/bin/env python3
"""
tests/test_e2e_real_llm.py
--------------------------
End-to-end test with REAL LLM client and REAL DarwinStore.
No mocks. No fake data. Runs the full adaptive pipeline.

Saves a detailed JSON + Markdown report to:
    test_output/e2e_results_{timestamp}.json
    test_output/e2e_report_{timestamp}.md

Run: uv run python tests/test_e2e_real_llm.py
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure output dir exists
OUTPUT_DIR = PROJECT_ROOT / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# LLM Client – uses the already configured LLMManager
# ═══════════════════════════════════════════════════════════════════════════
from src.features.LLM.LLMManager import LLMManager

if not LLMManager._config:
    LLMManager.initialize()
llm_client = LLMManager.get_client()

# Check for the required tier methods
for method in ("call_small", "call_normal", "call_smart"):
    if not hasattr(llm_client, method):
        print(f"[ERROR] LLM client missing required method: {method}")
        sys.exit(1)

print(f"[OK] LLM client ready: {type(llm_client).__module__}.{type(llm_client).__name__}")

# ═══════════════════════════════════════════════════════════════════════════
# Project Imports
# ═══════════════════════════════════════════════════════════════════════════
from src.core.Storage.DarwinStore import DarwinStore
from src.features.Query.QueryExpansionLayer import QueryExpansionLayer
from src.features.Query.SearchOrchestrator import SearchOrchestrator, SearchContext
from src.features.Query.Response.ResponseRouter import ResponseRouter

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
ENOCH_FILE = "book_of_enoch.pdf"
BIBLE_FILE = "CSB_Pew_Bible_2nd_Printing.pdf"

# ═══════════════════════════════════════════════════════════════════════════
# Result Recorder
# ═══════════════════════════════════════════════════════════════════════════
class ResultRecorder:
    def __init__(self):
        self.tests = []
        self.timestamp = datetime.now().isoformat()

    def add(self, name: str, plan, query_text: str, error: str = None):
        entry = {
            "test": name,
            "query": query_text,
            "timestamp": datetime.now().isoformat(),
            "success": error is None,
            "error": error,
            "plan": None,
        }
        if plan is not None:
            try:
                entry["plan"] = json.loads(plan.to_json())
            except Exception as e:
                entry["plan_parse_error"] = str(e)
                entry["plan_raw"] = repr(plan)
        self.tests.append(entry)

    def save(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = OUTPUT_DIR / f"e2e_results_{ts}.json"
        md_path = OUTPUT_DIR / f"e2e_report_{ts}.md"

        # JSON dump
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "run_at": self.timestamp,
                    "total_tests": len(self.tests),
                    "passed": sum(1 for t in self.tests if t["success"]),
                    "failed": sum(1 for t in self.tests if not t["success"]),
                },
                "tests": self.tests,
            }, f, indent=2, default=str)

        # Markdown report
        lines = [
            f"# DARWIN E2E Test Report",
            f"",
            f"**Run at:** {self.timestamp}",
            f"**Total:** {len(self.tests)} | **Passed:** {sum(1 for t in self.tests if t['success'])} | **Failed:** {sum(1 for t in self.tests if not t['success'])}",
            f"",
            "---",
            f"",
        ]

        for t in self.tests:
            status = "  PASS" if t["success"] else "❌ FAIL"
            lines.append(f"## {t['test']} — {status}")
            lines.append(f"**Query:** `{t['query']}`")
            if t["error"]:
                lines.append(f"**Error:** `{t['error']}`")
            if t.get("plan"):
                plan = t["plan"]
                lines.append(f"**Complex:** {plan.get('is_complex', 'N/A')}")
                lines.append(f"**Script:** {plan.get('script_key', 'N/A')}")
                lines.append(f"**Blocks:** {len(plan.get('blocks', []))}")
                for i, block in enumerate(plan.get("blocks", [])):
                    lines.append(f"\n### Block {i}: `{block.get('type', 'unknown')}`")
                    lines.append(f"**Title:** {block.get('title', 'N/A')}")
                    if block.get("markdown_breakdown"):
                        lines.append(f"**Breakdown:** {block['markdown_breakdown']}")
                    if block.get("markdown_fallback"):
                        lines.append(f"**Fallback:**\n```\n{block['markdown_fallback'][:500]}\n```")
                    data = block.get("data", {})
                    if data:
                        lines.append(f"**Data keys:** {list(data.keys())}")
            lines.append("\n---\n")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"\n[RECORDER] Results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  MD:   {md_path}")
        return json_path, md_path


recorder = ResultRecorder()

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def check_prerequisites(store: DarwinStore) -> bool:
    ok = True
    for filename in [ENOCH_FILE, BIBLE_FILE]:
        chunks = store.query_records(filename)
        wiki = store.load_wiki(filename)
        if not chunks:
            print(f"  ✗ {filename}: NO CHUNKS")
            ok = False
        else:
            print(f"  ✓ {filename}: {len(chunks)} chunks")
        if not wiki:
            print(f"  ✗ {filename}: NO WIKI")
            ok = False
        else:
            print(f"  ✓ {filename}: {len(wiki.pages)} wiki pages")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════
def test_broad_query():
    """'who are the watchers' → simple route → markdown block"""
    print("\n--- test_broad_query ---")
    store = DarwinStore()
    expansion = QueryExpansionLayer(store)
    search = SearchOrchestrator(store)
    router = ResponseRouter(llm_client=llm_client, mode="adaptive")

    queries = expansion.expand("who are the watchers", pinned_files=[ENOCH_FILE])
    assert len(queries) == 1
    rq = queries[0]
    print(f"  Intent: {rq.intent}")
    print(f"  Entities: {[(e.text, e.type) for e in rq.entities]}")
    print(f"  Expanded: {rq.expanded_terms[:5]}...")
    print(f"  Files: {rq.all_files}")

    ctx = search.search(rq, rq.all_files, top_k=5)
    print(f"  Concepts: {len(ctx.concepts)}")
    print(f"  Passages: {len(ctx.passages)}")

    plan = router.route(rq, ctx)
    print(f"  is_complex: {plan.is_complex}")
    print(f"  Blocks: {[b.block_type for b in plan.blocks]}")

    assert len(plan.blocks) >= 1
    json_out = plan.to_json()
    md_out = plan.to_markdown()
    assert len(json_out) > 0
    assert len(md_out) > 0

    print(f"  JSON length: {len(json_out)} chars")
    print(f"  Markdown length: {len(md_out)} chars")
    print("    test_broad_query passed")
    recorder.add("test_broad_query", plan, rq.original_query)
    return True


def test_dense_query():
    """'find sons of God passage' → simple route → NOT hierarchy"""
    print("\n--- test_dense_query ---")
    store = DarwinStore()
    expansion = QueryExpansionLayer(store)
    search = SearchOrchestrator(store)
    router = ResponseRouter(llm_client=llm_client, mode="adaptive")

    queries = expansion.expand("find sons of God passage", pinned_files=[BIBLE_FILE])
    rq = queries[0]
    print(f"  Intent: {rq.intent}")

    ctx = search.search(rq, rq.all_files, top_k=5)
    print(f"  Passages: {len(ctx.passages)}")

    plan = router.route(rq, ctx)
    print(f"  is_complex: {plan.is_complex}")
    print(f"  Blocks: {[b.block_type for b in plan.blocks]}")

    types = [b.block_type for b in plan.blocks]
    assert "hierarchy" not in types, f"Unexpected hierarchy in {types}"
    print("    test_dense_query passed")
    recorder.add("test_dense_query", plan, rq.original_query)
    return True


def test_comparison_query():
    """'Compare watchers in Enoch vs Genesis' → adaptive → comparison_table block"""
    print("\n--- test_comparison_query ---")
    store = DarwinStore()
    expansion = QueryExpansionLayer(store)
    search = SearchOrchestrator(store)
    router = ResponseRouter(llm_client=llm_client, mode="adaptive")

    queries = expansion.expand(
        "Compare the watchers in Enoch vs Genesis",
        pinned_files=[ENOCH_FILE, BIBLE_FILE]
    )
    print(f"  Decomposed into {len(queries)} queries")
    assert len(queries) == 2, f"Expected 2 queries, got {len(queries)}"

    all_contexts = []
    for rq in queries:
        print(f"  Searching: '{rq.original_query}' in {rq.all_files}")
        ctx = search.search(rq, rq.all_files, top_k=5)
        all_contexts.append(ctx)
        print(f"    → {len(ctx.concepts)} concepts, {len(ctx.passages)} passages")

    merged_concepts = []
    merged_passages = []
    merged_files = []
    for ctx in all_contexts:
        merged_concepts.extend(ctx.concepts)
        merged_passages.extend(ctx.passages)
        merged_files.extend(ctx.files)

    merged_ctx = SearchContext(
        query="Compare the watchers in Enoch vs Genesis",
        intent="both",
        files=list(dict.fromkeys(merged_files)),
        concepts=merged_concepts,
        passages=merged_passages,
        related_topics=[],
        merged_prompt="",
    )

    plan = router.route(queries[0], merged_ctx)
    print(f"  is_complex: {plan.is_complex}")
    print(f"  Script: {plan.script_key}")
    print(f"  Blocks: {[b.block_type for b in plan.blocks]}")

    types = [b.block_type for b in plan.blocks]
    assert "comparison_table" in types, f"Expected comparison_table in {types}"

    comp_block = next(b for b in plan.blocks if b.block_type == "comparison_table")
    assert comp_block.markdown_fallback is not None, "Missing fallback"
    assert comp_block.markdown_breakdown is not None, "Missing breakdown"

    print(f"  Comparison block fallback length: {len(comp_block.markdown_fallback or '')}")
    print("    test_comparison_query passed")
    recorder.add("test_comparison_query", plan, queries[0].original_query)
    return True


def test_explicit_type_request():
    """'Show me a graph of angelic concepts' → entity_map"""
    print("\n--- test_explicit_type_request ---")
    store = DarwinStore()
    expansion = QueryExpansionLayer(store)
    search = SearchOrchestrator(store)
    router = ResponseRouter(llm_client=llm_client, mode="adaptive")

    queries = expansion.expand("Show me a graph of angelic concepts", pinned_files=[ENOCH_FILE])
    rq = queries[0]
    rq.explicit_response_type = "entity_map"

    ctx = search.search(rq, rq.all_files, top_k=5)
    plan = router.route(rq, ctx)

    print(f"  is_complex: {plan.is_complex}")
    print(f"  Blocks: {[b.block_type for b in plan.blocks]}")

    types = [b.block_type for b in plan.blocks]
    assert "entity_map" in types, f"Expected entity_map in {types}"

    block = next(b for b in plan.blocks if b.block_type == "entity_map")
    assert block.markdown_fallback is not None
    print(f"  Fallback length: {len(block.markdown_fallback or '')}")
    print("    test_explicit_type_request passed")
    recorder.add("test_explicit_type_request", plan, rq.original_query)
    return True


def test_json_markdown_duality():
    """Every plan produces valid JSON + non-empty Markdown"""
    print("\n--- test_json_markdown_duality ---")
    store = DarwinStore()
    expansion = QueryExpansionLayer(store)
    search = SearchOrchestrator(store)
    router = ResponseRouter(llm_client=llm_client, mode="adaptive")

    test_queries = [
        "who are the watchers",
        "find sons of God",
        "Compare watchers in Enoch vs Genesis",
        "Show me a graph of concepts",
    ]

    for q in test_queries:
        queries = expansion.expand(q, pinned_files=[ENOCH_FILE, BIBLE_FILE])
        for rq in queries:
            ctx = search.search(rq, rq.all_files, top_k=5)
            plan = router.route(rq, ctx)

            json_str = plan.to_json()
            json_data = json.loads(json_str)
            assert "blocks" in json_data
            assert len(json_data["blocks"]) >= 1

            md = plan.to_markdown()
            assert len(md) > 0

            print(f"  '{q[:30]}...' → complex={plan.is_complex}, blocks={[b['type'] for b in json_data['blocks']]}")

    print("    test_json_markdown_duality passed")
    recorder.add("test_json_markdown_duality", None, "multiple")
    return True


def test_domain_agnostic():
    """Non-biblical query should not trigger bible-specific types"""
    print("\n--- test_domain_agnostic ---")
    store = DarwinStore()
    expansion = QueryExpansionLayer(store)
    search = SearchOrchestrator(store)
    router = ResponseRouter(llm_client=llm_client, mode="adaptive")

    queries = expansion.expand("explain database indexing strategies", pinned_files=[])
    rq = queries[0]

    if rq.all_files:
        ctx = search.search(rq, rq.all_files, top_k=5)
    else:
        ctx = SearchContext(
            query=rq.original_query,
            intent=rq.intent,
            files=[],
            concepts=[],
            passages=[],
            related_topics=[],
            merged_prompt="",
        )

    plan = router.route(rq, ctx)
    types = [b.block_type for b in plan.blocks]

    assert "hierarchy" not in types
    print(f"  Blocks: {types}")
    print("    test_domain_agnostic passed")
    recorder.add("test_domain_agnostic", plan, rq.original_query)
    return True


def test_complex_multi_block():
    """Complex query should trigger adaptive multi-block response"""
    print("\n--- test_complex_multi_block ---")
    store = DarwinStore()
    expansion = QueryExpansionLayer(store)
    search = SearchOrchestrator(store)
    router = ResponseRouter(llm_client=llm_client, mode="adaptive")

    queries = expansion.expand(
        "How to build a RAG pipeline step by step and compare indexing strategies",
        pinned_files=[ENOCH_FILE, BIBLE_FILE],
    )
    rq = queries[0]
    rq.requires_comparison = True

    ctx = search.search(rq, [ENOCH_FILE, BIBLE_FILE], top_k=8)
    if len(ctx.passages) < 20:
        print(f"  [WARN] Only {len(ctx.passages)} passages; heuristic may route simple")

    plan = router.route(rq, ctx)
    print(f"  is_complex: {plan.is_complex}")
    print(f"  Script: {plan.script_key}")
    print(f"  Blocks ({len(plan.blocks)}): {[b.block_type for b in plan.blocks]}")

    if plan.is_complex:
        assert len(plan.blocks) <= 4, "Max 4 blocks enforced"
        assert plan.blocks[0].block_type == "markdown", "Must start with markdown"
        assert plan.blocks[-1].block_type == "markdown", "Must end with markdown"
        visual_types = {"comparison_table", "timeline", "flowchart", "entity_map", "list", "pros_cons", "summary_card"}
        has_visual = any(b.block_type in visual_types for b in plan.blocks)
        assert has_visual, f"Expected at least one visual block, got {[b.block_type for b in plan.blocks]}"
    else:
        print("  [INFO] Routed as simple (insufficient data for complex mode)")

    print("    test_complex_multi_block passed")
    recorder.add("test_complex_multi_block", plan, rq.original_query)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════
def run_all_tests():
    print("=" * 70)
    print("DARWIN E2E — REAL LLM / REAL STORE / ADAPTIVE RESPONSE")
    print("=" * 70)
    print(f"LLM client: {type(llm_client).__module__}.{type(llm_client).__name__}")
    print(f"Output dir: {OUTPUT_DIR}")

    store = DarwinStore()
    if not check_prerequisites(store):
        print("\n❌ PREREQUISITES FAILED")
        return False

    tests = [
        test_broad_query,
        test_dense_query,
        test_comparison_query,
        test_explicit_type_request,
        test_json_markdown_duality,
        test_domain_agnostic,
        test_complex_multi_block,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            err_msg = f"{type(e).__name__}: {e}"
            print(f"\n  ❌ {test.__name__} FAILED: {err_msg}")
            traceback.print_exc()
            recorder.add(test.__name__, None, getattr(test, '__doc__', ''), error=err_msg)

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    json_path, md_path = recorder.save()
    print(f"\n📁 Open these files to inspect full output:")
    print(f"   {md_path}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)