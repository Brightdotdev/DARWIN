#!/usr/bin/env python3
"""
test_orchestrator_detailed.py
-----------------------------
Comprehensive test that logs every single retrieval detail into a JSON file.
Run:
    python test_orchestrator_detailed.py
"""
import sys
import json
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.Storage.DarwinStore import DarwinStore
from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator

STORE_DIR = "../../darwin_store_test"

# ----------------------------------------------------------------------
# Helper: convert RetrievalContext + inner objects to a JSON-friendly dict
# ----------------------------------------------------------------------
def _make_json_safe(obj: Any) -> Any:
    """Recursively convert dataclasses, enums, and non-serialisable types."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _make_json_safe(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if hasattr(obj, "value"):  # enums
        return obj.value
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def context_to_dict(ctx) -> Dict[str, Any]:
    """Convert a RetrievalContext (and all nested objects) to a plain dict."""
    return _make_json_safe(ctx)


# ----------------------------------------------------------------------
# Main test runner
# ----------------------------------------------------------------------
def main():
    print("\n" + "="*70)
    print("   DETAILED ORCHESTRATOR TEST – saving to test_orchestrator_results.json")
    print("="*70)

    store = DarwinStore(STORE_DIR)
    orch = RetrievalOrchestrator(store)

    results_log = {}          # query description -> full context dict
    test_index = 0

    # We'll store a list of test cases (description, kwargs, files, query, method)
    test_cases = [
        ("basic_hybrid_single_file", {"filter_structural": False, "merge_sources": False, "fill_gaps": False, "top_k": 5},
         "book_of_enoch.pdf", "hybrid", "who are the watchers"),
        ("structural_penalty_on", {"filter_structural": True, "merge_sources": False, "fill_gaps": False, "top_k": 5},
         "NVIDIA-Nemotron-3-Super-Technical-Report.pdf", "hybrid", "introduction"),
        ("gap_filling_and_merge", {"filter_structural": False, "merge_sources": True, "fill_gaps": True, "top_k": 5},
         "book_of_enoch.pdf", "hybrid", "what is azazel"),
        ("multi_file_merge", {"filter_structural": False, "merge_sources": True, "fill_gaps": True, "top_k": 6},
         ["cursor_paper.pdf", "NVIDIA-Nemotron-3-Super-Technical-Report.pdf"], "hybrid", "artificial intelligence"),
        ("keyword_retrieval", {"filter_structural": False, "merge_sources": False, "fill_gaps": False, "top_k": 5},
         "book_of_enoch.pdf", "keyword", "watchers"),
        ("semantic_retrieval", {"filter_structural": False, "merge_sources": False, "fill_gaps": False, "top_k": 5},
         "book_of_enoch.pdf", "semantic", "giants"),
        ("fallback_check", {"filter_structural": True, "merge_sources": False, "fill_gaps": False, "top_k": 5},
         "book_of_enoch.pdf", "keyword", "the"),
        ("truncation_check", {"filter_structural": False, "merge_sources": False, "fill_gaps": False, "top_k": 3, "max_chars": 1200},
         "CSB_Pew_Bible_2nd_Printing.pdf", "hybrid", "history"),
        ("multi_file_no_merge", {"filter_structural": False, "merge_sources": False, "fill_gaps": False, "top_k": 5},
         ["sumsum.pdf", "cursor_paper.pdf"], "hybrid", "science"),
        ("high_topk_body_only", {"filter_structural": True, "merge_sources": True, "fill_gaps": True, "top_k": 8},
         "book_of_enoch.pdf", "hybrid", "angel"),
    ]

    for desc, kwargs, files, method, query in test_cases:
        test_index += 1
        print(f"[{test_index}/{len(test_cases)}] Running: {desc}")

        ctx = orch.retrieve_and_resolve(
            query=query,
            files=files,
            method=method,
            **kwargs
        )

        # Build a rich dict representation
        ctx_dict = context_to_dict(ctx)

        # Also store the LLM prompt for human inspection
        prompt = orch.build_llm_prompt(ctx)
        ctx_dict["llm_prompt"] = prompt

        # Embed some metadata about the test run
        entry = {
            "test_id": test_index,
            "description": desc,
            "method": method,
            "query": query,
            "files": files if isinstance(files, list) else [files],
            "parameters": kwargs,
            "context": ctx_dict,
        }

        results_log[f"{test_index:02d}_{desc}"] = entry

    # Save to JSON
    output_path = Path("test_orchestrator_results.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)

    print(f"\n  All results saved to {output_path.resolve()}")
    print("   Open the file to inspect every passage, score, penalty, and audit detail.")


if __name__ == "__main__":
    main()