# ── Pure Python Tests (Final with Interactive Graph) ────────────────────
from src.features.Knowledge.WikiOrchestrator import WikiOrchestrator
from src.core.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType
from typing import Dict, Any, List


def display_entities_simple(entities: List[Dict[str, Any]]):
    print(f"{'Entity':<30} {'Type':<12} {'Freq':>5} {'Spread':>6} {'Salience':>8}  Example")
    print("-" * 100)
    for e in entities:
        example = e["mentions"][0]["text"][:40] if e["mentions"] else ""
        print(f"{e['canonical_name']:<30} {e['type']:<12} {e['frequency']:>5} {e['spread']:>6} {e['salience']:>8.3f}  {example}")


if __name__ == "__main__":
    darwin.configure()
    store = DarwinStore("../../darwin_store_test")
    orchestrator = WikiOrchestrator(store)

    FILE_NAMES = ["book_of_enoch.pdf", "OpenClawEbookFinal.pdf"]

    # ── Test 1: Build (or load) wiki for the first file ─────────────────
    print("\n--- Building wiki for book_of_enoch.pdf ---")
    wiki1 = orchestrator.build_wiki(FILE_NAMES[0], rerun=True)   # force fresh build

    # ── Test 2: Load existing wiki without rebuilding ─────────────────
    wiki1_loaded = orchestrator.get_wiki(FILE_NAMES[0])
    print("\n--- Entities from book_of_enoch.pdf ---")
    entities = wiki1_loaded.list_entities(sort_by="salience", limit=50)
    display_entities_simple([r.to_dict() for r in entities])

    # ── Test 3: get() returns NullWikiIndex when missing ────────────────
    wiki_null = orchestrator.get_wiki("missing.pdf")
    from src.features.Knowledge.WikiBuilder import NullWikiIndex
    if not isinstance(wiki_null, NullWikiIndex):
        darwin.log("Test 3: get() should return NullWikiIndex for missing file", LogType.ERROR)
    else:
        print("\n✓ test_get_returns_null")

    # ── Test 4: Build wiki for second file ─────────────────────────────
    print("\n--- Building wiki for OpenClawEbookFinal.pdf ---")
    wiki2 = orchestrator.build_wiki(FILE_NAMES[1], rerun=True)

    # ── Test 5: Multi-file entity listing ──────────────────────────────
    print("\n--- Combined entities from both files ---")
    all_entities = wiki2.list_entities(filenames=FILE_NAMES, sort_by="frequency")
    display_entities_simple([r.to_dict() for r in all_entities])

    # ── Test 6: Build and save the INTERACTIVE graph (with sliders) ────
    print("\n--- Building interactive entity graph for book_of_enoch.pdf ---")
    # Use the orchestrator method that saves the interactive HTML to store + local file
    success = orchestrator.build_and_save_interactive_graph(
        filename=FILE_NAMES[0],
        max_entities=100,
        title="Book of Enoch – Entity Graph"
    )
    if success:
        print("✓ Interactive graph saved to store and written to enoch_interactive.html")
    else:
        print("✗ Interactive graph build failed.")