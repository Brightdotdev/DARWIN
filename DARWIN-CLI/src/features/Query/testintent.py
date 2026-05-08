"""
battletest_QueryModule.py

Run: python battletest_QueryModule.py

Tests QueryModule with a diverse set of queries designed to break it.
Each test prints its result and any failures are counted at the end.
"""

import sys
import traceback
from dataclasses import asdict

# Make sure we can import from src
sys.path.insert(0, ".")

from src.features.Query.QueryRouter import organize_query, OrganizedQuery

# ---------------------------------------------------------------------------
# Test framework (tiny, self-contained)
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0
ERROR = 0

def test(name: str, fn):
    """Run a single test function and print the result."""
    global PASS, FAIL, ERROR
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        fn()
        PASS += 1
        print("  ✅ PASS")
    except AssertionError as e:
        FAIL += 1
        print(f"  ❌ FAIL: {e}")
        traceback.print_exc()
    except Exception as e:
        ERROR += 1
        print(f"  💥 ERROR: {e}")
        traceback.print_exc()


def assert_eq(actual, expected, msg=""):
    """Assert equality with a message."""
    if actual != expected:
        raise AssertionError(f"{msg}\n     Expected: {expected!r}\n     Got:      {actual!r}")


def assert_true(condition, msg=""):
    if not condition:
        raise AssertionError(f"Expected truthy, got {condition!r}. {msg}")


def assert_false(condition, msg=""):
    if condition:
        raise AssertionError(f"Expected falsy, got {condition!r}. {msg}")


def assert_in(item, container, msg=""):
    if item not in container:
        raise AssertionError(f"{item!r} not found in {container!r}. {msg}")


def assert_not_in(item, container, msg=""):
    if item in container:
        raise AssertionError(f"{item!r} found in {container!r}. {msg}")


def print_result(result: OrganizedQuery):
    """Pretty-print an OrganizedQuery for debugging."""
    d = asdict(result)
    for k, v in d.items():
        # Truncate long strings
        if isinstance(v, str) and len(v) > 100:
            v = v[:100] + "..."
        print(f"    {k}: {v}")


# ===================================================================
# Test cases
# ===================================================================

# --- URL Extraction ---

def test_url_inline_and_linked():
    result = organize_query(
        query="check https://example.com and https://example.com",
        linked_urls=["https://other.com"],
    )
    print_result(result)
    assert_true(result.has_urls)
    assert_eq(len(result.extracted_urls), 2)
    assert_in("https://example.com", result.extracted_urls)
    assert_in("https://other.com", result.extracted_urls)
    assert_eq(result.clean_query.count("[link]"), 2)

def test_url_no_protocol_not_detected():
    result = organize_query(query="check example.com for info")
    print_result(result)
    assert_false(result.has_urls)
    assert_in("example.com", result.clean_query)

def test_url_with_query_params():
    result = organize_query(query="see https://example.com/path?q=foo&bar=baz#section")
    print_result(result)
    assert_eq(len(result.extracted_urls), 1)
    assert_in("[link]", result.clean_query)

def test_url_at_end_of_sentence():
    result = organize_query(query="Visit https://example.com. Then continue.")
    print_result(result)
    assert_true(result.has_urls)
    # URL should not end with period
    for u in result.extracted_urls:
        assert_false(u.endswith("."), f"URL {u} ends with period")

def test_empty_query():
    result = organize_query(query="")
    print_result(result)
    assert_eq(result.clean_query, "")
    assert_false(result.has_urls)

# --- Pasted Text Classification ---

def test_pasted_code_python():
    code = "def foo():\n    x = 1\n    return x\n\nclass Bar:\n    pass\n"
    result = organize_query(query="explain this", pasted_text=code)
    print_result(result)
    assert_eq(result.pasted_text_type, "code")
    assert_eq(result.pasted_text, code)

def test_pasted_code_javascript():
    code = "const fn = () => {\n    return data.map(x => x.id);\n};\n"
    result = organize_query(query="what does this do", pasted_text=code)
    print_result(result)
    assert_eq(result.pasted_text_type, "code")

def test_pasted_data_csv():
    data = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCarol,35,Chicago\n"
    result = organize_query(query="analyze", pasted_text=data)
    print_result(result)
    assert_eq(result.pasted_text_type, "data")

def test_pasted_data_tsv():
    data = "name\tage\tcity\nAlice\t30\tNYC\nBob\t25\tLA\n"
    result = organize_query(query="analyze", pasted_text=data)
    print_result(result)
    assert_eq(result.pasted_text_type, "data")

def test_pasted_long_question():
    long_q = "What is the meaning of life? " * 10
    result = organize_query(query="", pasted_text=long_q)
    print_result(result)
    assert_eq(result.pasted_text_type, "long_question")

def test_short_pasted_appended():
    result = organize_query(query="hello", pasted_text="world")
    print_result(result)
    assert_in("world", result.clean_query)
    assert_true(result.has_pasted_text)

def test_barely_over_300_not_appended():
    text = "A" * 301
    result = organize_query(query="what is this", pasted_text=text)
    print_result(result)
    assert_eq(result.pasted_text_type, "document")
    assert_not_in("AAA", result.clean_query)  # pasted not merged into query

def test_document_fallback():
    text = "The quick brown fox jumps over the lazy dog. " * 10
    result = organize_query(query="summarize", pasted_text=text)
    print_result(result)
    assert_eq(result.pasted_text_type, "document")

def test_null_pasted_text():
    result = organize_query(query="hello", pasted_text=None)
    print_result(result)
    assert_false(result.has_pasted_text)
    assert_eq(result.pasted_text_type, None)

# --- Vague Reference Detection ---

def test_basic_vague_reference():
    result = organize_query(query="what is in the doc?")
    print_result(result)
    assert_true(result.needs_clarification)
    assert_true(result.vague_reference is not None)
    assert_in("vague_document_reference", result.clarification_reason)

def test_specific_file_not_vague():
    result = organize_query(query="summarize the doc Plan")
    print_result(result)
    assert_false(result.needs_clarification)

def test_that_file_is_vague():
    result = organize_query(query="that file has errors")
    print_result(result)
    assert_true(result.needs_clarification)
    assert_in("file", result.vague_reference.lower())

def test_vague_resolved_by_pinned():
    result = organize_query(query="summarize the document", pinned_file_ids=["f1"])
    print_result(result)
    assert_false(result.needs_clarification)

def test_vague_resolved_by_session():
    result = organize_query(query="open the spreadsheet", session_state={"pinned_files": ["f1"]})
    print_result(result)
    assert_false(result.needs_clarification)

def test_a_doc_not_vague():
    result = organize_query(query="I need a document")
    print_result(result)
    assert_false(result.needs_clarification)

# --- Scope & Clarification ---

def test_url_no_search_triggers_clarification():
    result = organize_query(query="check https://example.com", search_enabled=False)
    print_result(result)
    assert_true(result.needs_clarification)
    assert_eq(result.clarification_reason, "url_found_no_search")

def test_url_with_search_no_clarification():
    result = organize_query(query="check https://example.com", search_enabled=True)
    print_result(result)
    assert_false(result.needs_clarification)

def test_no_scope_at_all():
    result = organize_query(query="tell me about mars")
    print_result(result)
    assert_true(result.needs_clarification)
    assert_in("no_scope", result.clarification_reason)

def test_pasted_provides_scope():
    result = organize_query(query="summarize", pasted_text="A" * 400)
    print_result(result)
    assert_false(result.needs_clarification)

def test_context_slot_provides_scope():
    result = organize_query(query="explain", session_state={"context_slot": "some info"})
    print_result(result)
    assert_false(result.needs_clarification)

# --- Followup Short-circuit ---

def test_followup_skips_all():
    result = organize_query(query="yes use the doc", is_reply_to_followup=True)
    print_result(result)
    assert_true(result.is_followup)
    assert_false(result.needs_clarification)
    assert_eq(result.candidate_type, "plain_text")

def test_followup_still_processes_urls():
    result = organize_query(query="yes https://x.com", is_reply_to_followup=True)
    print_result(result)
    assert_true(result.has_urls)

# --- Question Type ---

def test_wh_question():
    result = organize_query(query="what is AI", search_enabled=True)
    print_result(result)
    assert_eq(result.question_type, "WH")

def test_yn_question():
    result = organize_query(query="Is the earth round", search_enabled=True)
    print_result(result)
    assert_eq(result.question_type, "YN")

def test_imperative():
    result = organize_query(query="explain gravity", search_enabled=True)
    print_result(result)
    assert_eq(result.question_type, "imperative")

def test_declarative():
    result = organize_query(query="I need info about gravity", search_enabled=True)
    print_result(result)
    assert_eq(result.question_type, "declarative")

# --- Multi-Question ---

def test_multi_question_true():
    result = organize_query(query="what is AI? how does it work?", search_enabled=True)
    print_result(result)
    assert_true(result.multi_question)

def test_single_question():
    result = organize_query(query="what is AI", search_enabled=True)
    print_result(result)
    assert_false(result.multi_question)

def test_three_questions():
    result = organize_query(query="what? who? when?", search_enabled=True)
    print_result(result)
    assert_true(result.multi_question)

# --- Temporal Signals ---

def test_date_entity_extracted():
    result = organize_query(query="revenue in 2020 and 2021", search_enabled=True)
    print_result(result)
    assert_true(len(result.temporal_signals) > 0)

def test_timeline_keyword():
    result = organize_query(query="show timeline of events", search_enabled=True)
    print_result(result)
    assert_in("timeline", result.temporal_signals)

# --- Comparison Subjects ---

def test_compare_two():
    result = organize_query(query="compare Japan and India", search_enabled=True)
    print_result(result)
    assert_true(result.comparison_subjects is not None)

def test_vs_keyword():
    result = organize_query(query="Python vs JavaScript", search_enabled=True)
    print_result(result)
    assert_true(result.comparison_subjects is not None)

def test_no_comparison():
    result = organize_query(query="tell me about Japan", search_enabled=True)
    print_result(result)
    assert_eq(result.comparison_subjects, None)

# --- Candidate Type Detection ---

def test_summary_high():
    result = organize_query(query="summarize the key points", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "summary_card")
    assert_eq(result.candidate_confidence, "high")

def test_tldr():
    result = organize_query(query="tldr of this article", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "summary_card")

def test_stats_grid():
    result = organize_query(query="how many users signed up", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "stats_grid")

def test_accordion():
    result = organize_query(query="what is AI? how does it work?", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "accordion")

def test_flowchart():
    result = organize_query(query="how does photosynthesis work", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "flowchart")

def test_timeline():
    result = organize_query(query="history of the internet timeline", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "timeline")

def test_bar_chart():
    result = organize_query(query="how often do errors occur by module", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "bar_chart")

def test_entity_map():
    result = organize_query(query="relationship between CEO and board", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "entity_map")

def test_network_view():
    result = organize_query(query="what connects to payment service", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "network_view")

def test_stepper_high():
    result = organize_query(query="walk me through the setup", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "stepper")
    assert_eq(result.candidate_confidence, "high")

def test_parametric_high():
    result = organize_query(query="what if we increase price by 20%", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "parametric")
    assert_eq(result.candidate_confidence, "high")

def test_parametric_low():
    result = organize_query(query="what if we change pricing", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "parametric")
    assert_eq(result.candidate_confidence, "low")

def test_fallback_plain_text():
    result = organize_query(query="interesting stuff", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "plain_text")
    assert_eq(result.candidate_confidence, "low")

# --- Root Verb ---

def test_root_verb_explain():
    result = organize_query(query="explain neural networks", search_enabled=True)
    print_result(result)
    assert_eq(result.root_verb, "explain")

def test_root_verb_compare():
    result = organize_query(query="compare two products", search_enabled=True)
    print_result(result)
    assert_eq(result.root_verb, "compare")

# --- Entities ---

def test_entities_present():
    result = organize_query(query="Apple revenue in 2023", search_enabled=True)
    print_result(result)
    # At least one entity expected
    assert_true(len(result.entities) > 0)

def test_no_entities():
    result = organize_query(query="hello world", search_enabled=True)
    print_result(result)
    assert_eq(len(result.entities), 0)

# --- Retrieval Hint ---

def test_hint_removes_stopwords():
    result = organize_query(query="what is the capital of France", search_enabled=True)
    print_result(result)
    hint = result.retrieval_hint
    assert_not_in("what", hint)
    assert_not_in("is", hint)
    assert_not_in("the", hint)
    assert_in("capital", hint)
    assert_in("france", hint)

def test_hint_includes_comparison():
    result = organize_query(query="compare Japan and India GDP", search_enabled=True)
    print_result(result)
    hint = result.retrieval_hint.lower()
    assert_in("japan", hint)
    assert_in("india", hint)

# --- Adversarial Inputs ---

def test_very_long_query():
    long_q = "what is " + "very " * 5000 + "long?"
    result = organize_query(query=long_q, search_enabled=True)
    print_result(result)
    assert_true(result.clean_query is not None)

def test_emoji_query():
    result = organize_query(query="🤔📊📈", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "plain_text")

def test_sql_injection_style():
    result = organize_query(query="SELECT * FROM docs WHERE topic='AI'", search_enabled=True)
    print_result(result)
    assert_true(result.clean_query is not None)

def test_repeated_punctuation():
    result = organize_query(query="what???? how??? when???", search_enabled=True)
    print_result(result)
    assert_true(result.multi_question)

def test_whitespace_only():
    result = organize_query(query="   \n\t   ", search_enabled=True)
    print_result(result)
    assert_eq(result.clean_query.strip(), "")

def test_url_as_entire_query():
    result = organize_query(query="https://example.com")
    print_result(result)
    assert_true(result.has_urls)
    assert_in("[link]", result.clean_query)

# --- Priority Ordering ---

def test_stepper_beats_flowchart():
    result = organize_query(query="walk me through how the system works", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "stepper")

def test_parametric_beats_stats():
    result = organize_query(query="what if we change price by 10% and how often to review", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "parametric")

def test_timeline_beats_summary():
    result = organize_query(query="summarize the timeline of events", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "timeline")

def test_comparison_beats_stats():
    result = organize_query(query="compare how many users in Japan vs India", search_enabled=True)
    print_result(result)
    assert_eq(result.candidate_type, "comparison_table")

# --- Edge Cases ---

def test_all_paths_populate_every_field():
    """Verify no NoneTypeAttribute errors downstream by checking all fields exist."""
    scenarios = [
        organize_query(query="hi", search_enabled=True),  # normal
        organize_query(query="the doc"),  # clarification
        organize_query(query="ok", is_reply_to_followup=True),  # followup
        organize_query(query="hi", pasted_text="A" * 400, search_enabled=True),  # pasted
    ]
    required_fields = [
        "clean_query", "entities", "root_verb", "question_type",
        "temporal_signals", "multi_question", "comparison_subjects",
        "has_pasted_text", "pasted_text", "pasted_text_type",
        "has_urls", "extracted_urls", "search_enabled", "pinned_file_ids",
        "is_followup", "needs_clarification", "clarification_reason",
        "vague_reference", "candidate_type", "candidate_confidence",
        "retrieval_hint",
    ]
    for i, result in enumerate(scenarios):
        for field in required_fields:
            assert_true(
                hasattr(result, field),
                f"Scenario {i}: missing field '{field}'"
            )

def test_vague_with_url_and_no_search():
    """Vague doc reference + URL + no search -> vague wins (priority)."""
    result = organize_query(query="the doc at https://x.com", search_enabled=False)
    print_result(result)
    assert_true(result.needs_clarification)
    assert_in("vague_document_reference", result.clarification_reason)

def test_pasted_code_with_comparison_query():
    code = "def foo():\n    x = 1\n    y = 2\n    return x + y\n" * 20  # make it > 300 chars
    result = organize_query(query="compare Python and JavaScript", pasted_text=code, search_enabled=True)
    print_result(result)
    assert_eq(result.pasted_text_type, "code")
    assert_true(result.comparison_subjects is not None)
    assert_eq(result.candidate_type, "comparison_table")

def test_vague_trumps_scope_check():
    """Vague reference should trigger before scope check even when scope exists."""
    result = organize_query(query="what does the doc say", search_enabled=True)
    # search_enabled=True provides scope, but vague ref is checked first
    print_result(result)
    assert_true(result.needs_clarification)
    assert_in("vague_document_reference", result.clarification_reason)

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    tests = [
        # URL extraction
        ("URL inline + linked deduplication", test_url_inline_and_linked),
        ("URL no protocol not detected", test_url_no_protocol_not_detected),
        ("URL with query params", test_url_with_query_params),
        ("URL at end of sentence", test_url_at_end_of_sentence),
        ("Empty query", test_empty_query),
        # Pasted text
        ("Pasted code (Python)", test_pasted_code_python),
        ("Pasted code (JavaScript)", test_pasted_code_javascript),
        ("Pasted data (CSV)", test_pasted_data_csv),
        ("Pasted data (TSV)", test_pasted_data_tsv),
        ("Pasted long question", test_pasted_long_question),
        ("Short pasted text appended", test_short_pasted_appended),
        ("Barely over 300 not appended", test_barely_over_300_not_appended),
        ("Document fallback", test_document_fallback),
        ("Null pasted text", test_null_pasted_text),
        # Vague reference
        ("Basic vague reference", test_basic_vague_reference),
        ("Specific file not vague", test_specific_file_not_vague),
        ("'that file' is vague", test_that_file_is_vague),
        ("Vague resolved by pinned", test_vague_resolved_by_pinned),
        ("Vague resolved by session", test_vague_resolved_by_session),
        ("'a doc' not vague", test_a_doc_not_vague),
        # Scope & clarification
        ("URL no search -> clarification", test_url_no_search_triggers_clarification),
        ("URL with search -> no clarification", test_url_with_search_no_clarification),
        ("No scope at all", test_no_scope_at_all),
        ("Pasted provides scope", test_pasted_provides_scope),
        ("Context slot provides scope", test_context_slot_provides_scope),
        # Followup
        ("Followup skips all", test_followup_skips_all),
        ("Followup processes URLs", test_followup_still_processes_urls),
        # Question type
        ("WH question", test_wh_question),
        ("YN question", test_yn_question),
        ("Imperative", test_imperative),
        ("Declarative", test_declarative),
        # Multi-question
        ("Multi-question true", test_multi_question_true),
        ("Single question", test_single_question),
        ("Three questions", test_three_questions),
        # Temporal
        ("Date entity extracted", test_date_entity_extracted),
        ("Timeline keyword", test_timeline_keyword),
        # Comparison
        ("Compare two", test_compare_two),
        ("VS keyword", test_vs_keyword),
        ("No comparison", test_no_comparison),
        # Candidate types
        ("Summary high confidence", test_summary_high),
        ("TLDR", test_tldr),
        ("Stats grid", test_stats_grid),
        ("Accordion", test_accordion),
        ("Flowchart", test_flowchart),
        ("Timeline", test_timeline),
        ("Bar chart", test_bar_chart),
        ("Entity map", test_entity_map),
        ("Network view", test_network_view),
        ("Stepper high", test_stepper_high),
        ("Parametric high", test_parametric_high),
        ("Parametric low", test_parametric_low),
        ("Fallback plain text", test_fallback_plain_text),
        # Root verb
        ("Root verb explain", test_root_verb_explain),
        ("Root verb compare", test_root_verb_compare),
        # Entities
        ("Entities present", test_entities_present),
        ("No entities", test_no_entities),
        # Retrieval hint
        ("Hint removes stopwords", test_hint_removes_stopwords),
        ("Hint includes comparison", test_hint_includes_comparison),
        # Adversarial
        ("Very long query", test_very_long_query),
        ("Emoji query", test_emoji_query),
        ("SQL injection style", test_sql_injection_style),
        ("Repeated punctuation", test_repeated_punctuation),
        ("Whitespace only", test_whitespace_only),
        ("URL as entire query", test_url_as_entire_query),
        # Priority ordering
        ("Stepper beats flowchart", test_stepper_beats_flowchart),
        ("Parametric beats stats", test_parametric_beats_stats),
        ("Timeline beats summary", test_timeline_beats_summary),
        ("Comparison beats stats", test_comparison_beats_stats),
        # Edge cases
        ("All fields populated", test_all_paths_populate_every_field),
        ("Vague with URL no search", test_vague_with_url_and_no_search),
        ("Pasted code with comparison", test_pasted_code_with_comparison_query),
        ("Vague trumps scope check", test_vague_trumps_scope_check),
    ]

    print("=" * 60)
    print("BATTLETEST: QueryModule.organize_query")
    print("=" * 60)

    for name, fn in tests:
        test(name, fn)

    total = PASS + FAIL + ERROR
    print("\n" + "=" * 60)
    print(f"RESULTS: {total} tests run")
    print(f"  ✅ {PASS} passed")
    print(f"  ❌ {FAIL} failed")
    print(f"  💥 {ERROR} errored")
    print("=" * 60)

    if FAIL > 0 or ERROR > 0:
        sys.exit(1)