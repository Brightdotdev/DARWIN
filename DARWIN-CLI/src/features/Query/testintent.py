from src.features.Query.IntentAnalyzer import analyze_query_legacy, UserMessage, analyze_query


# Mock store with filenames for document scope testing
class MockStore:
    class Registry:
        def list_all(self):
            return [
                "genesis.txt",
                "physics_notes.pdf",
                "history_book.docx",
                "economics_guide.pdf"
            ]
    
    def __init__(self):
        self.registry = self.Registry()


fuzz_queries = [

    # ───────── PARAMETRIC EDGE CASES ─────────
    "What if humans were 0 feet tall?",
    "What if humans were -50 meters tall?",
    "What if humans were 1,000,000 feet tall?",
    "Suppose gravity was 9.8 times stronger, what happens?",
    "If the earth was 50 percent bigger how would oceans behave?",
    "What if there were 7.5 billion people on Mars?",
    "Hypothetically if time was 0 seconds long what happens?",
    "If it was 300 what would happen?",  # missing unit
    "What if humans were three hundred feet tall?",  # word number (regex fail)
    "Suppose humans had 2 heads and 4 arms",  # no numeric match

    # ───────── MULTI-QUESTION TRAPS ─────────
    "Who is Enoch and why did he write a book",
    "What is gravity and how does it work and why does it exist",
    "Who built the pyramids & when was it built & how",
    "What is AI plus how does it learn",
    "Explain photosynthesis and respiration",
    "Why is the sky blue and grass green and water clear",
    "Who are you and what do you do and where do you live",
    "What is love and what is life and what is death",
    "How does a car engine work and explain each part",
    "Tell me about Rome and also Greece",

    # ───────── FOLLOWUP FALSE POSITIVES ─────────
    "Explain that",
    "Clarify",
    "Go on",
    "Continue",
    "This one?",
    "That thing you said",
    "It works how?",
    "Why is that",
    "Make it clearer",
    "Break it down",

    # ───────── FOLLOWUP FALSE NEGATIVES ─────────
    "Explain quantum mechanics in detail",  # no history but looks standalone
    "Tell me more about black holes",
    "Expand on photosynthesis process",
    "Give deeper explanation of gravity",
    "Elaborate on neural networks",

    # ───────── DOCUMENT SCOPE COLLISIONS ─────────
    "In genesis what happened to enoch",
    "From the history book explain world war 2",
    "According to physics notes what is energy",
    "In the guide explain economics basics",
    "From the report what are the conclusions",
    "In the book what is chapter one about",
    "From the text explain the introduction",
    "According to the document what is AI",
    "In genesis.txt what is creation story",
    "From physics_notes.pdf explain relativity",

    # ───────── INTENT CONFUSION ─────────
    "List why the sky is blue",
    "Explain how many planets exist",
    "Find why gravity exists",
    "Show how photosynthesis works",
    "Search what causes earthquakes",
    "Tell me number of continents",
    "Give me a summary of why wars happen",
    "Compare how many planets there are",
    "Summarize and explain gravity",
    "List and explain black holes",

    # ───────── RESPONSE TYPE COLLISIONS ─────────
    "Compare and summarize world war 2",
    "How does photosynthesis work step by step",
    "Timeline and explanation of roman empire",
    "Map all relationships in the bible",
    "Show frequency and compare usage of words",
    "Walk me through and summarize the process",
    "Explain steps and compare results",
    "What if humans were 300 feet tall and compare to now",
    "Show all connections and explain them",
    "Give me a full overview and step by step breakdown",

    # ───────── ENTITY EDGE CASES ─────────
    "Tell me about Apple",  # company vs fruit
    "Who is Jordan",       # person vs country
    "Explain Mercury",     # planet vs element
    "What is Python",      # language vs snake
    "Who is Washington",   # person vs place

    # ───────── NOISE / ADVERSARIAL ─────────
    "asdfghjkl",
    "???",
    "and and and",
    "what if if if 300 feet",
    "tell me something",
    "do it",
    "yes",
    "no",
    "ok",
    "random stuff here"
]


def run_fuzz_test():
    history = [
        {"role": "user", "content": "Tell me about Enoch"},
        {"role": "assistant", "content": "Enoch is a biblical figure"},
    ]

    store = MockStore()

    for i, query in enumerate(fuzz_queries):
        # Test legacy API (string-based, backward compat)
        analysis = analyze_query_legacy(
            query=query,
            history=history,
            store=store,
            enabled_tiers=["easy", "medium"],
            message_metadata={"type": "normal_reply"},
        )

        print(f"\n[{i}] {query}")
        print(f"intent={analysis.intent} | followup={analysis.is_followup} | multi={analysis.is_multi_question}")
        print(f"parametric={analysis.is_parametric} | doc={analysis.document_scope}")
        print(f"type={analysis.response_type} | search={analysis.requires_search}")
        print(f"garbage={analysis.is_garbage} | clarify={analysis.needs_clarification}")
        print(f"blocks={analysis.suggested_blocks}")
        if analysis.parametric_seed:
            print(f"seed={analysis.parametric_seed}")
        if analysis.clarification_options:
            print(f"clarify_options={[o['label'] for o in analysis.clarification_options]}")
        print("-" * 70)


def run_structured_test():
    """Test new UserMessage API with context/attachments."""
    store = MockStore()

    # Test with large context (should bypass search)
    msg = UserMessage(
        query="what do you think?",
        context="This is a very long pasted text " * 200,  # >500 chars
        metadata={"type": "normal_reply"},
    )
    analysis = analyze_query(
        message=msg,
        history=[],
        store=store,
        enabled_tiers=["easy", "medium"],
    )
    print("\n[STRUCTURED] Large context test")
    print(f"query={analysis.query} | source={analysis.context_source} | search={analysis.requires_search}")
    print("-" * 70)

    # Test with highlight reply (followup)
    msg2 = UserMessage(
        query="Can you simplify that?",
        context="",
        metadata={"type": "highlight_reply"},
    )
    analysis2 = analyze_query(
        message=msg2,
        history=[{"role": "assistant", "content": "Previous answer..."}],
        store=store,
        enabled_tiers=["easy"],
    )
    print("\n[STRUCTURED] Highlight reply test")
    print(f"followup={analysis2.is_followup} | search={analysis2.requires_search}")
    print("-" * 70)


if __name__ == "__main__":
    run_fuzz_test()
    run_structured_test()