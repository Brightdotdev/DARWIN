#!/usr/bin/env python3
"""
battle_test.py – Expanded stress‑test for handle_message with varied inputs.
Now includes 18 test cases covering dialogs and turn types.
"""

import asyncio
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.lib.KeyManager import load_keys_from_keychain
from src.features.LLM.LLMManager import LLMManager
from src.features.Storage.DarwinStore import DarwinStore
from src.features.Query.QueryPipeline import handle_message          # new entry point
from src.features.Query.QueryTypes import RawQuery, PastedText, URLRef
from src.core.Logger.DarwinLogger import darwin, LogType, OutputMode

# ═══ Configure the logger ═══
darwin.configure(mode=OutputMode.BOTH)

TEST_CASES = [
    # ── Original standalone queries (now using handle_message with no conversation state) ──
    {
        "name": "Fact – single file",
        "raw": RawQuery(
            query="What does the Book of Enoch say about the Watchers?",
            pinned_files=["book_of_enoch.pdf"],
        ),
        "conversation_id": None,   # no prior context
    },
    {
        "name": "Comparison – two files",
        "raw": RawQuery(
            query="Compare the creation story in Book of Enoch and Genesis (bible.pdf)",
            pinned_files=["book_of_enoch.pdf", "bible.pdf"],
        ),
    },
    {
        "name": "Instruction – pasted text",
        "raw": RawQuery(
            query="Summarise the steps for the ritual described in this text.",
            pasted_texts=[PastedText(content="Then the angel Uriel said: Take the censer and offer incense...", type="document")],
        ),
    },
    {
        "name": "Hypothetical – no file",
        "raw": RawQuery(
            query="What would happen if the Nephilim returned today?",
        ),
    },
    {
        "name": "Clarification – ambiguous",
        "raw": RawQuery(
            query="Tell me about the key.",
            pinned_files=["book_of_enoch.pdf"],
        ),
    },
    {
        "name": "Multi‑part – multiple files",
        "raw": RawQuery(
            query="First, what are the names of the fallen angels? Second, what punishments did they receive?",
            pinned_files=["book_of_enoch.pdf", "ancient_texts.pdf"],
        ),
    },
    {
        "name": "URL – with extraction (allow_search=False)",
        "raw": RawQuery(
            query="What does this article say?",
            extracted_urls=[URLRef(url="https://en.wikipedia.org/wiki/Book_of_Enoch", domain="wikipedia.org", placeholder="[article]")],
        ),
        "allow_search": False,
    },
    # ── Artifact‑targeted queries ─────────────────────────────────────
    {
        "name": "Profile card",
        "raw": RawQuery(
            query="Who is Enoch in the Book of Enoch?",
            pinned_files=["book_of_enoch.pdf"],
        ),
    },
    {
        "name": "Pros/Cons",
        "raw": RawQuery(
            query="What are the pros and cons of the Watchers' rebellion?",
            pinned_files=["book_of_enoch.pdf"],
        ),
    },
    {
        "name": "Timeline",
        "raw": RawQuery(
            query="Show a timeline of events in the Book of Enoch.",
            pinned_files=["book_of_enoch.pdf"],
        ),
    },
    {
        "name": "Flowchart",
        "raw": RawQuery(
            query="Explain the process of the Watchers' fall and punishment as a flowchart.",
            pinned_files=["book_of_enoch.pdf"],
        ),
    },
    {
        "name": "Stats Grid",
        "raw": RawQuery(
            query="What are the key numbers mentioned in Book of Enoch? e.g., number of angels, years.",
            pinned_files=["book_of_enoch.pdf"],
        ),
    },
    {
        "name": "Parameter Sandbox",
        "raw": RawQuery(
            query="What if the Nephilim returned today? Let me adjust the number of them and their size.",
            pinned_files=["book_of_enoch.pdf"],
        ),
    },
    # ── Dialog edge cases (will be run in sequence) ──────────────────
    # We'll simulate a short conversation by running these sequentially
]


async def run_conversation_tests(client, store):
    """Simulate a conversation with different turn types."""
    darwin.log("\n=== Conversation Simulation ===", LogType.INFO)
    # We'll use the same conversation id throughout
    convo_id = "battle_test_convo"

    # Turn 1: new query
    raw = RawQuery(
        query="Who are the Watchers in the Book of Enoch?",
        pinned_files=["book_of_enoch.pdf"],
    )
    darwin.log("Turn 1: new_query...", LogType.INFO)
    result = await handle_message(client, store, raw,
                                  conversation_id=convo_id,
                                  allow_search=True)
    darwin.log(f"Result: {result['type']} - {result.get('artifact_type', '')}", LogType.INFO)

    # Turn 2: re_render (make it a timeline)
    raw = RawQuery(query="Show me a timeline instead.")
    darwin.log("Turn 2: re_render...", LogType.INFO)
    result = await handle_message(client, store, raw,
                                  conversation_id=convo_id,
                                  allow_search=True)
    darwin.log(f"Result: {result['type']} - {result.get('artifact_type', '')}", LogType.INFO)

    # Turn 3: elaboration (tell me more about a point)
    raw = RawQuery(query="Tell me more about the Watchers' punishment.")
    darwin.log("Turn 3: elaboration...", LogType.INFO)
    result = await handle_message(client, store, raw,
                                  conversation_id=convo_id,
                                  allow_search=True)
    darwin.log(f"Result: {result['type']} - {result.get('artifact_type', '')}", LogType.INFO)

    # Turn 4: chitchat
    raw = RawQuery(query="Thanks!")
    darwin.log("Turn 4: chitchat...", LogType.INFO)
    result = await handle_message(client, store, raw,
                                  conversation_id=convo_id,
                                  allow_search=True)
    darwin.log(f"Result: {result['type']} - {result.get('text', '')[:80]}...", LogType.INFO)


async def run_tests(profile: str):
    darwin.log("=== Battle Test Starting ===", LogType.SUCCESS)
    darwin.log("Loading API keys from keychain...", LogType.INFO)
    load_keys_from_keychain()

    darwin.log("Initializing LLMManager...", LogType.INFO)
    LLMManager.initialize()

    if profile and LLMManager._config:
        try:
            LLMManager.switch_default_provider(profile)
            darwin.log(f"Using provider: {profile}", LogType.INFO)
        except Exception as e:
            darwin.log(str(e), LogType.ERROR)
            return

    client = LLMManager.get_client()
    darwin.log("LLM client ready.", LogType.SUCCESS)

    store = DarwinStore()
    darwin.log("DarwinStore initialized.", LogType.INFO)

    # Run standalone tests
    for tc in TEST_CASES:
        darwin.log(f"\n{'='*60}", LogType.INFO)
        darwin.log(f"TEST: {tc['name']}", LogType.INFO)
        darwin.log(f"Query: {tc['raw'].query[:80]}...", LogType.INFO)
        try:
            kwargs = {"allow_search": tc.get("allow_search", True)}
            kwargs["conversation_id"] = tc.get("conversation_id")  # may be None
            darwin.log("Calling handle_message...", LogType.INFO)
            result = await handle_message(client, store, tc["raw"], **kwargs)

            if result["type"] == "artifact":
                darwin.log(f"Artifact: {result['artifact_type']}", LogType.SUCCESS)
                import json
                snippet = json.dumps(result["data"], indent=2)[:500]
                darwin.log(f"Output snippet:\n{snippet}", LogType.INFO)
            elif result["type"] == "follow_up":
                darwin.log(f"Follow‑up: {result['data']['question']}", LogType.WARNING)
            elif result["type"] == "plain_text":
                darwin.log(f"Chat: {result['text'][:200]}", LogType.INFO)
            elif result["type"] == "error":
                darwin.log(f"ERROR: {result['message']}", LogType.ERROR)
        except Exception as e:
            darwin.log(f"Test crashed: {e}", LogType.ERROR)
            import traceback
            traceback.print_exc()

    # Run the dialog simulation
    await run_conversation_tests(client, store)

    darwin.log("\n=== All test cases completed ===", LogType.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None, help="Provider to use (e.g. ollama)")
    args = parser.parse_args()
    asyncio.run(run_tests(args.profile))