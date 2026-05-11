#!/usr/bin/env python3
"""
main_query.py — Interactive CLI harness for the Query → Artifact pipeline.

Usage:
    python main_query.py [--profile openai] [--no-search]

In the REPL you can:
    - Type a normal question.
    - Paste text by entering :paste, then your content, ending with a line containing only :endpaste.
    - Pin files with :pin biology.pdf, physics.pdf (comma or space separated).
    - Toggle external URL search with /search on|off.
    - Exit with /exit.

The full pipeline is executed and the final artifact JSON is printed.
"""

import sys
import asyncio
import argparse
from pathlib import Path
from typing import List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.lib.KeyManager import load_keys_from_keychain
from src.features.LLM.LLMManager import LLMManager
from src.features.Storage.DarwinStore import DarwinStore
from src.features.Query.QueryPipeline import handle_query
from src.features.Query.QueryTypes import RawQuery, PastedText
from src.core.Logger.DarwinLogger import darwin, LogType


async def interactive_loop(client, store, allow_search: bool):
    print("\n=== Darwin Query Pipeline REPL ===\n"
          "Type a question. Special commands:\n"
          "  :paste          – enter multi‑line text to attach\n"
          "  :endpaste       – finish pasting\n"
          "  :pin file1,...  – pin searchable files\n"
          "  /search on|off  – toggle URL content fetching\n"
          "  /exit           – quit\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # ── Handle special commands ────────────────────────────
        if user_input.startswith("/exit"):
            break
        if user_input.startswith("/search"):
            _, _, val = user_input.partition(" ")
            allow_search = val.strip().lower() == "on"
            print(f"URL search toggled {'ON' if allow_search else 'OFF'}")
            continue

        pasted_texts: List[PastedText] = []
        pinned_files: List[str] = []

        if user_input == ":paste":
            lines = []
            print("(Paste your text, end with :endpaste on a new line)")
            while True:
                line = input()
                if line.strip() == ":endpaste":
                    break
                lines.append(line)
            pasted_texts.append(PastedText(content="\n".join(lines), type="document"))
            # After pasting, ask for the actual query
            user_input = input("Now enter your question: ").strip()
            if not user_input:
                continue

        # Check for :pin directive inside the input (for simplicity, we'll ask separately)
        while ":pin" in user_input:
            start = user_input.find(":pin")
            end = user_input.find(" ", start)
            if end == -1:
                fnames_str = user_input[start+5:].strip()
                user_input = user_input[:start].strip()
            else:
                fnames_str = user_input[start+5:end].strip()
                user_input = (user_input[:start] + user_input[end:]).strip()
            pinned_files.extend([f.strip() for f in fnames_str.split(",") if f.strip()])

        raw = RawQuery(
            query=user_input,
            pasted_texts=pasted_texts,
            pinned_files=pinned_files,
        )

        print(f"\nProcessing: '{raw.query[:50]}...'")
        result = await handle_query(client, store, raw, allow_search=allow_search)

        if result["type"] == "artifact":
            print(f"\nArtifact type: {result['artifact_type']}")
            import json
            print(json.dumps(result["data"], indent=2))
            print(f"Confidence: {result.get('confidence', 'N/A')}")
        elif result["type"] == "follow_up":
            print("\nFollow‑up required:")
            print(result["data"]["question"])
            for opt in result["data"].get("options", []):
                print(f"  [{opt['id']}] {opt['label']}")
        elif result["type"] == "error":
            print(f"\nError: {result['message']}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Query Pipeline REPL")
    parser.add_argument("--profile", default=None, help="Default LLM provider")
    parser.add_argument("--no-search", action="store_true", help="Disable URL fetching")
    args = parser.parse_args()

    # Startup
    load_keys_from_keychain()
    LLMManager.initialize()
    if args.profile:
        LLMManager.switch_default_provider(args.profile)
    client = LLMManager.get_client()
    store = DarwinStore()          # assumes default project dir; adjust if needed

    await interactive_loop(client, store, allow_search=not args.no_search)
    return 0


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")