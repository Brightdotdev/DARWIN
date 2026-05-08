#!/usr/bin/env python3
"""
main.py — Darwin LLM Manager: onboarding, configuration, and test harness.
"""

import sys
import asyncio
import argparse
import requests
from pathlib import Path
from typing import Optional, List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.features.LLM.LLMManager import LLMManager
from src.features.LLM.KeyStore import KeyStore
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PROVIDER_CHOICES = [
    "openai", "anthropic", "google", "ollama", "groq",
    "mistral", "openrouter", "cohere", "together",
]
DEFAULT_OLLAMA_URL = "http://localhost:11434"


# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_ollama_models(base_url: str) -> Optional[List[str]]:
    """Fetch list of model names from an Ollama instance. Returns None on failure."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        if r.status_code == 200:
            data = r.json()
            return sorted(m["name"].split(":")[0] for m in data.get("models", []))
    except Exception:
        pass
    return None


def setup_ollama_interactive() -> Optional[str]:
    """
    Interactive Ollama provider setup.
    Returns the Ollama base URL if setup was successful, None if cancelled.
    """
    # 1. Try default
    models = fetch_ollama_models(DEFAULT_OLLAMA_URL)
    base_url = DEFAULT_OLLAMA_URL

    if models is None:
        darwin.log(
            f"Could not connect to Ollama at {DEFAULT_OLLAMA_URL}.",
            LogType.WARNING,
        )
        custom = input("Enter Ollama base URL (or press Enter to cancel): ").strip()
        if not custom:
            return None
        base_url = custom
        models = fetch_ollama_models(base_url)

    if models is None:
        darwin.log(
            f"Could not connect to Ollama at {base_url}.",
            LogType.ERROR,
        )
        return None

    if not models:
        darwin.log("No models found on your Ollama server.", LogType.WARNING)
        return None

    # 2. Show models and pick fast/smart
    darwin.log("Detected Ollama models:", LogType.INFO)
    for i, m in enumerate(models, 1):
        darwin.log(f"  {i}. {m}", LogType.INFO)

    try:
        fast_idx = int(input("Pick FAST model number: ")) - 1
        smart_idx = int(input("Pick SMART model number: ")) - 1
        fast_model = models[fast_idx]
        smart_model = models[smart_idx]
    except (ValueError, IndexError):
        darwin.log("Invalid selection.", LogType.ERROR)
        return None

    # 3. Register provider (Ollama does not use an API key)
    LLMManager.add_provider(
        "ollama",
        fast_model,
        smart_model,
        fast_temp=0.1,
        fast_max_tokens=1024,
        smart_temp=0.7,
        smart_max_tokens=4096,
        base_url=base_url,
    )
    return base_url


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP WIZARD
# ═══════════════════════════════════════════════════════════════════════════════

def run_setup_wizard() -> None:
    """Interactive multi‑provider onboarding wizard."""
    darwin.log("Darwin LLM Setup Wizard", LogType.INFO)
    keystore = KeyStore()
    keystore.load()

    while True:
        darwin.log("Available providers:", LogType.INFO)
        for i, p in enumerate(PROVIDER_CHOICES, 1):
            darwin.log(f"  {i}. {p}", LogType.INFO)

        choice = input("Enter number (or 'done' to finish): ").strip()
        if choice.lower() == "done":
            break

        try:
            idx = int(choice) - 1
            provider = PROVIDER_CHOICES[idx]
        except (ValueError, IndexError):
            darwin.log("Invalid choice.", LogType.WARNING)
            continue

        if provider == "ollama":
            result = setup_ollama_interactive()
            if result is None:
                continue  # back to provider list
        else:
            api_key = input(f"Enter API key for {provider}: ").strip()
            keystore.add_key(provider, api_key)

            fast_model = input("FAST model ID (e.g. gpt-4o-mini): ").strip()
            smart_model = input("SMART model ID (e.g. gpt-4o): ").strip()
            if not fast_model or not smart_model:
                darwin.log("Both model IDs are required.", LogType.WARNING)
                continue

            LLMManager.add_provider(provider, fast_model, smart_model,
                                    fast_temp=0.1, fast_max_tokens=1024,
                                    smart_temp=0.7, smart_max_tokens=4096)

        # Set as default if it's the first one added
        if len(LLMManager.list_providers()) == 1:
            LLMManager.switch_default_provider(provider)

        another = input("Add another provider? (y/n): ").lower()
        if another != "y":
            break

    LLMManager._save_config()
    darwin.log(f"Configuration saved to {LLMManager._config_path}", LogType.SUCCESS)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE REPL
# ═══════════════════════════════════════════════════════════════════════════════

def run_repl() -> None:
    """Simple interactive REPL for testing LLM calls and query routing."""
    client = LLMManager.get_client()
    darwin.log(
        "Darwin Test REPL. Type 'exit', '/switch <provider>', or '/test'.",
        LogType.INFO,
    )
    if LLMManager._config:
        darwin.log(
            f"Default provider: {LLMManager._config.default_provider}",
            LogType.INFO,
        )

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            break

        if user_input.startswith("/switch"):
            _, prov = user_input.split(maxsplit=1)
            try:
                LLMManager.switch_default_provider(prov)
                darwin.log(
                    f"Default provider changed to {prov}",
                    LogType.SUCCESS,
                )
            except Exception as e:
                darwin.log(str(e), LogType.ERROR)
            continue

        if user_input == "/test":
            darwin.log(
                "Testing fast + smart calls with 'Hello'",
                LogType.INFO,
            )
            for tier in ("fast", "smart"):
                try:
                    method = (
                        client.call_small
                        if tier == "fast"
                        else client.call_smart
                    )
                    resp = method([{"role": "user", "content": "Hello"}])
                    darwin.log(f"[{tier}] {resp}", LogType.SUCCESS)
                except Exception as e:
                    darwin.log(f"[{tier}] error: {e}", LogType.ERROR)
            continue

        from src.features.Query.IntentAnalyzer import (
            prepare_routing_prompt,
            parse_routing_response,
        )

        routing_prompt = prepare_routing_prompt(query=user_input)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(routing_prompt.user_message)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")


        raw_response = client.call_small(
            [{"role": "user", "content": routing_prompt.user_message}],
            system=routing_prompt.system_prompt,
            # provider="openai"   <-- removed; uses default provider
        )

        organized = parse_routing_response(raw_response, routing_prompt)
        print(organized.candidate_type)
        print(organized.comparison_subjects)
        print(organized.retrieval_hint)

# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Darwin LLM Manager – configuration, onboarding, and test harness."
    )
    parser.add_argument("--wizard", action="store_true", help="Force run setup wizard")
    parser.add_argument("--show-config", action="store_true", help="Print current LLM config")
    parser.add_argument("--test", action="store_true", help="Run a quick test call to all configured providers")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")
    parser.add_argument("--profile", type=str, default=None, help="Load a specific profile at startup")

    args = parser.parse_args()

    # Setup WebSocket server (optional, following the same lifecycle as ingestion)
    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        # ── Show config only ──────────────────────────────────────────────
        if args.show_config:
            if LLMManager.initialize():
                import json
                darwin.log(json.dumps(LLMManager._serialize_config(), indent=2), LogType.INFO)
            else:
                darwin.log("No configuration found. Run with --wizard to set up.", LogType.WARNING)
            return 0

        # ── Force wizard or no config? ────────────────────────────────────
        run_wizard = args.wizard
        if not run_wizard:
            exists = LLMManager.initialize()
            if not exists:
                darwin.log("No configuration found. Launching setup wizard...", LogType.WARNING)
                run_wizard = True
            else:
                darwin.log("Configuration loaded.", LogType.SUCCESS)

        if run_wizard:
            # Ensure minimal manager state for wizard
            if not LLMManager._keystore:
                LLMManager._keystore = KeyStore()
                LLMManager._keystore.load()
            run_setup_wizard()
            # Re‑initialize after wizard to load fresh config
            LLMManager.initialize()

        # ── Profile override ──────────────────────────────────────────────
        if args.profile and LLMManager._config:
            try:
                LLMManager.switch_default_provider(args.profile)
                darwin.log(f"Active profile: {args.profile}", LogType.INFO)
            except Exception as e:
                darwin.log(str(e), LogType.ERROR)
                return 1

        # ── Quick test ────────────────────────────────────────────────────
        if args.test:
            darwin.log("Running quick test...", LogType.INFO)
            client = LLMManager.get_client()
            for prov in LLMManager.list_providers():
                darwin.log(f"Testing provider: {prov}", LogType.INFO)
                for tier, method in [("fast", client.call_small), ("smart", client.call_smart)]:
                    try:
                        resp = method(
                            [{"role": "user", "content": "Hello"}],
                            provider=prov,
                        )
                        darwin.log(f"  [{tier}] ✓ {resp[:80]}...", LogType.SUCCESS)
                    except Exception as e:
                        darwin.log(f"  [{tier}] ✗ {e}", LogType.ERROR)
            return 0

        # ── Default: interactive REPL ─────────────────────────────────────
        run_repl()
        return 0

    except KeyboardInterrupt:
        darwin.log("")
        darwin.log("Goodbye!", LogType.INFO)
        return 130

    except Exception as e:
        import traceback
        darwin.log(f"Fatal error: {e}", LogType.ERROR)
        darwin.log("".join(traceback.format_exception(e)), LogType.ERROR)
        return 1

    finally:
        shutdown_websocket(ws_server)


if __name__ == "__main__":
    asyncio.run(main())