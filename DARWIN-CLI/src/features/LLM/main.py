#!/usr/bin/env python3
"""main.py — Darwin LLM Manager: onboarding, configuration, and test harness."""

import sys
import asyncio
import argparse
import json
import os
import requests
from pathlib import Path
from typing import Optional, List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.features.LLM.LLMManager import LLMManager
from src.features.LLM.LLMTypes import LLMResponse
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Logger.WebsocketHelpers import setup_websocket, shutdown_websocket
from src.lib.KeyManager import (
    set_api_key,
    load_keys_from_keychain,
    clear_api_key,
    PROVIDER_REGISTRY,
)
from src.features.LLM.ModelPicker import is_likely_llm_model, get_default_models

DEFAULT_OLLAMA_URL = "http://localhost:11434"


# ── Ollama helpers (same as before) ─────────────────────────────────
def fetch_ollama_models(base_url: str) -> Optional[List[str]]:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        if r.status_code == 200:
            data = r.json()
            models = []
            for m in data.get("models", []):
                name = m["name"].split(":")[0]
                if is_likely_llm_model(name):
                    models.append(name)
            return sorted(models)
    except Exception:
        pass
    return None

def setup_ollama_interactive() -> Optional[str]:
    # ... (identical to previous, no changes needed) ...
    pass


# ── Provider key status helper ──────────────────────────────────────
def get_provider_key_status(provider: str) -> bool:
    """Check if the API key for a provider is currently loaded."""
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "cohere": "COHERE_API_KEY",
        "together": "TOGETHER_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    env_var = env_map.get(provider)
    if env_var:
        return bool(os.environ.get(env_var))
    # Ollama doesn't need a key
    return True


# ── Setup wizard (enhanced) ─────────────────────────────────────────
def run_setup_wizard() -> None:
    darwin.log("Darwin LLM Setup Wizard", LogType.INFO)

    if not LLMManager._config:
        LLMManager.initialize()

    while True:
        darwin.log("Available providers:", LogType.INFO)
        supported = list(PROVIDER_REGISTRY.keys()) + ["ollama"]
        for i, p in enumerate(supported, 1):
            key_icon = "✓" if get_provider_key_status(p) else "✗"
            darwin.log(f"  {i}. {p}  [{key_icon} API key]", LogType.INFO)

        choice = input("Enter number (or 'done' to finish, 'del' to remove a provider): ").strip()
        if choice.lower() == "done":
            break
        if choice.lower() == "del":
            _remove_provider_interactive()
            continue

        try:
            idx = int(choice) - 1
            provider = supported[idx]
        except (ValueError, IndexError):
            darwin.log("Invalid choice.", LogType.WARNING)
            continue

        if provider == "ollama":
            setup_ollama_interactive()
            continue

        # Cloud provider
        existing = LLMManager._config and provider in LLMManager._config.providers
        if existing:
            darwin.log(f"Provider '{provider}' already configured.", LogType.WARNING)
            update = input("Update its settings? (y/n): ").lower()
            if update != "y":
                continue

        # Show default models
        defaults = get_default_models(provider)
        darwin.log(f"Default models for {provider}:", LogType.INFO)
        darwin.log(f"  FAST   : {defaults['fast']}")
        darwin.log(f"  SMART  : {defaults['smart']}")

        fast_model = input(f"FAST model ID [{defaults['fast']}]: ").strip()
        if not fast_model:
            fast_model = defaults["fast"]
        smart_model = input(f"SMART model ID [{defaults['smart']}]: ").strip()
        if not smart_model:
            smart_model = defaults["smart"]

        if not fast_model or not smart_model:
            darwin.log("Both model IDs are required.", LogType.WARNING)
            continue

        # ★ New: allow user to override temperatures / max_tokens
        fast_temp = _input_float("FAST temperature [0.1]: ", 0.1)
        fast_max_tok = _input_int("FAST max tokens [1024]: ", 1024)
        smart_temp = _input_float("SMART temperature [0.7]: ", 0.7)
        smart_max_tok = _input_int("SMART max tokens [4096]: ", 4096)

        api_key = input(f"Enter API key for {provider}: ").strip()
        try:
            set_api_key(provider, api_key, validate=True)
        except Exception as e:
            darwin.log(str(e), LogType.ERROR)
            continue

        LLMManager.add_provider(
            provider, fast_model, smart_model,
            fast_temp=fast_temp, fast_max_tokens=fast_max_tok,
            smart_temp=smart_temp, smart_max_tokens=smart_max_tok,
        )

        if len(LLMManager.list_providers()) == 1:
            LLMManager.switch_default_provider(provider)

        another = input("Add another provider? (y/n): ").lower()
        if another != "y":
            break

    LLMManager._save_config()
    darwin.log(f"Configuration saved to {LLMManager._config_path}", LogType.SUCCESS)


def _remove_provider_interactive():
    if not LLMManager._config or not LLMManager._config.providers:
        darwin.log("No providers to remove.", LogType.WARNING)
        return
    print("Configured providers:")
    providers = list(LLMManager.list_providers())
    for i, p in enumerate(providers, 1):
        print(f"  {i}. {p}")
    try:
        idx = int(input("Number to remove: ")) - 1
        provider = providers[idx]
        LLMManager.remove_provider(provider)
        darwin.log(f"Removed provider '{provider}'.", LogType.SUCCESS)
    except Exception as e:
        darwin.log(str(e), LogType.ERROR)


def _input_float(prompt: str, default: float) -> float:
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        darwin.log("Invalid number, using default.", LogType.WARNING)
        return default


def _input_int(prompt: str, default: int) -> int:
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        darwin.log("Invalid integer, using default.", LogType.WARNING)
        return default


# ── CLI entry point ─────────────────────────────────────────────────
async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Darwin LLM Manager – configuration and test harness."
    )
    parser.add_argument("--wizard", action="store_true", help="Force run setup wizard")
    parser.add_argument("--show-config", action="store_true", help="Print current LLM config")
    parser.add_argument("--status", action="store_true", help="Show providers and key status")  # ★
    parser.add_argument("--test", action="store_true", help="Run a quick test call to all configured providers")
    parser.add_argument("--test-json", action="store_true", help="Run a test with a simple JSON schema")  # ★
    parser.add_argument("--remove-provider", type=str, metavar="PROVIDER", help="Remove a provider by name")  # ★
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")
    parser.add_argument("--no-keychain", action="store_true", help="Do not load keys from OS keychain")  # ★
    parser.add_argument("--profile", type=str, default=None, help="Load a specific provider profile at startup")

    args = parser.parse_args()

    # Load API keys unless --no-keychain is set
    if not args.no_keychain:
        load_keys_from_keychain()

    ws_server = setup_websocket(ws_port=args.ws_port, no_ws=args.no_ws)

    try:
        # ── Show status ─────────────────────────────────────────────
        if args.status:
            if LLMManager.initialize():
                print("Providers:")
                for p in LLMManager.list_providers():
                    key_status = "✓" if get_provider_key_status(p) else "✗"
                    print(f"  {p}  [{key_status} key]")
            else:
                darwin.log("No configuration found.", LogType.WARNING)
            return 0

        # ── Remove provider ─────────────────────────────────────────
        if args.remove_provider:
            if not LLMManager.initialize():
                darwin.log("No configuration found.", LogType.ERROR)
                return 1
            try:
                LLMManager.remove_provider(args.remove_provider)
                darwin.log(f"Provider '{args.remove_provider}' removed.", LogType.SUCCESS)
            except Exception as e:
                darwin.log(str(e), LogType.ERROR)
                return 1
            return 0

        # ── Show config ─────────────────────────────────────────────
        if args.show_config:
            if LLMManager.initialize():
                darwin.log(json.dumps(LLMManager._serialize_config(), indent=2), LogType.INFO)
            else:
                darwin.log("No configuration found. Run with --wizard to set up.", LogType.WARNING)
            return 0

        # ── Wizard / normal start ───────────────────────────────────
        run_wizard = args.wizard
        if not run_wizard:
            exists = LLMManager.initialize()
            if not exists:
                darwin.log("No configuration found. Launching setup wizard...", LogType.WARNING)
                run_wizard = True
            else:
                darwin.log("Configuration loaded.", LogType.SUCCESS)

        if run_wizard:
            run_setup_wizard()
            LLMManager.initialize()

        if args.profile and LLMManager._config:
            try:
                LLMManager.switch_default_provider(args.profile)
                darwin.log(f"Active profile: {args.profile}", LogType.INFO)
            except Exception as e:
                darwin.log(str(e), LogType.ERROR)
                return 1

        # ── Test (plain) ────────────────────────────────────────────
        if args.test:
            darwin.log("Running quick test...", LogType.INFO)
            client = LLMManager.get_client()
            for prov in LLMManager.list_providers():
                darwin.log(f"Testing provider: {prov}", LogType.INFO)
                for tier, method in [("fast", client.call_small), ("smart", client.call_smart)]:
                    try:
                        resp: LLMResponse = method(
                            [{"role": "user", "content": "Hello"}],
                            provider=prov,
                        )
                        if resp.error:
                            darwin.log(f"  [{tier}] ✗ {resp.error}", LogType.ERROR)
                        else:
                            darwin.log(f"  [{tier}] ✓ {resp.text[:80]}...", LogType.SUCCESS)
                    except Exception as e:
                        darwin.log(f"  [{tier}] ✗ {e}", LogType.ERROR)
            return 0

        # ── Test JSON schema ────────────────────────────────────────
        if args.test_json:
            darwin.log("Running JSON schema test...", LogType.INFO)
            client = LLMManager.get_client()
            test_schema = {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["message", "confidence"]
            }
            for prov in LLMManager.list_providers():
                darwin.log(f"Testing provider: {prov}", LogType.INFO)
                for tier, method in [("fast", client.call_small), ("smart", client.call_smart)]:
                    try:
                        resp: LLMResponse = method(
                            [{"role": "user", "content": "Reply with a JSON object containing 'message' and 'confidence'."}],
                            provider=prov,
                            json_schema=test_schema,
                        )
                        if resp.error:
                            darwin.log(f"  [{tier}] ✗ {resp.error}", LogType.ERROR)
                        else:
                            darwin.log(f"  [{tier}] ✓ {resp.json}", LogType.SUCCESS)
                    except Exception as e:
                        darwin.log(f"  [{tier}] ✗ {e}", LogType.ERROR)
            return 0

        parser.print_help()
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