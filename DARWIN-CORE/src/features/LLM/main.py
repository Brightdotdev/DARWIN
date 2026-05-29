#!/usr/bin/env python3
"""
main.py — Darwin LLM Manager: onboarding, configuration, and test harness.
"""

import sys
import asyncio
import argparse
import json
import os
import warnings
import requests
from pathlib import Path
from typing import Optional, List

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


from src.features.LLM.LLMManager import LLMManager
from src.features.LLM.LLMTypes import LLMResponse
from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity
from src.lib.KeyManager import (
    set_api_key,
    load_keys_from_keychain,
    clear_api_key,
    PROVIDER_REGISTRY,
)
from src.features.LLM.ModelPicker import is_likely_llm_model, get_default_models
from src.features.LLM.LLMConfig import PROVIDER_ENV_MAP

DEFAULT_OLLAMA_URL = "http://localhost:11434"


# -------------------------------------------------------------------
# Input helpers
# -------------------------------------------------------------------
def _resolve_model_input(user_input: str, models: List[str], default: str) -> str:
    """
    Accept either a raw model name or a 1-based index from a displayed list.
    This prevents '3' from being saved as a model ID when the user means
    'the 3rd item in the list above'.
    """
    user_input = user_input.strip()
    if not user_input:
        return default
    if user_input.isdigit():
        idx = int(user_input) - 1
        if 0 <= idx < len(models):
            return models[idx]
        darwin.log(f"Invalid selection '{user_input}'; using default.", LogType.WARNING)
        return default
    return user_input


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


# -------------------------------------------------------------------
# Ollama helpers
# -------------------------------------------------------------------
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


def setup_ollama_interactive() -> None:
    base_url = input(f"Ollama base URL [{DEFAULT_OLLAMA_URL}]: ").strip()
    if not base_url:
        base_url = DEFAULT_OLLAMA_URL

    models = fetch_ollama_models(base_url)
    if not models:
        darwin.log("Could not fetch models from Ollama. Is it running?", LogType.ERROR)
        return

    darwin.log("Available models:", LogType.INFO)
    for i, m in enumerate(models, 1):
        darwin.log(f"  {i}. {m}", LogType.INFO)

    fast_default = models[0] if models else ""
    smart_default = models[-1] if models else ""

    # Accept either a model name or the list number
    fast_model = _resolve_model_input(input(f"FAST model [{fast_default}]: "), models, fast_default)
    smart_model = _resolve_model_input(input(f"SMART model [{smart_default}]: "), models, smart_default)
    normal_model = _resolve_model_input(input(f"NORMAL model [{smart_model}]: "), models, smart_model)

    if not fast_model or not smart_model or not normal_model:
        darwin.log("All three model IDs are required.", LogType.WARNING)
        return

    fast_temp = _input_float("FAST temperature [0.1]: ", 0.1)
    fast_max_tok = _input_int("FAST max tokens [1024]: ", 1024)
    smart_temp = _input_float("SMART temperature [0.7]: ", 0.7)
    smart_max_tok = _input_int("SMART max tokens [4096]: ", 4096)
    normal_temp = _input_float("NORMAL temperature [0.3]: ", 0.3)
    normal_max_tok = _input_int("NORMAL max tokens [2048]: ", 2048)

    try:
        LLMManager.add_provider(
            "ollama",
            fast_model,
            smart_model,
            normal_model=normal_model,
            fast_temp=fast_temp,
            fast_max_tokens=fast_max_tok,
            smart_temp=smart_temp,
            smart_max_tokens=smart_max_tok,
            normal_temp=normal_temp,
            normal_max_tokens=normal_max_tok,
            base_url=base_url,
        )
        darwin.log("Ollama provider added.", LogType.SUCCESS)
    except ValueError as e:
        darwin.log(f"Invalid model selection: {e}", LogType.ERROR)
    except Exception as e:
        darwin.log(f"Failed to add Ollama: {e}", LogType.ERROR)


# -------------------------------------------------------------------
# Provider key status
# -------------------------------------------------------------------
def get_provider_key_status(provider: str) -> bool:
    env_var = PROVIDER_ENV_MAP.get(provider)
    if env_var:
        return bool(os.environ.get(env_var))
    return True


# -------------------------------------------------------------------
# Setup wizard
# -------------------------------------------------------------------
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

        existing = LLMManager._config and provider in LLMManager._config.providers
        if existing:
            darwin.log(f"Provider '{provider}' already configured.", LogType.WARNING)
            update = input("Update its settings? (y/n): ").lower()
            if update != "y":
                continue

        defaults = get_default_models(provider)
        fast_default = defaults["fast"]
        smart_default = defaults["smart"]
        normal_default = defaults.get("normal", smart_default)

        darwin.log(f"Default models for {provider}:", LogType.INFO)
        darwin.log(f"  FAST   : {fast_default}")
        darwin.log(f"  NORMAL : {normal_default}")
        darwin.log(f"  SMART  : {smart_default}")

        fast_model = input(f"FAST model ID [{fast_default}]: ").strip()
        if not fast_model:
            fast_model = fast_default
        smart_model = input(f"SMART model ID [{smart_default}]: ").strip()
        if not smart_model:
            smart_model = smart_default
        normal_model = input(f"NORMAL model ID [{normal_default}]: ").strip()
        if not normal_model:
            normal_model = normal_default

        if not fast_model or not smart_model or not normal_model:
            darwin.log("All three model IDs are required.", LogType.WARNING)
            continue

        fast_temp = _input_float("FAST temperature [0.1]: ", 0.1)
        fast_max_tok = _input_int("FAST max tokens [1024]: ", 1024)
        smart_temp = _input_float("SMART temperature [0.7]: ", 0.7)
        smart_max_tok = _input_int("SMART max tokens [4096]: ", 4096)
        normal_temp = _input_float("NORMAL temperature [0.3]: ", 0.3)
        normal_max_tok = _input_int("NORMAL max tokens [2048]: ", 2048)

        api_key = input(f"Enter API key for {provider}: ").strip()
        try:
            set_api_key(provider, api_key, validate=True)
        except Exception as e:
            darwin.log(str(e), LogType.ERROR)
            continue

        try:
            LLMManager.add_provider(
                provider,
                fast_model,
                smart_model,
                normal_model=normal_model,
                fast_temp=fast_temp,
                fast_max_tokens=fast_max_tok,
                smart_temp=smart_temp,
                smart_max_tokens=smart_max_tok,
                normal_temp=normal_temp,
                normal_max_tokens=normal_max_tok,
            )
        except ValueError as e:
            darwin.log(f"Configuration error: {e}", LogType.ERROR)
            continue
        except Exception as e:
            darwin.log(f"Failed to add provider: {e}", LogType.ERROR)
            continue

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


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------
async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Darwin LLM Manager – configuration and test harness."
    )
    parser.add_argument("--wizard", action="store_true", help="Force run setup wizard")
    parser.add_argument("--show-config", action="store_true", help="Print current LLM config")
    parser.add_argument("--status", action="store_true", help="Show providers and key status")
    parser.add_argument("--test", action="store_true", help="Run a quick test call to all configured providers")
    parser.add_argument("--test-json", action="store_true", help="Run a test with a simple JSON schema")
    parser.add_argument("--remove-provider", type=str, metavar="PROVIDER", help="Remove a provider by name")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket server")
    parser.add_argument("--no-keychain", action="store_true", help="Do not load keys from OS keychain")
    parser.add_argument("--profile", type=str, default=None, help="Load a specific provider profile at startup")
    parser.add_argument("--verbosity", choices=["normal", "quiet"], default="normal",
                        help="Log verbosity (default: normal)")

    args = parser.parse_args()

    if not args.no_keychain:
        load_keys_from_keychain()

    verbosity_map = {
        "quiet": Verbosity.QUIET,
        "normal": Verbosity.NORMAL,
    }
    terminal_verbosity = verbosity_map[args.verbosity]

    darwin.configure(
        enable_websocket=not args.no_ws,
        ws_port=args.ws_port,
        enable_history=True,
        verbosity_default=terminal_verbosity,
        show_banner=True,
    )

    try:
        # ── Status
        if args.status:
            if LLMManager.initialize():
                print("Providers:")
                for p in LLMManager.list_providers():
                    key_status = "✓" if get_provider_key_status(p) else "✗"
                    print(f"  {p}  [{key_status} key]")
            else:
                darwin.log("No configuration found.", LogType.WARNING)
            return 0

        # ── Remove provider
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

        # ── Show config
        if args.show_config:
            if LLMManager.initialize():
                darwin.log(json.dumps(LLMManager._serialize_config(), indent=2), LogType.INFO)
            else:
                darwin.log("No configuration found. Run with --wizard to set up.", LogType.WARNING)
            return 0

        # ── Wizard / normal start
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

        # ── Plain test (now includes normal tier)
        if args.test:
            darwin.log("Running quick test...", LogType.INFO)
            client = LLMManager.get_client()
            for prov in LLMManager.list_providers():
                darwin.log(f"Testing provider: {prov}", LogType.INFO)
                tiers = [
                    ("fast",   client.call_small),
                    ("normal", client.call_normal),
                    ("smart",  client.call_smart),
                ]
                for tier_name, method in tiers:
                    try:
                        resp: LLMResponse = method(
                            [{"role": "user", "content": "Hello"}],
                            provider=prov,
                        )
                        if resp.error:
                            darwin.log(f"  [{tier_name}] ✗ {resp.error}", LogType.ERROR)
                        else:
                            darwin.log(f"  [{tier_name}] ✓ {resp.text[:80]}...", LogType.SUCCESS)
                    except Exception as e:
                        darwin.log(f"  [{tier_name}] ✗ {e}", LogType.ERROR)
            return 0

        # ── JSON schema test
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
                tiers = [
                    ("fast",   client.call_small),
                    ("normal", client.call_normal),
                    ("smart",  client.call_smart),
                ]
                for tier_name, method in tiers:
                    try:
                        resp: LLMResponse = method(
                            [{"role": "user",
                              "content": "Reply with a JSON object containing 'message' and 'confidence'."}],
                            provider=prov,
                            json_schema=test_schema,
                        )
                        if resp.error:
                            darwin.log(f"  [{tier_name}] ✗ {resp.error}", LogType.ERROR)
                        else:
                            darwin.log(f"  [{tier_name}] ✓ {resp.json}", LogType.SUCCESS)
                    except Exception as e:
                        darwin.log(f"  [{tier_name}] ✗ {e}", LogType.ERROR)
            return 0

        # No action → print help
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
        darwin.shutdown()


if __name__ == "__main__":
    asyncio.run(main())