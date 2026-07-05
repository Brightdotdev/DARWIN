import os
import sys
import argparse
import time
import traceback

from src.agent.LLM.LLMManager import LLMManager
from src.lib.KeyManager import load_keys_from_keychain
from src.core.Logger.DarwinLogger import darwin, LogType


def test_manager():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default=None, help="Force a provider (e.g. openrouter)")
    parser.add_argument("--debug", action="store_true", help="Show full traceback on error")
    args = parser.parse_args()

    try:
        load_keys_from_keychain(log=True)
        initialized = LLMManager.initialize()
        if not initialized:
            print("\n  No LLM configuration found.")
            print("   Run: uv run main.py --wizard\n")
            sys.exit(1)

        client = LLMManager.get_client()
        config = LLMManager._config
        provider = args.provider or config.default_provider

        print(f"\n▶ Active provider: {provider}")
        prov_cfg = config.providers.get(provider)
        if prov_cfg:
            print(f"▶ Fast model:  {prov_cfg.fast.model}")
            print(f"▶ Normal model: {prov_cfg.normal.model}")
            print(f"▶ Smart model:  {prov_cfg.smart.model}")
        else:
            print(f"▶ Provider '{provider}' is not configured.")
            print(f"   Configured providers: {', '.join(config.providers.keys()) or '(none)'}\n")
            sys.exit(1)
        print()

        messages = [{"role": "user", "content": "What is a large language model?"}]

        for idx in (1, 2):
            t0 = time.perf_counter()
            try:
                resp = client.call_small(messages, provider=provider)
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"--- Call {idx} ({elapsed:.4f}s) ---")
                print(f"ERROR: {e}")
                print(f"Success: False\n")
                # Send traceback to log backend, not to terminal
                darwin.log(f"call_small crashed:\n{traceback.format_exc()}", LogType.ERROR)
                continue

            elapsed = time.perf_counter() - t0
            print(f"--- Call {idx} ({elapsed:.4f}s) ---")
            if resp.error:
                print(f"ERROR: {resp.error}")
            else:
                print(f"TEXT: {resp.text[:200]}...")
            print(f"Success: {resp.success}\n")

    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user.\n")
        sys.exit(130)
    except Exception as e:
        # Friendly user-facing message
        print(f"\n  Something went wrong: {e}")
        if args.debug:
            print("\n--- Debug traceback ---")
            traceback.print_exc()
        else:
            print("   Run with --debug to see the full traceback.")
            # Full crash info still goes to the log stream (websocket / file / history)
            darwin.log(f"Unhandled exception in testllm.py:\n{traceback.format_exc()}", LogType.ERROR)
        print()
        sys.exit(1)


if __name__ == "__main__":
    test_manager()