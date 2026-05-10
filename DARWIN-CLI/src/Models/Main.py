# src/core/Models/cli.py

import argparse
import sys
from typing import Optional

from src.features.Hardware.SystemDetector import detect_system, recommend_defaults
from src.features.Hardware.DeviceManager import DeviceManager
from src.Models.ModelRegistry import ModelCategory, by_category, get_spec
from src.Models.ModelManager import ModelManager
from src.Models.ModelSetup import handle_download_error
from src.features.Storage.DarwinStore import DarwinStore
from src.lib.KeyManager import (
    set_api_key,
    clear_api_key,
    load_keys_from_keychain,
    PROVIDER_REGISTRY,
)


def handle_models(args):
    """Main CLI handler for model management commands."""
    store = DarwinStore()
    manager = ModelManager(darwin_store=store)

    if args.command == "key":
        handle_keys(args)
        return

    if args.command == "status":
        manager.status()

    elif args.command == "list":
        manager.list_models(getattr(args, "category", None))

    elif args.command == "install":
        if args.model:
            for model in args.model:
                manager.install(model, force=args.force)
                # If API model, check key availability
                spec = get_spec(model)
                if spec and spec.disk_mb == 0:  # API model
                    provider = model.split("/")[0]
                    env_var = PROVIDER_REGISTRY.get(provider, (None,))[0]
                    if env_var and not __import__("os").getenv(env_var):
                        print(f"⚠️  API key for {provider} is not set. "
                              f"Use 'python -m src.Models.cli key set {provider} YOUR_KEY' "
                              f"or the onboarding wizard.")
        elif args.category:
            manager.install_category(args.category, tier=getattr(args, "tier", "all") or "all")
        else:
            # No model or category specified — run smart setup with defaults
            ModelManager.setup()
            loader = ModelManager.get_loader()
            prefs = loader.get_preferences()
            print("\n✅ Installed recommended models:")
            for category, model_id in prefs.items():
                print(f"   {category}: {model_id}")

    elif args.command == "setup":
        if args.profile == "required":
            manager.setup_required()
        elif args.profile == "recommended":
            manager.setup_recommended()
        else:
            manager.setup_required()

    elif args.command == "uninstall":
        if not args.model:
            print("❌ Specify --model to uninstall")
            return
        for model in args.model:
            manager.uninstall(model)

    elif args.command == "upgrade":
        if args.model:
            for model in args.model:
                manager.upgrade(model)
        elif args.category:
            cat_models = by_category(args.category)
            for key in cat_models:
                if manager.model_store.is_downloaded(key):
                    manager.upgrade(key)
            print(f"✅ Upgraded all {args.category} models.")
        else:
            print("❌ Specify --model or --category to upgrade")

    elif args.command == "switch":
        if not args.category or not args.model:
            print("❌ Specify both --category and --model to switch")
            return
        manager.switch(args.category, args.model[0])

    elif args.command == "setup-custom":
        import asyncio
        from src.Models.ModelStartup import run_onboarding_async
        asyncio.run(run_onboarding_async())
        print("✅ Custom setup complete. Run 'status' to see your models.")

    elif args.command == "validate":
        from src.Models.ModelSetup import validate_model, validate_category
        system = detect_system()

        if args.model:
            for key in args.model:
                result = validate_model(key, system, manager.model_store)
                if result.can_run:
                    print(f"✅ {key} — OK")
                else:
                    print(f"❌ {key}: {', '.join(result.warnings)}")
        elif args.category:
            results = validate_category(args.category, system, manager.model_store)
            for r in results:
                if r.can_run:
                    print(f"✅ {r.model_key} — OK")
                else:
                    print(f"⚠️  {r.model_key}: {', '.join(r.warnings)}")


def handle_keys(args):
    """Handle API key subcommands."""
    # Load keys from keychain first to have current state
    load_keys_from_keychain()
    
    if args.key_command == "set":
        set_api_key(args.provider, args.key_value, validate=not args.no_validate)
        print(f"✅ API key for {args.provider} saved and activated.")
    elif args.key_command == "remove":
        clear_api_key(args.provider)
        print(f"✅ API key for {args.provider} removed.")
    elif args.key_command == "list":
        import os
        print("Providers and key status:")
        for provider, (env_var, _) in PROVIDER_REGISTRY.items():
            key_set = bool(os.getenv(env_var))
            status = "🔑 set" if key_set else "❌ missing"
            print(f"  {provider}: {status}")
    else:
        print("Unknown key command.")


def create_parser():
    """Create the argument parser for the models CLI."""
    parser = argparse.ArgumentParser(
        description="🧠 Darwin Model Manager — download, switch, and manage AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.Models.cli status
  python -m src.Models.cli install --tokenizer
  python -m src.Models.cli install --category embedding --tier light
  python -m src.Models.cli install --caption
  python -m src.Models.cli switch --category embedding --model BAAI/bge-m3
  python -m src.Models.cli upgrade --category embedding
  python -m src.Models.cli uninstall --model intfloat/e5-small-v2
  python -m src.Models.cli key set openai sk-... --no-validate
  python -m src.Models.cli key list
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- Key management ---
    key_parser = subparsers.add_parser("key", help="Manage API keys (OpenAI, Cohere, etc.)")
    key_sub = key_parser.add_subparsers(dest="key_command", required=True)

    key_set = key_sub.add_parser("set", help="Store an API key in the system keychain")
    key_set.add_argument("provider", choices=list(PROVIDER_REGISTRY.keys()),
                         help="Provider name (openai, cohere, voyage, gemini, openrouter)")
    key_set.add_argument("key_value", help="The API key string")
    key_set.add_argument("--no-validate", action="store_true",
                         help="Skip key validation test")

    key_remove = key_sub.add_parser("remove", help="Remove a stored API key")
    key_remove.add_argument("provider", choices=list(PROVIDER_REGISTRY.keys()))

    key_sub.add_parser("list", help="List all providers and key presence")

    # status
    subparsers.add_parser("status", help="Show model status and system info")

    # list
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--category", "-c", choices=[c.value for c in ModelCategory],
                             help="Filter by category")

    # install
    install_parser = subparsers.add_parser("install", help="Install models")
    install_parser.add_argument("--model", "-m", nargs="+", help="Model key(s) or model_id(s) to install")
    install_parser.add_argument("--category", "-c", choices=[c.value for c in ModelCategory],
                                help="Install all models in category")
    install_parser.add_argument("--tier", "-t", choices=["light", "medium", "large", "overkill"],
                                help="Filter category install by tier")
    install_parser.add_argument("--force", "-f", action="store_true", help="Force re-download")

    # uninstall
    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall models")
    uninstall_parser.add_argument("--model", "-m", nargs="+", required=True,
                                  help="Model key(s) or model_id(s) to uninstall")

    # upgrade
    upgrade_parser = subparsers.add_parser("upgrade", help="Upgrade models")
    upgrade_parser.add_argument("--model", "-m", nargs="+", help="Model key(s) to upgrade")
    upgrade_parser.add_argument("--category", "-c", choices=[c.value for c in ModelCategory],
                                help="Upgrade all installed models in category")

    # switch
    switch_parser = subparsers.add_parser("switch", help="Switch model in a category")
    switch_parser.add_argument("--category", "-c", required=True,
                               choices=[c.value for c in ModelCategory], help="Category to switch")
    switch_parser.add_argument("--model", "-m", nargs=1, required=True, help="New model key or model_id")

    # setup
    setup_parser = subparsers.add_parser("setup", help="Complete setup")
    setup_parser.add_argument("--profile", "-p", choices=["required", "recommended"],
                              default="required", help="Setup profile")

    # setup-custom
    subparsers.add_parser("setup-custom", help="Interactive setup with custom model selection")

    # validate
    validate_parser = subparsers.add_parser("validate", help="Check if models can run on your hardware")
    validate_parser.add_argument("--model", "-m", nargs="+", help="Model key(s) to validate")
    validate_parser.add_argument("--category", "-c", choices=[c.value for c in ModelCategory],
                                 help="Validate all models in category")

    return parser


def main():
    """Entry point for the models CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        handle_models(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        print("   Partial downloads are saved and will resume on next run.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.command == "install" and hasattr(args, "model") and args.model:
            for model in args.model:
                handle_download_error(e, model)
        else:
            print("\n💡 Try running with --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()