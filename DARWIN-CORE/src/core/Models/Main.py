# src/core/Models/cli.py
import argparse
import sys
import os
from typing import Optional

from src.features.Hardware.SystemDetector import detect_system
from src.core.Models.ModelRegistry import ModelCategory, get_spec, REGISTRY
from src.core.Models.ModelManager import ModelManager
from src.core.Models.ModelSetup import handle_download_error
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.KeyManager import (
    set_api_key,
    clear_api_key,
    load_keys_from_keychain,
    PROVIDER_REGISTRY,
)
from src.core.Logger.DarwinLogger import darwin, LogType,Verbosity
from src.core.Plugins.PluginManager import PluginRequiredError


def handle_models(args):
    darwin.configure(
        enable_websocket=False,
        enable_history=True,
        verbosity_default=Verbosity.NORMAL,
        show_banner=True,
    )
    # Graceful store initialization
    try:
        store = DarwinStore()
    except Exception:
        store = None
        darwin.log("⚠️  DarwinStore not available — migration features disabled", LogType.WARNING)

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
                try:
                    manager.install(model, force=args.force)
                except PluginRequiredError as e:
                    darwin.log(f"❌ {e}", LogType.ERROR)
                    darwin.log("   Install the required plugin from Settings or run:", LogType.INFO)
                    darwin.log("   python -m src.core.Plugins.cli install torch", LogType.INFO)
                    continue
                except Exception as e:
                    handle_download_error(e, model)

                # API key reminder for API models
                spec = get_spec(model)
                if spec and spec.disk_mb == 0:
                    provider = model.split("/")[0]
                    env_var = PROVIDER_REGISTRY.get(provider, (None,))[0]
                    if env_var and not os.getenv(env_var):
                        darwin.log(f"⚠️  API key for {provider} is not set. "
                                   f"Use 'python -m src.core.Models.cli key set {provider} YOUR_KEY'",
                                   LogType.WARNING)
        elif args.category:
            manager.install_category(args.category, tier=getattr(args, "tier", "all") or "all")
        else:
            # Install recommended defaults
            try:
                ModelManager.setup()
                loader = ModelManager.get_loader()
                prefs = loader.get_preferences()
                darwin.log("\n✅ Installed recommended models:")
                for category, model_id in prefs.items():
                    darwin.log(f"   {category}: {model_id}")
            except Exception as e:
                darwin.log(f"❌ Setup failed: {e}", LogType.ERROR)
                darwin.log("💡 Try running 'python -m src.core.Models.cli setup-custom' instead", LogType.INFO)

    elif args.command == "setup":
        if args.profile == "required":
            manager.setup_required()
        elif args.profile == "recommended":
            manager.setup_recommended()
        else:
            manager.setup_required()

    elif args.command == "uninstall":
        if not args.model:
            darwin.log("❌ Specify --model to uninstall", LogType.ERROR)
            return
        for model in args.model:
            manager.uninstall(model)

    elif args.command == "upgrade":
        if args.model:
            for model in args.model:
                manager.upgrade(model)
        elif args.category:
            cat_models = {
                k: v for k, v in REGISTRY.items() if v.category == args.category
            }
            for key in cat_models:
                if manager.model_store.is_downloaded(key):
                    manager.upgrade(key)
            darwin.log(f"✅ Upgraded all {args.category} models.")
        else:
            darwin.log("❌ Specify --model or --category to upgrade", LogType.ERROR)

    elif args.command == "switch":
        if not args.category:
            darwin.log("❌ Specify --category", LogType.ERROR)
            return
        migrate = getattr(args, "migrate", False)
        if args.interactive or not args.model:
            interactive_switch(manager, args.category, migrate=migrate)
        else:
            try:
                manager.switch(args.category, args.model[0], migrate=migrate)
            except PluginRequiredError as e:
                darwin.log(f"❌ {e}", LogType.ERROR)
                darwin.log("   Install the required plugin before switching.", LogType.INFO)

    elif args.command == "switch-format":
        if not args.category or not args.base_model:
            darwin.log("❌ Specify --category and --base-model", LogType.ERROR)
            return
        try:
            selected = manager.switch_with_format(
                args.category,
                args.base_model,
                prefer_format=getattr(args, "format", None),
                migrate=getattr(args, "migrate", False),
            )
            darwin.log(f"✅ Switched to {selected}")
        except (ValueError, PluginRequiredError) as e:
            darwin.log(f"❌ {e}", LogType.ERROR)

    elif args.command == "setup-custom":
        try:
            import asyncio
            from src.core.Models.ModelStartup import run_onboarding_async
            asyncio.run(run_onboarding_async())
            darwin.log("✅ Custom setup complete. Run 'status' to see your models.")
        except ImportError:
            darwin.log("❌ Onboarding module not available. Use setup instead:", LogType.ERROR)
            darwin.log("   python -m src.core.Models.cli setup --profile recommended", LogType.INFO)

    elif args.command == "validate":
        from src.core.Models.ModelSetup import validate_model, validate_category
        system = detect_system()
        if args.model:
            for key in args.model:
                result = validate_model(key, system, manager.model_store)
                if result.can_run:
                    darwin.log(f"✅ {key} — OK")
                else:
                    darwin.log(f"❌ {key}: {', '.join(result.warnings)}", LogType.ERROR)
        elif args.category:
            results = validate_category(args.category, system, manager.model_store)
            for r in results:
                if r.can_run:
                    darwin.log(f"✅ {r.model_key} — OK")
                else:
                    darwin.log(f"⚠️  {r.model_key}: {', '.join(r.warnings)}", LogType.WARNING)


def interactive_switch(manager: ModelManager, category: str, migrate: bool = False):
    """
    Interactive model selection using the same listing logic as the frontend.
    """
    models = manager.list_available_models(category)

    if not models:
        darwin.log(f"No models available for {category}.")
        return

    darwin.log(f"\n{'='*60}")
    darwin.log(f"Available models for {category}")
    darwin.log(f"{'='*60}")

    # Group by base model (model_id) to show variants together
    base_groups = {}
    for m in models:
        base = m["model_id"]
        if base not in base_groups:
            base_groups[base] = []
        base_groups[base].append(m)

    idx = 1
    index_map = {}
    for base, variants in sorted(base_groups.items()):
        darwin.log(f"\n📦 {base}")
        for m in variants:
            status = m["status"]
            tags = ", ".join(m["tags"])
            install_marker = " ✅" if status == "installed" else ""
            darwin.log(
                f"  [{idx}] {tags:<25} {m['key']:<30} "
                f"({m['tier']}, {m['disk_mb']}MB, RAM {m['ram_gb']}GB){install_marker}"
            )
            index_map[idx] = m["key"]
            idx += 1

    darwin.log(f"\n{'='*60}")
    choice = input(f"Select model number (1-{len(index_map)}) or q to cancel: ").strip()
    if choice.lower() == 'q':
        return

    try:
        idx = int(choice)
        if idx in index_map:
            selected_key = index_map[idx]
            darwin.log(f"Switching to {selected_key}...")
            manager.switch(category, selected_key, migrate=migrate)
        else:
            darwin.log("Invalid selection.", LogType.ERROR)
    except ValueError:
        darwin.log("Invalid input.", LogType.ERROR)


def handle_keys(args):
    try:
        load_keys_from_keychain()
    except Exception as e:
        darwin.log(f"⚠️  Could not load keychain: {e}", LogType.WARNING)

    if args.key_command == "set":
        set_api_key(args.provider, args.key_value, validate=not args.no_validate)
        darwin.log(f"✅ API key for {args.provider} saved and activated.")
    elif args.key_command == "remove":
        clear_api_key(args.provider)
        darwin.log(f"✅ API key for {args.provider} removed.")
    elif args.key_command == "list":
        darwin.log("Providers and key status:")
        for provider, (env_var, _) in PROVIDER_REGISTRY.items():
            key_set = bool(os.getenv(env_var))
            status = "🔑 set" if key_set else "❌ missing"
            darwin.log(f"  {provider}: {status}")
    else:
        darwin.log("Unknown key command.", LogType.ERROR)


def create_parser():
    parser = argparse.ArgumentParser(
        description="🧠 Darwin Model Manager — download, switch, and manage AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.Models.cli status
  python -m src.Models.cli list --category embedding
  python -m src.Models.cli install --model BAAI/bge-m3-onnx
  python -m src.Models.cli switch --category embedding
  python -m src.Models.cli switch-format --category embedding --base-model BAAI/bge-m3 --format onnx
  python -m src.Models.cli upgrade --category embedding
  python -m src.Models.cli uninstall --model intfloat/e5-small-v2-onnx
  python -m src.Models.cli key set openai sk-... --no-validate
  python -m src.Models.cli key list
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # key
    key_parser = subparsers.add_parser("key", help="Manage API keys")
    key_sub = key_parser.add_subparsers(dest="key_command", required=True)
    key_set = key_sub.add_parser("set", help="Store an API key in the system keychain")
    key_set.add_argument("provider", choices=list(PROVIDER_REGISTRY.keys()))
    key_set.add_argument("key_value", help="The API key string")
    key_set.add_argument("--no-validate", action="store_true")
    key_sub.add_parser("remove", help="Remove a stored API key").add_argument("provider", choices=list(PROVIDER_REGISTRY.keys()))
    key_sub.add_parser("list", help="List all providers and key presence")

    subparsers.add_parser("status", help="Show model status and system info")

    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--category", "-c", choices=[c.value for c in ModelCategory])

    install_parser = subparsers.add_parser("install", help="Install models")
    install_parser.add_argument("--model", "-m", nargs="+")
    install_parser.add_argument("--category", "-c", choices=[c.value for c in ModelCategory])
    install_parser.add_argument("--tier", "-t", choices=["light", "medium", "large", "overkill"])
    install_parser.add_argument("--force", "-f", action="store_true")

    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall models")
    uninstall_parser.add_argument("--model", "-m", nargs="+", required=True)

    upgrade_parser = subparsers.add_parser("upgrade", help="Upgrade models")
    upgrade_parser.add_argument("--model", "-m", nargs="+")
    upgrade_parser.add_argument("--category", "-c", choices=[c.value for c in ModelCategory])

    switch_parser = subparsers.add_parser("switch", help="Switch model in a category")
    switch_parser.add_argument("--category", "-c", required=True,
                               choices=[c.value for c in ModelCategory])
    switch_parser.add_argument("--model", "-m", nargs=1, required=False,
                               help="Model key or model_id (omit for interactive)")
    switch_parser.add_argument("--interactive", "-i", action="store_true",
                               help="Choose interactively")
    switch_parser.add_argument("--migrate", action="store_true",
                               help="Migrate embeddings after switching (embedding category only)")

    switch_fmt_parser = subparsers.add_parser("switch-format", help="Switch model with format preference (ONNX/PyTorch)")
    switch_fmt_parser.add_argument("--category", "-c", required=True,
                                   choices=[c.value for c in ModelCategory])
    switch_fmt_parser.add_argument("--base-model", "-b", required=True,
                                   help="Base model name, e.g., 'BAAI/bge-m3'")
    switch_fmt_parser.add_argument("--format", "-f", choices=["onnx", "pytorch"],
                                   help="Preferred format (auto-detect if omitted)")
    switch_fmt_parser.add_argument("--migrate", action="store_true")

    setup_parser = subparsers.add_parser("setup", help="Complete setup")
    setup_parser.add_argument("--profile", "-p", choices=["required", "recommended"], default="required")

    subparsers.add_parser("setup-custom", help="Interactive custom setup")

    validate_parser = subparsers.add_parser("validate", help="Check if models can run on your hardware")
    validate_parser.add_argument("--model", "-m", nargs="+")
    validate_parser.add_argument("--category", "-c", choices=[c.value for c in ModelCategory])

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    try:
        handle_models(args)
    except KeyboardInterrupt:
        darwin.log("\n\n⚠️  Operation cancelled by user", LogType.WARNING)
        darwin.log("   Partial downloads are saved and will resume on next run.", LogType.WARNING)
        sys.exit(1)
    except PluginRequiredError as e:
        darwin.log(f"\n❌ {e}", LogType.ERROR)
        darwin.log("   Install the required plugin from Settings or run:", LogType.INFO)
        darwin.log("   python -m src.core.Plugins.cli install torch", LogType.INFO)
        sys.exit(1)
    except Exception as e:
        darwin.log(f"\n❌ Error: {e}", LogType.ERROR)
        if args.command == "install" and hasattr(args, "model") and args.model:
            for model in args.model:
                handle_download_error(e, model)
        else:
            darwin.log("\n💡 Try running with --help for usage information.", LogType.INFO)
        sys.exit(1)


if __name__ == "__main__":
    main()