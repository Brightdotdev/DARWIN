# src/core/Models/cli.py
"""
🧠 Darwin Model Manager CLI — using the new unified ModelManager.
"""

import argparse
import sys
import os
from typing import Dict

from src.core.Models.Lib.ModelRegistry import ModelCategory, find_spec, MODEL_SPECS
from src.core.Models.ModelManager import ModelManager, ManagerStatus, ModelManagerError, modelManager

from src.lib.KeyManager import (
    set_api_key,
    clear_api_key,
    load_keys_from_keychain,
    PROVIDER_REGISTRY,
)
from src.core.Logger.DarwinLogger import darwin, LogType


def handle_models(args):
    # Use the singleton manager (or create fresh)
    mgr = modelManager

    if args.command == "key":
        handle_keys(args)
        return

    if args.command == "status":
        # Run initialize to get full health status
        status = mgr.initialize(auto_setup=False)
        print_status(status)

    elif args.command == "list":
        category = getattr(args, "category", None)
        if category:
            cat_enum = ModelCategory(category)
            available = mgr.list_available(cat_enum)
            darwin.log(f"\nAvailable models for {category}:")
            for mid in available:
                darwin.log(f"  {mid}")
            # Also show all registered models and their status
            all_models = {mid: spec for mid, spec in MODEL_SPECS.items() if spec.category == cat_enum}
            darwin.log(f"\nAll registered models in {category}:")
            for mid, spec in all_models.items():
                status = "✅" if mid in available else "⏳" if not spec.requires_provider else "🔑"
                darwin.log(f"  {status} {mid} ({spec.tier})")
        else:
            # Show all categories summary via initialize
            status = mgr.initialize(auto_setup=False)
            darwin.log("\nModel MODEL_SPECS Summary:")
            for cat_name, cat_status in status.categories.items():
                darwin.log(f"  {cat_name}: {len(cat_status.available_models)} available, active: {cat_status.active_model_id}")

    elif args.command == "install":
        # Install means: download and set as active (or just download?)
        # The new manager's setup will download and load+set active.
        if args.model:
            for model_id in args.model:
                try:
                    # Determine category from spec
                    spec = find_spec(model_id)
                    if not spec:
                        darwin.log(f"❌ Unknown model: {model_id}", LogType.ERROR)
                        continue
                    category = spec.category
                    # Setup: download, load, set active
                    result = mgr.setup(
                        category,
                        model_id=model_id,
                        auto_download=True,
                        force_download=args.force,
                        required=False,
                    )
                    if result.success:
                        darwin.log(f"✅ Installed and activated: {model_id}", LogType.SUCCESS)
                    else:
                        darwin.log(f"❌ Failed to install {model_id}: {result.error}", LogType.ERROR)
      
                except Exception as e:
                    darwin.log(f"❌ Failed to install {model_id}: {e}", LogType.ERROR)
                # API key reminder for API models
                spec = find_spec(model_id)
                if spec and spec.requires_provider:
                    provider = model_id.split("/")[0]  # crude, but similar
                    # Find provider in MODEL_SPECS
                    env_var = PROVIDER_REGISTRY.get(provider, (None,))[0]
                    if env_var and not os.getenv(env_var):
                        darwin.log(
                            f"⚠️  API key for {provider} is not set. "
                            f"Use 'python -m src.core.Models.cli key set {provider} YOUR_KEY'",
                            LogType.WARNING,
                        )
        elif args.category:
            cat_enum = ModelCategory(args.category)
            # Install all models in category? The old code installed recommended models.
            # We'll install the default (first available) model for that category.
            default_id = mgr._pick_default(cat_enum)
            if default_id:
                darwin.log(f"Installing default model for {args.category}: {default_id}")
                result = mgr.setup(
                    cat_enum,
                    model_id=default_id,
                    auto_download=True,
                    force_download=args.force,
                    required=False,
                )
                if result.success:
                    darwin.log(f"✅ Installed {default_id} as active for {args.category}", LogType.SUCCESS)
                else:
                    darwin.log(f"❌ Failed to install {default_id}: {result.error}", LogType.ERROR)
            else:
                darwin.log(f"❌ No default model found for {args.category}", LogType.ERROR)
        else:
            # No model or category: run full setup with recommended profile
            darwin.log("Running full setup with recommended models...", LogType.INFO)
            preferences = build_preferences_for_profile("recommended")
            results = mgr.setup_all(
                preferences=preferences,
                auto_download=True,
                force_download=args.force,
                required_categories=[ModelCategory.EMBEDDING],  # embedding is mandatory
            )
            for cat, res in results.items():
                if res.success:
                    darwin.log(f"✅ {cat.name}: {res.model_id} ready", LogType.SUCCESS)
                else:
                    darwin.log(f"❌ {cat.name}: {res.error}", LogType.ERROR)

    elif args.command == "setup":
        profile = getattr(args, "profile", "required")
        if profile == "required":
            preferences = build_preferences_for_profile("required")
        elif profile == "recommended":
            preferences = build_preferences_for_profile("recommended")
        else:
            preferences = {}
            darwin.log(f"Unknown profile: {profile}, using required", LogType.WARNING)
            preferences = build_preferences_for_profile("required")

        results = mgr.setup_all(
            preferences=preferences,
            auto_download=True,
            required_categories=[ModelCategory.EMBEDDING],
        )
        for cat, res in results.items():
            if res.success:
                darwin.log(f"✅ {cat.name}: {res.model_id} ready", LogType.SUCCESS)
            else:
                darwin.log(f"❌ {cat.name}: {res.error}", LogType.ERROR)

    elif args.command == "uninstall":
        if not args.model:
            darwin.log("❌ Specify --model to uninstall", LogType.ERROR)
            return
        for model_id in args.model:
            spec = find_spec(model_id)
            if not spec:
                darwin.log(f"❌ Unknown model: {model_id}", LogType.ERROR)
                continue
            category = spec.category
            # Discard with delete=True, and also clear preference if active
            # The discard function already clears active if deleting.
            mgr.discard_model(category, model_id, delete=True)
            darwin.log(f"✅ Uninstalled {model_id}", LogType.SUCCESS)

    elif args.command == "upgrade":
        # Upgrade = force re-download and reload
        if args.model:
            for model_id in args.model:
                spec = find_spec(model_id)
                if not spec:
                    darwin.log(f"❌ Unknown model: {model_id}", LogType.ERROR)
                    continue
                category = spec.category
                darwin.log(f"Upgrading {model_id}...")
                result = mgr.setup(
                    category,
                    model_id=model_id,
                    auto_download=True,
                    force_download=True,
                    required=False,
                )
                if result.success:
                    darwin.log(f"✅ Upgraded {model_id}", LogType.SUCCESS)
                else:
                    darwin.log(f"❌ Upgrade failed for {model_id}: {result.error}", LogType.ERROR)
        elif args.category:
            cat_enum = ModelCategory(args.category)
            # Upgrade all installed models in that category
            # List available models (downloaded)
            available = mgr.list_available(cat_enum)
            if not available:
                darwin.log(f"No models available in {args.category}", LogType.WARNING)
            for model_id in available:
                darwin.log(f"Upgrading {model_id}...")
                result = mgr.setup(
                    cat_enum,
                    model_id=model_id,
                    auto_download=True,
                    force_download=True,
                    required=False,
                )
                if result.success:
                    darwin.log(f"✅ Upgraded {model_id}", LogType.SUCCESS)
                else:
                    darwin.log(f"❌ Upgrade failed for {model_id}: {result.error}", LogType.ERROR)
        else:
            darwin.log("❌ Specify --model or --category to upgrade", LogType.ERROR)

    elif args.command == "switch":
        if not args.category:
            darwin.log("❌ Specify --category", LogType.ERROR)
            return
        cat_enum = ModelCategory(args.category)
        migrate = getattr(args, "migrate", False)
        if args.interactive or not args.model:
            interactive_switch(mgr, args.category, migrate=migrate)
        else:
            model_id = args.model[0]
            try:
                # Ensure the model is available; if not, attempt to download if allowed
                # Use setup to ensure it's downloaded and loaded
                result = mgr.setup(cat_enum, model_id=model_id, auto_download=True, required=False)
                if result.success:
                    mgr.set_preference(cat_enum, model_id, auto_download=False)
                    darwin.log(f"✅ Switched to {model_id}", LogType.SUCCESS)
                else:
                    darwin.log(f"❌ Failed to switch to {model_id}: {result.error}", LogType.ERROR)
            except Exception as e:
                darwin.log(f"❌ Switch failed: {e}", LogType.ERROR)

    elif args.command == "switch-format":
        # Deprecated: format selection is not in new manager; we'll adapt by finding a model with that format.
        darwin.log("switch-format is deprecated. Use 'switch' with the exact model key.", LogType.WARNING)
        if not args.category or not args.base_model:
            darwin.log("❌ Specify --category and --base-model", LogType.ERROR)
            return
        cat_enum = ModelCategory(args.category)
        format_pref = getattr(args, "format", None)
        # Find a model with that base and format
        candidates = []
        for mid, spec in MODEL_SPECS.items():
            if spec.category == cat_enum and mid.startswith(args.base_model):
                if format_pref:
                    if format_pref == "onnx" and mid.endswith("-onnx"):
                        candidates.append(mid)
                    elif format_pref == "pytorch" and mid.endswith("-pytorch"):
                        candidates.append(mid)
                else:
                    candidates.append(mid)
        if not candidates:
            darwin.log(f"No model found for base {args.base_model} with format {format_pref}", LogType.ERROR)
            return
        selected = candidates[0]  # simplest: pick first
        darwin.log(f"Selected {selected}")
        result = mgr.setup(cat_enum, model_id=selected, auto_download=True, required=False)
        if result.success:
            mgr.set_preference(cat_enum, selected, auto_download=False)
            darwin.log(f"✅ Switched to {selected}", LogType.SUCCESS)
        else:
            darwin.log(f"❌ Failed: {result.error}", LogType.ERROR)

    elif args.command == "setup-custom":
        # Keep interactive onboarding if available; otherwise fallback to setup required
        try:
            import asyncio
            from src.core.Models.ModelStartup import run_onboarding_async
            asyncio.run(run_onboarding_async())
            darwin.log("  Custom setup complete. Run 'status' to see your models.")
        except ImportError:
            darwin.log("❌ Onboarding module not available. Use setup instead:", LogType.ERROR)
            darwin.log("   python -m src.core.Models.cli setup --profile recommended", LogType.INFO)

    elif args.command == "validate":
        
        if args.model:
            for model_id in args.model:
                spec = find_spec(model_id)
                if not spec:
                    darwin.log(f"❌ Unknown model: {model_id}", LogType.ERROR)
                    continue
                category = spec.category
                if mgr.is_downloaded(category, model_id):
                    darwin.log(f"  {model_id} — OK")
                else:
                    darwin.log(f"❌ {model_id}: not downloaded", LogType.ERROR)
        elif args.category:
            cat_enum = ModelCategory(args.category)
            available = mgr.list_available(cat_enum)
            for model_id, spec in MODEL_SPECS.items():
                if spec.category == cat_enum:
                    if model_id in available:
                        darwin.log(f"  {model_id} — OK")
                    else:
                        darwin.log(f"⚠️  {model_id}: not available", LogType.WARNING)
        else:
            # Full validation: run initialize
            status = mgr.initialize(auto_setup=False)
            for cat_name, cat_status in status.categories.items():
                if cat_status.available:
                    darwin.log(f"  {cat_name}: OK (active: {cat_status.active_model_id})")
                else:
                    darwin.log(f"  {cat_name}: ❌ {cat_status.error}")

    else:
        darwin.log(f"Unknown command: {args.command}", LogType.ERROR)


def print_status(status: ManagerStatus):
    """Pretty print the manager status."""
    darwin.log(status.summary, LogType.INFO)
    if status.warnings:
        darwin.log("\nWarnings:", LogType.WARNING)
        for w in status.warnings:
            darwin.log(f"  ⚠️ {w}", LogType.WARNING)
    if status.missing_required:
        darwin.log("\nMissing required models:", LogType.ERROR)
        for cat in status.missing_required:
            darwin.log(f"  ❌ {cat}", LogType.ERROR)
        darwin.log("Run 'install' or 'setup' to resolve.", LogType.INFO)


def build_preferences_for_profile(profile: str) -> Dict[ModelCategory, str]:
    """
    Build a preferences dict for a given profile (required/recommended).
    Uses the MODEL_SPECS to pick reasonable models.
    """
    prefs = {}
    # Define mapping: for each category, a list of preferred model IDs (order matters)
    profile_map = {
        "required": {
            ModelCategory.EMBEDDING: ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5-onnx"],
            # No other categories required
        },
        "recommended": {
            ModelCategory.EMBEDDING: ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5-onnx"],
            ModelCategory.NLP: ["en_core_web_sm", "en_core_web_md"],
            ModelCategory.OCR: ["rapidocr", "SWHL/RapidOCR"],
            ModelCategory.RERANKER: ["ms-marco", "cross-encoder/ms-marco-MiniLM-L6-v2"],
        }
    }
    # For each category in the profile, try to find the first available model from the list
    for cat, model_list in profile_map.get(profile, {}).items():
        for mid in model_list:
            spec = find_spec(mid)
            if spec and spec.category == cat:
                prefs[cat] = mid
                break
        if cat not in prefs:
            # Fallback: pick any from MODEL_SPECS
            for spec in MODEL_SPECS:
                if spec.category == cat:
                    prefs[cat] = spec.model_id
                    break
    return prefs


def interactive_switch(mgr: ModelManager, category: str, migrate: bool = False):
    """
    Interactive model selection using available models.
    """
    cat_enum = ModelCategory(category)
    # Get available models
    available = mgr.list_available(cat_enum)
    if not available:
        darwin.log(f"No models available for {category}.", LogType.ERROR)
        return

    darwin.log(f"\n{'='*60}")
    darwin.log(f"Available models for {category}")
    darwin.log(f"{'='*60}")

    # Display with indices
    index_map = {}
    for idx, model_id in enumerate(available, 1):
        spec = find_spec(model_id)
        tags = f"{spec.tier} ({spec.disk_mb}MB, RAM {spec.ram_gb}GB)" if spec else ""
        darwin.log(f"  [{idx}] {model_id} {tags}")
        index_map[idx] = model_id

    darwin.log(f"\n{'='*60}")
    choice = input(f"Select model number (1-{len(index_map)}) or q to cancel: ").strip()
    if choice.lower() == 'q':
        return

    try:
        idx = int(choice)
        if idx in index_map:
            selected = index_map[idx]
            darwin.log(f"Switching to {selected}...")
            result = mgr.setup(cat_enum, model_id=selected, auto_download=True, required=False)
            if result.success:
                mgr.set_preference(cat_enum, selected, auto_download=False)
                darwin.log(f"✅ Switched to {selected}", LogType.SUCCESS)
            else:
                darwin.log(f"❌ Failed: {result.error}", LogType.ERROR)
        else:
            darwin.log("Invalid selection.", LogType.ERROR)
    except ValueError:
        darwin.log("Invalid input.", LogType.ERROR)


def handle_keys(args):
    # Keep unchanged
    try:
        load_keys_from_keychain()
    except Exception as e:
        darwin.log(f"⚠️  Could not load keychain: {e}", LogType.WARNING)

    if args.key_command == "set":
        set_api_key(args.provider, args.key_value, validate=not args.no_validate)
        darwin.log(f"  API key for {args.provider} saved and activated.")
    elif args.key_command == "remove":
        clear_api_key(args.provider)
        darwin.log(f"  API key for {args.provider} removed.")
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
  python -m src.core.Models.cli status
  python -m src.core.Models.cli list --category embedding
  python -m src.core.Models.cli install --model BAAI/bge-m3-onnx
  python -m src.core.Models.cli switch --category embedding
  python -m src.core.Models.cli switch-format --category embedding --base-model BAAI/bge-m3 --format onnx
  python -m src.core.Models.cli upgrade --category embedding
  python -m src.core.Models.cli uninstall --model intfloat/e5-small-v2-onnx
  python -m src.core.Models.cli key set openai sk-... --no-validate
  python -m src.core.Models.cli key list
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
        darwin.configure()
        handle_models(args)
    except KeyboardInterrupt:
        darwin.log("\n\n⚠️  Operation cancelled by user", LogType.WARNING)
        darwin.log("   Partial downloads are saved and will resume on next run.", LogType.WARNING)
        sys.exit(1)

    except Exception as e:
        darwin.log(f"\n❌ Error: {e}", LogType.ERROR)
        sys.exit(1)


if __name__ == "__main__":
    main()