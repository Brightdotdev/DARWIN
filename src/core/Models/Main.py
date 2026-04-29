# src/core/Models/cli.py

import argparse
import sys
from typing import Optional

from src.core.Models.SystemDetector import detect_system, recommend_defaults
from src.core.Models.ModelStore import ModelStore
from src.core.Models.ModelLoader import ModelLoader
from src.core.Models.ModelRegistry import ModelCategory, by_category
from src.core.Models.ModelManager import ModelManager
from src.core.Models.ModelUtils import handle_download_error
from src.core.Storage.DarwinStore import DarwinStore


# ═══════════════════════════════════════════════════════════════════════════
# CLI dispatch
# ═══════════════════════════════════════════════════════════════════════════

def handle_models(args):
    """Main CLI handler for model management commands."""
    store = DarwinStore()
    manager = ModelManager(darwin_store=store)

    if hasattr(args, 'command') and args.command == "install":
        if hasattr(args, 'tokenizer') and args.tokenizer:
            if not args.model:
                system = detect_system()
                recommended = recommend_defaults(system)
                default_key = recommended.get("tokenizer", "en_core_web_md")
                args.model = [default_key]
                print(f"\n📝 Installing recommended tokenizer: {default_key}")
                print(f"   (Use --category tokenizer to install all options)\n")

        elif hasattr(args, 'embedding') and args.embedding:
            if not args.model:
                system = detect_system()
                recommended = recommend_defaults(system)
                default_key = recommended.get("embedding", "BAAI/bge-m3")
                args.model = [default_key]
                print(f"\n📝 Installing recommended embedding: {default_key}")
                print(f"   (Use --category embedding to see all options)\n")

        elif hasattr(args, 'caption') and args.caption:
            if not args.model:
                system = detect_system()
                recommended = recommend_defaults(system)
                default_key = recommended.get("vision")
                if default_key:
                    args.model = [default_key]
                    print(f"\n📝 Installing recommended vision model: {default_key}")
                else:
                    args.model = ["HuggingFaceTB/SmolVLM-256M-Instruct"]
                    print(f"\n📝 Installing lightweight vision model: HuggingFaceTB/SmolVLM-256M-Instruct")
                print(f"   (Use --category vision to see all options)\n")

    if args.command == "status":
        manager.status()

    elif args.command == "list":
        manager.list_models(getattr(args, 'category', None))

    elif args.command == "install":
        if args.model:
            for model in args.model:
                manager.install(model, force=args.force)
        elif args.category:
            manager.install_category(args.category, tier=getattr(args, 'tier', 'all') or 'all')
        else:
            print("❌ Specify --model or --category")

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
                if manager.store.is_downloaded(key):
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
        from src.core.Models.ModelStartup import run_onboarding_async
        asyncio.run(run_onboarding_async())
        print("✅ Custom setup complete. Run 'status' to see your models.")

    elif args.command == "validate":
        from src.core.Models.ModelSetup import validate_model, validate_category
        system = detect_system()
        
        if args.model:
            for key in args.model:
                result = validate_model(key, system, manager.store)
                if result.can_run:
                    print(f"✅ {key} — OK")
                else:
                    print(f"❌ {key}: {', '.join(result.warnings)}")
        elif args.category:
            results = validate_category(args.category, system, manager.store)
            for r in results:
                if r.can_run:
                    print(f"✅ {r.model_key} — OK")
                else:
                    print(f"⚠️  {r.model_key}: {', '.join(r.warnings)}")


def create_parser():
    """Create the argument parser for the models CLI."""
    parser = argparse.ArgumentParser(
        description="🧠 Darwin Model Manager — download, switch, and manage AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.core.Models.cli status
  python -m src.core.Models.cli install --tokenizer
  python -m src.core.Models.cli install --category embedding --tier light
  python -m src.core.Models.cli install --caption
  python -m src.core.Models.cli switch --category embedding --model BAAI/bge-m3
  python -m src.core.Models.cli upgrade --category embedding
  python -m src.core.Models.cli uninstall --model intfloat/e5-small-v2
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

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
    install_parser.add_argument("--caption", action="store_true", help="Install vision models")
    install_parser.add_argument("--tokenizer", action="store_true", help="Install spaCy NLP model")
    install_parser.add_argument("--embedding", action="store_true", help="Install embedding model")

    # uninstall
    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall models")
    uninstall_parser.add_argument("--model", "-m", nargs="+", required=True, help="Model key(s) or model_id(s) to uninstall")

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
    validate_parser.add_argument("--category", "-c", choices=[c.value for c in ModelCategory], help="Validate all models in category")
    
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
        if args.command == "install" and hasattr(args, 'model') and args.model:
            for model in args.model:
                handle_download_error(e, model)
        else:
            print("\n💡 Try running with --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()