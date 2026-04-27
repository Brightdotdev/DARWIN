
# src/core/Models/cli.py

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.core.Models.SystemDetector import detect_system, recommend_defaults, default_models_dir
from src.core.Models.ModelStore import ModelStore
from src.core.Models.ModelLoader import ModelLoader
from src.core.Models.ModelRegistry import (
    REGISTRY, TIER_LABELS, get_spec, ModelCategory,
     by_category
)



# ═══════════════════════════════════════════════════════════════════════════
# Config sync
# ═══════════════════════════════════════════════════════════════════════════

def _get_config_path() -> Path:
    """Return the path to the app config file."""
    return default_models_dir().parent / "config.json"


def load_config() -> dict:
    """Load the persisted model preferences from config.json."""
    config_path = _get_config_path()
    if not config_path.exists():
        return {}
    try:
        import json
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️  Could not read config ({config_path}): {e}")
        return {}


def save_config(config: dict) -> None:
    """
    Persist model preferences to config.json.

    Called after every install, uninstall, or switch so that the main
    application always loads the right model on the next startup without
    manual intervention.
    """
    import json
    config_path = _get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"   💾 Config saved → {config_path}")
    except Exception as e:
        print(f"   ⚠️  Could not save config ({config_path}): {e}")



def sync_model_to_config(category: str, key: str) -> None:
    """
    Update the active model for a category in config.json.
    Stores the full model_id (e.g., 'BAAI/bge-m3').
    """
    config = load_config()
    spec = get_spec(key)
    config[category] = spec.model_id  # Store model_id, not key
    save_config(config)

def remove_model_from_config(category: str, key: str) -> None:
    """
    Clear a model from config.json on uninstall.
    """
    config = load_config()
    spec = get_spec(key)
    if config.get(category) == spec.model_id:
        config.pop(category, None)
        save_config(config)
        print(f"   ℹ️  Cleared '{category}' from config (was '{spec.model_id}').")

# ═══════════════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════════════

def handle_download_error(error: Exception, key: str, manager: 'ModelManager') -> None:
    """Handle download errors with actionable recovery suggestions."""
    from src.core.Models.ModelStore import DownloadError, IntegrityError

    print(f"\n❌ Error downloading {key}: {error}")

    model_path = manager.store.root / key

    if isinstance(error, IntegrityError):
        print("\n🔧 Recovery steps:")
        print(f"   1. Clear corrupted files: python -m src.core.Models.cli uninstall --model {key}")
        print(f"   2. Free up disk space")
        print(f"   3. Retry: python -m src.core.Models.cli install --model {key}")
        print(f"\n   Model location: {model_path}")

    elif isinstance(error, DownloadError):
        if error.recoverable:
            print(f"\n💡 {error.hint}")
            print(f"   Partial download saved. Re-run to resume.")
        else:
            print(f"\n💡 {error.hint}")
            print(f"   This is a required dependency. Please install it and try again.")
    else:
        print(f"\n💡 Check your internet connection and disk space.")
        print(f"   Error details: {str(error)}")


# ═══════════════════════════════════════════════════════════════════════════
# ModelManager
# ═══════════════════════════════════════════════════════════════════════════


class ModelManager:
    def __init__(self, darwin_store=None): 
        self.store = ModelStore()
        self.system = detect_system()
        self._darwin_store = darwin_store
        self._loader: Optional[ModelLoader] = None

    @property
    def loader(self) -> ModelLoader:
        if self._loader is None:
            
            from src.core.Models import get_loader
            self._loader = get_loader()
        return self._loader

    def status(self) -> None:
        """Show installed models and their status."""
        print("\n📊 Model Status\n")
        for key in REGISTRY:
            spec = REGISTRY[key]
            installed = self.store.is_downloaded(key)
            size = self.store.installed_size(key) if installed else "N/A"
            print(f"   {key:<25} {spec.model_id:<45} {'✅' if installed else '⬜'} {size}")

    def list_models(self, category: Optional[str] = None) -> None:
        """List models available for installation."""
        models = REGISTRY
        if category:
            models = {k: v for k, v in REGISTRY.items() if v.category == category}
        
        print(f"\n📋 Available Models ({category or 'all'}):\n")
        for key, spec in models.items():
            print(f"   {key:<25} {spec.model_id:<45} {spec.tier:<10} {spec.disk_mb}MB")
        print()

    def install_category(self, category: str, tier: str = 'all') -> None:
        """Install all models in a category, optionally filtered by tier."""
        models = {k: v for k, v in REGISTRY.items() if v.category == category}
        if tier != 'all':
            models = {k: v for k, v in models.items() if v.tier == tier}
        
        if not models:
            print(f"❌ No models found for category '{category}' (tier: {tier})")
            return
            
        for key in models:
            if not self.store.is_downloaded(key):
                self.install(key)

    def setup_required(self) -> None:
        """Install all required models."""
        required = {k: v for k, v in REGISTRY.items() if v.required}
        for key in required:
            if not self.store.is_downloaded(key):
                self.install(key)

    def setup_recommended(self) -> None:
        """Install recommended models based on system."""
        recommended = recommend_defaults(self.system)
        for category, key in recommended.items():
            if key and not self.store.is_downloaded(key):
                self.install(key)

    def upgrade(self, key: str) -> None:
        """Upgrade a model by forcing re-download."""
        if key not in REGISTRY:
            print(f"❌ Unknown model: {key}")
            return
            
        spec = REGISTRY[key]
        if not self.store.is_downloaded(key):
            print(f"⬜ {key} is not installed. Use install instead.")
            return
            
        print(f"⬆️  Upgrading {key}...")
        self.install(key, force=True)

    def switch(self, category: str, new_key: str) -> None:
        """Switch active model. new_key can be model_id or shorthand."""
        from src.core.Models.ModelRegistry import REGISTRY
        
        # Accept both model_id and shorthand
        id_to_key = {spec.model_id: k for k, spec in REGISTRY.items()}
        if new_key in id_to_key:
            key = id_to_key[new_key]
        elif new_key in REGISTRY:
            key = new_key
        else:
            print(f"❌ Unknown model: {new_key}")
            return
        
        spec = get_spec(key)
        if spec.category != category:
            print(f"{new_key} is not a {category} model")
            return

        if not self.store.is_downloaded(new_key):
            print(f"   Downloading {new_key} first...")
            self.store.download(new_key)

        self.loader.set_preference(category, spec.model_id)

        if self._darwin_store and category == "embedding":
            health = self._darwin_store.get_embedding_health()
            stale = health["needs_reembed"]
            if stale:
                print(f"⚠️  {len(stale)} file(s) embedded with the old model.")
                print(f"   They'll still search fine until you re-embed them.")
                print(f"   Run 're-embed --all' to upgrade them to {new_key}.")
    
        self.loader.set_preference(category, spec.model_id)
        self._darwin_store.update_model_preference(category, spec.model_id)
        print(f"✅ Switched {category} → {new_key}\n")

    def install(self, key: str, force: bool = False) -> None:
        """Install only puts model on disk — does not activate it."""
        if key not in REGISTRY:
            print(f"❌ Unknown model: {key}")
            return

        spec = get_spec(key)

        if not self.system.can_run(spec):
            print(f"⚠️  Warning: {key} needs {spec.ram_gb}GB RAM (you have {self.system.ram_gb}GB)")
            response = input("   Continue anyway? [y/N]: ").strip().lower()
            if response != 'y':
                return

        if not self.system.has_disk_space(spec):
            print(f"⚠️  Warning: Not enough disk space.")
            response = input("   Continue anyway? [y/N]: ").strip().lower()
            if response != 'y':
                return

        print(f"\n⬇️  Installing {key} ({spec.disk_mb}MB)...")
        self.store.download(key, force=force)
        print(f"✅ {key} installed. Run switch to activate it.\n")
        # ← No config write — install and activate are now separate steps

    def uninstall(self, key: str) -> None:
        if key not in REGISTRY:
            print(f"❌ Unknown model: {key}")
            return

        if not self.store.is_downloaded(key):
            print(f"⬜ {key} is not installed")
            return

        spec = get_spec(key)
        if spec.required:
            print(f"⚠️  {key} is required. Uninstalling may break functionality.")
            response = input("   Continue? [y/N]: ").strip().lower()
            if response != 'y':
                return

        self.store.delete(key)
        if self._darwin_store and spec.category == "embedding":
            by_model = self._darwin_store.get_embedded_files_by_model()
            entry = by_model["embedding"].get(spec.model_id, {})
            affected = entry.get("files", [])
            if affected:
                print(f"⚠️  {len(affected)} file(s) were embedded with this model.")
                print(f"   These files will lose search until re-embedded:")
                for f in affected[:5]:
                    print(f"   • {f}")
                if len(affected) > 5:
                    print(f"   ... and {len(affected) - 5} more")
                response = input("   Continue? [y/N]: ").strip().lower()
                if response != 'y':
                    return


        print(f"✅ {key} removed.\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI dispatch
# ═══════════════════════════════════════════════════════════════════════════

def handle_models(args, manager: ModelManager): 
    """Main CLI handler for model management commands."""
    
    # Handle shorthand flags - install only recommended models
    if hasattr(args, 'command') and args.command == "install":
        # Tokenizer shorthand: install recommended tokenizer only
        if hasattr(args, 'tokenizer') and args.tokenizer:
            if not args.model:
                recommended = recommend_defaults(manager.system)
                default_key = recommended.get("tokenizer", "en_core_web_md")
                args.model = [default_key]
                print(f"\n📝 Installing recommended tokenizer: {default_key}")
                print(f"   (Use --category tokenizer to install all options)\n")

        # Embedding shorthand: install recommended embedding only
        elif hasattr(args, 'embedding') and args.embedding:
            if not args.model:
                recommended = recommend_defaults(manager.system)
                default_key = recommended.get("embedding", "BAAI/bge-m3")
                args.model = [default_key]
                print(f"\n📝 Installing recommended embedding: {default_key}")
                print(f"   (Use --category embedding to see all options)\n")

        # Caption shorthand: install recommended vision model only
        elif hasattr(args, 'caption') and args.caption:
            if not args.model:
                recommended = recommend_defaults(manager.system)
                default_key = recommended.get("vision")  # May be None
                if default_key:
                    args.model = [default_key]
                    print(f"\n📝 Installing recommended vision model: {default_key}")
                else:
                    # Default to lightest vision model
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

    return parser


def main():
    """Entry point for the models CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return
    from src.core.Storage.DarwinStore import DarwinStore
    store = DarwinStore()
    manager = ModelManager(darwin_store=store)    
    
    try:
        handle_models(args,manager)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        print("   Partial downloads are saved and will resume on next run.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.command == "install" and hasattr(args, 'model') and args.model:
            for model in args.model:
                handle_download_error(e, model, manager)
        else:
            print("\n💡 Try running with --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()


