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
    required_models, by_category
)


def handle_download_error(error: Exception, key: str, manager: 'ModelManager') -> None:
    """Handle download errors with actionable recovery suggestions."""
    from src.core.Models.ModelStore import DownloadError, IntegrityError
    
    print(f"\n❌ Error downloading {key}: {error}")
    
    if isinstance(error, IntegrityError):
        print("\n🔧 Recovery steps:")
        print(f"   1. Clear corrupted files: python -m src.core.Models.cli uninstall --model {key}")
        print(f"   2. Free up disk space")
        print(f"   3. Retry: python -m src.core.Models.cli install --model {key}")
        print(f"\n   Model location: {manager.models_dir / key}")
        
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

class ModelManager:
    """High-level model management interface."""
    
    def __init__(self):
        self.store = ModelStore()
        self.system = detect_system()
        self._loader: Optional[ModelLoader] = None
    
    @property
    def loader(self) -> ModelLoader:
        if self._loader is None:
            preferences = recommend_defaults(self.system)
            self._loader = ModelLoader(self.store, preferences)
        return self._loader
    
    def status(self) -> None:
        """Show complete model status."""
        print("\n" + "=" * 70)
        print("📊 MODEL STATUS")
        print("=" * 70)
        print(f"\n💻 System: {self.system.ram_gb}GB RAM | {self.system.cpu_cores} cores | "
              f"{self.system.disk_free_gb:.1f}GB free")
        print()
        
        for category in ModelCategory:
            cat_models = by_category(category.value)
            print(f"── {category.value.upper()} ──" + "─" * (70 - len(category.value) - 8))
            
            for key, spec in cat_models.items():
                status = "✅ installed" if self.store.is_downloaded(key) else "⬜ not installed"
                req = "(required)" if spec.required else "(optional)"
                print(f"  {key:<25} {TIER_LABELS.get(spec.tier, '?'):<15} "
                      f"{spec.disk_mb:>5}MB  {status:<18} {req}")
            print()
    
    def list_models(self, category: Optional[str] = None) -> None:
        """List available models."""
        if category:
            models = by_category(category)
            print(f"\n📦 {category.upper()} MODELS")
        else:
            models = REGISTRY
            print("\n📦 ALL AVAILABLE MODELS")
        
        print("=" * 80)
        print(f"{'':2} {'Key':<25} {'Tier':<15} {'Size':>6} {'RAM':>5}  {'Req':<6} Notes")
        print("-" * 80)
        
        for key, spec in models.items():
            installed = "✅" if self.store.is_downloaded(key) else "⬜"
            print(f"{installed} {key:<25} {TIER_LABELS.get(spec.tier, '?'):<15} "
                  f"{spec.disk_mb:>4}MB {spec.ram_gb:>3}GB  "
                  f"{'YES' if spec.required else 'no':<6} {spec.notes}")
        print()
    
    def install(self, key: str, force: bool = False) -> None:
        """Install a model by key name."""
        if key not in REGISTRY:
            print(f"❌ Unknown model: {key}")
            print(f"   Available: {', '.join(REGISTRY.keys())}")
            return
        
        spec = get_spec(key)
        
        # Check system requirements
        if not self.system.can_run(spec):
            print(f"⚠️  Warning: {key} needs {spec.ram_gb}GB RAM (you have {self.system.ram_gb}GB)")
            response = input("   Continue anyway? [y/N]: ").strip().lower()
            if response != 'y':
                print("   Cancelled.")
                return
        
        if not self.system.has_disk_space(spec):
            print(f"⚠️  Warning: Not enough disk space. Need {spec.disk_mb/1024:.1f}GB, "
                  f"have {self.system.disk_free_gb:.1f}GB free")
            response = input("   Continue anyway? [y/N]: ").strip().lower()
            if response != 'y':
                print("   Cancelled.")
                return
        
        print(f"\n⬇️  Installing {key} ({spec.disk_mb}MB)...")
        self.store.download(key, force=force)
        print(f"✅ {key} installed successfully!\n")
    
    def install_category(self, category: str, tier: str = "all") -> None:
        """Install all models in a category."""
        cat_models = by_category(category)
        
        if not cat_models:
            print(f"❌ Unknown category: {category}")
            print(f"   Available: {[c.value for c in ModelCategory]}")
            return
        
        if tier != "all":
            cat_models = {k: v for k, v in cat_models.items() if v.tier == tier}
        
        if not cat_models:
            print(f"❌ No models found in category '{category}' with tier '{tier}'")
            return
        
        print(f"\n⬇️  Installing {len(cat_models)} model(s) from {category}...")
        for key in cat_models:
            if not self.store.is_downloaded(key):
                self.install(key)
            else:
                print(f"   ✅ {key} already installed")
        print(f"\n✅ {category} models ready!\n")
    
    def uninstall(self, key: str) -> None:
        """Uninstall a model."""
        if key not in REGISTRY:
            print(f"❌ Unknown model: {key}")
            return
        
        if not self.store.is_downloaded(key):
            print(f"⬜ {key} is not installed")
            return
        
        spec = get_spec(key)
        if spec.required:
            print(f"⚠️  {key} is a required model. Uninstalling may break functionality.")
            response = input("   Continue? [y/N]: ").strip().lower()
            if response != 'y':
                print("   Cancelled.")
                return
        
        print(f"\n🗑️  Uninstalling {key}...")
        self.store.delete(key)
        print(f"✅ {key} removed. Freed {spec.disk_mb}MB.\n")
    
    def upgrade(self, key: str) -> None:
        """Re-download a model to get the latest version."""
        if key not in REGISTRY:
            print(f"❌ Unknown model: {key}")
            return
        
        print(f"\n🔄 Upgrading {key}...")
        self.install(key, force=True)
    
    def switch(self, category: str, new_key: str) -> None:
        """Switch to a different model in the same category."""
        if new_key not in REGISTRY:
            print(f"❌ Unknown model: {new_key}")
            return
        
        spec = get_spec(new_key)
        if spec.category != category:
            print(f"❌ {new_key} is in category '{spec.category}', not '{category}'")
            return
        
        self.install(new_key)
        self.loader.set_preference(category, new_key)
        print(f"✅ Switched {category} to {new_key}\n")
    
    def setup_required(self) -> None:
        """Download all required models."""
        required = required_models()
        print(f"\n📦 Installing {len(required)} required models...")
        
        for key, spec in required.items():
            if not self.store.is_downloaded(key):
                self.install(key)
            else:
                print(f"   ✅ {key} already installed")
        
        print("✅ All required models ready!\n")
    
    def setup_recommended(self) -> None:
        """Download recommended models based on system."""
        recommendations = recommend_defaults(self.system)
        print(f"\n📦 Installing recommended models for your system...")
        
        for category, key in recommendations.items():
            if key and not self.store.is_downloaded(key):
                self.install(key)
        
        print("✅ Recommended models ready!\n")


def handle_models(args):
    """Main CLI handler for model management commands."""
    manager = ModelManager()
    
    # Handle shorthand flags - install only recommended models
    if hasattr(args, 'command') and args.command == "install":
        # Tokenizer shorthand: install recommended tokenizer only
        if hasattr(args, 'tokenizer') and args.tokenizer:
            if not args.model:
                recommended = recommend_defaults(manager.system)
                default_key = recommended.get("tokenizer", "spacy-en-core-md")
                args.model = [default_key]
                print(f"\n📝 Installing recommended tokenizer: {default_key}")
                print(f"   (Use --category tokenizer to install all options)\n")
        
        # Embedding shorthand: install recommended embedding only
        elif hasattr(args, 'embedding') and args.embedding:
            if not args.model:
                recommended = recommend_defaults(manager.system)
                default_key = recommended.get("embedding", "bge-m3")
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
                    args.model = ["smolvlm"]
                    print(f"\n📝 Installing lightweight vision model: smolvlm")
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
  python -m src.core.Models.cli switch --category embedding --model bge-m3
  python -m src.core.Models.cli upgrade --category embedding
  python -m src.core.Models.cli uninstall --model e5-small-v2
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
    install_parser.add_argument("--model", "-m", nargs="+", help="Model key(s) to install")
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
    uninstall_parser.add_argument("--model", "-m", nargs="+", required=True, help="Model key(s) to uninstall")
    
    # upgrade
    upgrade_parser = subparsers.add_parser("upgrade", help="Upgrade models")
    upgrade_parser.add_argument("--model", "-m", nargs="+", help="Model key(s) to upgrade")
    upgrade_parser.add_argument("--category", "-c", choices=[c.value for c in ModelCategory],
                               help="Upgrade all installed models in category")
    
    # switch
    switch_parser = subparsers.add_parser("switch", help="Switch model in a category")
    switch_parser.add_argument("--category", "-c", required=True,
                              choices=[c.value for c in ModelCategory], help="Category to switch")
    switch_parser.add_argument("--model", "-m", nargs=1, required=True, help="New model key")
    
    # setup
    setup_parser = subparsers.add_parser("setup", help="Complete setup")
    setup_parser.add_argument("--profile", "-p", choices=["required", "recommended"],
                             default="required", help="Setup profile")
    
    # Global
    
    return parser


def main():
    """Entry point for the models CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    manager = ModelManager()
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
                handle_download_error(e, model, manager)
        else:
            print("\n💡 Try running with --help for usage information.")
        sys.exit(1)
    

if __name__ == "__main__":
    main()