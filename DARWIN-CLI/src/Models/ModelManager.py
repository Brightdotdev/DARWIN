# src/core/Models/ModelManager.py

import sys
from pathlib import Path
from typing import Optional as Opt

from src.Models.ModelStore import ModelStore
from src.Models.ModelLoader import ModelLoader
from src.features.Hardware.SystemDetector import detect_system, recommend_defaults
from src.features.Storage.DarwinStore import DarwinStore
from src.Models.ModelRegistry import get_spec, REGISTRY
from src.core.Logger.DarwinLogger import darwin, LogType


_LOADER_INSTANCE = None
_PREFERENCES_CACHE = None
_SYSTEM_CACHE = None


class ModelManager:
    """User-facing orchestration for model management."""
    
    def __init__(self, darwin_store: Opt[DarwinStore] = None):
        global _SYSTEM_CACHE
        
        self.model_store = ModelStore()
        
        # Use cached system info if available, otherwise detect
        if _SYSTEM_CACHE is None:
            _SYSTEM_CACHE = detect_system()
        self.system = _SYSTEM_CACHE
        
        self._darwin_store = darwin_store
        self._loader: Opt[ModelLoader] = None

    @staticmethod
    def get_loader() -> ModelLoader:
        """Global ModelLoader singleton."""
        global _LOADER_INSTANCE, _PREFERENCES_CACHE, _SYSTEM_CACHE
        
        if _LOADER_INSTANCE is None:
            mode_store = ModelStore()
            
            if _SYSTEM_CACHE is None:
                _SYSTEM_CACHE = detect_system()
            
            if _PREFERENCES_CACHE is None:
                _PREFERENCES_CACHE = recommend_defaults(_SYSTEM_CACHE)
            
            _LOADER_INSTANCE = ModelLoader(mode_store, _PREFERENCES_CACHE)
        
        return _LOADER_INSTANCE



    @staticmethod
    def setup(preferences: dict = None, skip_download: bool = False) -> ModelLoader:
        """Initialize the singleton loader with optional preferences and model downloads."""
        global _LOADER_INSTANCE, _PREFERENCES_CACHE, _SYSTEM_CACHE

        if _SYSTEM_CACHE is None:
            _SYSTEM_CACHE = detect_system()

        from src.features.Hardware.DeviceManager import DeviceManager
        if DeviceManager._config is None:
            DeviceManager.init(_SYSTEM_CACHE)

        if preferences is not None:
            _PREFERENCES_CACHE = preferences

        if _PREFERENCES_CACHE is None:
            _PREFERENCES_CACHE = recommend_defaults(_SYSTEM_CACHE)

        if not skip_download:
            from src.Models.ModelStartup import download_required
            download_required(ModelStore(), _PREFERENCES_CACHE, _SYSTEM_CACHE)

        if _LOADER_INSTANCE is None:
            _LOADER_INSTANCE = ModelLoader(ModelStore(), _PREFERENCES_CACHE)

        return _LOADER_INSTANCE

    @property
    def loader(self) -> ModelLoader:
        if self._loader is None:
            self._loader = self.get_loader()
        return self._loader

    # ═══════════════════════════════════════════════════════════════════════
    # Switch methods
    # ═══════════════════════════════════════════════════════════════════════

    def switch_embedding(self, new_key: str, migrate: bool = True, file: Opt[str] = None) -> None:
        """Switch embedding model with optional migration."""
        from src.Models.ModelRegistry import REGISTRY as registry
        
        resolved = self._resolve_key(new_key, "embedding")
        if not resolved:
            return
        
        key, spec = resolved
        
        if not self.model_store.is_downloaded(key):
            darwin.log(f"Downloading {new_key} first...", LogType.INFO)
            try:
                self.model_store.download(key)
            except Exception as e:
                darwin.log(f"Download failed: {e}", LogType.ERROR)
                return

        self.loader.set_preference("embedding", spec.model_id)
        
        if self._darwin_store:
            self._darwin_store.update_model_preference("embedding", spec.model_id)
        
        darwin.log(f"Switched embedding → {new_key}", LogType.SUCCESS)

        if self._darwin_store:
            health = self._darwin_store.get_embedding_health()
            stale = health.get("needs_reembed", [])
            
            if stale:
                if migrate:
                    darwin.log(f"{len(stale)} file(s) need re-embedding. Migrating...", LogType.INFO)
                    self.migrate_embedding(file=file)
                else:
                    darwin.log(f"{len(stale)} file(s) need re-embedding. Run migrate_embedding().", LogType.WARNING)

    def switch_reranker(self, new_key: str) -> None:
        """Switch reranker model (no migration needed)."""
        self._switch_stateless("reranker", new_key)

    def switch_vision(self, new_key: str) -> None:
        """Switch vision model (no migration needed)."""
        self._switch_stateless("vision", new_key)

    def switch_tokenizer(self, new_key: str) -> None:
        """Switch tokenizer model (no migration needed)."""
        self._switch_stateless("tokenizer", new_key)

    def _switch_stateless(self, category: str, new_key: str) -> None:
        """Internal: switch stateless model categories."""
        resolved = self._resolve_key(new_key, category)
        if not resolved:
            return
        
        key, spec = resolved
        
        if not self.model_store.is_downloaded(key):
            darwin.log(f"Downloading {new_key} first...", LogType.INFO)
            try:
                self.model_store.download(key)
            except Exception as e:
                darwin.log(f"Download failed: {e}", LogType.ERROR)
                return

        self.loader.set_preference(category, spec.model_id)
        
        if self._darwin_store:
            self._darwin_store.update_model_preference(category, spec.model_id)
        
        darwin.log(f"Switched {category} → {new_key}", LogType.SUCCESS)

    def _resolve_key(self, key: str, expected_category: str) -> Opt[tuple]:
        """Resolve model identifier to (registry_key, spec). Returns None on failure."""
        from src.Models.ModelRegistry import REGISTRY as registry, get_spec
        
        id_to_key = {spec.model_id: k for k, spec in registry.items()}
        
        if key in id_to_key:
            registry_key = id_to_key[key]
        elif key in registry:
            registry_key = key
        else:
            darwin.log(f"Unknown model: {key}", LogType.ERROR)
            return None
        
        spec = get_spec(registry_key)
        if spec.category != expected_category:
            darwin.log(f"{key} is not a {expected_category} model", LogType.ERROR)
            return None
        
        return registry_key, spec

    # ═══════════════════════════════════════════════════════════════════════
    # Migration methods
    # ═══════════════════════════════════════════════════════════════════════

    def migrate_embedding(self, file: Opt[str] = None, force: bool = False) -> dict:
        """Re-embed files with current embedding model."""
        if not self._darwin_store:
            darwin.log("No DarwinStore available for migration", LogType.ERROR)
            return {"reembedded": 0, "failed": 1}
        
        return self.loader.migrate_embeddings(
            filename=file,
            force=force,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # User commands
    # ═══════════════════════════════════════════════════════════════════════

    def status(self) -> None:
        """Show installed models and their status."""
        print("\n📊 Model Status\n")
        for key in REGISTRY:
            spec = REGISTRY[key]
            installed = self.model_store.is_downloaded(key)
            size = self.model_store.installed_size(key) if installed else "N/A"
            status = "✅" if installed else "⬜"
            print(f"   {key:<25} {spec.model_id:<45} {status} {size}")

    def list_models(self, category: Opt[str] = None) -> None:
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
            darwin.log(f"No models found for category '{category}' (tier: {tier})", LogType.WARNING)
            return
            
        for key in models:
            if not self.model_store.is_downloaded(key):
                self.install(key)

    def setup_required(self) -> None:
        """Install all required models."""
        required = {k: v for k, v in REGISTRY.items() if v.required}
        for key in required:
            if not self.model_store.is_downloaded(key):
                self.install(key)

    def setup_recommended(self) -> None:
        """Install recommended models based on system."""
        recommended = recommend_defaults(self.system)
        for category, key in recommended.items():
            if key and not self.model_store.is_downloaded(key):
                self.install(key)

    def upgrade(self, key: str) -> None:
        """Upgrade a model by forcing re-download."""
        if key not in REGISTRY:
            darwin.log(f"Unknown model: {key}", LogType.ERROR)
            return
            
        if not self.model_store.is_downloaded(key):
            darwin.log(f"{key} is not installed. Use install instead.", LogType.WARNING)
            return
            
        darwin.log(f"Upgrading {key}...", LogType.INFO)
        self.install(key, force=True)

    def install(self, key: str, force: bool = False) -> None:
        """Install model to disk — does not activate it."""
        if key not in REGISTRY:
            darwin.log(f"Unknown model: {key}", LogType.ERROR)
            return

        spec = get_spec(key)

        if not self.system.can_run(spec):
            darwin.log(f"Warning: {key} needs {spec.ram_gb}GB RAM (you have {self.system.ram_gb}GB)", LogType.WARNING)
            response = input("   Continue anyway? [y/N]: ").strip().lower()
            if response != 'y':
                return

        if not self.system.has_disk_space(spec):
            darwin.log("Warning: Not enough disk space.", LogType.WARNING)
            response = input("   Continue anyway? [y/N]: ").strip().lower()
            if response != 'y':
                return

        darwin.log(f"Installing {key} ({spec.disk_mb}MB)...", LogType.INFO)
        self.model_store.download(key, force=force)
        darwin.log(f"{key} installed. Run switch to activate it.", LogType.SUCCESS)

    def uninstall(self, key: str) -> None:
        """Uninstall a model from disk."""
        if key not in REGISTRY:
            darwin.log(f"Unknown model: {key}", LogType.ERROR)
            return

        if not self.model_store.is_downloaded(key):
            darwin.log(f"{key} is not installed", LogType.WARNING)
            return

        spec = get_spec(key)
        if spec.required:
            darwin.log(f"{key} is required. Uninstalling may break functionality.", LogType.WARNING)
            response = input("   Continue? [y/N]: ").strip().lower()
            if response != 'y':
                return

        # Check for embedded files with this embedding model
        if self._darwin_store and spec.category == "embedding":
            by_model = self._darwin_store.get_embedded_files_by_model()
            entry = by_model["embedding"].get(spec.model_id, {})
            affected = entry.get("files", [])
            if affected:
                darwin.log(f"{len(affected)} file(s) were embedded with this model.", LogType.WARNING)
                darwin.log("These files will lose search until re-embedded:", LogType.WARNING)
                for f in affected[:5]:
                    darwin.log(f"   • {f}", LogType.WARNING)
                if len(affected) > 5:
                    darwin.log(f"   ... and {len(affected) - 5} more", LogType.WARNING)
                response = input("   Continue? [y/N]: ").strip().lower()
                if response != 'y':
                    return

        self.model_store.delete(key)
        darwin.log(f"{key} removed.", LogType.SUCCESS)

    def switch(self, category: str, model: str) -> None:
        """Generic switch dispatch."""
        if category == "embedding":
            self.switch_embedding(model)
        elif category == "reranker":
            self.switch_reranker(model)
        elif category == "vision":
            self.switch_vision(model)
        elif category == "tokenizer":
            self.switch_tokenizer(model)
        else:
            darwin.log(f"Unknown category: {category}", LogType.ERROR)


def set_preferences(preferences: dict) -> None:
    """Set model preferences before loader creation."""
    global _PREFERENCES_CACHE
    _PREFERENCES_CACHE = preferences