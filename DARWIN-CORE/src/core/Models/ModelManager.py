# src/core/Models/ModelManager.py

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

from src.core.Models.ModelStore import get_model_store
from src.core.Models.ModelLoader import ModelLoader
from src.features.Hardware.SystemDetector import detect_system, recommend_defaults, SystemInfo
from src.features.Hardware.DeviceManager import DeviceManager
from src.core.Models.ModelRegistry import (
    get_spec, REGISTRY, ModelCategory, REQUIRED_CATEGORIES, DEFAULT_PREFERRED
)
from src.core.Models.ModelConfig import set_active_model, load_active_models
from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.KeyManager import load_keys_from_keychain, PROVIDER_REGISTRY
from src.features.Hardware.Capabilities import Capabilities
from src.core.Plugins.TorchPlugin import TorchPlugin, PluginRequiredError


# ── Phase 3: Result dataclasses for clean API/CLI separation ──────────────
@dataclass
class InstallResult:
    key: str
    success: bool
    message: str
    skipped: bool = False

@dataclass
class SwitchResult:
    category: str
    key: str
    success: bool
    message: str

@dataclass
class UninstallResult:
    key: str
    removed: bool
    message: str


_LOADER_INSTANCE = None
_SYSTEM_CACHE = None


class ModelManager:
    def __init__(self, system: Optional[SystemInfo] = None):
        global _SYSTEM_CACHE

        # Load API keys from keychain (graceful if not set up)
        try:
            load_keys_from_keychain()
        except Exception as e:
            
            darwin.log(f"Keychain not available: {e}", LogType.WARNING)

        # Ensure DeviceManager is initialised once
        if DeviceManager._config is None:
            if _SYSTEM_CACHE is None:
                _SYSTEM_CACHE = detect_system()
            try:
                DeviceManager.init(_SYSTEM_CACHE)
            except RuntimeError:
                pass

        self.model_store = get_model_store()
        if _SYSTEM_CACHE is None:
            _SYSTEM_CACHE = detect_system()
        self.system = system or _SYSTEM_CACHE
        
        # Cache capabilities for repeated checks
        self.capabilities = Capabilities.detect()

        self._loader: Optional[ModelLoader] = None

    # ── Static helpers ───────────────────────────────────────────────
    @staticmethod
    def get_capabilities() -> Capabilities:
        """Detect and return current system capabilities."""
        return Capabilities.detect()

    @staticmethod
    def get_loader() -> ModelLoader:
        global _LOADER_INSTANCE, _SYSTEM_CACHE
        if _LOADER_INSTANCE is None:
            store = get_model_store()
            if _SYSTEM_CACHE is None:
                _SYSTEM_CACHE = detect_system()
            prefs = load_active_models()
            if not prefs:
                prefs = recommend_defaults(_SYSTEM_CACHE)
                for cat, model in prefs.items():
                    if model:
                        set_active_model(cat, model)
            _LOADER_INSTANCE = ModelLoader(
                store,
                prefs,
                system_info=_SYSTEM_CACHE,
            )
        return _LOADER_INSTANCE

    @staticmethod
    def setup(preferences: dict = None, skip_download: bool = False) -> ModelLoader:
        global _LOADER_INSTANCE, _SYSTEM_CACHE

        try:
            load_keys_from_keychain()
        except Exception as e:
            
            darwin.log(f"Keychain not available: {e}", LogType.WARNING)

        if _SYSTEM_CACHE is None:
            _SYSTEM_CACHE = detect_system()
        if DeviceManager._config is None:
            try:
                DeviceManager.init(_SYSTEM_CACHE)
            except RuntimeError:
                pass

        if preferences is not None:
            for cat, model in preferences.items():
                set_active_model(cat, model)
        else:
            preferences = load_active_models()
            if not preferences:
                preferences = recommend_defaults(_SYSTEM_CACHE)
                for cat, model in preferences.items():
                    if model:
                        set_active_model(cat, model)

        if not skip_download:
            try:
                from src.core.Models.ModelStartup import download_required
                download_required(get_model_store(), preferences, _SYSTEM_CACHE)
            except ImportError:
                
                darwin.log("ModelStartup not available, skipping auto-download", LogType.WARNING)

        if _LOADER_INSTANCE is None:
            _LOADER_INSTANCE = ModelLoader(
                get_model_store(),
                preferences,
                system_info=_SYSTEM_CACHE,
            )
        return _LOADER_INSTANCE

    @property
    def loader(self) -> ModelLoader:
        if self._loader is None:
            self._loader = ModelLoader(
                self.model_store,
                load_active_models(),
                system_info=self.system,
            )
        return self._loader

    # ── Model listing for UI ─────────────────────────────────────────
    def list_available_models(self, category: Optional[str] = None) -> list[dict]:
        """Return list of models with status, tags, and plugin requirements."""
        caps = self.capabilities
        torch_available = caps.torch_available

        models = []
        for key, spec in REGISTRY.items():
            if category and spec.category != category:
                continue

            status = "unavailable"
            tags = []

            if spec.loader.startswith("api_"):
                provider = key.split("/")[0]
                env_var = PROVIDER_REGISTRY.get(provider, (None,))[0]
                has_key = env_var and bool(os.getenv(env_var))
                status = "available" if has_key else "needs_key"
                tags.append("API" if has_key else "API — key required")

            elif spec.loader in ("onnx_embedding", "onnx_reranker"):
                is_cached = self.model_store.is_downloaded(key, "onnx")
                status = "installed" if is_cached else "available"
                tags.append("ONNX")

            elif spec.requires_plugin == "torch":
                if torch_available:
                    is_cached = self.model_store.is_downloaded(key, "pytorch")
                    status = "installed" if is_cached else "available"
                    tags.append("PyTorch")
                else:
                    status = "needs_plugin"
                    tags.append("PyTorch — torch plugin required")

            elif spec.loader == "spacy":
                is_cached = self.model_store.is_downloaded(key)
                status = "installed" if is_cached else "available"
                tags.append("Local")

            else:
                is_cached = self.model_store.is_downloaded(key)
                status = "installed" if is_cached else "available"
                tags.append("Local")

            models.append({
                "key": key,
                "model_id": spec.model_id,
                "category": spec.category,
                "tier": spec.tier,
                "status": status,
                "tags": tags,
                "disk_mb": spec.disk_mb,
                "ram_gb": spec.ram_gb,
                "requires_plugin": spec.requires_plugin,
                "notes": spec.notes,
            })

        return models

    # ── Switching with format auto-selection ─────────────────────────
    def switch_with_format(
        self,
        category: str,
        base_model: str,
        prefer_format: Optional[str] = None,
        migrate: bool = False,
    ) -> Optional[str]:
        """
        Switch to a model, auto‑selecting ONNX or PyTorch variant.
        Returns the actual registry key activated, or raises on failure.
        """
        variants = {}
        for key, spec in REGISTRY.items():
            if spec.model_id == base_model and spec.category == category:
                if spec.loader in ("onnx_embedding", "onnx_reranker"):
                    variants["onnx"] = key
                elif spec.requires_plugin == "torch":
                    variants["pytorch"] = key
                else:
                    variants["other"] = key

        if not variants:
            msg = f"No models found for {base_model} in category {category}"
            
            darwin.log(msg, LogType.ERROR)
            raise ValueError(msg)

        selected_format = prefer_format
        if selected_format is None:
            caps = self.capabilities
            if "onnx" in variants:
                selected_format = "onnx"
            elif "pytorch" in variants and caps.torch_available:
                selected_format = "pytorch"
            elif "other" in variants:
                selected_format = "other"
            else:
                selected_format = "pytorch" if "pytorch" in variants else list(variants.keys())[0]

        selected_key = variants.get(selected_format)
        if not selected_key:
            available = ", ".join(variants.keys())
            msg = f"Format '{selected_format}' not available. Available: {available}"
            
            darwin.log(msg, LogType.ERROR)
            raise ValueError(msg)

        # ── FIX: No silent swallowing — propagate errors ─────────────
        self.switch(category, selected_key, migrate=migrate)
        return selected_key

    # ── Individual switch methods ────────────────────────────────────
    def switch_embedding(self, new_key: str) -> None:
        try:
            resolved_key = self.loader.resolve_key(new_key, "embedding")
            spec = get_spec(resolved_key)
            is_ready = self._check_plugin_and_download(resolved_key, spec)
            if not is_ready:
                return
            self.loader.set_preference("embedding", resolved_key)
        except PluginRequiredError as e:
            
            darwin.log(str(e), LogType.ERROR)
            return

    def switch_reranker(self, new_key: str) -> None:
        self._switch_stateless("reranker", new_key)

    def switch_vision(self, new_key: str) -> None:
        self._switch_stateless("vision", new_key)

    def switch_tokenizer(self, new_key: str) -> None:
        self._switch_stateless("tokenizer", new_key)

    def _switch_stateless(self, category: str, new_key: str) -> None:
        try:
            resolved_key = self.loader.resolve_key(new_key, category)
            spec = get_spec(resolved_key)
            is_ready = self._check_plugin_and_download(resolved_key, spec)
            if not is_ready:
                return
            self.loader.set_preference(category, resolved_key)
            
            darwin.log(f"Switched {category} → {new_key}", LogType.SUCCESS)
        except PluginRequiredError as e:
            
            darwin.log(str(e), LogType.ERROR)

    # ── Helper: plugin check + download with hardware validation ─────
    def _check_plugin_and_download(self, key: str, spec) -> bool:
        """Raise PluginRequiredError if plugin missing or hardware cannot run the model.
        Otherwise download if not cached."""
        if spec.requires_plugin:
            torch_plugin = TorchPlugin()
            if not torch_plugin.is_installed():
                raise PluginRequiredError("torch", f"model {spec.model_id}")

        if spec.disk_mb > 0:
            if not self.system.can_run(spec):
                
                darwin.log(
                        f"{key} needs {spec.ram_gb}GB RAM (you have {self.system.ram_gb}GB)",
                        LogType.ERROR,
                    )
                return False
            if not self.system.has_disk_space(spec):
                
                darwin.log(
                        f"{key} needs {spec.disk_mb}MB disk (you have {self.system.disk_free_gb*1024:.0f}MB free)",
                        LogType.ERROR,
                    )
                return False

            if not self.model_store.is_downloaded(key):
                
                darwin.log(f"Downloading {key}...", LogType.INFO)
                try:
                    self.model_store.download(key)
                except Exception as e:
                    
                    darwin.log(f"Download failed for {key}: {e}", LogType.ERROR)
                    return False

        return True


    def unload_all_models(self) -> None:
        self.loader.unload_all()

    # ── Install / uninstall commands ─────────────────────────────────
    def install(self, key: str, force: bool = False) -> InstallResult:
        """
        Install a model. Pure API — no input() prompts.
        Returns InstallResult for CLI to format.
        """
        if key not in REGISTRY:
            msg = f"Unknown model: {key}"
            
            darwin.log(msg, LogType.ERROR)
            return InstallResult(key=key, success=False, message=msg)

        spec = get_spec(key)

        # Plugin + hardware validation
        try:
            is_ready = self._check_plugin_and_download(key, spec)
            if not is_ready:
                return InstallResult(
                    key=key, success=False, message="Hardware or plugin check failed", skipped=True
                )
        except PluginRequiredError as e:
            msg = str(e)
            
            darwin.log(msg, LogType.ERROR)
            return InstallResult(key=key, success=False, message=msg)

        if spec.disk_mb == 0:
            provider = key.split('/')[0]
            env_var = PROVIDER_REGISTRY.get(provider, (None,))[0]
            if env_var and not os.getenv(env_var):
                msg = f"{key} requires an API key for {provider}. Use 'darwin key set {provider} YOUR_KEY'"
                
                darwin.log(msg, LogType.WARNING)
                return InstallResult(key=key, success=True, message=msg, skipped=False)
            else:
                msg = f"{key} is an API model – no local download needed."
                
                darwin.log(msg, LogType.INFO)
                return InstallResult(key=key, success=True, message=msg, skipped=False)

        # Hardware warnings (no interactive prompts)
        if not self.system.can_run(spec):
            msg = f"Warning: {key} needs {spec.ram_gb}GB RAM (you have {self.system.ram_gb}GB)."
            
            darwin.log(msg, LogType.WARNING)
            if not force:
                return InstallResult(key=key, success=False, message=msg, skipped=True)

        if not self.system.has_disk_space(spec):
            msg = "Warning: Not enough disk space."
            
            darwin.log(msg, LogType.WARNING)
            if not force:
                return InstallResult(key=key, success=False, message=msg, skipped=True)

        
        darwin.log(f"Installing {key} ({spec.disk_mb}MB)...", LogType.INFO)
        self.model_store.download(key, force=force)
        msg = f"{key} installed. Run switch to activate it."
        
        darwin.log(msg, LogType.SUCCESS)
        return InstallResult(key=key, success=True, message=msg)

    # ── FIX C5: Implement upgrade() ──────────────────────────────────
    def upgrade(self, key: str) -> InstallResult:
        """Upgrade a model: uninstall(force) then install(force)."""
        if key not in REGISTRY:
            msg = f"Unknown model: {key}"
            
            darwin.log(msg, LogType.ERROR)
            return InstallResult(key=key, success=False, message=msg)

        spec = get_spec(key)
        if spec.disk_mb == 0:
            msg = f"{key} is an API model — nothing to upgrade."
            
            darwin.log(msg, LogType.INFO)
            return InstallResult(key=key, success=True, message=msg, skipped=True)

        
        darwin.log(f"Upgrading {key}...", LogType.INFO)

        self.uninstall(key, force=True)
        return self.install(key, force=True)

    def uninstall(self, key: str, force: bool = False) -> UninstallResult:
        """
        Uninstall a model. Returns UninstallResult.
        """
        if key not in REGISTRY:
            msg = f"Unknown model: {key}"
            
            darwin.log(msg, LogType.ERROR)
            return UninstallResult(key=key, removed=False, message=msg)

        spec = get_spec(key)

        # ── API models: no disk files, just deactivate ──────────────────────
        if spec.disk_mb == 0:
            active_key = self.loader.get_preference(spec.category)
            # ── FIX C20: Normalize both keys before comparing ────────────
            resolved_active = self.loader._resolve_key(active_key) if active_key else None
            resolved_target = self.loader._resolve_key(key)
            if resolved_active == resolved_target:
                self.loader.set_preference(spec.category, None)
                msg = f"{key} deactivated (API model)."
                
                darwin.log(msg, LogType.SUCCESS)
                return UninstallResult(key=key, removed=True, message=msg)
            msg = f"{key} is an API model; not active, nothing to do."
            
            darwin.log(msg, LogType.INFO)
            return UninstallResult(key=key, removed=False, message=msg)

        # ── Guard: required models ──────────────────────────────────────────
        if spec.required and not force:
            msg = f"{key} is required. Uninstalling will break functionality. Use force=True to override."
            
            darwin.log(msg, LogType.WARNING)
            return UninstallResult(key=key, removed=False, message=msg)

        # ── Guard: currently active ─────────────────────────────────────────
        active_key = self.loader.get_preference(spec.category)
        resolved_active = self.loader._resolve_key(active_key) if active_key else None
        resolved_target = self.loader._resolve_key(key)
        if resolved_active == resolved_target:
            msg = f"{key} is the currently active {spec.category} model. Switch to another model first."
            
            darwin.log(msg, LogType.ERROR)
            return UninstallResult(key=key, removed=False, message=msg)

        # ── Guard: not installed ────────────────────────────────────────────
        if not self.model_store.is_downloaded(key):
            msg = f"{key} is not installed."
            
            darwin.log(msg, LogType.WARNING)
            return UninstallResult(key=key, removed=False, message=msg)

        # ── Execute deletion ────────────────────────────────────────────────
        target_format = self.model_store.get_format(key)
        self.model_store.delete(key, format=target_format)
        msg = f"{key} removed."
        
        darwin.log(msg, LogType.SUCCESS)
        return UninstallResult(key=key, removed=True, message=msg)

    # ── Batch operations ─────────────────────────────────────────────
    def install_category(self, category: str, tier: str = 'all') -> None:
        models = {k: v for k, v in REGISTRY.items() if v.category == category}
        if tier != 'all':
            models = {k: v for k, v in models.items() if v.tier == tier}
        for key in models:
            if not self.model_store.is_downloaded(key):
                self.install(key)

    def setup_required(self) -> None:
        """Install the preferred model for each required category."""
        for category in REQUIRED_CATEGORIES:
            preferred_key = DEFAULT_PREFERRED.get(category)
            if not preferred_key:
                continue
            if not self.model_store.is_downloaded(preferred_key):
                result = self.install(preferred_key)
                if result.success:
                    self.loader.set_preference(category, preferred_key)

    def setup_recommended(self) -> None:
        for cat, key in recommend_defaults(self.system).items():
            if key and not self.model_store.is_downloaded(key):
                self.install(key)

    # ── Utility ──────────────────────────────────────────────────────
    def status(self) -> None:
        
        darwin.log("\n Model Status\n")
        for key, spec in REGISTRY.items():
            installed = self.model_store.is_downloaded(key)
            size = self.model_store.installed_size(key) if installed else "N/A"
            status = "++" if installed else "XX"
            format_tag = ""
            if spec.loader in ("onnx_embedding", "onnx_reranker"):
                format_tag = "[ONNX]"
            elif spec.requires_plugin == "torch":
                format_tag = "[PyTorch]"
            elif spec.loader.startswith("api_"):
                format_tag = "[API]"
            
            darwin.log(f"   {key:<30} {spec.model_id:<45} {format_tag:<10} {status} {size}")

    def list_models(self, category: Optional[str] = None) -> None:
        models = REGISTRY
        if category:
            models = {k: v for k, v in REGISTRY.items() if v.category == category}
        
        darwin.log(f"\n Available Models ({category or 'all'}):\n")
        for key, spec in models.items():
            format_tag = ""
            if spec.loader in ("onnx_embedding", "onnx_reranker"):
                format_tag = "[ONNX]"
            elif spec.requires_plugin == "torch":
                format_tag = "[PyTorch]"
            elif spec.loader.startswith("api_"):
                format_tag = "[API]"
            
            darwin.log(f"   {key:<30} {spec.model_id:<45} {format_tag:<10} {spec.tier:<10} {spec.disk_mb}MB")
        

    def switch(self, category: str, model: str, migrate: bool = False) -> None:
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