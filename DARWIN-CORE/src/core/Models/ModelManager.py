# src/core/Models/ModelManager.py

import os
from typing import Optional, Dict, Any

from src.core.Models.ModelStore import get_model_store, ModelStore
from src.core.Models.ModelLoader import ModelLoader
from src.features.Hardware.SystemDetector import detect_system, recommend_defaults, SystemInfo
from src.features.Hardware.DeviceManager import DeviceManager
from src.core.Models.ModelRegistry import get_spec, REGISTRY, ModelCategory
from src.core.Models.ModelConfig import set_active_model, load_active_models
from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.KeyManager import load_keys_from_keychain, PROVIDER_REGISTRY
from src.features.Hardware.Capabilities import Capabilities
from src.core.Plugins.TorchPlugin import TorchPlugin, PluginRequiredError
from src.core.Models.ModelUtils import HardwareInsufficientError

_LOADER_INSTANCE = None
_SYSTEM_CACHE = None


class ModelManager:
    def __init__(self, darwin_store=None):
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
                # Already initialized — fine
                pass

        self.model_store = get_model_store()
        if _SYSTEM_CACHE is None:
            _SYSTEM_CACHE = detect_system()
        self.system = _SYSTEM_CACHE
        self._darwin_store = darwin_store

        # Cache capabilities for repeated checks
        self.capabilities = Capabilities.detect()

        self._loader: Optional[ModelLoader] = None

    # ── Static helpers ───────────────────────────────────────────────
    @staticmethod
    def get_capabilities() -> Capabilities:
        """Detect and return current system capabilities."""
        return Capabilities.detect()

    @staticmethod
    def emit_capabilities() -> None:
        """Emit capabilities to stdout for Tauri handshake."""
        caps = Capabilities.detect()
        caps.emit()

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
                system_info=_SYSTEM_CACHE,  # pass system info for hardware validation
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

        try:
            ModelManager.emit_capabilities()
        except Exception as e:
            darwin.log(f"Failed to emit capabilities: {e}", LogType.WARNING)

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
        Returns the actual registry key activated, or None if failed.
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
            darwin.log(f"No models found for {base_model} in category {category}", LogType.ERROR)
            return None

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
            darwin.log(f"Format '{selected_format}' not available. Available: {available}", LogType.ERROR)
            return None

        try:
            self.switch(category, selected_key, migrate=migrate)
            return selected_key
        except Exception:
            return None

    # ── Individual switch methods ────────────────────────────────────
    def switch_embedding(self, new_key: str, migrate: bool = False, file: Optional[str] = None) -> None:
        try:
            resolved_key = self.loader.resolve_key(new_key, "embedding")
            spec = get_spec(resolved_key)
            self._check_plugin_and_download(resolved_key, spec)
            self.loader.set_preference("embedding", resolved_key)
        except HardwareInsufficientError as e:
            darwin.log(str(e), LogType.ERROR)
            return
        except PluginRequiredError as e:
            darwin.log(str(e), LogType.ERROR)
            return

        if migrate and self._darwin_store:
            active = load_active_models().get("embedding")
            health = self._darwin_store.get_embedding_health(active)
            if health.get("needs_reembed", []):
                darwin.log(f"{len(health['needs_reembed'])} file(s) need re-embedding. Migrating...", LogType.INFO)
                self.migrate_embedding(file=file)

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
            self._check_plugin_and_download(resolved_key, spec)
            self.loader.set_preference(category, resolved_key)
            darwin.log(f"Switched {category} → {new_key}", LogType.SUCCESS)
        except HardwareInsufficientError as e:
            darwin.log(str(e), LogType.ERROR)
        except PluginRequiredError as e:
            darwin.log(str(e), LogType.ERROR)

    # ── Helper: plugin check + download with hardware validation ─────
    def _check_plugin_and_download(self, key: str, spec) -> None:
        """Raise PluginRequiredError if plugin missing, or HardwareInsufficientError if
        hardware cannot run the model. Otherwise download if not cached."""
        # Plugin check
        if spec.requires_plugin:
            torch_plugin = TorchPlugin()
            if not torch_plugin.is_installed():
                raise PluginRequiredError("torch", f"model {spec.model_id}")

        # Hardware check before attempting download
        if spec.disk_mb > 0:
            if not self.system.can_run(spec):
                raise HardwareInsufficientError(
                    key,
                    f"Needs {spec.ram_gb}GB RAM (you have {self.system.ram_gb}GB)"
                )
            if not self.system.has_disk_space(spec):
                raise HardwareInsufficientError(
                    key,
                    f"Needs {spec.disk_mb}MB disk (you have {self.system.disk_free_gb*1024:.0f}MB free)"
                )

            if not self.model_store.is_downloaded(key):
                darwin.log(f"Downloading {key}...", LogType.INFO)
                self.model_store.download(key)

    # ── Migration ────────────────────────────────────────────────────
    def migrate_embedding(self, file: Optional[str] = None, force: bool = False) -> dict:
        if not self._darwin_store:
            darwin.log("No DarwinStore available for migration", LogType.ERROR)
            return {"reembedded": 0, "failed": 1}
        from src.features.Ingestion.main import handle_ingestion
        from src.features.Ingestion.Parsing.lib import ParsingArgs
        from src.features.Ingestion.Chunking.lib import ChunkingArgs
        from src.features.Indexing.lib import IndexingArgs
        from src.features.Indexing.main import handle_indexing
        store = self._darwin_store
        current_model = self.loader.get_model_key(ModelCategory.EMBEDDING)
        stats = {"reembedded": 0, "skipped": 0, "term_banks_updated": 0, "failed": 0}
        if file:
            files_to_process = [file]
        else:
            health = store.get_embedding_health()
            needs = health.get("needs_reembed", [])
            up_to_date = health.get("up_to_date", []) if force else []
            files_to_process = list(set(needs) | set(up_to_date))
        if not files_to_process:
            darwin.log("All files are up-to-date with current embedding model", LogType.SUCCESS)
            return stats
        darwin.log(f"Migrating {len(files_to_process)} file(s) to {current_model}", LogType.INFO)
        try:
            handle_ingestion(
                ParsingArgs(target=".", files=list(files_to_process), save=True, skip_if_markdown=True),
                ChunkingArgs(rerun=True, inspect=False),
            )
        except Exception as e:
            darwin.log(f"Ingestion phase failed — {type(e).__name__}: {e}", LogType.ERROR)
            stats["failed"] += len(files_to_process)
            return stats
        for fname in files_to_process:
            try:
                handle_indexing(IndexingArgs(path=".", file=[fname], rerun=True, inspect=False))
                stats["reembedded"] += 1
                darwin.log(f"{fname} — migrated", LogType.SUCCESS)
            except Exception as e:
                darwin.log(f"{fname} — {type(e).__name__}: {e}", LogType.ERROR)
                stats["failed"] += 1
        try:
            self.loader.unload(ModelCategory.EMBEDDING)
        except Exception:
            pass
        darwin.log(
            f"Migration complete: {stats['reembedded']} migrated · {stats['term_banks_updated']} term banks updated · {stats['failed']} failed",
            LogType.SUCCESS,
        )
        return stats

    # ── Install / uninstall commands ─────────────────────────────────
    def install(self, key: str, force: bool = False) -> None:
        if key not in REGISTRY:
            darwin.log(f"Unknown model: {key}", LogType.ERROR)
            return
        spec = get_spec(key)

        # Plugin + hardware validation
        try:
            self._check_plugin_and_download(key, spec)
        except HardwareInsufficientError as e:
            darwin.log(str(e), LogType.ERROR)
            return
        except PluginRequiredError as e:
            darwin.log(str(e), LogType.ERROR)
            return

        if spec.disk_mb == 0:
            provider = key.split('/')[0]
            env_var = PROVIDER_REGISTRY.get(provider, (None,))[0]
            if env_var and not os.getenv(env_var):
                darwin.log(f"{key} requires an API key for {provider}. Use 'darwin key set {provider} YOUR_KEY'", LogType.WARNING)
            else:
                darwin.log(f"{key} is an API model – no local download needed.", LogType.INFO)
            return

        if not self.system.can_run(spec):
            darwin.log(f"Warning: {key} needs {spec.ram_gb}GB RAM (you have {self.system.ram_gb}GB)", LogType.WARNING)
            if input("   Continue anyway? [y/N]: ").strip().lower() != 'y':
                return

        if not self.system.has_disk_space(spec):
            darwin.log("Warning: Not enough disk space.", LogType.WARNING)
            if input("   Continue anyway? [y/N]: ").strip().lower() != 'y':
                return

        darwin.log(f"Installing {key} ({spec.disk_mb}MB)...", LogType.INFO)
        self.model_store.download(key, force=force)
        darwin.log(f"{key} installed. Run switch to activate it.", LogType.SUCCESS)

    def uninstall(self, key: str) -> None:
        if key not in REGISTRY:
            darwin.log(f"Unknown model: {key}", LogType.ERROR)
            return
        spec = get_spec(key)
        if spec.disk_mb == 0:
            darwin.log(f"{key} is an API model; nothing to delete locally.", LogType.INFO)
            active = load_active_models().get(spec.category)
            if active == spec.model_id:
                darwin.log(f"⚠️  {key} is currently the active {spec.category} model.", LogType.WARNING)
                if input("   Deactivate it and clear preference? [y/N]: ").strip().lower() == 'y':
                    set_active_model(spec.category, None)
                    self.loader.set_preference(spec.category, "")
                    darwin.log(f"{key} deactivated.", LogType.SUCCESS)
            return

        if not self.model_store.is_downloaded(key):
            darwin.log(f"{key} is not installed", LogType.WARNING)
            return

        active = load_active_models().get(spec.category)
        if active == spec.model_id:
            darwin.log(f"⚠️  {key} is the currently active {spec.category} model. Switch first.", LogType.ERROR)
            return

        if spec.required:
            darwin.log(f"{key} is required. Uninstalling may break functionality.", LogType.WARNING)
            if input("   Continue? [y/N]: ").strip().lower() != 'y':
                return

        if self._darwin_store and spec.category == "embedding":
            by_model = self._darwin_store.get_embedded_files_by_model()
            entry = by_model["embedding"].get(spec.model_id, {})
            affected = entry.get("files", [])
            if affected:
                darwin.log(f"{len(affected)} file(s) were embedded with this model.", LogType.WARNING)
                for f in affected[:5]:
                    darwin.log(f"   • {f}", LogType.WARNING)
                if len(affected) > 5:
                    darwin.log(f"   ... and {len(affected) - 5} more", LogType.WARNING)
                if input("   Continue? [y/N]: ").strip().lower() != 'y':
                    return

        target_format = self.model_store.get_format(key)
        self.model_store.delete(key, format=target_format)
        darwin.log(f"{key} removed.", LogType.SUCCESS)

    # ── Batch operations ─────────────────────────────────────────────
    def install_category(self, category: str, tier: str = 'all') -> None:
        models = {k: v for k, v in REGISTRY.items() if v.category == category}
        if tier != 'all':
            models = {k: v for k, v in models.items() if v.tier == tier}
        for key in models:
            if not self.model_store.is_downloaded(key):
                self.install(key)

    def setup_required(self) -> None:
        for key in [k for k, v in REGISTRY.items() if v.required]:
            if not self.model_store.is_downloaded(key):
                self.install(key)

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
        darwin.log()

    def switch(self, category: str, model: str, migrate: bool = False) -> None:
        if category == "embedding":
            self.switch_embedding(model, migrate=migrate)
        elif category == "reranker":
            self.switch_reranker(model)
        elif category == "vision":
            self.switch_vision(model)
        elif category == "tokenizer":
            self.switch_tokenizer(model)
        else:
            darwin.log(f"Unknown category: {category}", LogType.ERROR)