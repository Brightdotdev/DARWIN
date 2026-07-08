# src/core/Models/ModelsOrchestrator.py
"""
ModelsOrchestrator — Programmatic model lifecycle management.
Registry-driven: never hardcodes model IDs. All recommendations,
defaults, and validations are resolved from the live ModelRegistry.
"""

import os
import shutil
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from src.core.Models.Lib.ModelRegistry import (
    ModelCategory,
    ModelTier,
    find_spec,
    MODEL_SPECS,
    get_specs_by_category,
    get_specs_by_tier,
    get_models_per_provider,
    get_registry_stats,
)
from src.core.Models.ModelManager import (
    ModelManager,
    ManagerStatus,
    SetupResult,
    modelManager,
    ModelManagerError,
)
from src.core.Models.ModelConfig import get_active_model
from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.KeyManager import (
    set_api_key,
    clear_api_key,
    load_keys_from_keychain,
    PROVIDER_REGISTRY,
    is_key_set,
)


# ── Structured Result Types ───────────────────────────────────────────

@dataclass
class OperationResult:
    """Generic result for single-model operations."""
    success: bool
    model_id: str
    category: Optional[str] = None
    message: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.success


@dataclass
class BatchResult:
    """Result for multi-model / batch operations."""
    results: Dict[str, OperationResult] = field(default_factory=dict)
    failed: List[str] = field(default_factory=list)
    succeeded: List[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return len(self.failed) == 0

    @property
    def summary(self) -> str:
        return f"{len(self.succeeded)} ok, {len(self.failed)} failed"


@dataclass
class SystemInfo:
    """Hardware capability snapshot."""
    ram_gb: float
    disk_free_gb: float
    cpu_cores: int
    has_gpu: bool = False
    gpu_name: str = ""
    tier: str = "light"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RecommendationProfile:
    """Per-tier recommended model set (all IDs resolved from registry)."""
    embedding: Optional[str] = None
    nlp: Optional[str] = None
    ocr: Optional[str] = None
    reranker: Optional[str] = None
    vision: Optional[str] = None


# ── Orchestrator ─────────────────────────────────────────────────────

class ModelsOrchestrator:
    """
    High-level programmatic API for Darwin model management.
    Registry-driven: all model IDs are resolved from the live ModelRegistry.
    """

    def __init__(
        self,
        manager: Optional[ModelManager] = None,
        logger=None,
        auto_load_keys: bool = True,
    ):
        self.mgr = manager or modelManager
        self.darwin = logger or darwin

        if auto_load_keys:
            try:
                load_keys_from_keychain()
            except Exception as e:
                self._log(f"Could not load keychain: {e}", LogType.WARNING)

    # ── Internal helpers ───────────────────────────────────────────

    def _log(self, msg: str, level: LogType = LogType.INFO):
        if self.darwin:
            self.darwin.log(msg, level)

    def _spec_or_raise(self, model_id: str):
        """Validate a model_id exists in the registry."""
        spec = find_spec(model_id)
        if not spec:
            registered = [s.model_id for s in MODEL_SPECS]
            raise ValueError(
                f"Unknown model: '{model_id}'. "
                f"Registered models: {registered}"
            )
        return spec

    def _wrap_setup_result(self, result: SetupResult) -> OperationResult:
        return OperationResult(
            success=result.success,
            model_id=result.model_id,
            category=result.category,
            message="OK" if result.success else (result.error or "Failed"),
            error=result.error if not result.success else None,
        )

    def _pick_best_for_category(
        self,
        category: ModelCategory,
        prefer_tier: Optional[ModelTier] = None,
    ) -> Optional[str]:
        """
        Pick the best registered model for a category from the registry.
        - Respects prefer_tier (closest match)
        - Prefers local models over API
        - Prefers smaller RAM footprint
        """
        specs = get_specs_by_category(category)
        if not specs:
            return None

        tier_order = {ModelTier.DOLPHIN: 0, ModelTier.ORCA: 1, ModelTier.WHALE: 2}

        def sort_key(spec):
            if prefer_tier:
                dist = abs(tier_order.get(spec.tier, 1) - tier_order.get(prefer_tier, 1))
            else:
                dist = 0
            is_local = 0 if spec.is_local else 1
            ram = spec.ram_gb if spec.ram_gb > 0 else 999
            return (dist, is_local, ram, spec.model_id)

        sorted_specs = sorted(specs, key=sort_key)
        return sorted_specs[0].model_id if sorted_specs else None

    # ── Status & Discovery ───────────────────────────────────────────

    def status(self, auto_setup: bool = False) -> ManagerStatus:
        """Full health check across all categories."""
        return self.mgr.initialize(auto_setup=auto_setup)

    def health(self, category: ModelCategory) -> Dict[str, Any]:
        """Quick health for a single category."""
        return self.mgr.health(category)

    def list_available(self, category: ModelCategory) -> List[str]:
        """List models ready to use in a category."""
        result = []
        for spec in MODEL_SPECS:
            if spec.category != category:
                continue
            if spec.requires_provider:
                try:
                    from src.lib.KeyManager import is_key_set, load_keys_from_keychain
                    load_keys_from_keychain(log=False)
                    if is_key_set(spec.requires_provider):
                        result.append(spec.model_id)
                except Exception:
                    pass
            else:
                if self.is_downloaded(category, spec.model_id):
                    result.append(spec.model_id)
        return result

    def list_registered(self, category: Optional[ModelCategory] = None) -> List[Dict[str, Any]]:
        """
        List all registered specs from the registry, optionally filtered by category.
        """
        out = []
        for spec in MODEL_SPECS:
            if category and spec.category != category:
                continue
            out.append({
                "model_id": spec.model_id,
                "category": spec.category.value,
                "tier": spec.tier.name,
                "loader": spec.loader,
                "disk_mb": spec.disk_mb,
                "ram_gb": spec.ram_gb,
                "requires_provider": spec.requires_provider,
                "is_local": spec.is_local,
                "is_api": spec.is_api,
                "onnx_available": spec.onnx_available,
                "repo_id": spec.repo_id,
                "notes": spec.notes,
            })
        return out

    def list_categories(self) -> List[str]:
        """Return all known category names from the registry."""
        return sorted({spec.category.value for spec in MODEL_SPECS})

    def list_providers(self) -> List[str]:
        """Return all providers referenced by registered models."""
        return sorted({spec.requires_provider for spec in MODEL_SPECS if spec.requires_provider})

    def get_active(self, category: ModelCategory) -> Optional[str]:
        """Get the active model ID for a category."""
        return self.mgr.get_active(category)

    def is_available(self, category: ModelCategory, model_id: str) -> bool:
        """Check if a specific model is ready to run."""
        return self.mgr.is_available(category, model_id)

    def is_downloaded(self, category: ModelCategory, model_id: str) -> bool:
        return self.mgr.is_downloaded(category, model_id)

    def is_loaded(self, category: ModelCategory, model_id: Optional[str] = None) -> bool:
        return self.mgr.is_loaded(category, model_id)

    # ── Installation ───────────────────────────────────────────────

    def install(
        self,
        model_id: str,
        *,
        auto_download: bool = True,
        force: bool = False,
        set_active: bool = True,
    ) -> OperationResult:
        """
        Download, load, and optionally activate a single model.
        """
        spec = self._spec_or_raise(model_id)
        category = spec.category

        self._log(f"Installing {model_id}...", LogType.INFO)
        result = self.mgr.setup(
            category,
            model_id=model_id,
            auto_download=auto_download,
            force_download=force,
            required=False,
        )

        if result.success and set_active:
            self.mgr.set_preference(category, model_id, auto_download=False)

        if spec.requires_provider:
            provider = spec.requires_provider
            env_var = PROVIDER_REGISTRY.get(provider, (None,))[0]
            if env_var and not os.getenv(env_var):
                self._log(
                    f"API key for {provider} is not set. "
                    f"Use set_api_key('{provider}', 'YOUR_KEY')",
                    LogType.WARNING,
                )

        op = self._wrap_setup_result(result)
        if not op.success:
            self._log(f"Failed to install {model_id}: {op.error}", LogType.ERROR)
        else:
            self._log(f"Installed {model_id}", LogType.SUCCESS)
        return op

    def install_batch(
        self,
        model_ids: List[str],
        *,
        auto_download: bool = True,
        force: bool = False,
        set_active: bool = True,
    ) -> BatchResult:
        """Install multiple models sequentially. Validates all against registry first."""
        for mid in model_ids:
            self._spec_or_raise(mid)

        batch = BatchResult()
        for mid in model_ids:
            op = self.install(mid, auto_download=auto_download, force=force, set_active=set_active)
            batch.results[mid] = op
            if op.success:
                batch.succeeded.append(mid)
            else:
                batch.failed.append(mid)
        return batch

    def install_category_default(
        self,
        category: ModelCategory,
        *,
        auto_download: bool = True,
        force: bool = False,
    ) -> OperationResult:
        """Install the best default model for a category from the registry."""
        default_id = self._pick_best_for_category(category)
        if not default_id:
            return OperationResult(
                success=False,
                model_id="none",
                category=category.value,
                error=f"No models registered for category {category.value}",
            )
        return self.install(default_id, auto_download=auto_download, force=force, set_active=True)

    # ── Uninstall ───────────────────────────────────────────────────

    def uninstall(self, model_id: str, delete_files: bool = True) -> OperationResult:
        """Remove a model from preference and optionally disk."""
        spec = self._spec_or_raise(model_id)
        category = spec.category

        success = self.mgr.discard(category, model_id, delete=delete_files)
        return OperationResult(
            success=success,
            model_id=model_id,
            category=category.value,
            message="Uninstalled" + (" (files deleted)" if delete_files else ""),
        )

    def uninstall_batch(self, model_ids: List[str], delete_files: bool = True) -> BatchResult:
        batch = BatchResult()
        for mid in model_ids:
            op = self.uninstall(mid, delete_files=delete_files)
            batch.results[mid] = op
            if op.success:
                batch.succeeded.append(mid)
            else:
                batch.failed.append(mid)
        return batch

    # ── Can Switch (Dry-run validation) ─────────────────────────────

    def can_switch(
        self,
        category: ModelCategory,
        model_id: str,
    ) -> OperationResult:
        """
        Dry-run validation: check if a model CAN be switched to without doing it.
        Checks registry membership, category match, API key, disk space, RAM fit.
        Returns detailed reason if blocked.
        """
        # 1. Registry existence
        spec = find_spec(model_id)
        if not spec:
            return OperationResult(
                success=False,
                model_id=model_id,
                category=category.value,
                error=f"Model '{model_id}' not found in registry",
            )

        # 2. Category match
        if spec.category != category:
            return OperationResult(
                success=False,
                model_id=model_id,
                category=category.value,
                error=f"Model '{model_id}' is registered under '{spec.category.value}', not '{category.value}'",
            )

        # 3. API key check
        if spec.requires_provider:
            try:
                from src.lib.KeyManager import is_key_set, load_keys_from_keychain
                load_keys_from_keychain(log=False)
                if not is_key_set(spec.requires_provider):
                    return OperationResult(
                        success=False,
                        model_id=model_id,
                        category=category.value,
                        error=f"API key for provider '{spec.requires_provider}' is not set",
                    )
            except Exception as e:
                return OperationResult(
                    success=False,
                    model_id=model_id,
                    category=category.value,
                    error=f"Could not verify API key for '{spec.requires_provider}': {e}",
                )

        # 4. Local model: disk space + RAM fit
        if spec.is_local:
            if spec.disk_mb > 0:
                try:
                    free_mb = shutil.disk_usage("/").free / (1024 * 1024)
                    if free_mb < spec.disk_mb:
                        return OperationResult(
                            success=False,
                            model_id=model_id,
                            category=category.value,
                            error=f"Insufficient disk space: need {spec.disk_mb}MB, have {free_mb:.0f}MB",
                        )
                except Exception:
                    pass

            if spec.ram_gb > 0:
                try:
                    import psutil
                    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
                    if ram_gb < spec.ram_gb:
                        return OperationResult(
                            success=False,
                            model_id=model_id,
                            category=category.value,
                            error=f"Insufficient RAM: need {spec.ram_gb}GB, have {ram_gb:.1f}GB",
                        )
                except ImportError:
                    pass

            if not self.is_downloaded(category, model_id):
                return OperationResult(
                    success=False,
                    model_id=model_id,
                    category=category.value,
                    error="Model not downloaded (auto_download available)",
                )

        return OperationResult(
            success=True,
            model_id=model_id,
            category=category.value,
            message="Model can be activated",
        )

    # ── Switch / Preference ──────────────────────────────────────────

    def switch(
        self,
        category: ModelCategory,
        model_id: str,
        *,
        auto_download: bool = True,
    ) -> OperationResult:
        """
        Switch active model for a category.
        Pre-validates with can_switch(), then executes setup + preference set.
        """
        check = self.can_switch(category, model_id)
        if not check.success:
            return check

        result = self.mgr.setup(
            category,
            model_id=model_id,
            auto_download=auto_download,
            required=False,
        )
        if result.success:
            self.mgr.set_preference(category, model_id, auto_download=False)
            self._log(f"Switched {category.value} -> {model_id}", LogType.SUCCESS)

        return self._wrap_setup_result(result)

    def set_preference(self, category: ModelCategory, model_id: str) -> bool:
        """Set preference without downloading or loading. Validates registry first."""
        self._spec_or_raise(model_id)
        return self.mgr.set_preference(category, model_id, auto_download=False)

    def clear_preference(self, category: ModelCategory) -> None:
        self.mgr.clear_preference(category)

    # ── Upgrade ────────────────────────────────────────────────────

    def upgrade(self, model_id: str, *, auto_download: bool = True) -> OperationResult:
        """Force re-download and reload a model."""
        spec = self._spec_or_raise(model_id)
        result = self.mgr.setup(
            spec.category,
            model_id=model_id,
            auto_download=auto_download,
            force_download=True,
            required=False,
        )
        op = self._wrap_setup_result(result)
        if op.success:
            self._log(f"Upgraded {model_id}", LogType.SUCCESS)
        return op

    def upgrade_category(self, category: ModelCategory) -> BatchResult:
        """Upgrade all downloaded models in a category."""
        available = self.mgr.list_available(category)
        batch = BatchResult()
        for mid in available:
            op = self.upgrade(mid)
            batch.results[mid] = op
            if op.success:
                batch.succeeded.append(mid)
            else:
                batch.failed.append(mid)
        return batch

    # ── Validation ───────────────────────────────────────────────────

    def validate(self, model_id: str) -> OperationResult:
        """Check if a model is downloaded and ready."""
        spec = self._spec_or_raise(model_id)
        ok = self.mgr.is_downloaded(spec.category, model_id)
        return OperationResult(
            success=ok,
            model_id=model_id,
            category=spec.category.value,
            message="OK" if ok else "Not downloaded",
            error=None if ok else "Model files not found",
        )

    def validate_category(self, category: ModelCategory) -> Dict[str, OperationResult]:
        """Validate all registered models in a category."""
        specs = get_specs_by_category(category)
        return {spec.model_id: self.validate(spec.model_id) for spec in specs}

    # ── Setup Profiles ───────────────────────────────────────────────

    def setup_profile(
        self,
        profile: str = "required",
        *,
        auto_download: bool = True,
        force_download: bool = False,
    ) -> Dict[ModelCategory, SetupResult] | None:
        """
        Setup a predefined profile: 'required' or 'recommended'.
        Profiles are computed dynamically from the registry.
        """
        preferences = self._build_preferences(profile)
        if not preferences:
            self._log(f"Profile '{profile}' yielded no preferences from registry", LogType.WARNING)

        return self.mgr.setup_all(
            preferences=preferences,
            auto_download=auto_download,
            force_download=force_download,
            required_categories=[ModelCategory.EMBEDDING],
        )

    def setup_all(
        self,
        preferences: Dict[ModelCategory, str],
        *,
        auto_download: bool = True,
        force_download: bool = False,
    ) -> Dict[ModelCategory, SetupResult] | None :
        """Setup arbitrary set of categories. Validates all model IDs against registry."""
    
        modelSetup = self.mgr.setup_all(
            preferences=preferences,
            auto_download=auto_download,
            force_download=force_download,
            required_categories=[ModelCategory.EMBEDDING],
        )
        if ModelCategory.EMBEDDING not in modelSetup or not modelSetup[ModelCategory.EMBEDDING].success:
            self._log("Embedding model is required but not set up successfully", LogType.ERROR)
            return None

    # ── Onboarding (Non-Interactive) ─────────────────────────────────

    def onboard(
        self,
        preferences: Optional[Dict[str, str]] = None,
        *,
        auto_download: bool = True,
        detect_system: bool = True,
    ) -> Dict[str, Any]:
        """
        One-shot setup with explicit preferences. No prompts.
        All model IDs are validated against the registry.
        """
        system = self.detect_system() if detect_system else None

        if not preferences and detect_system and system:
            recs = self.get_recommendations(system)
            preferences = {
                k: v
                for k, v in {
                    "embedding": recs.embedding,
                    "nlp": recs.nlp,
                    "ocr": recs.ocr,
                    "reranker": recs.reranker,
                    "vision": recs.vision,
                }.items()
                if v is not None
            }

        if not preferences:
            raise ValueError("No preferences provided and system detection disabled.")

        enum_prefs: Dict[ModelCategory, str] = {}
        for cat_str, mid in preferences.items():
            cat_enum = ModelCategory[cat_str.upper()] if isinstance(cat_str, str) else cat_str
            spec = find_spec(mid)
            if not spec:
                self._log(f"Unknown model '{mid}' in preferences, skipping", LogType.ERROR)
                continue
            if spec.category != cat_enum:
                self._log(
                    f"Model '{mid}' is registered under '{spec.category.value}', "
                    f"not '{cat_enum.value}'. Skipping.",
                    LogType.WARNING,
                )
                continue
            enum_prefs[cat_enum] = mid

        results = self.mgr.setup_all(
            preferences=enum_prefs,
            auto_download=auto_download,
            required_categories=(
                [ModelCategory.EMBEDDING]
                if ModelCategory.EMBEDDING in enum_prefs
                else []
            ),
        )

        summary = {
            "ready": [cat.value for cat, res in results.items() if res.success],
            "failed": {
                cat.value: res.error for cat, res in results.items() if not res.success
            },
        }

        return {
            "system": system.__dict__ if system else None,
            "preferences": preferences,
            "results": {
                cat.value.lower(): {
                    "success": res.success,
                    "model_id": res.model_id,
                    "error": res.error,
                }
                for cat, res in results.items()
            },
            "summary": summary,
        }

    # ── System Detection & Recommendations ───────────────────────────

    @staticmethod
    def detect_system() -> SystemInfo:
        """Detect hardware capabilities without interactive prompts."""
        try:
            import psutil
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            cpu_count = psutil.cpu_count(logical=True) or 1
            ram_gb = ram.total / (1024 ** 3)
            disk_free_gb = disk.free / (1024 ** 3)
        except ImportError:
            ram_gb = 8.0
            disk_free_gb = 50.0
            cpu_count = 4

        has_gpu = False
        gpu_name = ""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                has_gpu = True
                gpu_name = result.stdout.strip().split("\n")[0]
        except Exception:
            pass

        if ram_gb >= 32:
            tier = "overkill"
        elif ram_gb >= 16:
            tier = "large"
        elif ram_gb >= 8:
            tier = "medium"
        else:
            tier = "light"

        return SystemInfo(
            ram_gb=round(ram_gb, 1),
            disk_free_gb=round(disk_free_gb, 1),
            cpu_cores=cpu_count,
            has_gpu=has_gpu,
            gpu_name=gpu_name,
            tier=tier,
        )

    def get_recommendations(self, system: Optional[SystemInfo] = None) -> RecommendationProfile:
        """
        Compute model recommendations from the live registry based on hardware tier.
        Never hardcodes model IDs.
        """
        if system is None:
            system = self.detect_system()

        tier_map = {
            "light": ModelTier.DOLPHIN,
            "medium": ModelTier.DOLPHIN,
            "large": ModelTier.ORCA,
            "overkill": ModelTier.WHALE,
        }
        prefer_tier = tier_map.get(system.tier, ModelTier.DOLPHIN)

        def pick(cat: ModelCategory) -> Optional[str]:
            return self._pick_best_for_category(cat, prefer_tier=prefer_tier)

        return RecommendationProfile(
            embedding=pick(ModelCategory.EMBEDDING),
            nlp=pick(ModelCategory.NLP),
            ocr=pick(ModelCategory.OCR),
            reranker=pick(ModelCategory.RERANKER),
            vision=pick(ModelCategory.VISION),
        )

    # ── Key Management ─────────────────────────────────────────────

    def set_key(self, provider: str, key: str, validate: bool = True) -> bool:
        try:
            set_api_key(provider, key, validate=validate)
            return True
        except Exception as e:
            self._log(f"Failed to set key for {provider}: {e}", LogType.ERROR)
            return False

    def remove_key(self, provider: str) -> bool:
        try:
            clear_api_key(provider)
            return True
        except Exception:
            return False

    def list_key_status(self) -> Dict[str, bool]:
        return {
            provider: bool(os.getenv(env_var))
            for provider, (env_var, _) in PROVIDER_REGISTRY.items()
        }

    # ── Resource Management ─────────────────────────────────────────

    def unload(self, category: ModelCategory, model_id: Optional[str] = None) -> bool:
        return self.mgr.unload(category, model_id)

    def unload_all(self) -> int:
        return self.mgr.unload_all()

    def discard(
        self,
        category: ModelCategory,
        model_id: Optional[str] = None,
        *,
        delete: bool = False,
    ) -> bool:
        return self.mgr.discard(category, model_id, delete=delete)

    # ── Registry Introspection ─────────────────────────────────────

    def registry_stats(self) -> Dict[str, Any]:
        """Return high-level stats about the model registry."""
        return get_registry_stats()

    def get_models_by_provider(
        self,
        provider: str,
        category: Optional[ModelCategory] = None,
    ) -> List[Dict[str, Any]]:
        """Query registry for models by provider."""
        specs = get_models_per_provider(provider, category=category, as_dicts=True)
        return specs if isinstance(specs, list) else []

    def get_models_by_tier(self, tier: ModelTier) -> List[Dict[str, Any]]:
        """Query registry for models by tier."""
        from dataclasses import asdict
        return [asdict(s) for s in get_specs_by_tier(tier)]

    # ── Internal: preference builders ────────────────────────────────

    def _build_preferences(self, profile: str) -> Dict[ModelCategory, str]:
        """
        Build preferences dynamically from the registry.
        'required' -> embedding only.
        'recommended' -> embedding, nlp, ocr, reranker.
        """
        profile_categories = {
            "required": [ModelCategory.EMBEDDING],
            "recommended": [
                ModelCategory.EMBEDDING,
                ModelCategory.NLP,
                ModelCategory.OCR,
                ModelCategory.RERANKER,
            ],
        }
        prefs: Dict[ModelCategory, str] = {}
        for cat in profile_categories.get(profile, []):
            best = self._pick_best_for_category(cat)
            if best:
                prefs[cat] = best
        return prefs


# ── Singleton convenience ────────────────────────────────────────────

modelsOrchestrator = ModelsOrchestrator()