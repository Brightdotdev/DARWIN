"""
Unified high-level API for all model operations.
Single entry point: initialize -> ensure -> use -> unload.
"""

from typing import Optional, Dict, List, Any, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.core.Models.Lib.ModelRegistry import ModelCategory, find_spec, get_specs_by_category,MODEL_SPECS
from src.core.Models.ModelConfig import get_active_model, set_active_model
from src.core.Logger.DarwinLogger import darwin, LogType

# Import all factories
from src.core.Models.Infrence.EmbeddingInferenceFactory import (
    get_embedding_inference,
    set_embedding_preference,
    discard_embedding_model,
    get_embedding_model_path,
    is_embedding_model_downloaded,
)
from src.core.Models.Infrence.NLPInferenceFactory import (
    get_nlp_inference,
    set_nlp_preference,
    discard_nlp_model,
    get_nlp_model_path,
    is_nlp_model_downloaded,
)
from src.core.Models.Infrence.RerankerInferenceFactory import (
    get_reranker_inference,
    set_reranker_preference,
    discard_reranker_model,
    get_reranker_model_path,
    is_reranker_model_downloaded,
)
from src.core.Models.Infrence.VisionInfrenceFactory import (
    get_vision_inference,
    set_vision_preference,
    discard_vision_model,
    get_vision_model_path,
    is_vision_model_downloaded,
)
from src.core.Models.Infrence.OCRInferenceFactory import (
    get_ocr_inference,
    set_ocr_preference,
    discard_ocr_model,
    get_ocr_model_path,
    is_ocr_model_downloaded,
)
from src.lib.KeyManager import is_key_set, load_keys_from_keychain


@dataclass
class CategoryStatus:
    """Status for a single model category."""
    category: str
    active_model_id: Optional[str]
    available: bool
    available_models: List[str] = field(default_factory=list)
    error: Optional[str] = None
    is_required: bool = False


@dataclass
class SetupResult:
    """Result of a single model setup operation."""
    category: str
    model_id: str
    success: bool
    downloaded: bool = False
    loaded: bool = False
    error: Optional[str] = None
    model_info: Optional[dict] = None

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"SetupResult({status} {self.category}/{self.model_id})"


@dataclass
class ManagerStatus:
    """
    Full metadata snapshot returned by initialize().
    """
    timestamp: str
    categories: Dict[str, CategoryStatus]
    missing_required: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def all_ready(self) -> bool:
        """True if all required categories are available."""
        return len(self.missing_required) == 0

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["Model Manager Status"]
        lines.append(f"  Timestamp: {self.timestamp}")
        for cat_name, status in self.categories.items():
            icon = "" if status.available else " "
            required = " (required)" if status.is_required else ""
            active = f" -> {status.active_model_id}" if status.active_model_id else ""
            available = f" ({len(status.available_models)} ready)"
            error = f"  Error: {status.error}" if status.error else ""
            lines.append(f"  {icon} {cat_name.upper()}{required}{active}{available}")
            if error:
                lines.append(f"     {error}")
        if self.missing_required:
            lines.append(f"   Missing required: {', '.join(self.missing_required)}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"   {w}")
        return "\n".join(lines)


class ModelManagerError(Exception):
    """Raised when a required model operation fails."""
    def __init__(self, message: str, category: Optional[str] = None):
        self.category = category
        prefix = f"[{category}] " if category else ""
        super().__init__(f"{prefix}{message}")


class ModelManager:
    """
    Unified entry point for all model operations.
    """

    # Category -> factory mappings
    _FACTORIES = {
        ModelCategory.EMBEDDING: {
            "get": get_embedding_inference,
            "set": set_embedding_preference,
            "discard": discard_embedding_model,
            "path": get_embedding_model_path,
            "downloaded": is_embedding_model_downloaded,
        },
        ModelCategory.NLP: {
            "get": get_nlp_inference,
            "set": set_nlp_preference,
            "discard": discard_nlp_model,
            "path": get_nlp_model_path,
            "downloaded": is_nlp_model_downloaded,
        },
        ModelCategory.RERANKER: {
            "get": get_reranker_inference,
            "set": set_reranker_preference,
            "discard": discard_reranker_model,
            "path": get_reranker_model_path,
            "downloaded": is_reranker_model_downloaded,
        },
        ModelCategory.VISION: {
            "get": get_vision_inference,
            "set": set_vision_preference,
            "discard": discard_vision_model,
            "path": get_vision_model_path,
            "downloaded": is_vision_model_downloaded,
        },
        ModelCategory.OCR: {
            "get": get_ocr_inference,
            "set": set_ocr_preference,
            "discard": discard_ocr_model,
            "path": get_ocr_model_path,
            "downloaded": is_ocr_model_downloaded,
        },
    }

    REQUIRED_CATEGORIES = {ModelCategory.EMBEDDING}

    def __init__(self):
        self._loaded_models: Dict[str, Any] = {}  # cache_key -> model instance

    # ────────────────────────────────────────────────────────────────
    # INITIALIZE – full health check
    # ────────────────────────────────────────────────────────────────

    def initialize(
        self,
        *,
        auto_setup: bool = False,
        auto_download: bool = True,
        force_download: bool = False,
        preferences: Optional[Dict[ModelCategory, str]] = None,
    ) -> ManagerStatus:
        timestamp = datetime.now().isoformat(timespec='seconds')
        categories: Dict[str, CategoryStatus] = {}
        warnings: List[str] = []
        missing_required: List[str] = []

        for cat in ModelCategory:
            cat_name = cat.name.lower()
            is_required = cat in self.REQUIRED_CATEGORIES

            active_id = get_active_model(cat_name)
            all_specs = get_specs_by_category(cat)

            available_models = []
            for spec in all_specs:
                if self._is_model_available(cat, spec.model_id):
                    available_models.append(spec.model_id)

            active_available = active_id and (active_id in available_models)
            any_available = bool(available_models)

            # ── FIX: initialize error here ──
            error = None

            if active_available:
                available = True
            elif active_id and not any_available:
                available = False
                error = f"Active model '{active_id}' not available and no other model ready"
                warnings.append(f"[{cat_name}] Active model '{active_id}' is not available.")
            elif not active_id and any_available:
                available = True
                warnings.append(f"[{cat_name}] No active model set, but {len(available_models)} model(s) are ready. Set a preference.")
            else:
                available = False
                error = "No model available"

            if is_required and not available:
                missing_required.append(cat_name)

            categories[cat_name] = CategoryStatus(
                category=cat_name,
                active_model_id=active_id,
                available=available,
                available_models=available_models,
                error=error,
                is_required=is_required,
            )

        # Auto-setup missing required
        if auto_setup and missing_required:
            darwin.info(f"Auto-setup triggered for: {', '.join(missing_required)}")
            if preferences is None:
                preferences = {}

            for cat_name in missing_required:
                cat = ModelCategory[cat_name.upper()]
                if cat in preferences:
                    model_id = preferences[cat]
                else:
                    model_id = self._pick_default(cat)
                    if model_id:
                        preferences[cat] = model_id
                    else:
                        darwin.warning(f"Could not auto-select default for {cat_name}")
                        warnings.append(f"Could not auto-select default for {cat_name}")
                        continue

                result = self.setup(
                    cat,
                    model_id=model_id,
                    auto_download=auto_download,
                    force_download=force_download,
                    required=False,
                )
                if result.success:
                    categories[cat_name].available = True
                    categories[cat_name].active_model_id = model_id
                    categories[cat_name].error = None
                    if model_id not in categories[cat_name].available_models:
                        categories[cat_name].available_models.append(model_id)
                else:
                    warnings.append(f"Auto-setup failed for {cat_name}: {result.error}")

            missing_required = [
                cn for cn, st in categories.items()
                if st.is_required and not st.available
            ]

        status_obj = ManagerStatus(
            timestamp=timestamp,
            categories=categories,
            missing_required=missing_required,
            warnings=warnings,
        )

        W = 56
        darwin.info(f"{'-- Models ' + '-' * (W - 10)}", style="bold cyan")

        cat_order = ["embedding", "nlp", "ocr", "reranker", "vision"]
        for cat_name in cat_order:
            if cat_name not in status_obj.categories:
                continue
            cs = status_obj.categories[cat_name]
            required_tag = " (required)" if cs.is_required else ""

            if cs.available:
                status_icon = "+"
                status_style = "green bold"
                status_label = "ready"
            elif cs.error and "not available" in (cs.error or ""):
                status_icon = "!"
                status_style = "orange1 bold"
                status_label = "warn"
            else:
                status_icon = "x"
                status_style = "red bold"
                status_label = "missing"

            darwin.info(f"  {cat_name.upper()}{required_tag}", style="bold white")
            if cs.active_model_id:
                model_short = cs.active_model_id.split("/")[-1]
                darwin.info(f"    Model: {model_short}", style="cyan")
            darwin.info(
                f"    Status: [{status_icon}] {status_label}  |  "
                f"{len(cs.available_models)} available",
                style=status_style,
            )
            if cs.error:
                darwin.info(f"    Error: {cs.error}", style="red")

        if status_obj.missing_required:
            darwin.info("")
            darwin.warning(
                f"  Missing required: {', '.join(status_obj.missing_required)}. "
                "Some features will be disabled."
            )

        return status_obj

    # ────────────────────────────────────────────────────────────────
    # SETUP – full lifecycle
    # ────────────────────────────────────────────────────────────────

    def setup(
        self,
        category: ModelCategory,
        model_id: Optional[str] = None,
        *,
        auto_download: bool = True,
        force_download: bool = False,
        required: bool = False,
        **kwargs,
    ) -> SetupResult:
        cat_name = category.name.lower()
        resolved_id = model_id or get_active_model(cat_name)

        if not resolved_id:
            resolved_id = self._pick_default(category)
            if not resolved_id:
                error = f"No default model for {cat_name}"
                result = SetupResult(cat_name, "none", False, error=error)
                if required:
                    darwin.error(error)
                    return result
                return result
            darwin.info(f"Auto-selected default for {cat_name}: {resolved_id}")

        # Check cache
        cache_key = f"{cat_name}:{resolved_id}"
        if cache_key in self._loaded_models:
            model = self._loaded_models[cache_key]
            if hasattr(model, 'is_loaded') and model.is_loaded():
                darwin.info(f"{cat_name} model already loaded: {resolved_id}")
                return SetupResult(
                    cat_name, resolved_id, True,
                    loaded=True, model_info=self._get_model_info(model)
                )

        # Ensure availability
        if not self._ensure_model_available(category, resolved_id, auto_download, force_download):
            error = f"Model {resolved_id} not available"
            result = SetupResult(cat_name, resolved_id, False, error=error)
            if required:
                darwin.error(error)
                return result
            return result

        # Get factory and load
        factory = self._get_factory(category, "get")
        if factory is None:
            error = f"No factory for {cat_name}"
            result = SetupResult(cat_name, resolved_id, False, error=error)
            if required:
                darwin.error(error)
                return result
            return result

        try:
            model = factory(model_id=resolved_id, **kwargs)
            if model is None:
                error = f"Factory returned None for {resolved_id}"
                result = SetupResult(cat_name, resolved_id, False, error=error)
                if required:
                    darwin.error(error)
                    return result
                return result

            # Load if needed
            if hasattr(model, 'load'):
                if hasattr(model, 'is_loaded') and not model.is_loaded():
                    model.load()
                elif not hasattr(model, 'is_loaded'):
                    model.load()

            info = self._get_model_info(model)
            if info:
                darwin.success(f"{cat_name} model loaded:")
                for key, value in info.items():
                    darwin.info(f"  {key}: {value}")

            self._loaded_models[cache_key] = model
            set_active_model(cat_name, resolved_id)

            return SetupResult(
                cat_name, resolved_id, True,
                downloaded=False, loaded=True, model_info=info
            )

        except Exception as e:
            error = f"Failed to load {resolved_id}: {e}"
            darwin.error(error)
            result = SetupResult(cat_name, resolved_id, False, error=error)
            if required:
                darwin.error(error)
                return result 
            return result

    def setup_all(
        self,
        preferences: Optional[Dict[ModelCategory, str]] = None,
        *,
        auto_download: bool = True,
        force_download: bool = False,
        required_categories: Optional[List[ModelCategory]] = None,
        **kwargs,
    ) -> Dict[ModelCategory, SetupResult] | None:
        if preferences is None:
            preferences = {}
        if not preferences:
            for cat in ModelCategory:
                active = get_active_model(cat.name.lower())
                if active:
                    preferences[cat] = active

        required_set = set(required_categories or [])
        results: Dict[ModelCategory, SetupResult] = {}

        # Enforce embedding
        if ModelCategory.EMBEDDING not in preferences:
            darwin.error("Embedding model is required but not specified")
            if ModelCategory.EMBEDDING in required_set or not required_set:
                darwin.error("Embedding model is required but not specified")
                return

        for cat, mid in preferences.items():
            is_required = cat in required_set or cat == ModelCategory.EMBEDDING
            try:
                result = self.setup(
                    cat, mid,
                    auto_download=auto_download,
                    force_download=force_download,
                    required=is_required,
                    **kwargs,
                )
            except ModelManagerError as e:
                result = SetupResult(cat.name.lower(), mid or "unknown", False, error=str(e))
            results[cat] = result
            if result.success:
                darwin.success(f"{cat.name}: {mid} ready")
            else:
                darwin.error(f"{cat.name}: {result.error}")
        return results

    # ────────────────────────────────────────────────────────────────
    # CORE: get / ensure
    # ────────────────────────────────────────────────────────────────

    def ensure(
        self,
        category: ModelCategory,
        model_id: Optional[str] = None,
        auto_download: bool = True,
        **kwargs,
    ) -> Optional[Any]:
        try:
            return self.get(category, model_id, auto_download, **kwargs)
        except Exception as e:
            darwin.warning(f"Failed to get {category.name} model: {e}")
            return None

    def get(
        self,
        category: ModelCategory,
        model_id: Optional[str] = None,
        auto_download: bool = True,
        **kwargs,
    ) -> Any:
        cat_name = category.name.lower()
        resolved_id = model_id or get_active_model(cat_name)
        if not resolved_id:
            error = f"No model specified for {cat_name} and no active model set"
            darwin.error(error, cat_name)
            return SetupResult(cat_name, 'none', False, error=error)

        cache_key = f"{cat_name}:{resolved_id}"
        if cache_key in self._loaded_models:
            model = self._loaded_models[cache_key]
            if hasattr(model, 'is_loaded') and model.is_loaded():
                return model

        if not self._ensure_model_available(category, resolved_id, auto_download):
            error = f"Model {resolved_id} not available"
            darwin.error(error, cat_name)
            return SetupResult(cat_name, 'none', False, error=error)

        factory = self._get_factory(category, "get")
        if factory is None:
            error = f"No factory for {cat_name}"
            darwin.error(error, cat_name)
            return SetupResult(cat_name, 'none', False, error=error)

        try:
            model = factory(model_id=resolved_id, **kwargs)
            if model is None:
                error = f"Factory returned None for {resolved_id}"
                darwin.error(error, cat_name)
                return SetupResult(cat_name, 'none', False, error=error)

            if hasattr(model, 'load'):
                if hasattr(model, 'is_loaded') and not model.is_loaded():
                    model.load()
                elif not hasattr(model, 'is_loaded'):
                    model.load()

            self._loaded_models[cache_key] = model
            return model
        except Exception as e:
            error = f"Failed to load {resolved_id}: {e}"
            darwin.error(error, cat_name)
            return SetupResult(cat_name, 'none', False, error=error)

    # ────────────────────────────────────────────────────────────────
    # CONTEXT MANAGER
    # ────────────────────────────────────────────────────────────────

    @contextmanager
    def use(
        self,
        category: ModelCategory,
        model_id: Optional[str] = None,
        auto_download: bool = True,
        unload_on_exit: bool = True,
        **kwargs,
    ) -> Iterator[Optional[Any]]:
        model = None
        try:
            model = self.get(category, model_id, auto_download, **kwargs)
            yield model
        except Exception as e:
            darwin.error(f"Error in context: {e}")
            yield None
        finally:
            if unload_on_exit and model is not None:
                cat_name = category.name.lower()
                resolved_id = model_id or get_active_model(cat_name)
                if resolved_id:
                    self.unload(category, resolved_id)

    # ────────────────────────────────────────────────────────────────
    # PREFERENCE MANAGEMENT
    # ────────────────────────────────────────────────────────────────

    def set_preference(
        self,
        category: ModelCategory,
        model_id: str,
        auto_download: bool = True,
        force_download: bool = False,
    ) -> bool:
        factory = self._get_factory(category, "set")
        if factory is None:
            return False
        try:
            factory(model_id=model_id, auto_download=auto_download, force=force_download)
            return True
        except Exception as e:
            darwin.error(f"Failed to set preference: {e}")
            return False

    def clear_preference(self, category: ModelCategory) -> None:
        set_active_model(category.name.lower(), None)

    def get_active(self, category: ModelCategory) -> Optional[str]:
        return get_active_model(category.name.lower())

    # ────────────────────────────────────────────────────────────────
    # DISCARD — convenience: unload + remove preference + optionally delete
    # ────────────────────────────────────────────────────────────────

    def discard(
        self,
        category: ModelCategory,
        model_id: Optional[str] = None,
        *,
        delete: bool = False,
        unload_first: bool = True,
    ) -> bool:
        """
        Convenience method to fully discard a model.

        Steps:
          1. Unload from memory (if loaded).
          2. Call the factory discard handler (removes preference, optionally deletes files).

        Args:
            category: The model category.
            model_id: Specific model to discard. Uses active model if None.
            delete: If True, delete model files from disk.
            unload_first: If True, unload from cache before discarding.

        Returns:
            True if discard succeeded.
        """
        cat_name = category.name.lower()
        resolved_id = model_id or get_active_model(cat_name)

        if not resolved_id:
            darwin.warning(f"No model specified or active for {cat_name}, nothing to discard")
            return False

        # 1. Unload if requested
        if unload_first:
            self.unload(category, resolved_id)

        # 2. Call factory discard
        discard_factory = self._get_factory(category, "discard")
        if discard_factory is None:
            darwin.error(f"No discard factory for {cat_name}")
            return False

        try:
            discard_factory(model_id=resolved_id, delete=delete)
            darwin.info(f"Discarded {cat_name}/{resolved_id}" + (" (files deleted)" if delete else ""))
            return True
        except Exception as e:
            darwin.error(f"Failed to discard {cat_name}/{resolved_id}: {e}")
            return False

    def discard_all(
        self,
        categories: Optional[List[ModelCategory]] = None,
        *,
        delete: bool = False,
    ) -> Dict[str, bool]:
        """
        Discard models across multiple (or all) categories.

        Args:
            categories: List of categories to discard from. Defaults to all.
            delete: If True, delete files for all discarded models.

        Returns:
            Dict mapping "category/model_id" -> success bool.
        """
        targets = categories or list(ModelCategory)
        results: Dict[str, bool] = {}

        for cat in targets:
            active = get_active_model(cat.name.lower())
            if active:
                success = self.discard(cat, active, delete=delete)
                results[f"{cat.name.lower()}/{active}"] = success
            else:
                results[f"{cat.name.lower()}/none"] = True  # Nothing to discard

        return results

    # ────────────────────────────────────────────────────────────────
    # RESOURCE MANAGEMENT
    # ────────────────────────────────────────────────────────────────

    def unload(self, category: ModelCategory, model_id: Optional[str] = None) -> bool:
        cat_name = category.name.lower()
        resolved_id = model_id or get_active_model(cat_name)
        if not resolved_id:
            return False

        cache_key = f"{cat_name}:{resolved_id}"
        if cache_key not in self._loaded_models:
            return False

        model = self._loaded_models.pop(cache_key)
        if hasattr(model, 'unload'):
            try:
                model.unload()
            except Exception as e:
                darwin.warning(f"Error unloading {resolved_id}: {e}")
        darwin.info(f"Unloaded {cat_name}/{resolved_id}")
        return True

    def unload_all(self) -> int:
        count = 0
        for key in list(self._loaded_models.keys()):
            try:
                cat_name, model_id = key.split(":", 1)
                # Find category enum
                cat_enum = ModelCategory[cat_name.upper()]
                if self.unload(cat_enum, model_id):
                    count += 1
            except (KeyError, ValueError):
                # Fallback: just pop and unload
                model = self._loaded_models.pop(key, None)
                if model and hasattr(model, 'unload'):
                    try:
                        model.unload()
                    except Exception:
                        pass
                count += 1
        darwin.info(f"Unloaded {count} model(s)")
        return count

    def is_available(self, category: ModelCategory, model_id: str) -> bool:
        """
        Check if a specific model is available for use.
        - For local models: returns True if files are downloaded.
        - For API models: returns True if the API key is set.
        """
        return self._is_model_available(category, model_id)

    # ────────────────────────────────────────────────────────────────
    # STATUS & DISCOVERY
    # ────────────────────────────────────────────────────────────────

    def get_model_info(self, category: ModelCategory, model_id: Optional[str] = None) -> Optional[dict]:
        cat_name = category.name.lower()
        resolved_id = model_id or get_active_model(cat_name)
        if not resolved_id:
            return None
        cache_key = f"{cat_name}:{resolved_id}"
        if cache_key in self._loaded_models:
            return self._get_model_info(self._loaded_models[cache_key])
        return None

    def is_loaded(self, category: ModelCategory, model_id: Optional[str] = None) -> bool:
        cat_name = category.name.lower()
        resolved_id = model_id or get_active_model(cat_name)
        if not resolved_id:
            return False
        cache_key = f"{cat_name}:{resolved_id}"
        if cache_key not in self._loaded_models:
            return False
        model = self._loaded_models[cache_key]
        if hasattr(model, 'is_loaded'):
            return model.is_loaded()
        return True

    def is_downloaded(self, category: ModelCategory, model_id: str) -> bool:
        factory = self._get_factory(category, "downloaded")
        if factory is None:
            return False
        return factory(model_id=model_id)

    def get_model_path(self, category: ModelCategory, model_id: str) -> Optional[Path]:
        factory = self._get_factory(category, "path")
        if factory is None:
            return None
        return factory(model_id=model_id)

    def list_available(self, category: ModelCategory) -> List[str]:
        result = []
        for spec in MODEL_SPECS:
            if spec.category != category:
                continue
            if spec.requires_provider:
                try:
                    load_keys_from_keychain(log=False)
                    if is_key_set(spec.requires_provider):
                        result.append(spec.model_id)
                except Exception:
                    pass
            else:
                if self.is_downloaded(category, spec.model_id):
                    result.append(spec.model_id)
        return result

    def health(self, category: ModelCategory) -> Dict[str, Any]:
        model_id = get_active_model(category.name.lower())
        return {
            "category": category.name.lower(),
            "model_id": model_id,
            "has_preference": model_id is not None,
            "is_loaded": self.is_loaded(category),
            "is_downloaded": self.is_downloaded(category, model_id) if model_id else False,
            "loaded_models": list(self._loaded_models.keys()),
        }

    # ────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ────────────────────────────────────────────────────────────────

    def _get_factory(self, category: ModelCategory, action: str):
        cat_factories = self._FACTORIES.get(category)
        if not cat_factories:
            return None
        return cat_factories.get(action)

    def _pick_default(self, category: ModelCategory) -> Optional[str]:
        candidates = get_specs_by_category(category)
        if not candidates:
            return None

        # Try downloaded first
        for spec in candidates:
            if not spec.requires_provider and self.is_downloaded(category, spec.model_id):
                return spec.model_id

        # Try API with key
        for spec in candidates:
            if spec.requires_provider:
                try:
                    load_keys_from_keychain(log=False)
                    if is_key_set(spec.requires_provider):
                        return spec.model_id
                except Exception:
                    pass

        # Local (will trigger download)
        for spec in candidates:
            if not spec.requires_provider:
                return spec.model_id

        # First API
        return candidates[0].model_id if candidates else None

    def _ensure_model_available(
        self,
        category: ModelCategory,
        model_id: str,
        auto_download: bool = True,
        force_download: bool = False,
    ) -> bool:
        spec = find_spec(model_id)
        if spec is None or spec.category != category:
            return False

        if spec.requires_provider:
            try:
                load_keys_from_keychain(log=False)
                return is_key_set(spec.requires_provider)
            except Exception:
                return False

        if self.is_downloaded(category, model_id):
            return True

        if not auto_download:
            return False

        set_factory = self._get_factory(category, "set")
        if set_factory is None:
            return False

        try:
            set_factory(model_id=model_id, auto_download=True, force=force_download)
            return self.is_downloaded(category, model_id)
        except Exception as e:
            darwin.error(f"Download failed: {e}")
            return False

    def _is_model_available(self, category: ModelCategory, model_id: str) -> bool:
        spec = find_spec(model_id)
        if spec is None or spec.category != category:
            return False
        if spec.requires_provider:
            try:
                load_keys_from_keychain(log=False)
                return is_key_set(spec.requires_provider)
            except Exception:
                return False
        return self.is_downloaded(category, model_id)

    def _get_model_info(self, model) -> Optional[dict]:
        if hasattr(model, 'get_model_info'):
            try:
                return model.get_model_info()
            except Exception:
                pass
        return {
            "type": type(model).__name__,
            "loaded": hasattr(model, 'is_loaded') and model.is_loaded() if hasattr(model, 'is_loaded') else True,
        }


# ────────────────────────────────────────────────────────────────────
# SINGLETON
# ────────────────────────────────────────────────────────────────────

modelManager = ModelManager()