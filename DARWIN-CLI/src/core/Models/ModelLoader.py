"""
src/Models/ModelLoader.py
Lazy-loads models, caches in memory, handles unload.
Wrappers live in wrappers.py. Vision dispatch lives in vision_loader.py.
"""

from pathlib import Path
from typing import Optional, Any

from src.core.Models.ModelStore import ModelStore
from src.core.Models.ModelRegistry import get_spec, ModelCategory, REGISTRY
from src.core.Models.ModelLoading.Wrappers import (
    HFEmbeddingWrapper,
    APIEmbeddingWrapper,
    APIRerankerWrapper,
    APIVisionWrapper,
)
from src.core.Models.ModelLoading.vision_loader import load_vision_model
from src.core.Logger.DarwinLogger import darwin, LogType


class ModelLoader:
    """Lazy-loads models, caches in memory, handles unload."""

    def __init__(self, store: ModelStore, preferences: dict[str, str] = None):
        self.store = store
        self._preferences = preferences or {}
        self._cache: dict[str, Any] = {}

    # ── Key resolution ─────────────────────────────────────────────────────

    def _resolve_key(self, key: str) -> str:
        """Resolve any model identifier to its canonical registry key."""
        if key in REGISTRY:
            return key
        id_to_key = {spec.model_id: k for k, spec in REGISTRY.items()}
        return id_to_key.get(key, key)

    # ── Preferences ────────────────────────────────────────────────────────

    def set_preference(self, category: str, key: str, cleanup_old: bool = False) -> None:
        """Set model preference for a category."""
        old_key = self._preferences.get(category)
        if old_key and old_key != key:
            resolved_old = self._resolve_key(old_key)
            self._cache.pop(resolved_old, None)
            if cleanup_old:
                self.store.delete(resolved_old)
        self._preferences[category] = key

    @property
    def preferences(self) -> dict[str, str]:
        return dict(self._preferences)

    def get_preferences(self) -> dict[str, str]:
        return dict(self._preferences)

    # ── Model key helpers ──────────────────────────────────────────────────

    def get_model_key(self, category: ModelCategory) -> str:
        """
        Get the active model_id for a category.
        Returns 'not set' if nothing is configured.
        """
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str, "not set")
        registry_key = self._resolve_key(key)
        if registry_key in REGISTRY:
            return get_spec(registry_key).model_id
        return key

    def get_all_model_keys(self) -> dict[str, str]:
        """
        Return the active model_id for every category.

        Example:
            {
                "embedding": "BAAI/bge-m3",
                "reranker":  "cross-encoder/ms-marco-MiniLM-L-12-v2",
                "vision":    "not set",
                "tokenizer": "not set",
            }
        """
        return {
            category.value: self.get_model_key(category)
            for category in ModelCategory
        }

    def get_model_name(self, category: ModelCategory) -> str:
        """Get human-readable model name. Alias for get_model_key."""
        return self.get_model_key(category)

    # ── Load ───────────────────────────────────────────────────────────────

    def get(self, category: ModelCategory) -> Any:
        """Get loaded model instance for a category. Loads on first access."""
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str)

        if not key:
            raise ValueError(f"No model set for '{cat_str}'. Run darwin setup first.")

        registry_key = self._resolve_key(key)

        if registry_key not in REGISTRY:
            raise ValueError(
                f"Unknown model '{key}' for '{cat_str}'. "
                f"Available: {list(REGISTRY.keys())}"
            )

        if registry_key in self._cache:
            return self._cache[registry_key]

        spec = get_spec(registry_key)

        # ── API models — skip store.ensure entirely ──
        if spec.loader == "api_embedding":
            wrapper = self._load_api_embedding(spec.model_id)
            self._cache[registry_key] = wrapper
            return wrapper

        if spec.loader == "api_reranker":
            wrapper = self._load_api_reranker(spec.model_id)
            self._cache[registry_key] = wrapper
            return wrapper

        if spec.loader == "api_vision":
            wrapper = self._load_api_vision(spec.model_id)
            self._cache[registry_key] = wrapper
            return wrapper

        # ── Local models — resolve path first ──
        path = str(self.store.ensure(registry_key))

        if spec.loader == "embedding_st":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(path)
            self._cache[registry_key] = model
            return model

        elif spec.loader == "embedding_hf":
            import torch
            from transformers import AutoTokenizer, AutoModel

            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModel.from_pretrained(path)
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
            model.to(device)
            model.eval()
            wrapper = HFEmbeddingWrapper(model, tokenizer, device)
            self._cache[registry_key] = wrapper
            return wrapper

        elif spec.loader == "cross_encoder":
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(path)
            self._cache[registry_key] = model
            return model

        elif spec.loader == "spacy":
            from src.core.Models.ModelLoading.SpacyConceptExtractor import SpacyConceptExtractor
            model = SpacyConceptExtractor(path, spec.model_id)
            self._cache[registry_key] = model
            return model

        elif spec.loader == "transformers":
            result = load_vision_model(path)
            self._cache[registry_key] = result
            return result

        raise ValueError(f"Unknown loader type: {spec.loader}")

    # ── Unload ─────────────────────────────────────────────────────────────

    def unload(self, category: ModelCategory) -> None:
        """Unload model from memory and free GPU/CPU RAM."""
        from src.features.Hardware.DeviceManager import DeviceManager

        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str)

        if not key:
            return

        resolved_key = self._resolve_key(key)
        cached = self._cache.pop(resolved_key, None)

        if cached is None:
            return

        try:
            import torch

            if isinstance(cached, tuple):
                # Vision: (processor, model, device) — release all objects
                for obj in cached:
                    if hasattr(obj, "cpu"):
                        obj.cpu()
            else:
                if hasattr(cached, "cpu"):
                    cached.cpu()

            del cached

            cfg = DeviceManager.get()
            if cfg.use_gpu:
                torch.cuda.empty_cache()

        except Exception:
            pass

    # ── Readiness ──────────────────────────────────────────────────────────

    def ensure_ready(self, category: ModelCategory) -> bool:
        """Check model is downloaded and loadable. Returns True if ready."""
        try:
            self.get(category)
            return True
        except (FileNotFoundError, OSError, ValueError) as e:
            darwin.log(f"Model {category.value} not ready: {e}", LogType.ERROR)
            return False

    # ── API loader helpers ─────────────────────────────────────────────────

    def _load_api_embedding(self, model_id: str) -> APIEmbeddingWrapper:
        return APIEmbeddingWrapper(model_id)

    def _load_api_reranker(self, model_id: str) -> APIRerankerWrapper:
        return APIRerankerWrapper(model_id)

    def _load_api_vision(self, model_id: str) -> APIVisionWrapper:
        return APIVisionWrapper(model_id)