# src/core/Models/ModelLoader.py

from pathlib import Path
from typing import Optional, Any

from src.core.Models.ModelStore import get_model_store, ModelStore
from src.core.Models.ModelRegistry import get_spec, ModelCategory, REGISTRY
from src.core.Models.ModelConfig import load_active_models, set_active_model
from src.core.Models.ModelLoading.Wrappers import (
    HFEmbeddingWrapper,
    APIEmbeddingWrapper,
    APIRerankerWrapper,
    APIVisionWrapper,
)
from src.core.Models.ModelLoading.vision_loader import load_vision_model
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Plugins.PluginManager import PluginManager, PluginRequiredError
from src.core.Plugins.TorchPlugin import TorchPlugin
from src.core.Models.ModelDownloader import ModelDownloader


# Lazy torch check
_torch_available = None
def _check_torch() -> bool:
    global _torch_available
    if _torch_available is None:
        try:
            import torch
            _torch_available = True
        except ImportError:
            _torch_available = False
    return _torch_available


class ModelLoader:
    def __init__(self, store: Optional[ModelStore] = None, preferences: dict[str, str] = None):
        self.store = store or get_model_store()
        if preferences is not None:
            self._preferences = dict(preferences)
        else:
            self._preferences = {k: v or "" for k, v in load_active_models().items()}

        self._cache: dict[str, Any] = {}
        self.plugin_manager = PluginManager()
        self.torch_plugin = TorchPlugin(self.plugin_manager)
        self.downloader = ModelDownloader()

    # ── Key resolution ─────────────────────────────────────────────────────
    def _resolve_key(self, key: str) -> str:
        if key in REGISTRY:
            return key
        # Find all registry keys with this model_id
        matches = [k for k, spec in REGISTRY.items() if spec.model_id == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Prefer non-ONNX if torch available, otherwise ONNX
            try:
                from src.features.Hardware.Capabilities import Capabilities
                caps = Capabilities.detect()
                if caps.torch_available:
                    non_onnx = [k for k in matches if not k.endswith("-onnx")]
                    if non_onnx:
                        return non_onnx[0]
                else:
                    onnx = [k for k in matches if k.endswith("-onnx")]
                    if onnx:
                        return onnx[0]
            except Exception:
                pass
            # Fallback: prefer non-ONNX
            non_onnx = [k for k in matches if not k.endswith("-onnx")]
            if non_onnx:
                return non_onnx[0]
            return matches[0]
        return key

    # ── Preferences ────────────────────────────────────────────────────────
    def set_preference(self, category: str, key: str, cleanup_old: bool = False) -> None:
        old_key = self._preferences.get(category)
        if old_key and old_key != key:
            resolved_old = self._resolve_key(old_key)
            self._cache.pop(resolved_old, None)
            if cleanup_old:
                try:
                    old_spec = get_spec(resolved_old)
                    old_format = self.store._infer_format(old_spec)
                    self.store.delete(resolved_old, format=old_format)
                except Exception:
                    pass
        self._preferences[category] = key
        set_active_model(category, key)

    @property
    def preferences(self) -> dict[str, str]:
        return dict(self._preferences)

    def get_preferences(self) -> dict[str, str]:
        return dict(self._preferences)

    # ── Model key helpers ──────────────────────────────────────────────────
    def get_model_key(self, category: ModelCategory) -> str:
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str, "not set")
        registry_key = self._resolve_key(key)
        if registry_key in REGISTRY:
            return get_spec(registry_key).model_id
        return key

    def get_all_model_keys(self) -> dict[str, str]:
        return {
            category.value: self.get_model_key(category)
            for category in ModelCategory
        }

    def get_model_name(self, category: ModelCategory) -> str:
        return self.get_model_key(category)

    # ── Load ───────────────────────────────────────────────────────────────
    def get(self, category: ModelCategory) -> Any:
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str)
        if not key:
            raise ValueError(f"No model set for '{cat_str}'. Run darwin setup first.")
        registry_key = self._resolve_key(key)
        if registry_key not in REGISTRY:
            raise ValueError(f"Unknown model '{key}' for '{cat_str}'.")
        if registry_key in self._cache:
            return self._cache[registry_key]
        spec = get_spec(registry_key)

        # ── Plugin check ──
        if spec.requires_plugin:
            if spec.requires_plugin == "torch":
                if not self.torch_plugin.ensure_torch():
                    # Fallback: if ONNX version exists, switch to it
                    onnx_fallback = self._find_onnx_fallback(registry_key)
                    if onnx_fallback:
                        darwin.log(
                            f"Torch not available, falling back to ONNX: {onnx_fallback}",
                            LogType.WARNING
                        )
                        spec = get_spec(onnx_fallback)
                        registry_key = onnx_fallback
                    else:
                        raise PluginRequiredError("torch", f"model {spec.model_id}")
            else:
                if not self.plugin_manager.is_installed(spec.requires_plugin):
                    raise PluginRequiredError(spec.requires_plugin, f"model {spec.model_id}")

        # ── API models ──
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

        # ── ONNX models (embedding + reranker) ──
        elif spec.loader in ("onnx_embedding", "onnx_reranker"):
            from src.core.Models.ModelLoading.ONNXLoader import ONNXLoader
            from src.core.Models.ModelDownloader import ModelDownloader
            
            downloader = ModelDownloader(self.downloader.cache_root)
            if not downloader.is_cached(registry_key, "onnx"):
                darwin.log(f"Downloading ONNX model {registry_key}...", LogType.INFO)
                downloader.download_onnx(registry_key)
            
            model_dir = downloader._get_model_dir(spec.model_id, spec.category, "onnx")
            loader = ONNXLoader(self.downloader)
            wrapper = loader.load_from_path(model_dir, category=spec.category)
            self._cache[registry_key] = wrapper
            return wrapper

        # ── PyTorch models (torch plugin required) ──
        path = str(self.store.ensure(registry_key))

        if spec.loader == "embedding_st":
            self.torch_plugin.require_torch("sentence-transformers embedding")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(path)
            self._cache[registry_key] = model
            return model

        elif spec.loader == "embedding_hf":
            self.torch_plugin.require_torch("HuggingFace embedding")
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
            self.torch_plugin.require_torch("cross-encoder")
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
            self.torch_plugin.require_torch("vision transformers")
            result = load_vision_model(path)
            self._cache[registry_key] = result
            return result

        raise ValueError(f"Unknown loader type: {spec.loader}")

    def _find_onnx_fallback(self, key: str) -> Optional[str]:
        """Find ONNX equivalent for a PyTorch model key."""
        spec = get_spec(key)
        # Direct ONNX variant
        onnx_key = key + "-onnx"
        if onnx_key in REGISTRY:
            return onnx_key
        # Search by model_id match
        for k, s in REGISTRY.items():
            if s.loader in ("onnx_embedding", "onnx_reranker") and s.model_id == spec.model_id:
                return k
        return None
    
    def _find_onnx_key(self, model_id: str, category: str) -> str:
        """Find registry key for ONNX variant of a model."""
        for key, spec in REGISTRY.items():
            if spec.model_id == model_id and spec.category == category and spec.loader in ("onnx_embedding", "onnx_reranker"):
                return key
        # Fallback: append -onnx
        return f"{model_id}-onnx"

    def unload(self, category: ModelCategory) -> None:
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
            if hasattr(cached, "session"):
                # ONNX — just delete session reference
                del cached.session
            else:
                if hasattr(cached, "cpu"):
                    cached.cpu()
                del cached
                # Only touch torch if available
                if _check_torch():
                    import torch
                    cfg = DeviceManager.get()
                    if cfg.use_gpu:
                        torch.cuda.empty_cache()
        except Exception:
            pass

    def ensure_ready(self, category: ModelCategory) -> bool:
        try:
            self.get(category)
            return True
        except (FileNotFoundError, OSError, ValueError, PluginRequiredError) as e:
            darwin.log(f"Model {category.value} not ready: {e}", LogType.ERROR)
            return False

    def _load_api_embedding(self, model_id: str):
        return APIEmbeddingWrapper(model_id)

    def _load_api_reranker(self, model_id: str):
        return APIRerankerWrapper(model_id)

    def _load_api_vision(self, model_id: str):
        return APIVisionWrapper(model_id)