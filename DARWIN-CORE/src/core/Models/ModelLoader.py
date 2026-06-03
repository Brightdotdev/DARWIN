# src/core/Models/ModelLoader.py
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Any, Dict

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
from src.core.Models.ModelUtils import HardwareInsufficientError
from src.features.Hardware.SystemDetector import SystemInfo


class LRUCache:
    def __init__(self, maxsize: int = 5):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: str, value: Any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)
            self._cache[key] = value

    def pop(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._cache.pop(key, None)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache


class ModelLoader:
    """
    Loads models on‑demand with plugin management, ONNX fallback,
    Darwin logging, and an LRU cache.

    Delegates all download and storage to ModelStore.
    """

    def __init__(
        self,
        store: Optional[ModelStore] = None,
        preferences: Optional[Dict[str, str]] = None,
        cache_size: int = 5,
        system_info: Optional[SystemInfo] = None,
    ):
        self.store = store or get_model_store()
        if preferences is not None:
            self._preferences = dict(preferences)
        else:
            self._preferences = {k: v or "" for k, v in load_active_models().items()}

        self._cache = LRUCache(maxsize=cache_size)
        self.plugin_manager = PluginManager()
        self.torch_plugin = TorchPlugin(self.plugin_manager)
        self.system_info = system_info  # may be None (no validation)

    # ── Key resolution ─────────────────────────────────────────────────
    def _resolve_key(self, key: str) -> str:
        if key in REGISTRY:
            return key
        matches = [k for k, spec in REGISTRY.items() if spec.model_id == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            if self.torch_plugin.is_installed():
                non_onnx = [k for k in matches if not k.endswith("-onnx")]
                if non_onnx:
                    return non_onnx[0]
            else:
                onnx = [k for k in matches if k.endswith("-onnx")]
                if onnx:
                    return onnx[0]
            return matches[0]
        return key

    def resolve_key(self, key: str, category: str) -> str:
        registry_key = self._resolve_key(key)
        spec = get_spec(registry_key)
        if spec.category != category:
            raise ValueError(f"Model '{key}' is not a {category} model")
        return registry_key

    # ── Preferences ────────────────────────────────────────────────────
    def set_preference(self, category: str, key: str, cleanup_old: bool = False) -> None:
        old_key = self._preferences.get(category)
        if old_key and old_key != key:
            resolved_old = self._resolve_key(old_key)
            self._cache.pop(resolved_old)
            if cleanup_old:
                try:
                    fmt = self.store.get_format(resolved_old)
                    self.store.delete(resolved_old, format=fmt)
                except Exception:
                    pass
        self._preferences[category] = key
        set_active_model(category, key)

    @property
    def preferences(self) -> Dict[str, str]:
        return dict(self._preferences)

    def get_model_key(self, category: ModelCategory) -> str:
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str, "not set")
        registry_key = self._resolve_key(key)
        if registry_key in REGISTRY:
            return get_spec(registry_key).model_id
        return key

    def get_all_model_keys(self) -> Dict[str, str]:
        return {cat.value: self.get_model_key(cat) for cat in ModelCategory}

    # ── Main load ─────────────────────────────────────────────────────
    def get(self, category: ModelCategory, auto_download: bool = True) -> Any:
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str)
        if not key:
            raise ValueError(f"No model set for '{cat_str}'. Run darwin setup first.")

        registry_key = self._resolve_key(key)
        if registry_key not in REGISTRY:
            raise ValueError(f"Unknown model '{key}' for '{cat_str}'.")

        cached = self._cache.get(registry_key)
        if cached is not None:
            return cached

        spec = get_spec(registry_key)

        # Plugin check
        if spec.requires_plugin and spec.requires_plugin == "torch":
            self.torch_plugin.require_torch(f"model {spec.model_id}")
        elif spec.requires_plugin:
            if not self.plugin_manager.is_installed(spec.requires_plugin):
                raise PluginRequiredError(spec.requires_plugin, f"model {spec.model_id}")

        # Hardware validation before any download
        if spec.disk_mb > 0 and self.system_info is not None:
            if not self.system_info.can_run(spec):
                raise HardwareInsufficientError(
                    registry_key,
                    f"Needs {spec.ram_gb}GB RAM (you have {self.system_info.ram_gb}GB)"
                )
            if not self.system_info.has_disk_space(spec):
                raise HardwareInsufficientError(
                    registry_key,
                    f"Needs {spec.disk_mb}MB disk (you have {self.system_info.disk_free_gb*1024:.0f}MB free)"
                )

        # If not cached and auto_download is False, raise
        if not auto_download and spec.disk_mb > 0 and not self.store.is_downloaded(registry_key):
            raise FileNotFoundError(
                f"Model '{registry_key}' is not downloaded and auto_download is disabled."
            )

        # API models – no local path
        if spec.loader in ("api_embedding", "api_reranker", "api_vision"):
            wrapper = self._load_api_model(spec)
            self._cache.set(registry_key, wrapper)
            return wrapper

        # All local models – store.ensure handles download if needed
        local_path = str(self.store.ensure(registry_key))

        if spec.loader in ("onnx_embedding", "onnx_reranker"):
            model = self._load_onnx_model(registry_key, local_path)
        elif spec.loader == "spacy":
            model = self._load_spacy_model(spec, local_path)
        elif spec.loader in ("embedding_st", "embedding_hf", "cross_encoder", "transformers"):
            model = self._load_torch_model(spec, local_path)
        else:
            raise ValueError(f"Unknown loader type: {spec.loader}")

        self._cache.set(registry_key, model)
        return model

    # ── Sub‑loaders ────────────────────────────────────────────────────
    def _load_api_model(self, spec):
        model_id = spec.model_id
        if spec.loader == "api_embedding":
            return APIEmbeddingWrapper(model_id,
                                       max_length=spec.max_length,
                                       min_length=spec.min_length)
        elif spec.loader == "api_reranker":
            return APIRerankerWrapper(model_id,
                                      max_length=spec.max_length,
                                      min_length=spec.min_length)
        elif spec.loader == "api_vision":
            return APIVisionWrapper(model_id)
        else:
            raise ValueError(f"Unknown API loader: {spec.loader}")

    def _load_onnx_model(self, registry_key: str, local_path: str):
        # ONNXLoader needs the registry key for spec lookup
        from src.core.Models.ModelLoading.ONNXLoader import ONNXLoader
        loader = ONNXLoader()  # ONNXLoader will use its own downloader if needed, but store already downloaded
        return loader.load(registry_key)

    def _load_spacy_model(self, spec, local_path: str):
        from src.core.Models.ModelLoading.SpacyConceptExtractor import SpacyConceptExtractor
        darwin.log(f"Loading spaCy model from {local_path}")
        return SpacyConceptExtractor(local_path, spec.model_id)

    def _load_torch_model(self, spec, local_path: str):
        # TorchPlugin.require_torch already called; this is safe
        import torch
        loader = spec.loader
        darwin.log(f"Loading PyTorch model from {local_path}", LogType.INFO)

        if loader == "embedding_st":
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(local_path)
        elif loader == "embedding_hf":
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained(local_path)
            model = AutoModel.from_pretrained(local_path)
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
            model.to(device)
            model.eval()
            return HFEmbeddingWrapper(model, tokenizer, device,
                                      max_length=spec.max_length,
                                      min_length=spec.min_length,
                                      normalize=spec.normalize)
        elif loader == "cross_encoder":
            from sentence_transformers import CrossEncoder
            return CrossEncoder(local_path)
        elif loader == "transformers":
            return load_vision_model(local_path)
        else:
            raise ValueError(f"Unknown PyTorch loader: {loader}")

    # ── Unload ─────────────────────────────────────────────────────────
    def unload(self, category: ModelCategory) -> None:
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str)
        if not key:
            return
        resolved_key = self._resolve_key(key)
        cached = self._cache.pop(resolved_key)
        if cached is None:
            return
        try:
            if hasattr(cached, "session"):
                del cached.session
            elif hasattr(cached, "model") and hasattr(cached.model, "cpu"):
                cached.model.cpu()
                del cached.model
            else:
                del cached
            if self.torch_plugin.is_installed():
                import torch
                from src.features.Hardware.DeviceManager import DeviceManager
                cfg = DeviceManager.get()
                if cfg.use_gpu:
                    torch.cuda.empty_cache()
        except Exception:
            pass

    def ensure_ready(self, category: ModelCategory) -> Optional[Any]:
        try:
            return self.get(category)
        except (FileNotFoundError, OSError, ValueError, PluginRequiredError) as e:
            darwin.log(f"Model {category.value} not ready: {e}", LogType.ERROR)
            return None