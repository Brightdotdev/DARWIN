# src/core/Models/ModelLoader.py
import gc
import threading
import uuid
from collections import OrderedDict
from typing import Optional, Any, Dict, List, Callable

import numpy as np
import psutil

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
from src.features.Hardware.SystemDetector import SystemInfo


class ManagedModelCache:
    """
    Thread-safe cache with reference counting, memory-pressure gating,
    and aggressive resource cleanup. Guarantees a model is loaded once
    per process and evicted only when refs hit zero (or forced).
    """

    def __init__(self, maxsize: int = 5, memory_pressure_threshold: float = 0.90):
        self.maxsize = maxsize
        self.memory_pressure_threshold = memory_pressure_threshold
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._ref_count: Dict[str, int] = {}
        self._loading: Dict[str, threading.Event] = {}   # ── FIX T1 ──
        self._lock = threading.Lock()

    def acquire(self, key: str, factory: Callable[[], Any]) -> Any:
        """
        Get or create a cached model. Increments reference count.
        Factory runs *outside* the lock to avoid blocking other threads
        during heavy model initialization.
        """
        with self._lock:
            if key in self._cache and self._cache[key] is not None:
                self._cache.move_to_end(key)
                self._ref_count[key] = self._ref_count.get(key, 0) + 1
                return self._cache[key]

            # If another thread is loading this key, wait for it
            if key in self._loading:
                event = self._loading[key]
                self._lock.release()
                try:
                    event.wait()
                finally:
                    self._lock.acquire()
                # After waiting, the model should be in cache
                if key in self._cache and self._cache[key] is not None:
                    self._cache.move_to_end(key)
                    self._ref_count[key] = self._ref_count.get(key, 0) + 1
                    return self._cache[key]
                # If loading failed, fall through to load ourselves

            # Memory pressure gate: evict zero-ref models before loading
            if self._memory_pressure() > self.memory_pressure_threshold:
                self._evict_lru_zero_ref()
                if len(self._cache) >= self.maxsize:
                    self._evict_lru()

            while len(self._cache) >= self.maxsize:
                self._evict_lru()

            # Reserve slot and mark as loading
            self._loading[key] = threading.Event()
            self._cache[key] = None
            self._ref_count[key] = 1

        # ── Load outside lock ───────────────────────────────────────
        try:
            model = factory()
        except Exception:
            with self._lock:
                self._cache.pop(key, None)
                self._ref_count.pop(key, None)
                self._loading.pop(key, None).set()
            raise

        if model is None:
            with self._lock:
                self._cache.pop(key, None)
                self._ref_count.pop(key, None)
                self._loading.pop(key, None).set()
                
                darwin.log(f"Model factory returned None for {key}", LogType.WARNING)
            return None

        with self._lock:
            self._cache[key] = model
            self._loading.pop(key, None).set()
            
            darwin.log(f"Model loaded into cache: {key}", LogType.INFO)
            return model

    def release(self, key: str, force: bool = False) -> None:
        """Decrement ref count. Evict if zero or forced."""
        with self._lock:
            self._ref_count[key] = max(0, self._ref_count.get(key, 0) - 1)
            if force or self._ref_count.get(key, 0) == 0:
                self._evict(key)

    def _evict_lru_zero_ref(self) -> None:
        """Evict models with zero references first (safest)."""
        for k in list(self._cache.keys()):
            if self._ref_count.get(k, 0) == 0:
                self._evict(k)
                return

    def _evict_lru(self) -> None:
        """Evict least-recently-used model regardless of refs."""
        if not self._cache:
            return
        key, model = self._cache.popitem(last=False)
        self._ref_count.pop(key, None)
        self._cleanup(model)

    def _evict(self, key: str) -> None:
        if key not in self._cache:
            return
        model = self._cache.pop(key)
        self._ref_count.pop(key, None)
        self._cleanup(model)

    def _cleanup(self, model: Any) -> None:
        """Aggressive resource cleanup: session deletion, GPU cache flush, GC."""
        try:
            if hasattr(model, "session") and model.session is not None:
                del model.session
            if hasattr(model, "model") and model.model is not None:
                if hasattr(model.model, "cpu"):
                    model.model.cpu()
                del model.model
            del model
        except Exception:
            pass

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        
        darwin.log("Model cache entry evicted and memory cleaned", LogType.INFO)

    def _memory_pressure(self) -> float:
        """Return 0.0-1.0 memory pressure ratio."""
        return psutil.virtual_memory().percent / 100.0

    def clear(self) -> None:
        with self._lock:
            for model in list(self._cache.values()):
                self._cleanup(model)
            self._cache.clear()
            self._ref_count.clear()
            self._loading.clear()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache


class ModelLoader:
    """
    Loads models on‑demand with plugin management, ONNX fallback,
    Darwin logging, and a managed cache with memory-pressure eviction.

    Owns: singleton lifecycle, memory-aware batch sizing, OOM recovery,
    stream encoding, and aggressive resource cleanup.
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

        self._cache = ManagedModelCache(maxsize=cache_size)
        self.plugin_manager = PluginManager()
        self.torch_plugin = TorchPlugin(self.plugin_manager)
        self.system_info = system_info

    # ── Key resolution ─────────────────────────────────────────────────
    def _resolve_key(self, key: str, prefer_pytorch: Optional[bool] = False) -> str:

        if key in REGISTRY:
            return key
        matches = [k for k, spec in REGISTRY.items() if spec.model_id == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            if prefer_pytorch is True:
                non_onnx = [k for k in matches if not k.endswith("-onnx")]
                if non_onnx:
                    return non_onnx[0]
            elif prefer_pytorch is False:
                onnx = [k for k in matches if k.endswith("-onnx")]
                if onnx:
                    return onnx[0]
            else:
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

    def resolve_key(self, key: str, category: str, prefer_pytorch: Optional[bool] = None) -> str:
        registry_key = self._resolve_key(key)
        spec = get_spec(registry_key)
        if spec.category != category:
            
            darwin.log(f"Model '{key}' is not a {category} model", LogType.ERROR)
        return registry_key

    # ── Preferences ────────────────────────────────────────────────────
    def set_preference(self, category: str, key: str, cleanup_old: bool = False) -> None:
        old_key = self._preferences.get(category)
        if old_key and old_key != key:
            resolved_old = self._resolve_key(old_key)
            self._cache.release(resolved_old, force=True)
            if cleanup_old:
                try:
                    fmt = self.store.get_format(resolved_old)
                    self.store.delete(resolved_old, format=fmt)
                except Exception:
                    pass
        self._preferences[category] = key
        set_active_model(category, key)

    def get_preference(self, category: str) -> Optional[str]:
        """Return the active model key for a category."""
        return self._preferences.get(category)

    @property
    def preferences(self) -> Dict[str, str]:
        return dict(self._preferences)

    def get_model_key(self, category: ModelCategory, prefer_pytorch: Optional[bool] = None) -> str:
        """Return the registry key (including format suffix) for the active model."""
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str, "not set")
        registry_key = self._resolve_key(key)
        return registry_key  # ── FIX: return registry key, not model_id ──

    def get_all_model_keys(self) -> Dict[str, str]:
        return {cat.value: self.get_model_key(cat) for cat in ModelCategory}

    # ── Memory & batch sizing ─────────────────────────────────────────
    @staticmethod
    def memory_pressure() -> float:
        """Return system memory pressure as 0.0-1.0."""
        return psutil.virtual_memory().percent / 100.0

    def adaptive_batch_size(self, base_size: int, vector_dim: int, num_items: int) -> int:
        """
        Scale batch size down if memory pressure is high or output would be huge.
        Prevents OOM on low-RAM systems.
        """
        pressure = self.memory_pressure()
        if pressure > 0.90:
            return max(1, base_size // 8)
        if pressure > 0.75:
            return max(1, base_size // 4)
        if pressure > 0.60:
            return max(1, base_size // 2)

        est_output_mb = (base_size * vector_dim * 4) / (1024 * 1024)
        if est_output_mb > 512:
            return max(1, base_size // 2)
        return base_size

    def stream_encode(
        self,
        model: Any,
        texts: List[str],
        base_batch_size: int,
        file_name: str = "",
        normalize: bool = True,
        vector_dim: int = 384,
        **encode_kwargs,
    ) -> np.ndarray:
        """
        Stream-encode texts in adaptive batches with OOM recovery.

        Progress is tracked by chunks processed rather than batch count,
        because adaptive batching makes the total number of batches unknown.

        Returns:
            np.ndarray of shape (N, D) dtype=float32
        """
        if not texts:
            return np.array([], dtype=np.float32)

        total_chunks = len(texts)
        # ── FIX T3: Unique progress ID per call ────────────────────────
        progress_id = f"embed_{file_name}_{uuid.uuid4().hex[:8]}"

        
        darwin.progress_start(
                progress_id,
                total_chunks,
                f"Embedding {file_name}",
            )

        all_vectors: List[np.ndarray] = []
        idx = 0
        processed = 0
        batch_num = 0
        current_batch = base_batch_size

        while idx < total_chunks:
            batch_size = self.adaptive_batch_size(
                current_batch,
                vector_dim,
                total_chunks - idx,
            )

            batch_texts = texts[idx : idx + batch_size]
            batch_num += 1

            try:
                kwargs = dict(encode_kwargs)
                kwargs.setdefault("batch_size", len(batch_texts))
                kwargs.setdefault("show_progress_bar", False)
                kwargs.setdefault("convert_to_numpy", True)

                vecs = model.encode(batch_texts, **kwargs)

                if hasattr(vecs, "astype"):
                    all_vectors.append(
                        vecs.astype(np.float32, copy=False)
                    )
                else:
                    try:
                        import torch

                        if isinstance(vecs, torch.Tensor):
                            all_vectors.append(
                                vecs.detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32, copy=False)
                            )
                        else:
                            all_vectors.append(
                                np.array(vecs, dtype=np.float32)
                            )
                    except ImportError:
                        all_vectors.append(
                            np.array(vecs, dtype=np.float32)
                        )

                # Success — advance idx AFTER confirming batch succeeded
                idx += len(batch_texts)
                processed += len(batch_texts)

                
                darwin.progress_update(
                        progress_id,
                        processed,
                        (
                            f"Batch {batch_num} "
                            f"(size={len(batch_texts)}) • "
                            f"{processed}/{total_chunks} chunks"
                        ),
                    )

                # Restore preferred size after successful batch
                current_batch = base_batch_size

            except MemoryError:
                
                darwin.log(
                        f"OOM at batch_size={len(batch_texts)}, halving and retrying",
                        LogType.WARNING,
                    )

                if len(batch_texts) <= 1:
                    
                    darwin.log(
                            "Cannot reduce batch below 1, skipping remaining chunks",
                            LogType.ERROR,
                        )
                    break

                current_batch = max(1, len(batch_texts) // 2)
                batch_num -= 1  # retry shouldn't count as a completed batch
                continue

            except Exception:
                if self.memory_pressure() > 0.95:
                    
                    darwin.log(
                            (
                                f"System memory critical "
                                f"({psutil.virtual_memory().percent}%), "
                                f"halving batch"
                            ),
                            LogType.WARNING,
                        )

                    if len(batch_texts) <= 1:
                        break

                    current_batch = max(1, len(batch_texts) // 2)
                    batch_num -= 1
                    continue

                
                darwin.progress_end(progress_id)
                raise

        
        darwin.progress_end(progress_id)

        if not all_vectors:
            return np.array([], dtype=np.float32)

        return np.vstack(all_vectors)

    # ── Main load ─────────────────────────────────────────────────────
    def get(self, category: ModelCategory, auto_download: bool = True, prefer_pytorch: Optional[bool] = None) -> Any | None:
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str)
        if not key:
            
            darwin.log(f"No model set for '{cat_str}'. Run darwin setup first.", LogType.ERROR)
            return None

        registry_key = self._resolve_key(key)
        if registry_key not in REGISTRY:
            
            darwin.log(f"Unknown model '{key}' for '{cat_str}'.", LogType.ERROR)
            return None

        def _factory():
            return self._load_model(registry_key, auto_download, prefer_pytorch)

        return self._cache.acquire(registry_key, _factory)

    def _load_model(self, registry_key: str, auto_download: bool, prefer_pytorch: Optional[bool] = None):
        """Internal load logic separated from cache management."""
        spec = get_spec(registry_key)

        if spec.requires_plugin and spec.requires_plugin == "torch":
            self.torch_plugin.require_torch(f"model {spec.model_id}")
        elif spec.requires_plugin:
            if not self.plugin_manager.is_installed(spec.requires_plugin):
                raise PluginRequiredError(spec.requires_plugin, f"model {spec.model_id}")

        if spec.disk_mb > 0 and self.system_info is not None:
            if not self.system_info.can_run(spec):
                
                darwin.log(
                        f"{registry_key} needs {spec.ram_gb}GB RAM (you have {self.system_info.ram_gb}GB)",
                        LogType.ERROR,
                    )
                return None
            if not self.system_info.has_disk_space(spec):
                
                darwin.log(
                        f"{registry_key} needs {spec.disk_mb}MB disk (you have {self.system_info.disk_free_gb*1024:.0f}MB free)",
                        LogType.ERROR,
                    )
                return None

        if not auto_download and spec.disk_mb > 0 and not self.store.is_downloaded(registry_key):
            
            darwin.log(f"Model '{registry_key}' is not downloaded and auto_download is disabled.", LogType.ERROR)
            return None

        if spec.loader in ("api_embedding", "api_reranker", "api_vision"):
            return self._load_api_model(spec)

        local_path = str(self.store.ensure(registry_key))

        if spec.loader in ("onnx_embedding", "onnx_reranker"):
            model = self._load_onnx_model(registry_key, local_path)
        elif spec.loader == "spacy":
            model = self._load_spacy_model(spec, local_path)
        elif spec.loader in ("embedding_st", "embedding_hf", "cross_encoder", "transformers"):
            model = self._load_torch_model(spec, local_path)
        else:
            
            darwin.log(f"Unknown loader type: {spec.loader}", LogType.ERROR)
            model = None

        # Warmup to avoid first-batch penalty
        if hasattr(model, "warmup") and callable(getattr(model, "warmup")):
            try:
                model.warmup()
            except Exception:
                pass

        return model

    # ── Sub‑loaders ────────────────────────────────────────────────────
    def _load_api_model(self, spec):
        model_id = spec.model_id
        if spec.loader == "api_embedding":
            return APIEmbeddingWrapper(
                model_id,
                max_length=getattr(spec, "max_length", None),
                min_length=getattr(spec, "min_length", 1),
            )
        elif spec.loader == "api_reranker":
            return APIRerankerWrapper(
                model_id,
                max_length=getattr(spec, "max_length", None),
                min_length=getattr(spec, "min_length", 1),
            )
        elif spec.loader == "api_vision":
            return APIVisionWrapper(model_id)
        else:
            
            darwin.log(f"Unknown API loader: {spec.loader}")
            return None

    def _load_onnx_model(self, registry_key: str, local_path: str):
        from src.core.Models.ModelLoading.ONNXLoader import ONNXLoader
        loader = ONNXLoader()
        return loader.load(registry_key, auto_download=False)

    def _load_spacy_model(self, spec, local_path: str):
        from src.core.Models.ModelLoading.SpacyConceptExtractor import SpacyConceptExtractor
        
        darwin.log(f"Loading spaCy model from {local_path}")
        return SpacyConceptExtractor(local_path, spec.model_id)

    def _load_torch_model(self, spec, local_path: str):
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
            return HFEmbeddingWrapper(
                model, tokenizer, device,
                max_length=getattr(spec, "max_length", None),
                min_length=getattr(spec, "min_length", 1),
                normalize=getattr(spec, "normalize", True),
            )
        elif loader == "cross_encoder":
            from sentence_transformers import CrossEncoder
            return CrossEncoder(local_path)
        elif loader == "transformers":
            return load_vision_model(local_path)
        else:
            
            darwin.log(f"Unknown PyTorch loader: {loader}")
            return None

    # ── Unload ─────────────────────────────────────────────────────────
    def unload(self, category: ModelCategory, force: bool = False) -> None:
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str)
        if not key:
            return
        resolved_key = self._resolve_key(key)
        self._cache.release(resolved_key, force=force)

    def unload_all(self) -> None:
        """Emergency unload of all cached models."""
        self._cache.clear()

    def ensure_ready(self, category: ModelCategory) -> Optional[Any]:
        try:
            return self.get(category)
        except (FileNotFoundError, OSError, ValueError, PluginRequiredError) as e:
            
            darwin.log(f"Model {category.value} not ready: {e}", LogType.ERROR)
            return None