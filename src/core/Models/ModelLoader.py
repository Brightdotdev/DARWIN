# src/core/Models/ModelLoader.py

import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from src.core.Models.ModelStore import ModelStore
from src.core.Models.ModelRegistry import get_spec, ModelCategory, REGISTRY
from src.core.Models.SpacyConceptExtractor import SpacyConceptExtractor
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Storage.DarwinStore import DarwinStore


class ModelLoader:
    """Lazy-loads models, caches in memory, handles unload."""
    
    def __init__(self, store: ModelStore, preferences: dict[str, str] = None):
        self.store = store
        self._preferences = preferences or {}
        self._cache: dict[str, Any] = {}

    def _resolve_key(self, key: str) -> str:
        """Resolve any model identifier to its canonical registry key."""
        if key in REGISTRY:
            return key

        id_to_key = {spec.model_id: k for k, spec in REGISTRY.items()}
        if key in id_to_key:
            return id_to_key[key]

        return key

    def set_preference(self, category: str, key: str, cleanup_old: bool = False) -> None:
        """Set model preference for a category."""
        old_key = self._preferences.get(category)

        if old_key and old_key != key:
            resolved_old = self._resolve_key(old_key)
            self._cache.pop(resolved_old, None)

            if cleanup_old:
                self.store.delete(resolved_old)

        self._preferences[category] = key

    def get(self, category: ModelCategory) -> Any:
        """Get loaded model instance for a category."""
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
        path = str(self.store.ensure(registry_key))

        # SentenceTransformers Embeddings
        if spec.loader == "embedding_st":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(path)
            self._cache[registry_key] = model
            return model

        # HF Native Embeddings
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

            wrapper = self._HFEmbeddingWrapper(model, tokenizer, device)
            self._cache[registry_key] = wrapper
            return wrapper

        # CrossEncoder (Reranker)
        elif spec.loader == "cross_encoder":
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(path)
            self._cache[registry_key] = model
            return model

        # spaCy
        elif spec.loader == "spacy":
            model = SpacyConceptExtractor(path, spec.model_id)
            self._cache[registry_key] = model
            return model

        # Vision Models
        elif spec.loader == "transformers":
            result = self._load_vision_model(path, registry_key)
            self._cache[registry_key] = result
            return result

        raise ValueError(f"Unknown loader type: {spec.loader}")

    def _load_vision_model(self, path: str, key: str) -> Tuple[Any, Any, str]:
        """Load vision model (processor, model, device)."""
        from src.core.Hardware.DeviceManager import DeviceManager
        import torch

        cfg = DeviceManager.get()
        device = cfg.device
        dtype = torch.float16 if cfg.use_gpu else torch.float32
        config_path = Path(path) / "config.json"

        with open(config_path) as f:
            config = json.load(f)

        model_type = config.get("model_type", "").lower()

        try:
            if "idefics" in model_type:
                from transformers import Idefics3ForConditionalGeneration, AutoProcessor
                processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
                model = Idefics3ForConditionalGeneration.from_pretrained(
                    path, dtype=dtype, low_cpu_mem_usage=True,
                    device_map=None, trust_remote_code=True
                )
                model.to(cfg.device)
            elif "llava" in model_type:
                from transformers import LlavaForConditionalGeneration, AutoProcessor
                processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
                model = LlavaForConditionalGeneration.from_pretrained(
                    path, dtype=dtype, low_cpu_mem_usage=True,
                    device_map=None, trust_remote_code=True
                )
                model.to(cfg.device)
            elif "moondream" in model_type:
                from transformers import AutoModelForCausalLM, AutoProcessor
                processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    path, dtype=dtype, low_cpu_mem_usage=True,
                    device_map=None, trust_remote_code=True
                )
                model.to(cfg.device)
            else:
                from transformers import AutoModelForImageTextToText, AutoProcessor
                processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
                model = AutoModelForImageTextToText.from_pretrained(
                    path, dtype=dtype, low_cpu_mem_usage=True,
                    device_map=None, trust_remote_code=True
                )
                model.to(cfg.device)

            darwin.log(f"Vision model loaded on {device}", LogType.INFO)
            return processor, model, device

        except ImportError as e:
            raise ImportError(
                f"Missing required class for {model_type} model. "
                f"Update transformers: pip install --upgrade transformers\nError: {e}"
            )

    class _HFEmbeddingWrapper:
        """Wrapper for HuggingFace embedding models."""
        
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device

        def encode(self, texts, batch_size=32, convert_to_numpy=True, **kwargs):
            import torch

            if isinstance(texts, str):
                texts = [texts]

            embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1)
                    emb = emb / emb.norm(dim=1, keepdim=True)

                embeddings.append(emb.cpu())

            result = torch.cat(embeddings, dim=0)

            if convert_to_numpy:
                return result.numpy()
            return result

    def unload(self, category: ModelCategory) -> None:
        """Unload model from memory to free RAM."""
        from src.core.Hardware.DeviceManager import DeviceManager

        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str)

        if key:
            resolved_key = self._resolve_key(key)
            cached = self._cache.pop(resolved_key, None)

            if cached is not None:
                try:
                    import torch
                    obj = cached[1] if isinstance(cached, tuple) else cached

                    if hasattr(obj, "cpu"):
                        obj.cpu()

                    del obj, cached


                    cfg = DeviceManager.get()

                    if cfg.use_gpu:
                        torch.cuda.empty_cache()


                except Exception:
                    pass




    @property
    def preferences(self) -> dict[str, str]:
        return dict(self._preferences)

    def get_model_name(self, category: ModelCategory) -> str:
        """Get human-readable model name."""
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str)

        if not key:
            return "not set"

        registry_key = self._resolve_key(key)

        if registry_key in REGISTRY:
            return get_spec(registry_key).model_id
        return key

    def get_model_key(self, category: ModelCategory) -> str:
        """Get canonical registry key."""
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str, "not set")
        registry_key = self._resolve_key(key)

        if registry_key in REGISTRY:
            return get_spec(registry_key).model_id
        return key

        # In src/core/Models/ModelLoader.py
    def ensure_ready(self, category: ModelCategory) -> bool:
        """Ensure model is downloaded and loadable. Returns True if ready."""
        try:
            self.get(category)
            return True
        except (FileNotFoundError, OSError, ValueError) as e:
            darwin.log(f"Model {category.value} not ready: {e}", LogType.ERROR)
            return False
   
    


    # ═══════════════════════════════════════════════════════════════════════
    # Embedding Migration
    # ═══════════════════════════════════════════════════════════════════════

    def migrate_embeddings(
        self,
        filename: Optional[str] = None,
        force: bool = False,
    ) -> dict:
        """Migrate embeddings to current model."""
        from src.core.Storage.DarwinStore import DarwinStore
        store = DarwinStore()

        current_model = self.get_model_key(ModelCategory.EMBEDDING)
        stats = {"reembedded": 0, "skipped": 0, "term_banks_updated": 0, "failed": 0}

        if filename:
            files_to_process = [filename]
        else:
            health = store.get_embedding_health()
            files_to_process = health.get("needs_reembed", [])

            if force:
                files_to_process.extend(health.get("up_to_date", []))

        if not files_to_process:
            darwin.log("All files are up-to-date with current embedding model", LogType.SUCCESS)
            return stats

        darwin.log(f"Migrating {len(files_to_process)} file(s) to {current_model}", LogType.INFO)

        for fname in files_to_process:
            try:
                md = store.load_markdown(fname)

                if not md:
                    darwin.log(f"{fname} — no markdown found, skipping", LogType.WARNING)
                    stats["failed"] += 1
                    continue

                from src.core.Chunking.main import handle_chunking
                from src.core.Chunking.lib import  ChunkingArgs

                chunks = handle_chunking(ChunkingArgs(file=fname, inspect=False, rerun=True))

                from src.core.Embedding.lib import EmbeddingArgs
                from src.core.Embedding.main import handle_embedding

                handle_embedding(EmbeddingArgs(file=fname, inspect=False, rerun=True))

                stats["reembedded"] += 1
                darwin.log(f"{fname} — migrated ({len(chunks)} chunks)", LogType.SUCCESS)
                
            except Exception as e:
                darwin.log(f"{fname} — {type(e).__name__}: {e}", LogType.ERROR)
                stats["failed"] += 1

        try:
            self.unload(ModelCategory.EMBEDDING)
        except Exception:
            pass

        darwin.log(
            f"Migration complete: {stats['reembedded']} migrated · "
            f"{stats['term_banks_updated']} term banks updated · "
            f"{stats['failed']} failed",
            LogType.SUCCESS,
        )

        return stats