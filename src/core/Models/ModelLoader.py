# src/core/models/loader.py

from pathlib import Path
from src.core.Models.ModelStore import ModelStore
from src.core.Models.ModelRegistry import (get_spec, ModelCategory)


# ═══════════════════════════════════════════════════════════════════════════
# HF Embedding Wrapper (NEW)
# ═══════════════════════════════════════════════════════════════════════════

class HFEmbeddingWrapper:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, texts, batch_size=32, **kwargs):
        import torch

        if isinstance(texts, str):
            texts = [texts]

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

                # Mean pooling
                emb = outputs.last_hidden_state.mean(dim=1)

                # Normalize (important for E5)
                emb = emb / emb.norm(dim=1, keepdim=True)

            embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# Model Loader
# ═══════════════════════════════════════════════════════════════════════════

class ModelLoader:
    def __init__(self, store: ModelStore, preferences: dict[str, str] = None):
        self.store = store
        self._preferences = preferences or {}
        self._cache: dict[str, object] = {}

    def set_preference(self, category: str, key: str,
                       cleanup_old: bool = False) -> None:
        old_key = self._preferences.get(category)

        if old_key and old_key != key:
            self._cache.pop(old_key, None)
            if cleanup_old:
                self.store.delete(old_key)

        self._preferences[category] = key

    def get(self, category: ModelCategory):
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str)

        if not key:
            raise ValueError(f"No model set for '{cat_str}'.")

        # Cache hit
        if key in self._cache:
            return self._cache[key]

        spec = get_spec(key)
        path = str(self.store.ensure(key))

        # ═══════════════════════════════════════════════════════════════
        # 1. SentenceTransformers Embeddings (BGE, GritLM, etc.)
        # ═══════════════════════════════════════════════════════════════
        if spec.loader == "embedding_st":
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(path)
            self._cache[key] = model
            return model

        # ═══════════════════════════════════════════════════════════════
        # 2. HF Native Embeddings (E5, etc.)  ← FIX
        # ═══════════════════════════════════════════════════════════════
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

            self._cache[key] = wrapper
            return wrapper

        # ═══════════════════════════════════════════════════════════════
        # 3. CrossEncoder (Reranker)
        # ═══════════════════════════════════════════════════════════════
        elif spec.loader == "cross_encoder":
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(path)
            self._cache[key] = model
            return model

        # ═══════════════════════════════════════════════════════════════
        # 4. Vision Models (unchanged)
        # ═══════════════════════════════════════════════════════════════
            
            
        elif spec.loader == "transformers":
            # ── VISION LOADING ──
            import torch
            import json
            from pathlib import Path
            
            # Device detection
            if torch.cuda.is_available():
                device, dtype = "cuda", torch.float16
            elif torch.backends.mps.is_available():
                device, dtype = "mps", torch.float16
            else:
                device, dtype = "cpu", torch.float32
            
            # Read config to detect architecture
            config_path = Path(path) / "config.json"
            with open(config_path) as f:
                config = json.load(f)
            model_type = config.get("model_type", "").lower()
            
            print(f"   Detected model_type: {model_type}")
            
            try:
                # Different architectures need different loading approaches
                if "idefics" in model_type:
                    from transformers import Idefics3ForConditionalGeneration, AutoProcessor
                    
                    processor = AutoProcessor.from_pretrained(
                        path,
                        trust_remote_code=True
                    )
                    
                    model = Idefics3ForConditionalGeneration.from_pretrained(
                        path,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    
                elif "llava" in model_type:
                    from transformers import LlavaForConditionalGeneration, AutoProcessor
                    
                    processor = AutoProcessor.from_pretrained(
                        path,
                        trust_remote_code=True
                    )
                    
                    model = LlavaForConditionalGeneration.from_pretrained(
                        path,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    
                elif "moondream" in model_type:
                    from transformers import AutoModelForCausalLM, AutoProcessor
                    
                    processor = AutoProcessor.from_pretrained(
                        path,
                        trust_remote_code=True
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        path,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    
                else:
                    # Generic fallback for newer models
                    from transformers import AutoModelForImageTextToText, AutoProcessor
                    
                    processor = AutoProcessor.from_pretrained(
                        path,
                        trust_remote_code=True
                    )
                    
                    model = AutoModelForImageTextToText.from_pretrained(
                        path,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True,
                        device_map="auto",
                        trust_remote_code=True
                    )
                
                self._cache[key] = (processor, model, device)
                print(f"   ✅ Vision model loaded on {device}")
                return self._cache[key]
                
            except ImportError as e:
                raise ImportError(
                    f"Missing required class for {model_type} model. "
                    f"Update transformers: pip install --upgrade transformers\n"
                    f"Error: {e}"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load vision model from {path}. "
                    f"Error: {e}\n"
                    f"Try: pip install --upgrade transformers accelerate"
                )
        raise ValueError(f"Unknown loader type: {spec.loader}")

    # ═══════════════════════════════════════════════════════════════════
    # Unload (unchanged but safe)
    # ═══════════════════════════════════════════════════════════════════
    def unload(self, category: "ModelCategory | str") -> None:
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str)

        if key:
            cached = self._cache.pop(key, None)

            if cached is not None:
                try:
                    import torch

                    if isinstance(cached, tuple):
                        obj = cached[1]
                    else:
                        obj = cached

                    if hasattr(obj, "cpu"):
                        obj.cpu()

                    del obj, cached

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception:
                    pass

    @property
    def preferences(self) -> dict[str, str]:
        return dict(self._preferences)

    def get_model_name(self, category: ModelCategory) -> str:
        cat_str = category.value if isinstance(category, ModelCategory) else category
        key = self._preferences.get(cat_str)

        if not key:
            return "not set"

        spec = get_spec(key)
        return spec.model_id

    def get_model_key(self, category: ModelCategory) -> str:
        cat_str = category.value if isinstance(category, ModelCategory) else category
        return self._preferences.get(cat_str, "not set")