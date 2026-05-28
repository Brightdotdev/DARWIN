# src/core/Models/__init__.py

"""Models module for Darwin — embedding, reranker, vision, tokenizer management."""

from src.core.Models.ModelStore import ModelStore
from src.core.Models.ModelLoader import ModelLoader
from src.core.Models.ModelRegistry import ModelCategory, REGISTRY, get_spec
from src.core.Models.ModelManager import ModelManager
from src.core.Models.ModelSetup import validate_model, validate_category, handle_download_error


from src.core.Models.EmbeddingConfig import get_preset,ModelPreset





__all__ = [
    "ModelStore",
    "ModelLoader", 
    "ModelManager",
    "ModelCategory",
    "REGISTRY",
    "get_spec",
    "validate_model",
    "validate_category",
    "handle_download_error",
    "get_preset",
    "ModelPreset"
]