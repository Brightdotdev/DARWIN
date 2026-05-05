# src/core/Models/__init__.py

"""Models module for Darwin — embedding, reranker, vision, tokenizer management."""

from src.Models.ModelStore import ModelStore
from src.Models.ModelLoader import ModelLoader
from src.Models.ModelRegistry import ModelCategory, REGISTRY, get_spec
from src.Models.ModelManager import ModelManager,set_preferences
from src.Models.ModelSetup import validate_model, validate_category, handle_download_error


from src.Models.EmbeddingConfig import get_preset,ModelPreset





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
    "set_preferences",
    "get_preset",
    "ModelPreset"
]