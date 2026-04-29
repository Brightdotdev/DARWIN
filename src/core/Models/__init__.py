# src/core/Models/__init__.py

"""Models module for Darwin — embedding, reranker, vision, tokenizer management."""

from src.core.Models.ModelStore import ModelStore
from src.core.Models.ModelLoader import ModelLoader
from src.core.Models.ModelRegistry import ModelCategory, REGISTRY, get_spec
from src.core.Models.SystemDetector import detect_system, recommend_defaults
from src.core.Models.ModelManager import ModelManager
from src.core.Models.ModelSetup import validate_model, validate_category
from src.core.Models.SpacyConceptExtractor import SpacyConceptExtractor
from src.core.Models.ModelUtils import handle_download_error


_LOADER_INSTANCE = None
_PREFERENCES_CACHE = None


def get_loader() -> ModelLoader:
    """Global ModelLoader singleton."""
    global _LOADER_INSTANCE, _PREFERENCES_CACHE
    
    if _LOADER_INSTANCE is None:
        store = ModelStore()
        system = detect_system()
        
        if _PREFERENCES_CACHE is None:
            _PREFERENCES_CACHE = recommend_defaults(system)
        
        _LOADER_INSTANCE = ModelLoader(store, _PREFERENCES_CACHE)
    
    return _LOADER_INSTANCE


def set_preferences(preferences: dict) -> None:
    """Set model preferences before loader creation."""
    global _PREFERENCES_CACHE
    _PREFERENCES_CACHE = preferences


__all__ = [
    "ModelStore",
    "ModelLoader", 
    "ModelManager",
    "ModelCategory",
    "REGISTRY",
    "get_spec",
    "get_loader",
    "detect_system",
    "recommend_defaults",
    "validate_model",
    "validate_category",
    "SpacyConceptExtractor",
    "handle_download_error",
]