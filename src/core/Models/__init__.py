# src/core/Models/__init__.py

"""Models module for Darwin — embedding, reranker, vision, tokenizer management."""

from src.core.Models.ModelStore import ModelStore
from src.core.Models.ModelLoader import ModelLoader
from src.core.Models.ModelRegistry import ModelCategory, REGISTRY, get_spec
from src.core.Models.SystemDetector import detect_system, recommend_defaults
from src.core.Models.ModelManager import ModelManager
from src.core.Models.ModelSetup import validate_model, validate_category, handle_download_error
from src.core.Models.SpacyConceptExtractor import SpacyConceptExtractor
from src.core.Logger.DarwinLogger import darwin, LogType


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


def setup(preferences: dict = None, skip_download: bool = False) -> ModelLoader:
    """
    Initialize the model system with optional preferences.
    
    Args:
        preferences: Dict of category -> model key (e.g., {"embedding": "BAAI/bge-m3"})
        skip_download: If True, don't download missing models
    
    Returns:
        ModelLoader instance
    """
    global _LOADER_INSTANCE, _PREFERENCES_CACHE
    
    if preferences is not None:
        _PREFERENCES_CACHE = preferences
    
    store = ModelStore()
    system = detect_system()
    
    if _PREFERENCES_CACHE is None:
        _PREFERENCES_CACHE = recommend_defaults(system)
    
    from src.core.Models.ModelStartup import download_required
    
    if not skip_download:
        download_required(store, _PREFERENCES_CACHE, system)
    
    _LOADER_INSTANCE = ModelLoader(store, _PREFERENCES_CACHE)
    return _LOADER_INSTANCE


__all__ = [
    "ModelStore",
    "ModelLoader", 
    "ModelManager",
    "ModelCategory",
    "REGISTRY",
    "get_spec",
    "get_loader",
    "set_preferences",
    "setup",
    "detect_system",
    "recommend_defaults",
    "validate_model",
    "validate_category",
    "SpacyConceptExtractor",
    "handle_download_error",
]