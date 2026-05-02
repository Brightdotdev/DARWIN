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
_SYSTEM_CACHE = None 



def set_preferences(preferences: dict) -> None:
    """Set model preferences before loader creation."""
    global _PREFERENCES_CACHE
    _PREFERENCES_CACHE = preferences


def get_loader() -> ModelLoader:
    """Global ModelLoader singleton."""
    global _LOADER_INSTANCE, _PREFERENCES_CACHE, _SYSTEM_CACHE
    
    if _LOADER_INSTANCE is None:
        store = ModelStore()
        
        if _SYSTEM_CACHE is None:          
            _SYSTEM_CACHE = detect_system()
        
        if _PREFERENCES_CACHE is None:
            _PREFERENCES_CACHE = recommend_defaults(_SYSTEM_CACHE)
        
        _LOADER_INSTANCE = ModelLoader(store, _PREFERENCES_CACHE)
    
    return _LOADER_INSTANCE


def setup(preferences: dict = None, skip_download: bool = False) -> ModelLoader:
    global _LOADER_INSTANCE, _PREFERENCES_CACHE, _SYSTEM_CACHE

    if _SYSTEM_CACHE is None:              
        _SYSTEM_CACHE = detect_system()

    if preferences is not None:
        _PREFERENCES_CACHE = preferences

    if _PREFERENCES_CACHE is None:
        _PREFERENCES_CACHE = recommend_defaults(_SYSTEM_CACHE)

    if not skip_download:
        from src.core.Models.ModelStartup import download_required
        download_required(ModelStore(), _PREFERENCES_CACHE, _SYSTEM_CACHE)

    if _LOADER_INSTANCE is None:           # ← don't overwrite existing instance
        _LOADER_INSTANCE = ModelLoader(ModelStore(), _PREFERENCES_CACHE)

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