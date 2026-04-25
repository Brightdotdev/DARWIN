# src/core/models/__init__.py

from pathlib import Path
from typing import Optional

_loader: "ModelLoader | None" = None
_system: "SystemInfo | None" = None


def setup(models_dir: Path, preferences: dict = None, skip_download: bool = False, force: bool = False):
    global _loader, _system

    # Skip if already initialized (e.g. forked subprocess on Linux/macOS)
    if _loader is not None and not force:
        return _loader, _system

    from src.core.Models.ModelStartup import create_loader, run_onboarding

    _loader, _system = create_loader(models_dir, preferences, skip_download=skip_download)

    return _loader, _system


def get_loader() -> "ModelLoader":
    """Get the global loader. Must call setup() first."""
    if _loader is None:
        raise RuntimeError(
            "Models not initialized. Call models.setup(models_dir) at app startup."
        )
    return _loader


def get_system() -> "SystemInfo":
    """Get detected system info. Must call setup() first."""
    if _system is None:
        raise RuntimeError(
            "Models not initialized. Call models.setup(models_dir) at app startup."
        )
    return _system


# Re-export commonly used types
from src.core.Models.ModelRegistry import ModelCategory, REGISTRY, TIER_LABELS
from src.core.Models.ModelLoader import ModelLoader
from src.core.Models.ModelStartup import ModelStore
from src.core.Models.SystemDetector import SystemInfo

__all__ = [
    "setup",
    "get_loader",
    "get_system",
    "ModelCategory",
    "ModelLoader",
    "ModelStore",
    "REGISTRY",
    "TIER_LABELS",
    "SystemInfo",
]