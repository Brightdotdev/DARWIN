# src/core/models/__init__.py

from pathlib import Path
from typing import Optional

_loader: "ModelLoader | None" = None
_system: "SystemInfo | None" = None


def setup(
    preferences: Optional[dict] = None,
    skip_download: bool = False,
    force: bool = False,
):
    """
    Initialize the global model loader.

    Preference resolution order (highest → lowest priority):
      1. `preferences` dict passed in directly (e.g. from a test or CLI override)
      2. Persisted config.json  (written by cli.py on every install/switch)
      3. Auto-detected hardware defaults from SystemDetector

    This means that once a user runs `cli switch --category embedding --model bge-m3`,
    that choice is saved to config.json and automatically picked up here on the
    next startup — no manual wiring needed.

    Args:
        preferences:    Optional override dict, e.g. {"embedding": "e5-small-v2"}.
                        Merged on top of config+hardware defaults.
        skip_download:  If True, skip downloading missing models (useful in tests).
        force:          If True, re-initialize even if already set up (e.g. hot-reload).
    """
    global _loader, _system

    # Skip if already initialized (e.g. forked subprocess on Linux/macOS)
    if _loader is not None and not force:
        return _loader, _system

    from src.core.Models.ModelStartup import create_loader
    from src.core.Models.Main import load_config

    # ── Merge preferences: config file wins over hardware defaults,
    #    caller-supplied preferences win over everything. ──────────────────
    merged: dict = {}

    # 1. Start with hardware-based defaults
    from src.core.Models.SystemDetector import detect_system, recommend_defaults
    detected_system = detect_system()
    merged.update(recommend_defaults(detected_system))

    # 2. Layer in persisted user choices from config.json
    persisted = load_config()
    merged.update({k: v for k, v in persisted.items() if v})

    # 3. Layer in any explicit caller overrides
    if preferences:
        merged.update({k: v for k, v in preferences.items() if v})

    _loader, _system = create_loader(merged, skip_download=skip_download)

    return _loader, _system


def get_loader() -> "ModelLoader":
    """Get the global loader. Must call setup() first."""
    if _loader is None:
        raise RuntimeError(
            "Models not initialized. Call models.setup() at app startup."
        )
    return _loader


def get_system() -> "SystemInfo":
    """Get detected system info. Must call setup() first."""
    if _system is None:
        raise RuntimeError(
            "Models not initialized. Call models.setup() at app startup."
        )
    return _system


# Re-export commonly used types
from src.core.Models.ModelRegistry import ModelCategory, REGISTRY, TIER_LABELS
from src.core.Models.ModelLoader import ModelLoader
from src.core.Models.ModelStore import ModelStore
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