# src/core/Models/ModelLoading/API/NLPInferenceFactory.py
"""
Factory for creating spaCy NLP inference instances (local only).
Manages model path, downloads, and active preferences.
NLP is optional – deleting the active model is allowed.
"""

from pathlib import Path
from typing import Optional, Dict, List, Union, Iterator

from src.core.Models.Lib.ModelRegistry import find_spec, ModelCategory
from src.core.Models.ModelConfig import (
    get_active_model,
    set_active_model,
    get_model_path,
    ensure_model_path,
)
from src.core.Models.Infrence.Spacy.SpacyNLPInference import SpacyNLPInference
from src.core.Models.ModelDownloading.SpacyNLPModelDownloader import SpacyNLPModelDownloader  # assume exists
from src.core.Logger.DarwinLogger import darwin, LogType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NLP_CATEGORY = ModelCategory.NLP.name.lower()  # "nlp"


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------

def get_nlp_inference(
    model_id: Optional[str] = None,
    disable_components: Optional[List[str]] = None,
) -> Optional[SpacyNLPInference]:
    """
    Return a spaCy NLP inference instance for the given model_id.
    If model_id is None, uses the currently active NLP model.

    Args:
        model_id: Model identifier from MODEL_REGISTRY, or None for active.
        disable_components: List of pipeline components to disable (e.g., ["ner"]).

    Returns:
        SpacyNLPInference instance ready to use, or None if model not found/loaded.
    """
    # Resolve model_id
    if not model_id:
        model_id = get_active_model(NLP_CATEGORY)
        if model_id is None:
            darwin.log("No NLP model specified and no active model set.", LogType.ERROR)
            return None

    # Look up spec
    spec = find_spec(model_id, category=ModelCategory.NLP)
    if spec is None:
        darwin.log(f"Unknown NLP model: {model_id}", LogType.ERROR)
        return None

    # Get local path
    model_path = get_model_path(model_id, None, NLP_CATEGORY)
    if model_path is None:
        darwin.log(f"Could not determine path for NLP model: {model_id}", LogType.ERROR)
        return None

    # Ensure model files exist
    if not model_path.exists():
        darwin.log(f"NLP model not found at: {model_path}", LogType.ERROR)
        darwin.log("Please download it first using set_nlp_preference()", LogType.ERROR)
        return None

    # Instantiate and load
    try:
        inference = SpacyNLPInference(
            model_dir=model_path,
            model_name=spec.model_id,
            disable_components=disable_components or [],
        )
        inference.load()
        return inference
    except Exception as e:
        darwin.log(f"Failed to load NLP model {model_id}: {e}", LogType.ERROR)
        return None


def set_nlp_preference(
    model_id: str,
    auto_download: bool = True,
    force: bool = False,
) -> None:
    """
    Set the active NLP model preference, optionally downloading if missing.

    Args:
        model_id: The model identifier.
        auto_download: If True and model missing, download it.
        force: If True, re-download even if already present.
    """
    spec = find_spec(model_id, category=ModelCategory.NLP)
    if spec is None:
        raise ValueError(f"Unknown NLP model: {model_id}")

    # Local model only – no API path
    model_path = ensure_model_path(model_id, category=NLP_CATEGORY)
    if model_path is None:
        darwin.log(f"Failed to get path for NLP model: {model_id}", LogType.ERROR)
        darwin.log("Model will not be set as active", LogType.ERROR)
        return

    # Check if already downloaded
    if not model_path.exists():
        if not auto_download:
            darwin.log(f"NLP model not found at: {model_path}", LogType.ERROR)
            darwin.log("Model will not be set as active", LogType.ERROR)
            return
        darwin.log(f"NLP model missing. Downloading...", LogType.INFO)

    # Download/ensure
    downloader = SpacyNLPModelDownloader(
        model_name=spec.model_id ,
        target_dir=model_path,
    )
    success = downloader.ensure(force=force)
    if not success:
        darwin.log(f"Failed to download NLP model: {model_id}", LogType.ERROR)
        darwin.log("Model will not be set as active", LogType.ERROR)
        return

    # Set active
    set_active_model(NLP_CATEGORY,spec.model_id)
    darwin.log(f"NLP model set as active: {model_path}", LogType.SUCCESS)


def discard_nlp_model(
    model_id: str,
    delete: bool = False,
) -> None:
    """
    Discard or delete a local NLP model from preferences and optionally disk.

    Since NLP is optional (nice‑to‑have), deleting an active model is allowed.
    If delete=True, the files are removed and the active preference is cleared.
    If delete=False, only the active preference is cleared; files are kept.

    Args:
        model_id: The model identifier.
        delete: If True, delete the model files from disk.
                If False, only remove from preferences (if active).
    """
    import shutil

    spec = find_spec(model_id, category=ModelCategory.NLP)
    if spec is None:
        darwin.log(f"Unknown NLP model: {model_id}", LogType.ERROR)
        return

    current = get_active_model(NLP_CATEGORY)
    model_path = get_model_path(model_id, None, NLP_CATEGORY)

    # Check if this model is currently the active one
    is_active = (current == model_id) or (model_path and str(model_path) == current)

    # ── Delete files if requested ──────────────────────────────────────────
    if delete:
        if model_path and model_path.exists():
            try:
                shutil.rmtree(model_path)
                darwin.log(f"NLP model deleted from disk: {model_path}", LogType.INFO)
            except Exception as e:
                darwin.log(f"Failed to delete NLP model: {e}", LogType.ERROR)
                # Still proceed to clear active preference if needed
        else:
            darwin.log(f"NLP model path not found: {model_path}", LogType.WARNING)
    else:
        # If not deleting, just mention we're keeping files
        darwin.log(f"NLP model removed from preferences only (files kept)", LogType.INFO)

    # ── Clear active preference if it was the active model ────────────────
    if is_active:
        set_active_model(NLP_CATEGORY,spec.model_id)
        darwin.log(f"Removed {model_id} from active NLP preferences", LogType.INFO)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_nlp_model_path(model_id: str) -> Optional[Path]:
    """
    Get the file system path for an NLP model.

    Returns None if the model is not found or has no local path.
    """
    spec = find_spec(model_id, category=ModelCategory.NLP)
    if spec is None:
        darwin.log(f"Unknown NLP model: {model_id}", LogType.ERROR)
        return None
    return get_model_path(model_id, None, NLP_CATEGORY)


def is_nlp_model_downloaded(model_id: str) -> bool:
    """
    Check whether the NLP model files exist locally.
    """
    path = get_nlp_model_path(model_id)
    if path is None:
        return False
    return path.exists()


