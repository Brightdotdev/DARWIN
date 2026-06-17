# src/core/Models/ModelLoading/API/OCRInferenceFactory.py
"""
Factory for creating OCR inference instances (local ONNX only).
Manages model path, downloads, and active preferences.
"""

from pathlib import Path
from typing import Optional

from src.core.Models.Lib.ModelRegistry import find_spec, ModelCategory
from src.core.Models.ModelConfig import (
    get_active_model,
    set_active_model,
    get_model_path,
    ensure_model_path,
)
from src.core.Models.Infrence.Onnx.RapidOCRInference import RapidOCRInference
from src.core.Models.ModelDownloading.OnnxOCRDownloader import OnnxOCRDownloader
from src.core.Logger.DarwinLogger import darwin, LogType


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OCR_CATEGORY = ModelCategory.OCR.name.lower()  # "ocr"


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------

def get_ocr_inference(
    model_id: Optional[str] = None,
) -> Optional[RapidOCRInference]:
    """
    Return an OCR inference instance for the given model_id.
    If model_id is None, uses the currently active OCR model.

    Args:
        model_id: Model identifier from MODEL_REGISTRY, or None for active.

    Returns:
        RapidOCRInference instance ready to use, or None if model not found/loaded.
    """
    if not model_id:
        model_id = get_active_model(OCR_CATEGORY)
        if model_id is None:
            darwin.log("No OCR model specified and no active model set.", LogType.ERROR)
            return None

    spec = find_spec(model_id, category=ModelCategory.OCR)
    if spec is None:
        darwin.log(f"Unknown OCR model: {model_id}", LogType.ERROR)
        return None

    model_path = get_model_path(model_id, None, OCR_CATEGORY)
    if model_path is None:
        darwin.log(f"Could not determine path for OCR model: {model_id}", LogType.ERROR)
        return None

    # Ensure model files exist and are valid
    if not _is_ocr_model_valid(model_path):
        darwin.log(f"OCR model not found or invalid at: {model_path}", LogType.ERROR)
        darwin.log(f"Please download it first using set_ocr_preference()", LogType.ERROR)
        return None

    try:
        inference = RapidOCRInference(model_dir=model_path)
        inference.load()
        return inference
    except Exception as e:
        darwin.log(f"Failed to load OCR model {model_id}: {e}", LogType.ERROR)
        return None


def set_ocr_preference(
    model_id: str,
    auto_download: bool = True,
    force: bool = False,
) -> None:
    """
    Set the active OCR model preference, optionally downloading if missing.
    """
    spec = find_spec(model_id, category=ModelCategory.OCR)
    if spec is None:
        raise ValueError(f"Unknown OCR model: {model_id}")

    model_path = ensure_model_path(model_id, category=OCR_CATEGORY)
    if model_path is None:
        darwin.log(f"Failed to get path for OCR model: {model_id}", LogType.ERROR)
        darwin.log("Model will not be set as active", LogType.ERROR)
        return

    downloader = OnnxOCRDownloader(target_dir=model_path, repo_id=spec.repo_id)

    # Check if already downloaded and valid
    if model_path.exists() and downloader.verify_integrity():
        if not force:
            darwin.log(f"OCR model already valid: {model_id}", LogType.INFO)
            set_active_model(OCR_CATEGORY,spec.model_id)
            darwin.log(f"OCR model set as active: {model_path}", LogType.SUCCESS)
            return
        else:
            darwin.log(f"Force re-download requested for {model_id}", LogType.INFO)

    # If directory exists but files are invalid, we need to clean up
    if model_path.exists() and not downloader.verify_integrity():
        darwin.log(f"OCR model files corrupted or incomplete. Re-downloading...", LogType.WARNING)

    if not auto_download:
        darwin.log(f"OCR model not available and auto_download=False", LogType.ERROR)
        darwin.log("Model will not be set as active", LogType.ERROR)
        return

    darwin.log(f"Downloading OCR model {model_id}...", LogType.INFO)
    success = downloader.ensure(force=force)
    if not success:
        darwin.log(f"Failed to download OCR model: {model_id}", LogType.ERROR)
        darwin.log("Model will not be set as active", LogType.ERROR)
        return

    # Verify after download
    if not downloader.verify_integrity():
        darwin.log(f"Integrity check failed after download for {model_id}", LogType.ERROR)
        return

    set_active_model(OCR_CATEGORY,spec.model_id)
    darwin.log(f"OCR model set as active: {model_path}", LogType.SUCCESS)


def discard_ocr_model(
    model_id: str,
    delete: bool = False,
) -> None:
    """
    Discard or delete a local OCR model from preferences and optionally disk.
    """
    import shutil

    spec = find_spec(model_id, category=ModelCategory.OCR)
    if spec is None:
        darwin.log(f"Unknown OCR model: {model_id}", LogType.ERROR)
        return

    current = get_active_model(OCR_CATEGORY)
    model_path = get_model_path(model_id, None, OCR_CATEGORY)

    is_active = (current == model_id) or (model_path and str(model_path) == current)

    if delete:
        if model_path and model_path.exists():
            try:
                shutil.rmtree(model_path)
                darwin.log(f"OCR model deleted from disk: {model_path}", LogType.INFO)
            except Exception as e:
                darwin.log(f"Failed to delete OCR model: {e}", LogType.ERROR)
        else:
            darwin.log(f"OCR model path not found: {model_path}", LogType.WARNING)
    else:
        darwin.log(f"OCR model removed from preferences only (files kept)", LogType.INFO)

    if is_active:
        set_active_model(OCR_CATEGORY,spec.model_id)
        darwin.log(f"Removed {model_id} from active OCR preferences", LogType.INFO)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_ocr_model_path(model_id: str) -> Optional[Path]:
    spec = find_spec(model_id, category=ModelCategory.OCR)
    if spec is None:
        darwin.log(f"Unknown OCR model: {model_id}", LogType.ERROR)
        return None
    return get_model_path(model_id, None, OCR_CATEGORY)


def is_ocr_model_downloaded(model_id: str) -> bool:
    """
    Check whether the OCR model files exist locally and are valid.
    """
    path = get_ocr_model_path(model_id)
    if path is None:
        return False
    return _is_ocr_model_valid(path)


def _is_ocr_model_valid(model_path: Path) -> bool:
    """
    Check that the required ONNX files exist and have reasonable size.
    Uses the downloader's integrity check to avoid duplication.
    """
    if not model_path.exists():
        return False
    # Use downloader to validate files
    downloader = OnnxOCRDownloader(target_dir=model_path)
    return downloader.verify_integrity()