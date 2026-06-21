"""
OCR Inference Factory — mirrors VisionInferenceFactory exactly.

Provides registry-aware routing for local OCR backends. Consumed by ModelManager.
"""

import shutil
from pathlib import Path
from typing import Optional

from src.core.Models.Lib.ModelRegistry import find_spec, ModelCategory
from src.core.Models.ModelConfig import (
    get_active_model,
    set_active_model,
    get_model_path,
    ensure_model_path,
)
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Models.ModeInterfaces import OCRInferenceInterface

# IMPORT HERE - not inside functions
from src.core.Models.Infrence.Onnx.RapidOCRInference import RapidOCRInference, MODELS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OCR_CATEGORY = ModelCategory.OCR.name.lower()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_ocr_inference(
    model_id: str = None,
    **kwargs,
) -> Optional[OCRInferenceInterface]:
    """
    Return the correct OCR inference backend for the given model_id.

    Routing logic:
      - rapidocr (default) -> RapidOCRInference (local ONNX)

    Args:
        model_id: Key into MODEL_REGISTRY or raw identifier. Defaults to active model.
        **kwargs: Passed to backend constructor:
            - confidence_threshold (float, default 0.7)
            - det_db_thresh (float, default 0.2)
            - det_db_box_thresh (float, default 0.4)

    Returns:
        An OCRInferenceInterface implementation, ready after .load().
    """
    # Use active model if none specified
    if not model_id:
        model_id = get_active_model(OCR_CATEGORY)
        if not model_id:
            darwin.log("No OCR model specified and no active model set.", LogType.ERROR)
            return None

    spec = find_spec(model_id, category=ModelCategory.OCR)

    if spec is None:
        darwin.log(f"Unknown OCR model: {model_id}", LogType.ERROR)
        return None

    # Resolve local path
    model_path = get_model_path(model_id, None, OCR_CATEGORY)
    if model_path is None:
        model_path = ensure_model_path(model_id, OCR_CATEGORY)

    # Route to local RapidOCR
    ok, err = RapidOCRInference.verify_integrity(model_path)

    if not ok:
        darwin.log(f"OCR models missing or corrupt: {err}", LogType.WARNING)
        return None

    inference = RapidOCRInference(
        model_dir=model_path,
        confidence_threshold=kwargs.get("confidence_threshold", 0.7),
        det_db_thresh=kwargs.get("det_db_thresh", 0.2),
        det_db_box_thresh=kwargs.get("det_db_box_thresh", 0.4),
    )
    inference.load()
    return inference


def set_ocr_preference(
    model_id: str,
    auto_download: bool = True,
    force: bool = False,
    **kwargs,
) -> None:
    """
    Set the active OCR model preference.

    Verifies model integrity before setting the preference. If models are missing
    or corrupt and auto_download is True, attempts download first.

    Args:
        model_id: The model identifier to set as active.
        auto_download: Whether to automatically download the model if missing locally.
        force: Whether to force re-download even if model exists.
        **kwargs: Additional config for the backend.
    """
    model_spec = find_spec(model_id, category=ModelCategory.OCR)

    if model_spec is None and model_id.lower() not in ("rapidocr", "default"):
        raise ValueError(f"Unknown OCR model: {model_id}")

    # Resolve path
    model_path = get_model_path(model_id, None, OCR_CATEGORY)
    if model_path is None:
        model_path = ensure_model_path(model_id, OCR_CATEGORY)

    # Integrity check before setting preference
    ok, err = RapidOCRInference.verify_integrity(model_path)

    if not ok:
        darwin.log(f"OCR models missing or corrupt: {err}", LogType.WARNING)
        if not auto_download:
            darwin.log(f"Auto-download disabled. Cannot set preference for {model_id}.", LogType.ERROR)
            return

        # Download missing/corrupt models
        darwin.log(f"Auto-downloading OCR models for {model_id}...", LogType.INFO)
        try:
            # Trigger a temporary load which handles download + re-verify
            temp = RapidOCRInference(model_dir=model_path, **kwargs)
            temp.load()
            # Re-verify after download
            ok, err = RapidOCRInference.verify_integrity(model_path)
            if not ok:
                darwin.log(f"OCR model verification failed after download: {err}", LogType.ERROR)
                return
        except Exception as e:
            darwin.log(f"Auto-download failed for {model_id}: {e}", LogType.ERROR)
            return

    elif force and auto_download:
        # Force re-download: nuke existing files and re-download
        darwin.log(f"Force re-download requested for {model_id}", LogType.INFO)
        for key in ("det", "rec", "cls"):
            f = model_path / MODELS[key]["file"]
            if f.exists():
                f.unlink()
                darwin.log(f"Removed {f.name} for re-download", LogType.DEBUG)

        try:
            temp = RapidOCRInference(model_dir=model_path, **kwargs)
            temp.load()
            ok, err = RapidOCRInference.verify_integrity(model_path)
            if not ok:
                darwin.log(f"OCR model verification failed after forced re-download: {err}", LogType.ERROR)
                return
        except Exception as e:
            darwin.log(f"Forced re-download failed for {model_id}: {e}", LogType.ERROR)
            return

    # All clear — set preference
    set_active_model(OCR_CATEGORY, model_id)
    darwin.log(f"OCR model set as active: {model_id}", LogType.INFO)


def discard_ocr_model(model_id: str, delete: bool = False) -> None:
    """
    Discard or delete a local OCR model.

    Args:
        model_id: The model identifier.
        delete: If True, delete model files from disk.
    """
    model_spec = find_spec(model_id, category=ModelCategory.OCR)

    if model_spec is None and model_id.lower() not in ("rapidocr", "default"):
        darwin.log(f"Unknown OCR model: {model_id}", LogType.ERROR)
        return

    model_path = get_model_path(model_id, None, OCR_CATEGORY)
    if model_path is None:
        darwin.log(f"Model path not found for: {model_id}", LogType.ERROR)
        return

    current_model = get_active_model(OCR_CATEGORY)
    if current_model == model_id:
        if delete:
            darwin.log(f"Cannot delete currently active OCR model: {model_id}", LogType.ERROR)
            darwin.log("Set a different OCR model as active first.", LogType.ERROR)
            return
        set_active_model(OCR_CATEGORY, None)
        darwin.log(f"Removed {model_id} from active OCR preferences", LogType.INFO)

    if delete and model_path.exists():
        try:
            if model_path.is_dir():
                shutil.rmtree(model_path)
            else:
                model_path.unlink()
            darwin.log(f"Deleted OCR model files: {model_path}", LogType.INFO)
        except Exception as e:
            darwin.log(f"Failed to delete OCR model: {e}", LogType.ERROR)


def get_ocr_model_path(model_id: str) -> Optional[Path]:
    """Get filesystem path for a local OCR model."""
    spec = find_spec(model_id, category=ModelCategory.OCR)
    if spec is None and model_id.lower() not in ("rapidocr", "default"):
        darwin.log(f"Unknown OCR model: {model_id}", LogType.ERROR)
        return None
    return get_model_path(model_id, None, OCR_CATEGORY)


def is_ocr_model_downloaded(model_id: str) -> bool:
    """
    Check if a local OCR model is downloaded, verified, and available.

    Returns True only if all required ONNX files exist AND pass integrity checks.
    """
    spec = find_spec(model_id, category=ModelCategory.OCR)
    if spec is None and model_id.lower() not in ("rapidocr", "default"):
        return False

    model_path = get_model_path(model_id, None, OCR_CATEGORY)
    if model_path is None:
        return False

    if not model_path.exists():
        return False

    # Full integrity verification - no import needed, already imported
    ok, _ = RapidOCRInference.verify_integrity(model_path)
    return ok