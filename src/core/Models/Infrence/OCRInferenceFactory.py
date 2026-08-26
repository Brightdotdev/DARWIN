"""
OCR Inference Factory — mirrors VisionInferenceFactory exactly.

Provides registry-aware routing for local OCR backends. Consumed by ModelManager.
"""

import shutil
from pathlib import Path
from typing import Optional

from core.Models.Lib.ModelRegistry import find_spec, ModelCategory
from core.Models.ModelConfig import (
    get_active_model,
    set_active_model,
    get_model_path,
    ensure_model_path,
)
from core.Logger.DarwinLogger import darwin
from core.Models.ModeInterfaces import OCRInferenceInterface

from core.Models.Infrence.Onnx.RapidOCRInference import RapidOCRInference, MODELS


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
            darwin.error("No OCR model specified and no active model set")
            return None

    spec = find_spec(model_id, category=ModelCategory.OCR)

    if spec is None:
        darwin.error(f"Unknown OCR model: {model_id}")
        return None

    # Resolve local path
    model_path = get_model_path(model_id, None, OCR_CATEGORY)
    if model_path is None:
        model_path = ensure_model_path(model_id, OCR_CATEGORY)

    # Route to local RapidOCR
    ok, err = RapidOCRInference.verify_integrity(model_path)

    if not ok:
        darwin.warning(f"OCR models missing or corrupt: {err}")
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
        darwin.error(f"Unknown OCR model: {model_id}")
        return

    # Resolve path
    model_path = get_model_path(model_id, None, OCR_CATEGORY)
    if model_path is None:
        model_path = ensure_model_path(model_id, OCR_CATEGORY)

    # Integrity check before setting preference
    ok, err = RapidOCRInference.verify_integrity(model_path)

    if not ok:
        darwin.warning(f"OCR models missing or corrupt: {err}")
        if not auto_download:
            darwin.error(f"Auto-download disabled, cannot set preference for {model_id}")
            return

        # Download missing/corrupt models
        darwin.info(f"Auto-downloading OCR models for {model_id}...")
        try:
            # Trigger a temporary load which handles download + re-verify
            temp = RapidOCRInference(model_dir=model_path, **kwargs)
            temp.load()
            # Re-verify after download
            ok, err = RapidOCRInference.verify_integrity(model_path)
            if not ok:
                darwin.error(f"OCR model verification failed after download: {err}")
                return
        except Exception as e:
            darwin.error(f"Auto-download failed for {model_id}: {e}")
            return

    elif force and auto_download:
        # Force re-download: nuke existing files and re-download
        darwin.info(f"Force re-download requested for {model_id}")
        for key in ("det", "rec", "cls"):
            f = model_path / MODELS[key]["file"]
            if f.exists():
                f.unlink()
                darwin.debug(f"Removed {f.name} for re-download")

        try:
            temp = RapidOCRInference(model_dir=model_path, **kwargs)
            temp.load()
            ok, err = RapidOCRInference.verify_integrity(model_path)
            if not ok:
                darwin.error(f"OCR model verification failed after forced re-download: {err}")
                return
        except Exception as e:
            darwin.error(f"Forced re-download failed for {model_id}: {e}")
            return

    # All clear — set preference
    set_active_model(OCR_CATEGORY, model_id)
    darwin.info(f"OCR model set as active: {model_id}")


def discard_ocr_model(model_id: str, delete: bool = False) -> None:
    """
    Discard or delete a local OCR model.

    Args:
        model_id: The model identifier.
        delete: If True, delete model files from disk.
    """
    model_spec = find_spec(model_id, category=ModelCategory.OCR)

    if model_spec is None and model_id.lower() not in ("rapidocr", "default"):
        darwin.error(f"Unknown OCR model: {model_id}")
        return

    model_path = get_model_path(model_id, None, OCR_CATEGORY)
    if model_path is None:
        darwin.error(f"Model path not found for: {model_id}")
        return

    current_model = get_active_model(OCR_CATEGORY)
    if current_model == model_id:
        if delete:
            darwin.error(f"Cannot delete currently active OCR model: {model_id}")
            darwin.error("Set a different OCR model as active first")
            return
        set_active_model(OCR_CATEGORY, None)
        darwin.info(f"Removed {model_id} from active OCR preferences")

    if delete and model_path.exists():
        try:
            if model_path.is_dir():
                shutil.rmtree(model_path)
            else:
                model_path.unlink()
            darwin.info(f"Deleted OCR model files: {model_path}")
        except Exception as e:
            darwin.error(f"Failed to delete OCR model: {e}")


def get_ocr_model_path(model_id: str) -> Optional[Path]:
    """Get filesystem path for a local OCR model."""
    spec = find_spec(model_id, category=ModelCategory.OCR)
    if spec is None and model_id.lower() not in ("rapidocr", "default"):
        darwin.error(f"Unknown OCR model: {model_id}")
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