# src/core/Models/ModelLoading/Vision/VisionInferenceFactory.py
"""
Vision Inference Factory — mirrors EmbeddingInferenceFactory pattern.

Provides model registry-aware routing between API (LiteLLM) and local (ONNX)
vision backends. Consumed directly by ModelManager.

Usage:
    from src.core.Models.ModelLoading.Vision.VisionInferenceFactory import get_vision_inference

    inference = get_vision_inference("openai/gpt-4o-mini")
    # or
    inference = get_vision_inference("HuggingFaceTB/SmolVLM-256M-Instruct")
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
from src.lib.KeyManager import is_key_set, load_keys_from_keychain
from src.core.Logger.DarwinLogger import darwin
from src.core.Models.ModeInterfaces import VisionInferenceInterface


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VISION_CATEGORY = ModelCategory.VISION.name.lower()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_vision_inference(
    model_id: str,
    **kwargs,
) -> Optional[VisionInferenceInterface]:
    """
    Return the correct vision inference backend for the given model_id.

    Routing logic:
      - If requires_provider is set -> APIVisionInference (LiteLLM)
      - Otherwise -> SmolVLMOnnxInference (local ONNX)

    Args:
        model_id: Key into MODEL_REGISTRY or a raw model identifier.
        **kwargs: Additional args passed to the backend constructor:
            - For API: max_image_size, jpeg_quality
            - For LOCAL: decoder_precision, vision_precision, embed_precision,
                         max_new_tokens, temperature, top_p, top_k, seed

    Returns:
        A VisionInferenceInterface implementation, ready to .load(),
        or None if the model cannot be loaded.
    """
    # Use active model if none specified
    if model_id is None or model_id == "":
        model_id = get_active_model(VISION_CATEGORY)
        if model_id is None:
            darwin.error("No vision model specified and no active model set")
            return None

    spec = find_spec(model_id, category=ModelCategory.VISION)

    if spec is None:
        darwin.error(f"Unknown vision model: {model_id}")
        return None

    # Route to appropriate backend
    # API-based model (OpenAI, Anthropic, etc.)
    load_keys_from_keychain(log=False)
    if not is_key_set(spec.requires_provider):
        darwin.error(f"Missing API key for provider: {spec.requires_provider}")
        darwin.error("Model loading failed")
        return None

    from src.core.Models.Infrence.Api.ApiVisionInfrence import APIVisionInference
    api_vision = APIVisionInference(
        model_id=spec.model_id,
        max_image_size=kwargs.get("max_image_size", 1280),
        jpeg_quality=kwargs.get("jpeg_quality", 85),
    )
    api_vision.load()
    return api_vision




def set_vision_preference(
    model_id: str,
    auto_download: bool = True,
    force: bool = False,
    **kwargs,
) -> None:
    """
    Set the active vision model preference.

    Args:
        model_id: The model identifier to set as active.
        auto_download: Whether to automatically download the model if missing locally.
        force: Whether to force re-download even if model exists.
        **kwargs: Additional config for local models (decoder_precision, etc.)
    """
    model_spec = find_spec(model_id, category=ModelCategory.VISION)

    if model_spec is None:
        darwin.error(f"Unknown vision model: {model_id}")
        return

    # Handle API-based models
    load_keys_from_keychain(log=False)
    if not is_key_set(model_spec.requires_provider):
        darwin.error(f"Missing API key for provider: {model_spec.requires_provider}")
        darwin.error("Model will not be set as active")
        return

    set_active_model(VISION_CATEGORY, model_spec.model_id)
    darwin.info(f"Model set as active: {model_spec.model_id}")
    return



def discard_vision_model(model_id: str, delete: bool = False) -> None:
    """
    Discard or delete a local vision model.

    Args:
        model_id: The model identifier to discard.
        delete: If True, delete the model files from disk.
                If False, only remove from preferences.
    """
    import shutil

    # Validate model exists in registry
    model_spec = find_spec(model_id, category=ModelCategory.VISION)
    if model_spec is None:
        darwin.error(f"Unknown model: {model_id}")
        return

    # Check if it's an API-based model
    if model_spec.requires_provider:
        darwin.info(f"Model {model_id} is API-based, nothing to discard locally")
        darwin.info(f"To remove this model, clear the key for {model_spec.requires_provider}")

        # Still remove from preferences if it's the active model
        current_model = get_active_model(VISION_CATEGORY)
        if current_model == model_id:
            set_active_model(VISION_CATEGORY, None)
            darwin.info(f"Removed {model_id} from active preferences")
        return

    # Handle local model
    model_path = get_model_path(model_id, None, VISION_CATEGORY)

    if model_path is None:
        darwin.error(f"Model path not found for: {model_id}")
        return

    # Check if we're trying to delete the active model
    current_model = get_active_model(VISION_CATEGORY)
    if current_model == str(model_path) or current_model == model_id:
        if delete:
            darwin.error(f"Cannot delete the currently active model: {model_id}")
            darwin.error("Set a different model as active first")
            return
        else:
            # Just remove from preferences if not deleting
            set_active_model(VISION_CATEGORY, None)
            darwin.info(f"Removed {model_id} from active preferences")
            return

    # Delete if requested
    if delete:
        if model_path.exists():
            try:
                if model_path.is_dir():
                    shutil.rmtree(model_path)
                    darwin.info(f"Model directory deleted: {model_path}")
                else:
                    model_path.unlink()
                    darwin.info(f"Model file deleted: {model_path}")

                darwin.info(f"Model {model_id} deleted from disk")
            except Exception as e:
                darwin.error(f"Failed to delete model at {model_path}: {e}")
                return
        else:
            darwin.warning(f"Model path does not exist: {model_path}")
            darwin.info("Nothing to delete")

    # Remove from preferences if it was set as active (shouldn't be, but just in case)
    if current_model == str(model_path) or current_model == model_id:
        set_active_model(VISION_CATEGORY, None)
        darwin.info(f"Removed {model_id} from active preferences")


def get_vision_model_path(model_id: str) -> Optional[Path]:
    """
    Get the file system path for a local vision model.

    Args:
        model_id: The model identifier.

    Returns:
        Path to the model directory, or None if the model is API-based or not found.
    """
    spec = find_spec(model_id, category=ModelCategory.VISION)

    if spec is None:
        darwin.error(f"Unknown vision model: {model_id}")
        return None

    if spec.requires_provider:
        darwin.info(f"Model {model_id} is API-based and has no local path")
        return None

    return get_model_path(model_id, None, VISION_CATEGORY)


def is_vision_model_downloaded(model_id: str) -> bool:
    """
    Check if a local vision model is downloaded and available.

    Args:
        model_id: The model identifier.

    Returns:
        True if the model exists locally, False otherwise.
    """
    spec = find_spec(model_id, category=ModelCategory.VISION)

    if spec is None:
        return False

    if spec.requires_provider:
        load_keys_from_keychain(log=False)
        if not is_key_set(spec.requires_provider):
            darwin.error(f"Missing API key for provider: {spec.requires_provider}, model not available")
            return False
        

        return True  # API models are always "available" if the key is set

    model_path = get_model_path(model_id, None, VISION_CATEGORY)
    if model_path is None:
        return False
 
    return model_path.exists()