from pathlib import Path
from typing import Optional
from core.Models.Lib.ModelRegistry import find_spec, ModelCategory
from core.Models.ModelConfig import get_active_model, set_active_model, get_model_path, ensure_model_path
from lib.KeyManager import is_key_set, load_keys_from_keychain
from core.Models.Infrence.Api.APIEmbeddingInference import APIEmbeddingInference
from core.Models.Infrence.Onnx.OnnxEmbeddingInference import OnnxEmbeddingInference
from core.Models.ModelDownloading.OnnxEmbeddingDownloader import OnnxEmbeddingDownloader
from core.Models.ModeInterfaces import EmbeddingInferenceInterface
from core.Logger.DarwinLogger import darwin


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_CATEGORY = ModelCategory.EMBEDDING.name.lower()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedding_inference(
    model_id: str,
    normalize: bool = True,
) -> Optional[EmbeddingInferenceInterface]:
    """
    Return the correct embedding inference backend for the given model_id.
    
    Routing logic:
      - If requires_provider is set → APIEmbeddingInference
      - Otherwise → OnnxEmbeddingInference (local)
    
    Args:
        model_id: Key into MODEL_REGISTRY or a raw model identifier.
        normalize: Whether to L2-normalize output embeddings.
    
    Returns:
        An EmbeddingInferenceInterface implementation, ready to .load(),
        or None if the model cannot be loaded.
    """
    # Use active model if none specified
    if model_id is None or model_id == "":
        model_id = get_active_model(EMBEDDING_CATEGORY)
        if model_id is None:
            darwin.error("No embedding model specified and no active model set")
            return None
    
    spec = find_spec(model_id, category=ModelCategory.EMBEDDING)

    if spec is None:
        darwin.error(f"Unknown embedding model: {model_id}")
        return None

    # Route to appropriate backend
    if spec.requires_provider:
        # API-based model
        load_keys_from_keychain(log=False)
        if not is_key_set(spec.requires_provider):
            darwin.error(f"Missing API key for provider: {spec.requires_provider}")
            darwin.error("Model loading failed")
            return None
        api_embed =APIEmbeddingInference(model_id=spec.model_id, normalize=normalize)
        api_embed.load()
        return api_embed
    else:
        # Local ONNX model
        model_path = ensure_model_path(model_id, category=EMBEDDING_CATEGORY)
        if model_path is None:
            darwin.error(f"No local path found for model: {model_id}")
            darwin.error("Model loading failed")
            return None
            
        # Check if model file exists
        if not model_path.exists():
            darwin.error(f"Model file not found at: {model_path}")
            darwin.error("Model loading failed")
            return None
            
        onnx_embed = OnnxEmbeddingInference(model_id=spec.model_id,model_path=model_path, normalize=normalize)
        onnx_embed.load()
        return onnx_embed


def set_embedding_preference(model_id: str, auto_download: bool = True, force: bool = False) -> None:
    """
    Set the active embedding model preference.
    
    Args:
        model_id: The model identifier to set as active.
        auto_download: Whether to automatically download the model if missing locally.
        force: Whether to force re-download even if model exists.
    """
    model_spec = find_spec(model_id, category=ModelCategory.EMBEDDING)

    if model_spec is None:
        darwin.error(f"Unknown embedding model: {model_id}")
        return

    # Handle API-based models
    if model_spec.requires_provider:
        load_keys_from_keychain(log=False)
        if not is_key_set(model_spec.requires_provider):
            darwin.error(f"Missing API key for provider: {model_spec.requires_provider}")
            darwin.error("Model will not be set as active")
            return
        
        set_active_model(EMBEDDING_CATEGORY,model_spec.model_id,)
        darwin.info(f"Model set as active: {model_spec.model_id}")
        return

    # Handle local ONNX models
    # First, ensure the directory exists and get the path
    model_path = ensure_model_path(model_id, category=EMBEDDING_CATEGORY)
    
    if model_path is None:
        darwin.error(f"No local path found for model: {model_id}")
        darwin.error("Model will not be set as active")
        return

    # Check if model exists, and download if needed
    model_downloader = OnnxEmbeddingDownloader(target_dir=model_path)
    
    if not model_path.exists():
        if not auto_download:
            darwin.error(f"Model not found at: {model_path}")
            darwin.error("Model will not be set as active")
            return
        
        darwin.info("Model not found locally, attempting auto-download...")
    
    # Ensure the model is downloaded
    is_downloaded = model_downloader.ensure(force=force)
    
    if not is_downloaded:
        darwin.error(f"Failed to download model: {model_id}")
        darwin.error("Model will not be set as active")
        darwin.error("Try using recache or force to re-download")
        return
    
    # Set as active model
    set_active_model(EMBEDDING_CATEGORY,model_spec.model_id)
    darwin.info(f"Model set as active: {model_path}")


def discard_embedding_model(model_id: str, delete: bool = False) -> None:
    """
    Discard or delete a local embedding model.
    
    Args:
        model_id: The model identifier to discard.
        delete: If True, delete the model files from disk.
                If False, only remove from preferences.
    """
    import shutil
    
    # Validate model exists in registry
    model_spec = find_spec(model_id, category=ModelCategory.EMBEDDING)
    if model_spec is None:
        darwin.error(f"Unknown model: {model_id}")
        return
    
    # Check if it's an API-based model
    if model_spec.requires_provider:
        darwin.info(f"Model {model_id} is API-based, nothing to discard locally")
        darwin.info(f"To remove this model, clear the key for {model_spec.requires_provider}")
        
        # Still remove from preferences if it's the active model
        current_model = get_active_model(EMBEDDING_CATEGORY)
        if current_model == model_id:
            set_active_model(EMBEDDING_CATEGORY,None)
            darwin.info(f"Removed {model_id} from active preferences")
        return
    
    # Handle local model
    model_path = get_model_path(model_id, None, EMBEDDING_CATEGORY)
    
    if model_path is None:
        darwin.error(f"Model path not found for: {model_id}")
        return
    
    # Check if we're trying to delete the active model
    current_model = get_active_model(EMBEDDING_CATEGORY)
    if current_model == str(model_path) or current_model == model_id:
        if delete:
            darwin.error(f"Cannot delete the currently active model: {model_id}")
            darwin.error("Set a different model as active first")
            return
        else:
            # Just remove from preferences if not deleting
            set_active_model(EMBEDDING_CATEGORY,None)
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
        set_active_model(EMBEDDING_CATEGORY,None)
        darwin.info(f"Removed {model_id} from active preferences")


def get_embedding_model_path(model_id: str) -> Optional[Path]:
    """
    Get the file system path for a local embedding model.
    
    Args:
        model_id: The model identifier.
        
    Returns:
        Path to the model directory, or None if the model is API-based or not found.
    """
    spec = find_spec(model_id, category=ModelCategory.EMBEDDING)
    
    if spec is None:
        darwin.error(f"Unknown embedding model: {model_id}")
        return None
    
    if spec.requires_provider:
        darwin.info(f"Model {model_id} is API-based and has no local path")
        return None
    
    return get_model_path(model_id, None, EMBEDDING_CATEGORY)


def is_embedding_model_downloaded(model_id: str) -> bool:
    """
    Check if a local embedding model is downloaded and available.
    
    Args:
        model_id: The model identifier.
        
    Returns:
        True if the model exists locally, False otherwise.
    """
    spec = find_spec(model_id, category=ModelCategory.EMBEDDING)
    
    if spec is None:
        return False
    
    if spec.requires_provider:
        return True  # API models are always "available" if the key is set
    
    model_path = get_model_path(model_id, None, EMBEDDING_CATEGORY)
    if model_path is None:
        return False
    
    return model_path.exists()


