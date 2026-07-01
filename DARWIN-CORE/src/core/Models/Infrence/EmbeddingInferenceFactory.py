from pathlib import Path
from typing import Optional
from src.core.Models.Lib.ModelRegistry import find_spec, ModelCategory
from src.core.Models.ModelConfig import get_active_model, set_active_model, get_model_path, ensure_model_path
from src.lib.KeyManager import is_key_set, load_keys_from_keychain
from src.core.Models.Infrence.Api.APIEmbeddingInference import APIEmbeddingInference
from src.core.Models.Infrence.Onnx.OnnxEmbeddingInference import OnnxEmbeddingInference
from src.core.Models.ModelDownloading.OnnxEmbeddingDownloader import OnnxEmbeddingDownloader
from src.core.Models.ModeInterfaces import EmbeddingInferenceInterface
from src.core.Logger.DarwinLogger import darwin, LogType


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
            darwin.log("No embedding model specified and no active model set.", LogType.ERROR)
            return None
    
    spec = find_spec(model_id, category=ModelCategory.EMBEDDING)

    if spec is None:
        darwin.log(f"Unknown embedding model: {model_id}", LogType.ERROR)
        return None

    # Route to appropriate backend
    if spec.requires_provider:
        # API-based model
        load_keys_from_keychain(log=False)
        if not is_key_set(spec.requires_provider):
            darwin.log(f"Missing key for provider: {spec.requires_provider}", LogType.ERROR)
            darwin.log(f"Model Loading Failed", LogType.ERROR)
            return None
        api_embed =APIEmbeddingInference(model_id=spec.model_id, normalize=normalize)
        api_embed.load()
        return api_embed
    else:
        # Local ONNX model
        model_path = ensure_model_path(model_id, category=EMBEDDING_CATEGORY)
        if model_path is None:
            darwin.log(f"Failed to get path for model: {model_id}", LogType.ERROR)
            darwin.log(f"Model Loading Failed", LogType.ERROR)
            return None
            
        # Check if model file exists
        if not model_path.exists():
            darwin.log(f"Model file not found at: {model_path}", LogType.ERROR)
            darwin.log(f"Model Loading Failed", LogType.ERROR)
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
        raise ValueError(f"Unknown embedding model: {model_id}")

    # Handle API-based models
    if model_spec.requires_provider:
        load_keys_from_keychain(log=False)
        if not is_key_set(model_spec.requires_provider):
            darwin.log(f"Missing key for provider: {model_spec.requires_provider}", LogType.ERROR)
            darwin.log(f"Model will not be set as active", LogType.ERROR)
            return
        
        set_active_model(EMBEDDING_CATEGORY,model_spec.model_id,)
        darwin.log(f"Model set as active: {model_spec.model_id}", LogType.INFO)
        return

    # Handle local ONNX models
    # First, ensure the directory exists and get the path
    model_path = ensure_model_path(model_id, category=EMBEDDING_CATEGORY)
    
    if model_path is None:
        darwin.log(f"Failed to get path for model: {model_id}", LogType.ERROR)
        darwin.log(f"Model will not be set as active", LogType.ERROR)
        return

    # Check if model exists, and download if needed
    model_downloader = OnnxEmbeddingDownloader(target_dir=model_path)
    
    if not model_path.exists():
        if not auto_download:
            darwin.log(f"Model not found at: {model_path}", LogType.ERROR)
            darwin.log(f"Model will not be set as active", LogType.ERROR)
            return
        
        darwin.log(f"Model not found. Attempting auto-download...", LogType.INFO)
    
    # Ensure the model is downloaded
    is_downloaded = model_downloader.ensure(force=force)
    
    if not is_downloaded:
        darwin.log(f"Failed to download model: {model_id}", LogType.ERROR)
        darwin.log(f"Model will not be set as active", LogType.ERROR)
        darwin.log(f"Try using the recache or force options to cleanup and redownload", LogType.ERROR)
        return
    
    # Set as active model
    set_active_model(EMBEDDING_CATEGORY,model_spec.model_id)
    darwin.log(f"Model set as active: {model_path}", LogType.INFO)


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
        darwin.log(f"Unknown model: {model_id}", LogType.ERROR)
        return
    
    # Check if it's an API-based model
    if model_spec.requires_provider:
        darwin.log(f"Model {model_id} is an API-based model. Nothing to discard locally.", LogType.INFO)
        darwin.log(f"To remove this model, clear the key for {model_spec.requires_provider}", LogType.INFO)
        
        # Still remove from preferences if it's the active model
        current_model = get_active_model(EMBEDDING_CATEGORY)
        if current_model == model_id:
            set_active_model(EMBEDDING_CATEGORY,None)
            darwin.log(f"Removed {model_id} from active preferences", LogType.INFO)
        return
    
    # Handle local model
    model_path = get_model_path(model_id, None, EMBEDDING_CATEGORY)
    
    if model_path is None:
        darwin.log(f"Model path not found for: {model_id}", LogType.ERROR)
        return
    
    # Check if we're trying to delete the active model
    current_model = get_active_model(EMBEDDING_CATEGORY)
    if current_model == str(model_path) or current_model == model_id:
        if delete:
            darwin.log(f"Cannot delete the currently active model: {model_id}", LogType.ERROR)
            darwin.log(f"Please set a different model as active first", LogType.ERROR)
            return
        else:
            # Just remove from preferences if not deleting
            set_active_model(EMBEDDING_CATEGORY,None)
            darwin.log(f"Removed {model_id} from active preferences", LogType.INFO)
            return
    
    # Delete if requested
    if delete:
        if model_path.exists():
            try:
                if model_path.is_dir():
                    shutil.rmtree(model_path)
                    darwin.log(f"Model directory deleted: {model_path}", LogType.INFO)
                else:
                    model_path.unlink()
                    darwin.log(f"Model file deleted: {model_path}", LogType.INFO)
                    
                darwin.log(f"Model {model_id} has been deleted from disk", LogType.INFO)
            except Exception as e:
                darwin.log(f"Failed to delete model at {model_path}: {e}", LogType.ERROR)
                return
        else:
            darwin.log(f"Model path does not exist: {model_path}", LogType.WARNING)
            darwin.log(f"Nothing to delete", LogType.INFO)
    
    # Remove from preferences if it was set as active (shouldn't be, but just in case)
    if current_model == str(model_path) or current_model == model_id:
        set_active_model(EMBEDDING_CATEGORY,None)
        darwin.log(f"Removed {model_id} from active preferences", LogType.INFO)


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
        darwin.log(f"Unknown embedding model: {model_id}", LogType.ERROR)
        return None
    
    if spec.requires_provider:
        darwin.log(f"Model {model_id} is API-based and has no local path", LogType.INFO)
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


