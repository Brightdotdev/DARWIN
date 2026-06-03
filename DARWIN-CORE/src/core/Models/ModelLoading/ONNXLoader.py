"""
src/core/Models/ModelLoading/ONNXLoader.py
Robust loader for ONNX models – uses the enhanced OnnxModelDownloader
and Darwin event system for logging. Reads model parameters from ModelSpec.
"""

from pathlib import Path
from typing import Union

from src.core.Models.ModelDownloading.OnnxModelDownloader import OnnxModelDownloader
from src.core.Models.EmbeddingConfig import get_preset
from src.core.Logger.DarwinLogger import darwin, LogType
from .ONNXWrapper import ONNXEmbeddingWrapper, ONNXRerankerWrapper


class ONNXLoader:
    """
    Loads ONNX embedding/reranker models, downloading if necessary.
    Model parameters (max_length, min_length, normalize) come from ModelSpec.
    """

    def __init__(self, downloader: OnnxModelDownloader = None):
        self.downloader = downloader or OnnxModelDownloader()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(self, key: str) -> Union[ONNXEmbeddingWrapper, ONNXRerankerWrapper]:
        """
        Ensure model is cached (download if needed), then load into a wrapper.

        Args:
            key: Registry key (e.g. 'BAAI/bge-m3-onnx').

        Returns:
            Wrapper instance ready for inference.
        """
        spec = get_preset(key)
        model_dir: Path

        if self.downloader.is_cached(key, "onnx"):
            model_dir = self.downloader.get_model_path(key, "onnx")
            darwin.log(f"ONNX model {key} found in cache ({model_dir})")
        else:
            darwin.log(f"ONNX model {key} not cached – downloading …")
            model_dir = self.downloader.download_onnx(key)   # raises on failure

        # Extract params from spec with safe defaults
        max_length = spec.max_tokens
        min_length = spec.min_tokens
        
        normalize = getattr(spec, "normalize", True)

        wrapper = self.load_from_path(
            model_dir,
            category=spec.category,
            max_length=max_length,
            min_length=min_length,
            normalize=normalize,
        )

        darwin.log("ONNX model loaded successfully: {key}",LogType.SUCCESS,)
        return wrapper

    def load_from_path(
        self,
        model_dir: Path,
        category: str = "embedding",
        max_length: int = 512,
        min_length: int = 1,
        normalize: bool = True,
    ) -> Union[ONNXEmbeddingWrapper, ONNXRerankerWrapper]:
        """
        Instantiate a wrapper from an already‑cached model directory.

        Args:
            model_dir: Directory containing model.onnx and tokenizer files.
            category: 'embedding' or 'reranker'.
            max_length: Maximum token count before truncation.
            min_length: Minimum token count before warning.
            normalize: Whether to L2‑normalize output vectors.

        Returns:
            Wrapper instance.
        """
        if category == "reranker":
            return ONNXRerankerWrapper(
                model_dir,
                max_length=max_length,
                min_length=min_length,
            )
        return ONNXEmbeddingWrapper(
            model_dir,
            max_length=max_length,
            min_length=min_length,
            normalize=normalize,
        )