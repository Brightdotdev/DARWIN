# src/core/Models/ModelLoading/ONNXLoader.py

from pathlib import Path
from typing import Optional

from src.core.Models.ModelDownloader import ModelDownloader
from .ONNXWrapper import ONNXEmbeddingWrapper, ONNXRerankerWrapper


class ONNXLoader:
    def __init__(self, downloader: Optional[ModelDownloader] = None):
        self.downloader = downloader or ModelDownloader()

    def load(self, key: str):
        """
        Load ONNX model from cache or download it.
        Auto-detects embedding vs reranker from registry.
        """
        from src.core.Models.ModelRegistry import get_spec
        
        spec = get_spec(key)
        
        if not self.downloader.is_cached(key, "onnx"):
            self.downloader.download_onnx(key)
        
        model_dir = self.downloader._get_model_dir(spec.model_id, spec.category, "onnx")
        return self.load_from_path(model_dir, category=spec.category)

    def load_from_path(self, model_dir: Path, category: str = "embedding"):
        if category == "reranker":
            return ONNXRerankerWrapper(model_dir)
        return ONNXEmbeddingWrapper(model_dir)