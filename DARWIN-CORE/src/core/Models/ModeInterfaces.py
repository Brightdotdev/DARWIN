# src/core/Models/ModelLoading/API/EmbeddingInferenceInterface.py
"""
Unified interface for all embedding inference backends.
Both local (ONNX) and remote (API) implementations must conform to this contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

from src.core.Logger.DarwinLogger import darwin


import numpy as np


@dataclass(frozen=True)
class UnifiedEmbeddingMetadata:
    """Normalized metadata available from any backend."""
    vector_size: int
    inference_time_ms: float
    num_inputs: int
    device: str
    backend: str  # "onnx" | "api"
    model_id: str
    normalize: bool


@dataclass(frozen=True)
class UnifiedEmbeddingResult:
    """Normalized result from any embedding backend."""
    embeddings: np.ndarray  # shape: (num_inputs, vector_size)
    metadata: UnifiedEmbeddingMetadata


class EmbeddingInferenceInterface(ABC):
    """
    Abstract base for embedding inference engines.
    Guarantees consistent lifecycle (load → encode → unload) across ONNX and API.
    """

    def __init__(self, model_id: str, normalize: bool = True):
        self.model_id = model_id
        self.normalize = normalize

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> "EmbeddingInferenceInterface":
        """Load the model/tokenizer/session. Chainable."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release all resources."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the backend is ready to encode."""
        ...

    # ── Inference ────────────────────────────────────────────────────────────

    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
    ) -> UnifiedEmbeddingResult:
        """
        Encode text(s) into embeddings.
        Accepts single str or list of str. Returns normalized result.
        """
        ...

    # ── Introspection ──────────────────────────────────────────────────────────

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return static/dynamic info about the loaded model."""
        ...

    # ── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _coerce_texts(texts: Union[str, List[str]]) -> List[str]:
        """Normalize input to a list of strings."""
        if isinstance(texts, str):
            return [texts]
        return list(texts)

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors along the last axis."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, a_min=1e-12, a_max=None)




@dataclass(frozen=True)
class UnifiedVisionMetadata:
    """Normalized metadata available from any vision backend."""
    model_id: str
    inference_time_ms: float
    num_tokens_generated: int
    device: str
    backend: str  # "onnx" | "api"
    image_size: Optional[tuple[int, int]] = None
    prompt_length: int = 0


@dataclass
class UnifiedVisionResult:
    """Normalized result from any vision backend."""
    text: str
    tokens: List[int]
    metadata: UnifiedVisionMetadata


class VisionInferenceInterface(ABC):
    """
    Abstract base for vision inference engines.
    Guarantees consistent lifecycle (load → generate → unload) across ONNX and API.
    """

    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> "VisionInferenceInterface":
        """Load the model/tokenizer/session. Chainable."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release all resources."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the backend is ready to generate."""
        ...

    # ── Inference ────────────────────────────────────────────────────────────

    @abstractmethod
    def generate(
        self,
        image: Union[str, Path, bytes, np.ndarray, Image.Image],
        prompt: str = "Describe this image.",
        max_new_tokens: Optional[int] = None,
    ) -> UnifiedVisionResult:
        """
        Generate text from image + prompt.
        """
        ...

    # ── Introspection ──────────────────────────────────────────────────────────

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return static/dynamic info about the loaded model."""
        ...

    # ── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _load_image(image: Union[str, Path, bytes, np.ndarray, Image.Image]) -> Image.Image:
        """Load image to PIL RGB."""
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            from io import BytesIO
            img = Image.open(BytesIO(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                img = Image.fromarray(image).convert("RGB")
            elif image.ndim == 3 and image.shape[2] == 4:
                img = Image.fromarray(image[:, :, :3]).convert("RGB")
            else:
                darwin.error(f"Unsupported numpy array shape: {image.shape}")
                return None
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            darwin.error(f"Unsupported image type: {type(image)}")
            return None
        return img

    @staticmethod
    def _format_chat(messages: List[Dict[str, str]]) -> str:
        """Format chat messages into prompt string."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"
            else:
                formatted += f"{role}: {content}\n"
        formatted += "Assistant:"
        return formatted
    





class OCRInferenceInterface(ABC):
    """
    OCR inference interface.
    Implementations handle model lifecycle, preprocessing, and text extraction.
    """

    @abstractmethod
    def load(self) -> None:
        """Download models if missing and initialize the engine."""

    @abstractmethod
    async def extract_text(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Extract text from an image.
        Returns empty string if no text found or confidence is too low.
        """