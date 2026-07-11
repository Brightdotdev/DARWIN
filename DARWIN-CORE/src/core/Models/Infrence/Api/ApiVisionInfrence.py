import time
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
from PIL import Image

from src.core.Logger.DarwinLogger import darwin
from src.core.Models.ModeInterfaces import (
    VisionInferenceInterface,
    UnifiedVisionMetadata,
    UnifiedVisionResult,
)


class APIVisionInference(VisionInferenceInterface):
    """API vision model using LiteLLM. Implements VisionInferenceInterface."""

    def __init__(
        self,
        model_id: str = "openai/gpt-4o-mini",
        max_image_size: int = 1280,
        jpeg_quality: int = 85,
    ):
        super().__init__(model_id=model_id)
        self.max_image_size = max_image_size
        self.jpeg_quality = jpeg_quality
        self._ready = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> "APIVisionInference":
        """Verify model_id is valid. No heavy loading for API. Chainable."""
        from src.core.Models.Lib.ModelRegistry import find_spec
        spec = find_spec(self.model_id)
        if not spec:
            darwin.error(f"Model {self.model_id} not found in registry")
            return self
        self._ready = True
        darwin.info(f"{self.model_id} ready")
        return self

    def unload(self) -> None:
        self._ready = False
        darwin.info(f"{self.model_id} unloaded")

    def is_loaded(self) -> bool:
        return self._ready

    # ── Inference ────────────────────────────────────────────────────────────

    def generate(
        self,
        image: Union[str, Path, bytes, np.ndarray, Image.Image],
        prompt: str = "Describe this image.",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream_callback: Optional[callable] = None,
    ) -> UnifiedVisionResult:
        """
        Generate text from image + prompt via API.

        Note: temperature, top_p, top_k, stream_callback are ignored for API.
        The API provider handles its own sampling.
        """
        if not self.is_loaded():
            self.load()

        start = time.perf_counter()

        try:
            from src.core.Models.Infrence.Api.InfrenceApi import vision_sync
            text = vision_sync(
                image=image,
                prompt=prompt,
                model=self.model_id,
                max_image_size=self.max_image_size,
                jpeg_quality=self.jpeg_quality,
            )
        except Exception as e:
            darwin.error(f"Vision inference failed for {self.model_id}: {e}")
            raise

        inference_time_ms = (time.perf_counter() - start) * 1000

        # API doesn't give token-level info by default
        tokens: List[int] = []

        metadata = UnifiedVisionMetadata(
            model_id=self.model_id,
            inference_time_ms=round(inference_time_ms, 3),
            num_tokens_generated=0,  # API opaque
            device="API",
            backend="api",
            image_size=None,
            prompt_length=len(prompt),
        )

        return UnifiedVisionResult(
            text=text,
            tokens=tokens,
            metadata=metadata,
        )

    # ── Async variant ────────────────────────────────────────────────────────

    async def agenerate(
        self,
        image: Union[str, Path, bytes, np.ndarray, Image.Image],
        prompt: str = "Describe this image.",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream_callback: Optional[callable] = None,
    ) -> UnifiedVisionResult:
        """Async generate. Same contract, but non-blocking."""
        if not self.is_loaded():
            self.load()

        from src.core.Models.Infrence.Api.InfrenceApi import vision

        start = time.perf_counter()

        try:
            text = await vision(
                image=image,
                prompt=prompt,
                model=self.model_id,
                max_image_size=self.max_image_size,
                jpeg_quality=self.jpeg_quality,
            )
        except Exception as e:
            darwin.error(f"Vision inference failed for {self.model_id}: {e}")
            raise

        inference_time_ms = (time.perf_counter() - start) * 1000

        metadata = UnifiedVisionMetadata(
            model_id=self.model_id,
            inference_time_ms=round(inference_time_ms, 3),
            num_tokens_generated=0,
            device="API",
            backend="api",
            image_size=None,
            prompt_length=len(prompt),
        )

        return UnifiedVisionResult(
            text=text,
            tokens=[],
            metadata=metadata,
        )

    # ── Introspection ────────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        return {
            "status": "ready" if self._ready else "unloaded",
            "model_id": self.model_id,
            "backend": "api",
            "max_image_size": self.max_image_size,
            "jpeg_quality": self.jpeg_quality,
        }

    # ── Context Manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "APIVisionInference":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()