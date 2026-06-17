# src/core/Models/ModelLoading/API/APIVisionInference.py
"""
API-based vision inference using LiteLLM.
Implements the unified VisionInferenceInterface.
"""

import time
from pathlib import Path
from typing import  Optional, Union
import numpy as np
from PIL import Image

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Models.Infrence.Api.InfrenceApi import vision_sync

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
            raise ValueError(f"Model {self.model_id} not found in registry")
        self._ready = True
        darwin.log(f"APIVisionInference ready: {self.model_id}", LogType.INFO)
        return self

    def unload(self) -> None:
        self._ready = False
        darwin.log(f"APIVisionInference unloaded: {self.model_id}", LogType.INFO)

    def is_loaded(self) -> bool:
        return self._ready

    # ── Inference ────────────────────────────────────────────────────────────

    def generate(
        self,
        image: Union[str, Path, bytes, np.ndarray, Image.Image],
        prompt: str = "Describe this image.",
        max_new_tokens: Optional[int] = None,
    ) -> UnifiedVisionResult:
        """
        Generate text from image + prompt via API.
        """
        if not self.is_loaded():
            self.load()

        start = time.perf_counter()

        try:
            text = vision_sync(
                image=image,
                prompt=prompt,
                model=self.model_id,
                max_image_size=self.max_image_size,
                jpeg_quality=self.jpeg_quality,
            )
        except Exception as e:
            darwin.log(f"Vision inference failed for {self.model_id}: {e}", LogType.ERROR)
            raise

        inference_time_ms = (time.perf_counter() - start) * 1000

        # API doesn't give us token-level info, so we approximate
        # You can swap this for actual token counting if your API provider supports it
        tokens = []  # API doesn't expose tokens via LiteLLM by default

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


    # ── Introspection ────────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        return {
            "status": "ready" if self._ready else "unloaded",
            "model_id": self.model_id,
            "backend": "api",
            "max_image_size": self.max_image_size,
            "jpeg_quality": self.jpeg_quality,
        }

    # ── Async variant (API-specific convenience) ───────────────────────────

    async def agenerate(
        self,
        image: Union[str, Path, bytes, np.ndarray, Image.Image],
        prompt: str = "Describe this image.",
        max_new_tokens: Optional[int] = None,
    ) -> UnifiedVisionResult:
        """
        Async generate. Same contract, but non-blocking.
        """
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
            darwin.log(f"Vision inference failed for {self.model_id}: {e}", LogType.ERROR)
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