import time
from typing import List, Optional, Union
import numpy as np
from core.Logger.DarwinLogger import darwin
from core.Models.Infrence.Api.InfrenceApi import embed_sync
from core.Models.EmbeddingConfig import get_preset
from core.Models.ModeInterfaces import (
    EmbeddingInferenceInterface,
    UnifiedEmbeddingMetadata,
    UnifiedEmbeddingResult,
)


class APIEmbeddingInference(EmbeddingInferenceInterface):
    """API-based embedding using LiteLLM (sync wrapper)."""

    def __init__(self, model_id: str, normalize: bool = True):
        super().__init__(model_id=model_id, normalize=normalize)
        self._ready = True
        preset = get_preset(model_id)
        self.api_batch_limit = preset.api_batch_limit if preset else None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> "APIEmbeddingInference":
        """No-op for API: always ready. Chainable."""
        self._ready = True
        darwin.info(f"{self.model_id} ready (API)")
        return self

    def unload(self) -> None:
        """No-op for API: nothing to release."""
        self._ready = False
        darwin.info(f"{self.model_id} unloaded (API)")

    def is_loaded(self) -> bool:
        return self._ready

    # ── Inference ────────────────────────────────────────────────────────────

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
    ) -> UnifiedEmbeddingResult:
        """
        Encode text(s) via API. Returns unified result.
        """
        if not self.is_loaded():
            darwin.error(f"{self.model_id} not ready")
            darwin.error(f"APIEmbeddingInference for {self.model_id} not ready")
            return self._empty_result(0)

        texts_list = self._coerce_texts(texts)
        if not texts_list:
            return self._empty_result(0)

        start = time.perf_counter()

        try:
            embeddings_list = embed_sync(
                texts_list,
                model=self.model_id,
                batch_size=batch_size,
            )
        except Exception as e:
            darwin.error(f"API embedding failed for {self.model_id}: {e}")
            raise

        embeddings = np.array(embeddings_list, dtype=np.float32)

        if self.normalize:
            embeddings = self._normalize_vectors(embeddings)

        inference_time_ms = (time.perf_counter() - start) * 1000

        metadata = UnifiedEmbeddingMetadata(
            vector_size=embeddings.shape[-1],
            inference_time_ms=round(inference_time_ms, 3),
            num_inputs=len(texts_list),
            device="API",
            backend="api",
            model_id=self.model_id,
            normalize=self.normalize,
        )

        return UnifiedEmbeddingResult(embeddings=embeddings, metadata=metadata)

    # ── Introspection ────────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        return {
            "status": "ready" if self._ready else "unloaded",
            "model_id": self.model_id,
            "backend": "api",
            "normalize": self.normalize,
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _empty_result(self, num_inputs: int) -> UnifiedEmbeddingResult:
        return UnifiedEmbeddingResult(
            embeddings=np.zeros((num_inputs, 0), dtype=np.float32),
            metadata=UnifiedEmbeddingMetadata(
                vector_size=0,
                inference_time_ms=0.0,
                num_inputs=num_inputs,
                device="API",
                backend="api",
                model_id=self.model_id,
                normalize=self.normalize,
            ),
        )