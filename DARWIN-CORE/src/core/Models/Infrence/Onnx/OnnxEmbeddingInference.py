import gc
import time
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Models.ModeInterfaces import (
    EmbeddingInferenceInterface,
    UnifiedEmbeddingResult,
    UnifiedEmbeddingMetadata,
)


class OnnxEmbeddingInference(EmbeddingInferenceInterface):
    """
    ONNX embedding inference with tokenizers library (no transformers).
    Implements EmbeddingInferenceInterface.
    """

    def __init__(
        self,
        model_id: str,
        model_path: Path,
        normalize: bool = True,
        max_length: int = 512,
    ):
        super().__init__(model_id, normalize)
        self.model_path = Path(model_path)
        self.max_length = max_length

        self._session = None
        self._tokenizer = None
        self._vector_size = 0
        self._device = "CPU"
        self._onnx_runtime_version = "unknown"

    def _log(self, msg: str, level: LogType = LogType.INFO):
        darwin.log(f"[ONNX Embedding] {msg}", level)

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> "OnnxEmbeddingInference":
        if self.is_loaded():
            self._log("Already loaded, skipping.")
            return self

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        self._log(f"Loading model from {self.model_path}")
        self._load_tokenizer()
        self._load_onnx_session()
        self._log("Model loaded successfully.", LogType.SUCCESS)
        return self

    def unload(self) -> None:
        if not self.is_loaded():
            return
        self._log("Unloading model...")
        self._session = None
        self._tokenizer = None
        gc.collect()
        self._log("Model unloaded.", LogType.SUCCESS)

    def is_loaded(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    # ── Internal loading helpers ───────────────────────────────────────────

    def _load_tokenizer(self) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError:
            self._log("tokenizers library required. pip install tokenizers", LogType.ERROR)
            raise ImportError("tokenizers library required. pip install tokenizers")

        tokenizer_path = self.model_path / "tokenizer.json"
        if not tokenizer_path.exists():
            self._log(f"Tokenizer not found in {self.model_path}", LogType.ERROR)
            raise FileNotFoundError(f"Tokenizer not found in {self.model_path}")

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._log(f"Tokenizer loaded from {tokenizer_path}")

    def _load_onnx_session(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            self._log("onnxruntime required", LogType.ERROR)
            raise ImportError("onnxruntime required")

        model_file = self.model_path / "model.onnx"
        if not model_file.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_file}")

        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]
        self._device = providers[0].replace("ExecutionProvider", "")
        self._onnx_runtime_version = ort.__version__

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_file),
            sess_options=sess_options,
            providers=providers,
        )

        # Infer vector size from model output
        output = self._session.get_outputs()[0]
        shape = output.shape
        if len(shape) == 2:
            self._vector_size = shape[1]
        else:
            self._log("Could not infer vector size from model, using default 384", LogType.WARNING)
            self._vector_size = 384

        self._log(
            f"ONNX session loaded: {model_file} | "
            f"providers={providers} | vector_size={self._vector_size}"
        )

    # ── Inference (interface implementation) ──────────────────────────────

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
    ) -> UnifiedEmbeddingResult:
        if isinstance(texts, str):
            texts = [texts]

        if not self.is_loaded():
            self._log("Model not loaded. Call .load() first.", LogType.ERROR)
            raise RuntimeError("Model not loaded. Call .load() first.")

        self._log(f"Encoding {len(texts)} text(s)")

        start_time = time.perf_counter()
        all_embeddings = []
        bs = batch_size or 32

        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            embeddings = self._encode_batch(batch)
            all_embeddings.append(embeddings)

        embeddings = np.vstack(all_embeddings)

        if self.normalize:
            embeddings = self._normalize_vectors(embeddings)

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        metadata = UnifiedEmbeddingMetadata(
            vector_size=self._vector_size,
            inference_time_ms=round(inference_time_ms, 3),
            num_inputs=len(texts),
            device=self._device,
            backend="onnx",
            model_id=self.model_id,
            normalize=self.normalize,
        )

        return UnifiedEmbeddingResult(embeddings=embeddings, metadata=metadata)

    def _encode_batch(self, batch: List[str]) -> np.ndarray:
        """Tokenize and run ONNX inference. Auto-detects pooled vs raw output."""
        encodings = self._tokenizer.encode_batch(batch, add_special_tokens=True)

        max_len = max(len(e.ids) for e in encodings)
        max_len = min(max_len, self.max_length)

        pad_token_id = self._tokenizer.token_to_id("") or 0

        input_ids = np.array([
            e.ids[:max_len] + [pad_token_id] * (max_len - len(e.ids[:max_len]))
            for e in encodings
        ], dtype=np.int64)

        attention_mask = np.array([
            [1] * min(len(e.ids), max_len) + [0] * (max_len - min(len(e.ids), max_len))
            for e in encodings
        ], dtype=np.int64)

        token_type_ids = np.array([
            e.type_ids[:max_len] + [0] * (max_len - len(e.type_ids[:max_len]))
            for e in encodings
        ], dtype=np.int64)

        outputs = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })

        embeddings = outputs[0]

        # AUTO-DETECT: pooled (2D) vs raw hidden states (3D)
        if embeddings.ndim == 3:
            # Raw hidden states: (batch, seq_len, hidden_dim)
            # Mean pool using attention mask
            mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
            sum_embeddings = np.sum(embeddings * mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
            embeddings = sum_embeddings / sum_mask
            
            self._log(f"Applied mean pooling: {embeddings.shape}")

        elif embeddings.ndim == 2:
            # Already pooled: (batch, hidden_dim) — use as-is
            pass

        else:
            raise ValueError(
                f"Unexpected embedding dimensions: {embeddings.ndim}D shape {embeddings.shape}. "
                f"Expected 2D (pooled) or 3D (raw hidden states)."
            )

        return embeddings

    # ── Introspection ──────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        if not self.is_loaded():
            return {"status": "not_loaded", "model_dir": str(self.model_path)}

        return {
            "status": "loaded",
            "model_dir": str(self.model_path),
            "vector_size": self._vector_size,
            "device": self._device,
            "onnx_runtime_version": self._onnx_runtime_version,
            "normalize": self.normalize,
            "backend": "onnx",
            "model_id": self.model_id,
        }