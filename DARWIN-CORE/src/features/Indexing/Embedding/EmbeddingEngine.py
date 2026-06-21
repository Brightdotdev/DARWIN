
# ═══════════════════════════════════════════════════════════════════════════════
# src/features/Indexing/Embedding/EmbeddingEngine.py
# ═══════════════════════════════════════════════════════════════════════════════
"""
Pure embedding engine: stream encoding, tensor conversion, progress tracking,
and OOM recovery.

KNOWS NOTHING about:
- model loading / key resolution
- presets / prefixes
- caching / storage
- chunk filtering

Receives a *loaded* model and a list of strings. Returns an ndarray.
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

import numpy as np

from src.core.Logger.DarwinLogger import darwin, LogType
from .MemoryMonitor import MemoryMonitor
from .BatchingStrategy import BatchingStrategy


class EmbeddingEngine:
    """
    Pure embedding orchestration.

    Lifecycle:
        engine = EmbeddingEngine()
        vectors = engine.encode_stream(model, texts, base_batch_size=32)
    """

    def __init__(
        self,
        memory_monitor: Optional[MemoryMonitor] = None,
        strategy: Optional[BatchingStrategy] = None,
    ):
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.strategy = strategy or BatchingStrategy(self.memory_monitor)

    @staticmethod
    def _detect_device(model: Any) -> str:
        """Best-effort device detection from a loaded model."""
        try:
            import torch
            if hasattr(model, "device"):
                d = str(getattr(model, "device"))
                if "cuda" in d:
                    return "cuda"
                if "mps" in d:
                    return "mps"
            if hasattr(model, "model") and hasattr(model.model, "device"):
                d = str(model.model.device)
                if "cuda" in d:
                    return "cuda"
                if "mps" in d:
                    return "mps"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _to_numpy(vecs: Any) -> np.ndarray:
        """Convert encoder output to float32 numpy, zero-copy where possible."""
        if hasattr(vecs, "astype"):
            return vecs.astype(np.float32, copy=False)

        try:
            import torch
            if isinstance(vecs, torch.Tensor):
                return (
                    vecs.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
        except Exception:
            pass

        return np.array(vecs, dtype=np.float32)

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        """Heuristic: is this exception a memory error?"""
        msg = str(exc).lower()
        if "out of memory" in msg or "oom" in msg:
            return True
        if isinstance(exc, MemoryError):
            return True
        try:
            import torch
            if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(
                exc, torch.cuda.OutOfMemoryError
            ):
                return True
        except Exception:
            pass
        return False

    def encode_stream(
        self,
        model: Any,
        texts: List[str],
        *,
        base_batch_size: int = 32,
        file_name: str = "",
        **encode_kwargs,
    ) -> np.ndarray:
        """
        Encode a list of texts in adaptive batches with OOM recovery.

        Args:
            model: Loaded encoder model (must expose `.encode()`).
            texts: List of strings to embed.
            base_batch_size: Starting batch size.
            file_name: Identifier for progress tracking.
            **encode_kwargs: Passed through to model.encode().

        Returns:
            np.ndarray of shape (N, D) dtype=float32.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        total = len(texts)
        progress_id = f"embed_{file_name}_{uuid.uuid4().hex[:8]}"
        darwin.progress_start(progress_id, total, f"Embedding {file_name}")

        device = self._detect_device(model)
        model_key = getattr(model, "model_id", file_name) or file_name

        current_size = self.strategy.initial_size(base_batch_size, model_key, device)

        all_vectors: List[np.ndarray] = []
        idx = 0
        processed = 0
        batch_num = 0

        while idx < total:
            remaining = total - idx
            batch_size = min(current_size, remaining)
            batch_texts = texts[idx : idx + batch_size]
            batch_num += 1

            try:
                kwargs = dict(encode_kwargs)
                kwargs.setdefault("batch_size", len(batch_texts))
                kwargs.setdefault("show_progress_bar", False)
                kwargs.setdefault("convert_to_numpy", True)

                vecs = model.encode(batch_texts, **kwargs)
                np_vecs = self._to_numpy(vecs)
                all_vectors.append(np_vecs)

                # Only advance on *confirmed* success
                idx += len(batch_texts)
                processed += len(batch_texts)

                darwin.progress_update(
                    progress_id,
                    processed,
                    f"Batch {batch_num} (size={len(batch_texts)}) • "
                    f"{processed}/{total} chunks",
                )

                # Record success and potentially grow back toward base
                current_size = self.strategy.on_success(batch_size, model_key)
                current_size = self.strategy.recover(current_size, base_batch_size)
                current_size = min(current_size, remaining)  # don't overshoot tail

            except Exception as exc:
                if not self._is_oom_error(exc):
                    darwin.progress_end(progress_id)
                    raise

                darwin.log(
                    f"OOM at batch_size={batch_size}, reducing and retrying",
                    LogType.WARNING,
                )

                if batch_size <= 1:
                    darwin.log(
                        "Cannot reduce batch below 1, skipping remaining chunks",
                        LogType.ERROR,
                    )
                    break

                current_size = self.strategy.on_failure(
                    batch_size, model_key, device, remaining
                )
                # Safety: ensure we actually shrink
                if current_size >= batch_size:
                    current_size = max(1, batch_size // 2)
                continue

        darwin.progress_end(progress_id)

        if not all_vectors:
            return np.array([], dtype=np.float32)

        return np.vstack(all_vectors)
