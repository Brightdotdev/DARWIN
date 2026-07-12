from __future__ import annotations

import time
from typing import Any, List, Optional

import numpy as np

from src.core.Logger.DarwinLogger import darwin
from src.core.Pipeline.Indexing.Embedding.MemoryMonitor import MemoryMonitor
from src.core.Pipeline.Indexing.Embedding.BatchingStrategy import BatchingStrategy


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
        # FIXED: use shared singleton strategy so history persists (Audit #3)
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.strategy = strategy or BatchingStrategy(self.memory_monitor)

    @staticmethod
    def _detect_device(model: Any) -> str:
        """Best-effort device detection from a loaded model."""
        for attr_path in ("device", "model.device"):
            obj = model
            for part in attr_path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            else:
                d = str(obj).lower()
                if "cuda" in d:
                    return "cuda"
                if "mps" in d:
                    return "mps"
        return "cpu"

    @staticmethod
    def _to_numpy(vecs: Any, expected_dim: Optional[int] = None) -> np.ndarray:
        """Convert encoder output to float32 numpy with shape validation."""
        arr = vecs.astype(np.float32, copy=False) if hasattr(vecs, "astype") else np.array(vecs, dtype=np.float32)

        if arr.ndim != 2:
            darwin.error(
                f"Expected 2D embeddings (batch, dim), got {arr.ndim}D shape {arr.shape}. "
                f"The model may be outputting raw hidden states instead of pooled embeddings."
            )
            return None

        if expected_dim and arr.shape[1] != expected_dim:
            darwin.error(
                f"Embedding dimension mismatch: expected {expected_dim}, got {arr.shape[1]}. "
                f"Shape: {arr.shape}"
            )
            return None

        return arr

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        """Heuristic: is this exception a memory error?"""
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "oom" in msg
            or isinstance(exc, MemoryError)
        )

    def encode_stream(
        self,
        model: Any,
        texts: List[str],
        *,
        base_batch_size: int = 32,
        file_name: str = "",
        expected_dim: Optional[int] = None,
        use_length_batching: bool = True,
        token_budget: int = 2048,
        **encode_kwargs,
    ) -> np.ndarray:
        """
        Encode a list of texts in adaptive batches with OOM recovery.

        Args:
            model: Loaded encoder model (must expose `.encode()`).
            texts: List of strings to embed.
            base_batch_size: Starting batch size (item count, not token count).
            file_name: Identifier for progress tracking.
            expected_dim: Expected embedding dimension for validation.
            use_length_batching: If True, group by token budget instead of item count.
            token_budget: Max estimated tokens per batch (only if length_batching=True).
            **encode_kwargs: Passed through to model.encode().

        Returns:
            np.ndarray of shape (N, D) dtype=float32.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        total = len(texts)
        # FIXED: removed UUID overhead (Audit #10)
        progress_id = f"embed_{file_name}"
        darwin.progress_start(progress_id, total, f"Embedding {file_name}")

        device = self._detect_device(model)
        model_key = getattr(model, "model_id", file_name) or file_name

        # ── Warmup profiling (Audit #6) ──────────────────────────────────
        if model_key not in self.strategy._profiled and total >= 10:
            sample = texts[:min(50, total)]
            base_size = self.strategy.initial_size(base_batch_size, model_key, device)
            profiled_size = self.strategy.profile_batch_size(
                model, sample, base_size, model_key, device
            )
            darwin.info(
                f"Profiled {model_key}: optimal batch size ≈ {profiled_size}"
            )

        # ── Determine batching mode ────────────────────────────────────
        if use_length_batching:
            # FIXED: group by token budget, not item count (Audit #2)
            index_batches = self.strategy.make_length_batches(
                texts, token_budget, preset_max_tokens=token_budget
            )
            darwin.info(
                f"Length-aware batching: {len(texts)} texts → {len(index_batches)} batches "
                f"(budget={token_budget} tokens)"
            )
        else:
            # Legacy item-count mode
            current_size = self.strategy.initial_size(base_batch_size, model_key, device)
            index_batches = []
            idx = 0
            while idx < total:
                size = min(current_size, total - idx)
                index_batches.append(list(range(idx, idx + size)))
                idx += size

        all_vectors: List[np.ndarray] = []
        processed = 0
        batch_num = 0
        current_size = base_batch_size

        for batch_indices in index_batches:
            batch_num += 1
            batch_texts = [texts[i] for i in batch_indices]
            batch_size = len(batch_texts)

            # Sort within batch by length to minimize padding
            sorted_pairs = sorted(enumerate(batch_texts), key=lambda x: len(x[1]))
            sorted_texts = [t for _, t in sorted_pairs]
            sort_map = [orig for orig, _ in sorted_pairs]
            unsort_map = [0] * len(sort_map)
            for new_pos, orig_pos in enumerate(sort_map):
                unsort_map[orig_pos] = new_pos

            try:
                start_t = time.perf_counter()
                kwargs = dict(encode_kwargs)
                kwargs.setdefault("batch_size", len(sorted_texts))

                vecs = model.encode(sorted_texts, **kwargs)
                np_vecs = self._to_numpy(vecs.embeddings, expected_dim)
                elapsed = time.perf_counter() - start_t

                # Restore original order
                np_vecs = np_vecs[unsort_map]
                all_vectors.append(np_vecs)

                processed += batch_size
                darwin.progress_update(
                    progress_id,
                    processed,
                    f"Batch {batch_num}/{len(index_batches)} (size={batch_size}) • {processed}/{total} chunks",
                )

                # FIXED: record throughput for adaptive sizing (Audit #7)
                throughput = batch_size / elapsed if elapsed > 0 else 0
                current_size = self.strategy.on_success(batch_size, model_key, elapsed)
                current_size = self.strategy.recover(current_size, base_batch_size)

            except Exception as exc:
                if not self._is_oom_error(exc):
                    darwin.progress_end(progress_id)
                    raise

                darwin.warning(f"OOM at batch_size={batch_size}, reducing and retrying")

                if batch_size <= 1:
                    darwin.error("Cannot reduce batch below 1, skipping remaining chunks")
                    break

                current_size = self.strategy.on_failure(batch_size, model_key, device, len(batch_indices))
                if current_size >= batch_size:
                    current_size = max(1, batch_size // 2)

                # Re-split remaining work with new size
                if use_length_batching:
                    remaining_texts = [texts[i] for i in batch_indices]
                    new_batches = self.strategy.make_length_batches(
                        remaining_texts, token_budget // 2, token_budget // 2
                    )
                    for nb in new_batches:
                        index_batches.append([batch_indices[i] for i in nb])
                else:
                    for i in range(0, len(batch_indices), current_size):
                        index_batches.append(batch_indices[i:i + current_size])
                continue

        darwin.progress_end(progress_id)

        if not all_vectors:
            return np.array([], dtype=np.float32)

        return np.vstack(all_vectors)