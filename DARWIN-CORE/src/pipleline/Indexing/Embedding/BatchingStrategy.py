
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any

from src.pipleline.Indexing.Embedding.MemoryMonitor import MemoryMonitor


class _ThroughputSample:
    """Single throughput measurement."""
    __slots__ = ("size", "throughput", "memory_mb")

    def __init__(self, size: int, throughput: float, memory_mb: float):
        self.size = size
        self.throughput = throughput
        self.memory_mb = memory_mb


class BatchingStrategy:
    """
    Decides optimal batch size for embedding streams.

    Rules:
    - Uses history (last successful size) to inform initial size
    - Profiles throughput on first run per model to find optimal
    - Only shrinks on *actual* OOM / MemoryError
    - Floor: CPU=16, GPU=8 (never preemptively shrink to 1)
    - Recovers by +25 % after each success so we trend back to base
    - Remembers last successful size per model_key across calls
    - Token-budget mode for length-aware batching
    """

    DEFAULT_MIN_CPU: int = 16
    DEFAULT_MIN_GPU: int = 8

    # Singleton instance for cross-call history persistence
    _instance: Optional["BatchingStrategy"] = None

    def __new__(cls, *args, **kwargs) -> "BatchingStrategy":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        memory_monitor: Optional[MemoryMonitor] = None,
        min_batch_cpu: int = DEFAULT_MIN_CPU,
        min_batch_gpu: int = DEFAULT_MIN_GPU,
        history: Optional[Dict[str, int]] = None,
        profile_samples: int = 3,
    ):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.min_batch_cpu = min_batch_cpu
        self.min_batch_gpu = min_batch_gpu
        self._history: Dict[str, int] = history or {}
        self._profile_samples = profile_samples
        self._profiles: Dict[str, List[_ThroughputSample]] = {}
        self._profiled: set[str] = set()

    def _floor(self, device: str = "cpu") -> int:
        if device in ("cuda", "gpu", "mps"):
            return self.min_batch_gpu
        return self.min_batch_cpu

    # ──────────────────────────────────────────────────────────────────────
    # NEW: Token-budget batching for length-aware grouping (Audit #2)
    # ──────────────────────────────────────────────────────────────────────

    def make_length_batches(
        self,
        texts: List[str],
        token_budget: int,
        preset_max_tokens: int,
    ) -> List[List[int]]:
        """
        Group text indices into batches where total token count <= token_budget.
        Uses a simple character-length heuristic (chars ≈ tokens * 4).

        Returns list of index batches, e.g. [[0,1,2], [3,4], [5,6,7,8]...]
        """
        if not texts:
            return []

        # Estimate tokens from chars (rough: 1 token ≈ 4 chars for English)
        estimates = [max(1, len(t) // 4) for t in texts]
        max_est = max(estimates) if estimates else 1

        # Safety: if a single item exceeds budget, it gets its own batch
        budget = max(token_budget, max_est)

        # Sort by estimated length descending for better packing
        sorted_indices = sorted(range(len(texts)), key=lambda i: estimates[i], reverse=True)

        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_sum = 0

        for idx in sorted_indices:
            est = estimates[idx]
            if current_sum + est <= budget and len(current_batch) < 256:
                current_batch.append(idx)
                current_sum += est
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_sum = est

        if current_batch:
            batches.append(current_batch)

        return batches

    # ──────────────────────────────────────────────────────────────────────
    # NEW: Warmup / profiling (Audit #6)
    # ──────────────────────────────────────────────────────────────────────

    def profile_batch_size(
        self,
        model: Any,
        sample_texts: List[str],
        base_size: int,
        model_key: str,
        device: str,
    ) -> int:
        """
        Test a few batch sizes and pick the one with highest throughput.
        Only runs once per model_key. Returns the winning size.
        """
        if model_key in self._profiled or len(sample_texts) < 10:
            return base_size

        test_sizes = self._generate_test_sizes(base_size)
        samples: List[_ThroughputSample] = []

        for size in test_sizes:
            if size > len(sample_texts):
                continue
            batch = sample_texts[:size]
            start = time.perf_counter()
            try:
                result = model.encode(batch)
                elapsed = time.perf_counter() - start
                throughput = size / elapsed if elapsed > 0 else 0
                samples.append(_ThroughputSample(size, throughput, 0.0))
            except Exception:
                break

        if samples:
            best = max(samples, key=lambda s: s.throughput)
            self._history[model_key] = best.size
            self._profiles[model_key] = samples
            self._profiled.add(model_key)
            return best.size

        return base_size

    def _generate_test_sizes(self, base: int) -> List[int]:
        """Generate sizes to test around the base size."""
        candidates = {base}
        for mult in (0.5, 0.75, 1.0, 1.5, 2.0):
            candidates.add(max(1, int(base * mult)))
        return sorted(candidates)

    # ──────────────────────────────────────────────────────────────────────
    # Core sizing — FIXED to read history (Audit #3)
    # ──────────────────────────────────────────────────────────────────────

    def initial_size(self, base_size: int, model_key: str, device: str = "cpu") -> int:
        """
        Pick the first batch size to try.
        FIXED: now reads history if available (was ignored before).
        """
        if model_key in self._history:
            size = self._history[model_key]
        else:
            size = base_size

        ceiling = self.memory_monitor.suggest_batch_ceiling(size)
        return min(ceiling, size)

    def on_success(self, size: int, model_key: str, elapsed_sec: float = 0.0) -> int:
        """Record success and update history."""
        self._history[model_key] = size
        return size

    def recover(self, current_size: int, base_size: int) -> int:
        """
        After a success, trend back toward base_size if we were reduced.
        FIXED: was a no-op because current_size was never < base_size after on_success.
        """
        if current_size < base_size:
            return min(base_size, int(current_size * 1.25))
        return min(current_size, base_size * 2)

    def on_failure(
        self,
        current_size: int,
        model_key: str,
        device: str = "cpu",
        remaining_items: int = 0,
    ) -> int:
        """
        Return the next batch size to try after an OOM.
        """
        floor = self._floor(device)

        if remaining_items > 0 and remaining_items < current_size:
            return max(1, remaining_items)

        if current_size > floor:
            return max(floor, int(current_size * 0.75))

        return max(1, current_size // 2)

    def get_history(self) -> Dict[str, int]:
        return dict(self._history)

    def get_profile(self, model_key: str) -> Optional[List[Tuple[int, float]]]:
        """Return (size, throughput) tuples for a model's profile."""
        samples = self._profiles.get(model_key)
        if not samples:
            return None
        return [(s.size, s.throughput) for s in samples]