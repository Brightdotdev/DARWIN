
# ═══════════════════════════════════════════════════════════════════════════════
# src/features/Indexing/Embedding/BatchingStrategy.py
# ═══════════════════════════════════════════════════════════════════════════════
"""
Unified adaptive batch sizing with hardware-aware floors, OOM recovery,
and per-model history.

This is the ONLY place batch-size decisions are made.
"""

from __future__ import annotations

from typing import Dict, Optional

from .MemoryMonitor import MemoryMonitor


class BatchingStrategy:
    """
    Decides optimal batch size for embedding streams.

    Rules:
    - Starts at base_size (trusts caller; ignores warm history for initial)
    - Only shrinks on *actual* OOM / MemoryError
    - Floor: CPU=16, GPU=8  (never preemptively shrink to 1)
    - Recovers by +25 % after each success so we trend back to base
    - Remembers last successful size per model_key across calls
    """

    DEFAULT_MIN_CPU: int = 16
    DEFAULT_MIN_GPU: int = 8

    def __init__(
        self,
        memory_monitor: Optional[MemoryMonitor] = None,
        min_batch_cpu: int = DEFAULT_MIN_CPU,
        min_batch_gpu: int = DEFAULT_MIN_GPU,
        history: Optional[Dict[str, int]] = None,
    ):
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.min_batch_cpu = min_batch_cpu
        self.min_batch_gpu = min_batch_gpu
        self._history: Dict[str, int] = history or {}

    def _floor(self, device: str = "cpu") -> int:
        if device in ("cuda", "gpu", "mps"):
            return self.min_batch_gpu
        return self.min_batch_cpu

    def initial_size(self, base_size: int, model_key: str, device: str = "cpu") -> int:
        """
        Pick the first batch size to try.
        1. Respect the caller's base_size.
        2. Apply memory-monitor ceiling (only if critically high).
        3. Do NOT apply floor here — if the user wants base=4, let them.
        """
        ceiling = self.memory_monitor.suggest_batch_ceiling(base_size)
        return min(ceiling, base_size)

    def on_success(self, size: int, model_key: str) -> int:
        """Record success and return the same size (recovery happens elsewhere)."""
        self._history[model_key] = size
        return size

    def recover(self, current_size: int, base_size: int) -> int:
        """After a success, trend back toward base_size if we were reduced."""
        if current_size < base_size:
            return min(base_size, int(current_size * 1.25))
        return current_size

    def on_failure(
        self,
        current_size: int,
        model_key: str,
        device: str = "cpu",
        remaining_items: int = 0,
    ) -> int:
        """
        Return the next batch size to try after an OOM.
        - Above floor: reduce by 25 % but don't drop below floor.
        - At/below floor: halve (can go below floor because we *actually* OOM'd).
        - Never below 1.
        """
        floor = self._floor(device)

        if remaining_items > 0 and remaining_items < current_size:
            # Last partial batch — just use exactly what's left
            return max(1, remaining_items)

        if current_size > floor:
            # We haven't hit the floor yet; be conservative
            return max(floor, int(current_size * 0.75))

        # We OOM'd at or below the floor — something is genuinely tight.
        return max(1, current_size // 2)

    def get_history(self) -> Dict[str, int]:
        return dict(self._history)

