import psutil
from typing import Optional


class MemoryMonitor:
    """
    Advisory memory-pressure detection.  Provides data; does NOT dictate
    batch sizes.  The BatchingStrategy decides what to do with the signal.
    """

    CRITICAL: float = 0.95
    HIGH: float = 0.92
    WARNING: float = 0.80

    def get_pressure(self) -> float:
        """Return 0.0-1.0 memory pressure ratio."""
        return psutil.virtual_memory().percent / 100.0

    def is_critical(self) -> bool:
        return self.get_pressure() > self.CRITICAL

    def is_high(self) -> bool:
        return self.get_pressure() > self.HIGH

    def is_warning(self) -> bool:
        return self.get_pressure() > self.WARNING

    def suggest_batch_ceiling(self, base_size: int) -> int:
        """
        Only cap batch size at *extreme* pressure.  Even then we are
        conservative – a high-memory system sitting at 93 % is usually
        fine for a 32-item embedding batch.
        """
        pressure = self.get_pressure()
        if pressure > self.CRITICAL:
            return max(16, base_size // 2)
        if pressure > self.HIGH:
            return max(24, int(base_size * 0.75))
        return base_size

    def suggest_eviction(self) -> bool:
        """Tell the model cache it should drop zero-ref entries."""
        return self.is_critical()

    def __repr__(self) -> str:
        return f"MemoryMonitor(pressure={self.get_pressure():.2%})"