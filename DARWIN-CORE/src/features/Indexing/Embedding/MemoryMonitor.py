# ═══════════════════════════════════════════════════════════════════════════════
# src/features/Indexing/Embedding/MemoryMonitor.py
# ═══════════════════════════════════════════════════════════════════════════════
"""
Hardware-aware memory monitoring for adaptive batch sizing.
Detects CPU RAM, NVIDIA GPU (via nvidia-smi), and ONNX execution providers.
No PyTorch required – lightweight and dependency‑minimal.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class DeviceProfile:
    """Immutable snapshot of available compute resources."""
    platform: str
    cpu_count: int
    ram_total_gb: float
    ram_free_gb: float
    gpu_vendor: Optional[str]          # "nvidia" or None
    vram_total_gb: float
    vram_free_gb: float
    onnx_providers: List[str]
    is_cuda: bool                      # always False without torch
    is_mps: bool                       # always False without torch

    @property
    def has_gpu(self) -> bool:
        return self.vram_free_gb > 0.5

    @property
    def primary_device(self) -> str:
        if self.is_cuda:
            return "cuda"
        if self.is_mps:
            return "mps"
        return "cpu"


class MemoryMonitor:
    """
    Singleton-style hardware profiler.
    Caches profile after first call; call refresh() to re-detect.
    """

    _instance: Optional["MemoryMonitor"] = None
    _profile: Optional[DeviceProfile] = None

    def __new__(cls) -> "MemoryMonitor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.refresh()

    # ──────────────────────────────────────────────────────────────────────
    # Detection helpers (no PyTorch)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_gpu() -> Tuple[Optional[str], float, float]:
        """
        Returns (vendor, vram_total_gb, vram_free_gb).
        Currently only supports NVIDIA via nvidia-smi.
        """
        # nvidia-smi fallback (works without PyTorch)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total,memory.free",
                 "--format=csv,nounits,noheader"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            # Output: "total_mb, free_mb" (e.g., "8192, 4096")
            parts = out.strip().split(",")
            if len(parts) == 2:
                total_mb, free_mb = map(float, parts)
                return "nvidia", total_mb / 1024, free_mb / 1024
        except Exception:
            pass

        # No GPU detected
        return None, 0.0, 0.0

    @staticmethod
    def _detect_ram() -> Tuple[float, float]:
        """Returns (total_gb, free_gb) using psutil."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            return vm.total / (1024 ** 3), vm.available / (1024 ** 3)
        except Exception:
            # Safe fallback values (16 GB total, 8 GB free)
            return 16.0, 8.0

    @staticmethod
    def _detect_onnx_providers() -> List[str]:
        try:
            import onnxruntime as ort
            return ort.get_available_providers()
        except Exception:
            return ["CPUExecutionProvider"]

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def refresh(self) -> DeviceProfile:
        """Re-detect hardware and cache profile."""
        ram_total, ram_free = self._detect_ram()
        gpu_vendor, vram_total, vram_free = self._detect_gpu()
        providers = self._detect_onnx_providers()

        self._profile = DeviceProfile(
            platform=platform.system().lower(),
            cpu_count=os.cpu_count() or 4,
            ram_total_gb=ram_total,
            ram_free_gb=ram_free,
            gpu_vendor=gpu_vendor,
            vram_total_gb=vram_total,
            vram_free_gb=vram_free,
            onnx_providers=providers,
            is_cuda=False,          # no PyTorch → no CUDA
            is_mps=False,           # no PyTorch → no MPS
        )
        return self._profile

    @property
    def profile(self) -> DeviceProfile:
        if self._profile is None:
            self.refresh()
        return self._profile  # type: ignore[return-value]

    def suggest_batch_ceiling(self, requested: int) -> int:
        """
        Apply a *hard* ceiling only if memory is critically low.
        Otherwise trust the caller.
        """
        prof = self.profile
        if prof.has_gpu and prof.vram_free_gb < 1.0:
            return min(requested, 8)
        if not prof.has_gpu and prof.ram_free_gb < 2.0:
            return min(requested, 16)
        return requested

    def can_fit_model(self, *, ram_gb: float = 0.0, vram_gb: float = 0.0) -> bool:
        prof = self.profile
        if vram_gb > 0 and prof.has_gpu:
            return prof.vram_free_gb >= vram_gb
        return prof.ram_free_gb >= ram_gb

    def log_summary(self) -> None:
        """Pretty-print hardware summary via darwin logger."""
        from src.core.Logger.DarwinLogger import darwin, LogType
        p = self.profile
        darwin.log(f"Platform: {p.platform}", LogType.INFO)
        darwin.log(f"CPUs: {p.cpu_count}, RAM: {p.ram_total_gb:.1f}GB total, {p.ram_free_gb:.1f}GB free", LogType.INFO)
        if p.gpu_vendor:
            darwin.log(f"GPU: {p.gpu_vendor}, VRAM: {p.vram_free_gb:.1f}GB free", LogType.SUCCESS)
        else:
            darwin.log("GPU: not detected", LogType.INFO)
        darwin.log(f"ONNX providers: {p.onnx_providers}", LogType.INFO)