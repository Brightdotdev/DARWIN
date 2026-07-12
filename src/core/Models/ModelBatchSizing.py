# src/core/Models/ModelBatchSizing.py
"""
Hardware-aware batch sizing for local models.
Uses: psutil, onnxruntime providers, nvidia-ml (optional), platform tools.
"""

import os
import platform
import subprocess
import sys
from functools import lru_cache
from typing import Optional, Dict, Any, Tuple

from src.core.Models.EmbeddingConfig import ModelPreset, get_preset
from src.core.Logger.DarwinLogger import darwin, LogType


# ═══════════════════════════════════════════════════════════════════════════
# Cached device info (computed once, reused)
# ═══════════════════════════════════════════════════════════════════════════

_DEVICE_INFO: Optional[Dict[str, Any]] = None


def _get_device_info() -> Dict[str, Any]:
    global _DEVICE_INFO
    if _DEVICE_INFO is not None:
        return _DEVICE_INFO

    info = {
        "platform": platform.system().lower(),
        "cpu_count": 0,
        "ram_total_gb": 0.0,
        "ram_available_gb": 0.0,
        "gpu_available": False,
        "gpu_vendors": [],
        "gpu_vram_gb": 0.0,
        "ort_providers": [],
    }

    # ── CPU / RAM via psutil ─────────────────────────────────────────────
    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count(logical=True) or 0
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = mem.total / (1024 ** 3)
        info["ram_available_gb"] = mem.available / (1024 ** 3)
    except ImportError:
        # psutil not available — try /proc/meminfo on Linux
        if info["platform"] == "linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            info["ram_total_gb"] = kb / (1024 * 1024)
                        elif line.startswith("MemAvailable:"):
                            kb = int(line.split()[1])
                            info["ram_available_gb"] = kb / (1024 * 1024)
            except Exception:
                pass

    # ── ONNX Runtime providers ────────────────────────────────────────────
    try:
        import onnxruntime as ort
        info["ort_providers"] = ort.get_available_providers()
    except ImportError:
        pass

    _detect_nvidia(info)
    _detect_amd(info)
    _detect_apple(info)
    _detect_intel(info)

    _DEVICE_INFO = info
    return info


def _detect_nvidia(info: Dict[str, Any]) -> None:
    """Detect NVIDIA GPU via nvidia-ml (lightweight) or nvidia-smi."""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        total_vram = 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_vram += mem_info.free / (1024 ** 3)
        if device_count > 0:
            info["gpu_available"] = True
            info["gpu_vendors"].append("nvidia")
            info["gpu_vram_gb"] = total_vram
        pynvml.nvmlShutdown()
        return
    except Exception:
        pass

    # Fallback: nvidia-smi subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            if lines:
                total_free = sum(float(l) for l in lines) / 1024  # MB → GB
                info["gpu_available"] = True
                info["gpu_vendors"].append("nvidia")
                info["gpu_vram_gb"] = total_free
    except Exception:
        pass


def _detect_amd(info: Dict[str, Any]) -> None:
    """Detect AMD GPU via rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "VRAM"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and "VRAM" in result.stdout:
            info["gpu_available"] = True
            info["gpu_vendors"].append("amd")
    except Exception:
        pass

    # Linux sysfs fallback
    if info["platform"] == "linux":
        try:
            kfd_path = "/sys/class/kfd/kfd/topology/nodes/"
            if os.path.exists(kfd_path):
                info["gpu_available"] = True
                info["gpu_vendors"].append("amd")
        except Exception:
            pass


def _detect_apple(info: Dict[str, Any]) -> None:
    """Detect Apple Silicon GPU (shared memory with CPU)."""
    if info["platform"] != "darwin":
        return

    # Check for Apple Silicon
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip() == "1":
            info["gpu_available"] = True
            info["gpu_vendors"].append("apple")
            # On Apple Silicon, GPU uses unified memory with CPU
            # Available GPU memory ≈ available RAM
            info["gpu_vram_gb"] = info["ram_available_gb"]
    except Exception:
        pass

    # Also check hw.memsize for total
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            bytes_total = int(result.stdout.strip())
            info["ram_total_gb"] = bytes_total / (1024 ** 3)
    except Exception:
        pass


def _detect_intel(info: Dict[str, Any]) -> None:
    """Detect Intel GPU (OpenVINO, Arc, etc.)."""
    # Check for Intel GPU in sysfs
    if info["platform"] == "linux":
        try:
            for i in range(8):
                vendor_path = f"/sys/class/drm/card{i}/device/vendor"
                if os.path.exists(vendor_path):
                    with open(vendor_path) as f:
                        vendor = f.read().strip()
                        if vendor == "0x8086":  # Intel PCI vendor ID
                            info["gpu_available"] = True
                            info["gpu_vendors"].append("intel")
                            break
        except Exception:
            pass

    # Windows: check for Intel GPU via wmic
    if info["platform"] == "windows":
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "Name"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and "Intel" in result.stdout:
                info["gpu_available"] = True
                info["gpu_vendors"].append("intel")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Batch sizing logic
# ═══════════════════════════════════════════════════════════════════════════

def get_optimal_batch_size(requested_size: int = 32) -> int:
    """
    Return a hardware-aware batch size.
    """
    info = _get_device_info()

    # GPU path
    if info["gpu_available"]:
        vram = info.get("gpu_vram_gb", 0)
        if vram > 8:
            return min(requested_size, 128)
        elif vram > 4:
            return min(requested_size, 64)
        else:
            return min(requested_size, 32)

    # CPU path
    ram_avail = info.get("ram_available_gb", 0)
    if ram_avail < 2.0:
        return min(requested_size, 8)
    elif ram_avail > 8.0:
        return min(requested_size, 64)
    else:
        return min(requested_size, 16)


def get_model_batch_size(
    preset: Optional[ModelPreset] = None,
    model_key: Optional[str] = None,
    requested_size: int = 32,
) -> int:
    """
    Batch size adjusted for model-specific constraints + hardware.
    For API models, respects preset.api_batch_limit.
    """
    if preset is None and model_key is not None:
        preset = get_preset(model_key)

    if preset is None:
        return get_optimal_batch_size(requested_size)

    # API models: use their batch limit
    if preset.api_batch_limit is not None:
        return min(requested_size, preset.api_batch_limit)

    # Vision models: check preset for batch override
    if preset.category == "vision":
        # Most vision models process 1 image at a time
        # But some can batch — allow override via negative overlap sentinel
        # or we could add a batch_size field to ModelPreset
        return 1

    base_size = get_optimal_batch_size(requested_size)
    model_name = preset.model_name.lower()

    # Model-specific tuning
    if "e5-small" in model_name:
        return min(base_size * 2, 256)
    elif "minilm" in model_name:
        return min(base_size * 4, 512)
    elif "e5-large" in model_name or "e5" in model_name:
        return min(base_size, 64)

    return base_size


# ═══════════════════════════════════════════════════════════════════════════
# Model memory estimation (using ModelRegistry data)
# ═══════════════════════════════════════════════════════════════════════════

def can_fit_model(ram_gb: float, disk_mb: float, safety_factor: float = 1.5) -> bool:
    """
    Check if system can fit a model in memory.
    Uses ModelRegistry ram_gb/disk_mb if available.
    """
    info = _get_device_info()
    ram_avail = info.get("ram_available_gb", 0)
    # Rough heuristic: model needs ~1.5x its disk size in RAM when loaded
    estimated_ram_needed = (disk_mb / 1024) * safety_factor
    return ram_avail >= estimated_ram_needed


def estimate_model_batch_from_registry(model_id: str) -> int:
    """
    Look up model in ModelRegistry and return conservative batch size
    based on its declared RAM requirements.
    """
    try:
        from src.core.Models.Lib.ModelRegistry import find_spec
        spec = find_spec(model_id)
        if spec is None:
            return get_optimal_batch_size(32)

        info = _get_device_info()
        ram_avail = info.get("ram_available_gb", 0)
        model_ram = spec.ram_gb

        if model_ram <= 0:
            return get_optimal_batch_size(32)

        # Conservative: how many copies fit in available RAM?
        # Leave 30% headroom for OS + other processes
        usable_ram = ram_avail * 0.7
        max_batches = int(usable_ram / model_ram)
        return max(1, min(max_batches, 64))
    except ImportError:
        return get_optimal_batch_size(32)


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostics / introspection
# ═══════════════════════════════════════════════════════════════════════════

def get_hardware_summary() -> Dict[str, Any]:
    """Return full hardware detection results for logging/debugging."""
    return _get_device_info().copy()


def log_hardware_summary() -> None:
    """Log hardware detection results via Darwin."""
    info = _get_device_info()
    darwin.info(f"Platform: {info['platform']}")
    darwin.info(f"CPUs: {info['cpu_count']}, RAM: {info['ram_total_gb']:.1f}GB total, {info['ram_available_gb']:.1f}GB free")
    if info['gpu_available']:
        darwin.success(f"GPU: {', '.join(info['gpu_vendors'])}, VRAM: {info['gpu_vram_gb']:.1f}GB free")
    else:
        darwin.warning("GPU: not detected")
    darwin.info(f"ONNX providers: {info['ort_providers']}")


# ═══════════════════════════════════════════════════════════════════════════
# Force refresh (for testing / after hardware changes)
# ═══════════════════════════════════════════════════════════════════════════

def refresh_device_info() -> Dict[str, Any]:
    """Clear cache and re-detect hardware."""
    global _DEVICE_INFO
    _DEVICE_INFO = None
    return _get_device_info()