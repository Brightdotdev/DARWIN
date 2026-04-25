# src/core/Models/SystemDetector.py

import os
import platform
import psutil
from dataclasses import dataclass
from pathlib import Path
from src.core.Models.ModelRegistry import REGISTRY

@dataclass
class SystemInfo:
    ram_gb: int
    cpu_cores: int
    has_gpu: bool
    gpu_name: str = ""
    disk_free_gb: float = 0.0

    @property
    def tier(self) -> str:
        """Auto-detect best tier for this hardware."""
        if self.ram_gb >= 32:
            return "overkill"
        elif self.ram_gb >= 16:
            return "large"
        elif self.ram_gb >= 8:
            return "medium"
        else:
            return "light"

    def can_run(self, spec) -> bool:
        """Can this system comfortably run the given ModelSpec?"""
        return self.ram_gb >= spec.ram_gb

    def has_disk_space(self, spec) -> bool:
        """Enough free disk for this model?"""
        return self.disk_free_gb >= (spec.disk_mb / 1024) + 1  # +1GB buffer

    def has_disk_space_for_all(self, specs: list) -> bool:
        """Enough free disk for a list of models?"""
        total_gb = sum(s.disk_mb for s in specs) / 1024
        return self.disk_free_gb >= total_gb + 1



def detect_system() -> SystemInfo:
    """Read actual hardware specs. Works on Linux, macOS, Windows."""
    ram = psutil.virtual_memory()
    ram_gb = int(ram.total / (1024**3))
    disk = psutil.disk_usage(str(Path.home()))  # ← str() here
    disk_free_gb = disk.free / (1024**3)


    cpu_cores = psutil.cpu_count(logical=True)

    has_gpu = False
    gpu_name = ""
    try:
        import torch
        if torch.cuda.is_available():
            has_gpu = True
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return SystemInfo(
        ram_gb=ram_gb,
        cpu_cores=cpu_cores,
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        disk_free_gb=disk_free_gb,
    )


def recommend_defaults(system: SystemInfo) -> dict[str, str | None]:
    """
    Given hardware, pick the best model for each category.

    Returns:
        {
            "embedding":  "bge-m3" | "e5-small-v2" | None,
            "reranker":   "reranker-medium",           # always installed
            "vision":     None,                        # optional — opt-in
            "tokenizer":  None,                        # optional — opt-in
        }
    """
    recommendations: dict[str, str | None] = {
        "embedding":  None,
        "reranker":   "reranker-medium",   # always
        "vision":     "smolvlm",                # optional
        "tokenizer":  None,                # optional
    }

    # ── Embedding: pick best that fits in RAM ──
    for key in ["bge-m3", "e5-small-v2"]:
        spec = REGISTRY.get(key)
        if spec and spec.required and system.can_run(spec):
            recommendations["embedding"] = key
            break

    # Vision and tokenizer are optional — not auto-selected
    # User opts in during onboarding

    return recommendations



def default_models_dir() -> Path:
    """Platform-appropriate global models directory."""
    system = platform.system()

    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "Darwin" / "models"

    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Darwin" / "models"

    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
        return base / "darwin" / "models"