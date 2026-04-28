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
    disk = psutil.disk_usage(str(Path.home()))
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


def recommend_defaults(system: SystemInfo) -> dict[str, str]:
    """
    Recommend best models based on system capabilities.
    Returns a dict of category -> canonical model_id (registry key).
    
    Uses full canonical model_id strings that match REGISTRY keys
    (e.g., 'BAAI/bge-m3' not 'bge-m3').
    """
    recommendations = {}
    
    # Reranker — always the same tiny cross-encoder
    recommendations["reranker"] = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # Embedding based on RAM
    if system.ram_gb >= 32 and system.has_gpu:
        recommendations["embedding"] = "GritLM/GritLM-7B"
    elif system.ram_gb >= 16:
        recommendations["embedding"] = "BAAI/bge-m3"
    else:
        recommendations["embedding"] = "intfloat/e5-small-v2"
    
    # Vision based on RAM
    if system.ram_gb >= 16:
        recommendations["vision"] = "vikhyatk/moondream2"
    elif system.ram_gb >= 4:
        recommendations["vision"] = "HuggingFaceTB/SmolVLM-256M-Instruct"
    else:
        recommendations["vision"] = None  # Skip vision for very low RAM
    
    # Tokenizer (spaCy) — medium is the best default balance
    if system.ram_gb >= 16 and system.disk_free_gb > 2:
        recommendations["tokenizer"] = "en_core_web_lg"
    else:
        recommendations["tokenizer"] = "en_core_web_md"
    
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