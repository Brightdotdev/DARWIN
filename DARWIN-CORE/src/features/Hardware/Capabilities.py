"""
Single source of truth for what this sidecar can do.
Reported to Tauri on startup via stdout handshake.
"""

import json
import sys
import subprocess
from dataclasses import dataclass, asdict
from typing import List
from src.core.Logger.DarwinLogger import darwin
from src.lib.AppPaths import default_plugins_dir, default_models_dir
@dataclass
class Capabilities:
    torch_available: bool
    cuda_available: bool
    onnx_available: bool
    gpu_detected: bool
    gpu_name: str
    ram_gb: int
    plugins_installed: List[str]
    models_cached: List[str]

    @classmethod
    def detect(cls) -> "Capabilities":
        torch_avail, cuda_avail = _check_torch()
        onnx_avail = _check_onnx()
        gpu, gpu_name = _check_gpu_without_torch()
        ram = _get_ram()
        plugins = _list_plugins()
        models = _list_cached_models()

        return cls(
            torch_available=torch_avail,
            cuda_available=cuda_avail,
            onnx_available=onnx_avail,
            gpu_detected=gpu,
            gpu_name=gpu_name,
            ram_gb=ram,
            plugins_installed=plugins,
            models_cached=models,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def emit(self) -> None:
        """Print to stdout for Tauri to capture. Format: DARWIN_CAPABILITIES:{...}"""
        darwin.log(f"DARWIN_CAPABILITIES")
        darwin.logjson(self.to_json())


def _check_torch() -> tuple[bool, bool]:
    try:
        import torch
        return True, torch.cuda.is_available()
    except ImportError:
        return False, False


def _check_onnx() -> bool:
    try:
        import onnxruntime as ort
        return len(ort.get_available_providers()) > 0
    except ImportError:
        return False


def _check_gpu_without_torch() -> tuple[bool, str]:
    # Try GPUtil
    try:
        import gputil
        gpus = gputil.getGPUs()
        if gpus:
            return True, gpus[0].name
    except Exception:
        pass

    # Windows WMI fallback
    if sys.platform == "win32":
        try:
            import wmi
            c = wmi.WMI()
            for gpu in c.Win32_VideoController():
                if gpu.AdapterRAM and int(gpu.AdapterRAM) > 512 * 1024 * 1024:
                    return True, gpu.Name
        except Exception:
            pass

    # macOS Apple Silicon
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=5
            )
            if "Apple" in result.stdout:
                return True, "Apple Silicon GPU"
        except Exception:
            pass

    return False, ""


def _get_ram() -> int:
    import psutil
    return int(psutil.virtual_memory().total / (1024 ** 3))


def _list_plugins() -> List[str]:
    plugin_dir = default_plugins_dir()
    if not plugin_dir.exists():
        return []
    return [d.name for d in plugin_dir.iterdir() if d.is_dir()]


def _list_cached_models() -> List[str]:
    models_dir = default_models_dir()
    if not models_dir.exists():
        return []

    cached = []
    for org_dir in models_dir.iterdir():
        if not org_dir.is_dir():
            continue
        for model_dir in org_dir.iterdir():
            if not model_dir.is_dir():
                continue
            onnx_path = model_dir / "onnx"
            pytorch_path = model_dir / "pytorch"
            if onnx_path.exists() and any(onnx_path.glob("*.onnx")):
                cached.append(f"{org_dir.name}/{model_dir.name}/onnx")
            if pytorch_path.exists() and (any(pytorch_path.glob("*.safetensors")) or any(pytorch_path.glob("*.bin"))):
                cached.append(f"{org_dir.name}/{model_dir.name}/pytorch")
    return cached