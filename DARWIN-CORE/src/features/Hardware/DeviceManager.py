# src/core/Hardware/DeviceManager.py

from dataclasses import dataclass

from src.features.Hardware.SystemDetector import SystemInfo


@dataclass(frozen=True)
class DeviceConfig:
    device: str          # "cuda" | "mps" | "cpu"
    dtype: str           # "float16" | "float32"
    use_gpu: bool
    gpu_name: str = ""


class DeviceManager:
    """
    SINGLE SOURCE OF TRUTH for device selection.
    Uses SystemInfo from SystemDetector — no scattered torch.cuda.is_available().
    """

    _config: DeviceConfig | None = None

    @classmethod
    def init(cls, system: SystemInfo) -> DeviceConfig:
        """
        Initialize device config from a SystemInfo object.
        Type-checked and idempotent.
        """
        if not isinstance(system, SystemInfo):
            raise TypeError(f"Expected SystemInfo, got {type(system).__name__}")

        if cls._config is not None:
            raise RuntimeError(
                "DeviceManager already initialized. Call reinit() to change configuration."
            )

        cfg = DeviceConfig(
            device=system.device,
            dtype=system.dtype,
            use_gpu=system.has_gpu,
            gpu_name=system.gpu_name,
        )

        cls._config = cfg
        return cfg

    @classmethod
    def reinit(cls, system: SystemInfo) -> DeviceConfig:
        """Force re-initialization (e.g., after plugin install or GPU hot-swap)."""
        if not isinstance(system, SystemInfo):
            raise TypeError(f"Expected SystemInfo, got {type(system).__name__}")

        cfg = DeviceConfig(
            device=system.device,
            dtype=system.dtype,
            use_gpu=system.has_gpu,
            gpu_name=system.gpu_name,
        )
        cls._config = cfg
        return cfg

    @classmethod
    def get(cls) -> DeviceConfig:
        if cls._config is None:
            raise RuntimeError("DeviceManager not initialized. Call init(system) at startup.")
        return cls._config

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._config is not None