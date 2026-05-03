# src/core/Hardware/DeviceManager.py

from dataclasses import dataclass


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
    def init(cls, system) -> DeviceConfig:
        """
        Initialize device config from a SystemInfo object.
        The system parameter now actually gets used.
        """
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