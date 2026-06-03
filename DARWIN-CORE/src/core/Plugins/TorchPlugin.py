# src/core/Plugins/TorchPlugin.py
import platform
from pathlib import Path
from typing import Optional
from .PluginManager import PluginManager, PluginRequiredError

class TorchPlugin:
    def __init__(self, manager: Optional[PluginManager] = None):
        self.manager = manager or PluginManager()
        self._available: Optional[bool] = None

    def _get_download_url(self, version: str = "2.1.0") -> str:
        system = platform.system().lower()
        arch = platform.machine().lower()
        has_cuda = self._detect_cuda()
        suffix = "cuda" if has_cuda else "cpu"
        arch_map = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
        arch_str = arch_map.get(arch, arch)
        zip_name = f"darwin-torch-{suffix}-{system}-{arch_str}-v{version}.zip"
        return f"https://github.com/Brightdotdev/darwin/releases/download/torch-plugin-v{version}/{zip_name}"

    def _detect_cuda(self) -> bool:
        if platform.system() == "Linux":
            return Path("/usr/local/cuda").exists()
        elif platform.system() == "Windows":
            import subprocess
            try:
                subprocess.run(["nvidia-smi"], capture_output=True, timeout=5, check=True)
                return True
            except Exception:
                return False
        return False

    def is_installed(self) -> bool:
        """Check if the torch plugin is installed and importable."""
        if self._available is not None:
            return self._available
        self._available = self.manager.is_installed("torch") and self._try_import()
        return self._available

    def _try_import(self) -> bool:
        if not self.manager.add_to_path("torch"):
            return False
        try:
            import torch  # noqa
            return True
        except ImportError:
            return False

    def install_if_needed(self) -> None:
        """Download and install the torch plugin. Raises on failure."""
        if self.is_installed():
            return
        url = self._get_download_url()
        self.manager.install_from_zip("torch", url, progress_id="install_torch")
        # clear cache after install
        self._available = None
        if not self.is_installed():
            raise PluginRequiredError("torch", "Installation failed")

    def require_torch(self, feature: str = "") -> None:
        """Raise PluginRequiredError if torch is not installed. Never auto-installs."""
        if not self.is_installed():
            raise PluginRequiredError("torch", feature)