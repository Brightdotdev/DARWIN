"""
src/core/Plugins/TorchPlugin.py
--------------------------------
Specific logic for the torch plugin (darwin-torch-cpu.zip / darwin-torch-cuda.zip).
"""

import platform
from pathlib import Path
from typing import Optional

from .PluginManager import PluginManager, PluginRequiredError


class TorchPlugin:
    def __init__(self, manager: Optional[PluginManager] = None):
        self.manager = manager or PluginManager()
    
    def _get_download_url(self, version: str = "2.1.0") -> str:
        """Construct GitHub Releases URL for torch plugin."""
        base = "https://github.com/Brightdotdev/darwin/releases/download"
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Determine CUDA vs CPU
        if system == "windows":
            # Check for CUDA via nvidia-smi or registry
            has_cuda = self._check_cuda_windows()
            suffix = "cuda" if has_cuda else "cpu"
        elif system == "linux":
            # Check /usr/local/cuda
            has_cuda = Path("/usr/local/cuda").exists()
            suffix = "cuda" if has_cuda else "cpu"
        else:
            suffix = "cpu"
        
        zip_name = f"darwin-torch-{suffix}-{system}-{machine}-v{version}.zip"
        return f"{base}/torch-plugin-v{version}/{zip_name}"
    
    def _check_cuda_windows(self) -> bool:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def ensure_torch(self, auto_install: bool = False) -> bool:
        """
        Check if torch is available. If not, optionally install plugin.
        Returns True if torch is available after attempt.
        """
        if self.manager.verify_torch():
            return True
        
        if auto_install:
            url = self._get_download_url()
            self.manager.install_from_zip("torch", url)
            return self.manager.verify_torch()
        
        return False
    
    def require_torch(self, feature: str = "") -> None:
        """Raise if torch not available."""
        if not self.ensure_torch():
            raise PluginRequiredError("torch", feature)