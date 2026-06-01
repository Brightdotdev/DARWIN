"""
src/core/Plugins/PluginManager.py
-----------------------------------
Manages optional plugin installation (torch, etc.).
Plugins live in ~/.darwin/plugins/{name}/
"""

import sys
import zipfile
from pathlib import Path
from typing import Optional, Dict
from urllib.request import urlopen
from src.lib.AppPaths import default_plugins_dir



class PluginRequiredError(ImportError):
    """Raised when a feature needs a plugin that isn't installed."""
    def __init__(self, plugin_name: str, feature: str = ""):
        msg = f"Plugin '{plugin_name}' is required"
        if feature:
            msg += f" for {feature}"
        msg += f". Install it from Settings."
        super().__init__(msg)
        self.plugin_name = plugin_name


class PluginManager:
    PLUGIN_ROOT = default_plugins_dir()

    def __init__(self):
        self.PLUGIN_ROOT.mkdir(parents=True, exist_ok=True)

    def is_installed(self, name: str) -> bool:
        plugin_dir = self.PLUGIN_ROOT / name
        if not plugin_dir.exists():
            return False
        marker = plugin_dir / ".installed"
        return marker.exists()

    def get_path(self, name: str) -> Optional[Path]:
        if not self.is_installed(name):
            return None
        return self.PLUGIN_ROOT / name

    def add_to_path(self, name: str) -> bool:
        """Add plugin's python/ directory to sys.path so imports work."""
        plugin_dir = self.get_path(name)
        if not plugin_dir:
            return False
        python_dir = plugin_dir / "python"
        if python_dir.exists() and str(python_dir) not in sys.path:
            sys.path.insert(0, str(python_dir))
        site_packages = plugin_dir / "python" / "site-packages"
        if site_packages.exists() and str(site_packages) not in sys.path:
            sys.path.insert(0, str(site_packages))
        return True

    def install_from_zip(
        self,
        name: str,
        zip_url: str,
        extract_root: Optional[Path] = None,
    ) -> bool:
        """Download and extract a plugin zip."""
        target = extract_root or (self.PLUGIN_ROOT / name)
        target.mkdir(parents=True, exist_ok=True)

        zip_path = target / "download.zip"

        try:
            with urlopen(zip_url) as response:
                with open(zip_path, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target)

            zip_path.unlink()

            (target / ".installed").write_text(f"url: {zip_url}\n")
            return True

        except Exception as e:
            if zip_path.exists():
                zip_path.unlink()
            raise RuntimeError(f"Failed to install plugin {name}: {e}")

    def uninstall(self, name: str) -> bool:
        import shutil
        target = self.PLUGIN_ROOT / name
        if target.exists():
            shutil.rmtree(target)
            return True
        return False

    def list_installed(self) -> Dict[str, Path]:
        return {
            d.name: d for d in self.PLUGIN_ROOT.iterdir()
            if d.is_dir() and (d / ".installed").exists()
        }

    def verify_torch(self) -> bool:
        """Try importing torch after adding plugin to path."""
        self.add_to_path("torch")
        try:
            import torch
            return True
        except ImportError:
            return False