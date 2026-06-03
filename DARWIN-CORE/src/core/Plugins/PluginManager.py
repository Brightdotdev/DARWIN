# src/core/Plugins/PluginManager.py

import sys
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Dict
from urllib.request import urlopen

from src.lib.AppPaths import default_plugins_dir
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Logger.DarwinLogger import SpinnerStartEvent, SpinnerEndEvent


class PluginRequiredError(ImportError):
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
        return (self.PLUGIN_ROOT / name / ".installed").exists()

    def get_path(self, name: str) -> Optional[Path]:
        if self.is_installed(name):
            return self.PLUGIN_ROOT / name
        return None

    def add_to_path(self, name: str) -> bool:
        """Add plugin's site-packages to sys.path (with conflict check)."""
        plugin_dir = self.get_path(name)
        if not plugin_dir:
            return False

        # Ensure critical modules are not already imported
        if name == "torch":
            if "torch" in sys.modules:
                darwin.log("torch is already loaded; cannot activate plugin.", LogType.WARNING)
                return False

        site_packages = plugin_dir / "python" / "site-packages"
        if site_packages.exists() and str(site_packages) not in sys.path:
            sys.path.insert(0, str(site_packages))

        # For DLL loading on Windows, add the plugin directory to PATH
        import os
        if sys.platform == "win32":
            os.environ["PATH"] = str(plugin_dir) + os.pathsep + os.environ.get("PATH", "")
        return True

    def install_from_zip(
        self,
        name: str,
        zip_url: str,
        extract_root: Optional[Path] = None,
        progress_id: str = None,
    ) -> bool:
        """Download and extract a plugin zip with Darwin progress feedback."""
        target = extract_root or (self.PLUGIN_ROOT / name)
        target.mkdir(parents=True, exist_ok=True)

        zip_path = target / "download.zip"
        if progress_id is None:
            progress_id = f"plugin_{name}"

        darwin.spinner_start(progress_id,f"Downloading {name} plugin...")

        try:
            # Download with retries
            self._download_file(zip_url, zip_path, progress_id)

            darwin.spinner_start(f"{progress_id}_extract","Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target)
            darwin.spinner_stop(f"{progress_id}_extract", success=True)

            zip_path.unlink()
            (target / ".installed").write_text(f"url: {zip_url}\n")
            darwin.spinner_stop(progress_id, success=True)
            return True

        except Exception as e:
            darwin.spinner_stop(progress_id, success=False)
            if zip_path.exists():
                zip_path.unlink()
            raise RuntimeError(f"Failed to install plugin {name}: {e}")

    def _download_file(self, url: str, dest: Path, progress_id: str):
        """Robust download with retries and progress."""
        import time
        from urllib.error import URLError

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with urlopen(url, timeout=60) as response:
                    total_size = int(response.headers.get("Content-Length", 0))
                    downloaded = 0

                    if total_size > 0:
                        darwin.progress_start(progress_id,total_size,f"Downloading {dest.name}")

                    with open(dest, "wb") as f:
                        while True:
                            chunk = response.read(65536)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                darwin.progress_update(progress_id,downloaded,f"{downloaded/1024/1024:.1f} MB")
                    if total_size > 0:
                        darwin.progress_end(progress_id)
                    return
            except (URLError, ConnectionError, TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                darwin.log(f"Download attempt {attempt+1} failed: {e}. Retrying in {wait}s...", LogType.WARNING)
                time.sleep(wait)

    def uninstall(self, name: str) -> bool:
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
        if not self.add_to_path("torch"):
            return False
        try:
            import torch
            return True
        except ImportError:
            return False