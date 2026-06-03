# src/core/Models/ModelStore.py
"""
ModelStore – Orchestrator for model downloads, verification, and path management.
All heavy logic is delegated to specialised, testable components.
Progress is reported through the Darwin event system (via the downloaders).
"""

from pathlib import Path
import shutil
from typing import Optional, Set

from src.core.Models.ModelRegistry import get_spec, REGISTRY
from src.core.Models.ModelUtils import DownloadError, IntegrityError
from src.lib.AppPaths import default_models_dir
from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Models.ModelDownloading.OnnxModelDownloader import OnnxModelDownloader
from src.core.Models.ModelDownloading.PyTorchModelDownloader import PyTorchModelDownloader
from src.core.Models.ModelDownloading.SpacyModelDownloader import SpacyModelDownloader
from src.core.Models.ModelStoreHelpers import ModelPathResolver, ModelVerifier


class ModelStore:
    """Orchestrates path resolution, verification, and downloads – no direct I/O."""

    def __init__(self) -> None:
        root = default_models_dir()
        self.resolver = ModelPathResolver(root)
        self.verifier = ModelVerifier(self.resolver)
        self.pytorch_downloader = PyTorchModelDownloader()
        self.spacy_downloader = SpacyModelDownloader()
        self._active_downloads: Set[str] = set()

    # ── Public API ────────────────────────────────────────────────────
    def model_path(self, key: str, format: Optional[str] = None) -> Path:
        """Return the canonical local path for a model (API models return __api__)."""
        spec = get_spec(key)
        return self.resolver.model_path(spec, format)

    def is_downloaded(self, key: str, format: Optional[str] = None) -> bool:
        """Check if a model is fully downloaded and valid."""
        spec = get_spec(key)
        if format is None:
            format = self.resolver.infer_format(spec)
        return self.verifier.is_downloaded(spec, key, format)

    def ensure(self, key: str, force: bool = False) -> Path:
        """
        Guarantee a model is downloaded and verified.
        Returns the local path (or __api__ for API models).
        """
        spec = get_spec(key)
        if spec.loader.startswith("api_"):
            return Path("__api__")

        target_format = self.resolver.infer_format(spec)
        if not self.is_downloaded(key, target_format) or force:
            self.download(key, force=force)

        # Prefer the new nested path, fallback to legacy
        new_path = self.resolver.model_path(spec, target_format)
        if self.verifier._verify_at_path(new_path, spec):
            return new_path
        legacy = self.resolver.legacy_model_path(spec)
        if self.verifier._verify_at_path(legacy, spec):
            return legacy
        raise IntegrityError(f"Model '{key}' verification failed after download.")

    def download(self, key: str, force: bool = False, minimal: bool = True) -> Path:
        """
        Download a model, respecting the format determined by its loader.
        Skips API models and already downloaded models unless forced.
        """
        spec = get_spec(key)
        if spec.disk_mb == 0 or spec.loader.startswith("api_"):
            darwin.log(f"{key} is an API model — nothing to download.", LogType.INFO)
            return Path("__api__")

        target_format = self.resolver.infer_format(spec)
        if self.is_downloaded(key, target_format) and not force:
            return self.resolver.model_path(spec, target_format)

        if key in self._active_downloads:
            raise DownloadError(f"Download already in progress for {key}")

        self._active_downloads.add(key)
        try:
            if target_format == "onnx":
                return self._download_onnx(key)
            elif target_format == "pytorch":
                target = self.resolver.model_path(spec, "pytorch")
                return self.pytorch_downloader.download(spec, target, minimal)
            elif spec.loader == "spacy":
                target = self.resolver.model_path(spec, "spacy")
                return self.spacy_downloader.download(spec.model_id, target)
            else:
                # Unknown local loaders default to PyTorch layout
                target = self.resolver.model_path(spec, "pytorch")
                return self.pytorch_downloader.download(spec, target, minimal)
        finally:
            self._active_downloads.discard(key)
            # Clean up incomplete marker (onnx handled internally)
            if target_format != "onnx":
                target = self.resolver.model_path(spec, target_format)
                (target / ".incomplete").unlink(missing_ok=True)

    def delete(self, key: str, format: Optional[str] = None) -> None:
        """Remove a model. If format is None, all formats and legacy paths are removed."""
        spec = get_spec(key)
        if format:
            path = self.resolver.model_path(spec, format)
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        else:
            for p in (self.resolver.model_path(spec), self.resolver.legacy_model_path(spec)):
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)

    def list_installed(self) -> list[str]:
        """List all registry keys of fully downloaded and valid models."""
        return [k for k in REGISTRY if self.is_downloaded(k)]

    def installed_size(self, key: str, format: Optional[str] = None) -> str:
        """Human‑readable size of an installed model."""
        spec = get_spec(key)
        if spec.disk_mb == 0:
            return "API (0MB)"

        if format is None:
            format = self.resolver.infer_format(spec)

        path = self.resolver.model_path(spec, format)
        if not path.exists():
            path = self.resolver.legacy_model_path(spec)
        if not path.exists():
            return "0 MB"

        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        for unit in ["B", "KB", "MB", "GB"]:
            if total < 1024:
                return f"{total:.1f} {unit}"
            total /= 1024
        return f"{total:.1f} TB"

    def get_size_info(self) -> dict:
        """Detailed size and file count for all installed models."""
        info = {}
        for key in self.list_installed():
            spec = get_spec(key)
            fmt = self.resolver.infer_format(spec)
            path = self.resolver.model_path(spec, fmt)
            if not path.exists():
                path = self.resolver.legacy_model_path(spec)
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            file_count = sum(1 for f in path.rglob("*") if f.is_file())
            info[key] = {
                "size_bytes": total,
                "size_human": self.installed_size(key, fmt),
                "files": file_count,
                "path": str(path),
            }
        return info

    def cleanup_failed_downloads(self) -> int:
        """Remove corrupted/incomplete downloads. Returns bytes freed."""
        freed = 0
        for key in REGISTRY:
            spec = get_spec(key)
            fmt = self.resolver.infer_format(spec)
            for path in (self.resolver.model_path(spec, fmt), self.resolver.legacy_model_path(spec)):
                if path.exists() and not self.verifier._verify_at_path(path, spec):
                    freed += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    shutil.rmtree(path, ignore_errors=True)
        return freed

    # ── ONNX delegation (kept separate for clarity) ──────────────────
    def _download_onnx(self, key: str) -> Path:
        """Delegate ONNX download to the dedicated OnnxModelDownloader."""
        downloader = OnnxModelDownloader(self.resolver.root)
        return downloader.download_onnx(key)

    def get_format(self, key: str) -> str:
        """Return the storage format for a model (onnx, pytorch, spacy)."""
        spec = get_spec(key)
        return self.resolver.infer_format(spec)


# ── Singleton ────────────────────────────────────────────────────────
_STORE_INSTANCE: Optional[ModelStore] = None

def get_model_store() -> ModelStore:
    """Return the global ModelStore singleton."""
    global _STORE_INSTANCE
    if _STORE_INSTANCE is None:
        _STORE_INSTANCE = ModelStore()
    return _STORE_INSTANCE