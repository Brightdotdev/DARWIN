from pathlib import Path
import shutil
from src.core.Models.ModelUtils import DownloadError, IntegrityError
from src.core.Models.lib import ALLOW_PATTERNS, IGNORE_PATTERNS
from src.core.Logger.DarwinLogger import darwin

class PyTorchModelDownloader:
    """Downloads PyTorch models from Hugging Face Hub with Darwin feedback."""

    def download(self, spec, target_path: Path, minimal: bool, force: bool = False) -> Path:
        # ── FIX C1: Do not unconditionally rmtree ──────────────────────────
        if target_path.exists():
            if not force:
                # Verify existing cache before deciding to wipe
                if self._quick_verify(target_path, spec):
                    darwin.log(f"{spec.model_id} already cached and valid — skipping download")
                    return target_path
            # Only wipe if force=True or verification failed
            shutil.rmtree(target_path, ignore_errors=True)

        target_path.mkdir(parents=True, exist_ok=True)
        (target_path / ".incomplete").touch()

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise DownloadError(
                "huggingface_hub package required",
                "Install with: pip install huggingface-hub",
                recoverable=False,
            )

        progress_id = f"dl_{spec.model_id.replace('/', '_')}"
        darwin.spinner_start(progress_id, f"Downloading {spec.model_id}")

        try:
            kwargs = {
                "repo_id": spec.model_id,
                "local_dir": str(target_path),
                "local_files_only": False,
            }
            if minimal and spec.category != "vision":
                kwargs.update({
                    "allow_patterns": ALLOW_PATTERNS,
                    "ignore_patterns": IGNORE_PATTERNS,
                })

            snapshot_download(**kwargs)

            # Quick sanity check
            if not any(target_path.iterdir()):
                raise IntegrityError("Download completed but directory is empty")

            darwin.spinner_stop(progress_id, success=True)
            darwin.log(f"Downloaded {spec.model_id}")
            return target_path

        except Exception as e:
            darwin.spinner_stop(progress_id, success=False)
            raise DownloadError(f"HuggingFace download failed: {e}")

    def _quick_verify(self, path: Path, spec) -> bool:
        """Lightweight check: does the directory look like a valid model?"""
        if not path.exists() or (path / ".incomplete").exists():
            return False
        # Check for at least one critical file
        has_config = (path / "config.json").exists()
        has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))
        return has_config and has_weights