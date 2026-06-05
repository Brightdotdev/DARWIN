"""
src/core/Models/OnnxModelDownloader.py
Production‑ready downloader for ONNX models from Hugging Face Hub.
Integrates with Darwin event system for logging and progress feedback.
"""

import time
import shutil
from pathlib import Path
from typing import List, Optional

from huggingface_hub import snapshot_download
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    EntryNotFoundError,
    HfHubHTTPError,
)

from src.lib.AppPaths import default_models_dir
from src.core.Models.ModelRegistry import get_spec
from src.core.Logger.DarwinLogger import (
    darwin,
    LogType,
)


class OnnxModelDownloadError(RuntimeError):
    """Aggregated error when ONNX download fails across all repo candidates."""

    def __init__(self, key: str, candidates: List[str], errors: List[Exception]):
        self.key = key
        self.candidates = candidates
        self.errors = errors
        details = "\n".join(
            f"  [{cand}] {err}" for cand, err in zip(candidates, errors)
        )
        super().__init__(
            f"Failed to download ONNX model '{key}' (tried {len(candidates)} repos):\n{details}"
        )


class OnnxModelDownloader:
    """
    Downloads and caches ONNX models with retries, Darwin progress feedback,
    and optional post‑download validation.
    """

    def __init__(
        self,
        cache_root: Optional[Path] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        enable_validation: bool = False,
    ):
        self.cache_root = cache_root or default_models_dir()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_validation = enable_validation

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def is_cached(self, key: str, format: str = "onnx") -> bool:
        """Check if model files exist in the nested cache directory."""
        spec = get_spec(key)
        model_dir = self._model_dir(spec.model_id, spec.category, format)
        return model_dir.exists() and any(model_dir.iterdir())

    def download_onnx(
        self,
        key: str,
        repo_id: Optional[str] = None,
        validate_onnx: Optional[bool] = None,
    ) -> Path:
        """
        Download ONNX model files (model.onnx + required JSON/tokenizer files).

        Args:
            key: Registry key for the model.
            repo_id: Override the Hugging Face repo to search.
            validate_onnx: If True, loads the model with ONNX Runtime
                           to verify integrity. Defaults to class setting.

        Returns:
            Path to the local cache directory containing the model.
        """
        spec = get_spec(key)
        target = self._model_dir(spec.model_id, spec.category, "onnx")
        target.mkdir(parents=True, exist_ok=True)

        # Build list of repo candidates
        repo_candidates = self._repo_candidates(spec, repo_id)

        do_validate = validate_onnx if validate_onnx is not None else self.enable_validation

        # ------------------------------------------------------------------ #
        # Download with retries
        # ------------------------------------------------------------------ #
        download_id = f"onnx_dl_{key.replace('/', '_')}"
        darwin.spinner_start(download_id, f"Downloading {key} …")

        try:
            model_path = self._retry_download(
                key, repo_candidates, target, download_id
            )
            darwin.spinner_stop(download_id, success=True)
            darwin.log(f"ONNX model cached at {model_path}", log_type=LogType.SUCCESS)

            # Optional validation
            if do_validate:
                darwin.log( "Validating ONNX model …")
                self._validate_onnx_model(model_path / "model.onnx", key)

            return model_path

        except Exception as exc:
            darwin.spinner_stop(download_id, success=False)
            darwin.log("Download failed: {exc}", log_type=LogType.ERROR)
            raise




    def delete_model(self, key: str, format: str = "onnx") -> bool:
        """Remove a cached model from disk."""
        spec = get_spec(key)
        path = self._model_dir(spec.model_id, spec.category, format)
        if path.exists():
            shutil.rmtree(path)
            return True
        return False

    def get_model_path(self, key: str, format: str = "onnx") -> Path:
        spec = get_spec(key)
        return self._model_dir(spec.model_id, spec.category, format)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _model_dir(self, model_id: str, category: str, format: str) -> Path:
        org, name = model_id.split("/", 1)
        
        # Strip format-related suffixes from the model name for clean directory paths
        base_name = name
        for suffix in ["-onnx", "_onnx", "-quantized", "_quantized"]:
            if base_name.lower().endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break
        
        return self.cache_root / category / org / base_name / format

    def _repo_candidates(self, spec, custom_repo: Optional[str]) -> List[str]:
        """Prioritised list of Hugging Face repos to try."""
        candidates = []
        if custom_repo:
            candidates.append(custom_repo)
        
        # Derive base model name (strip -onnx suffix if present)
        model_id = spec.model_id  # e.g. "intfloat/e5-small-v2-onnx"
        org, name = model_id.split("/", 1)
        
        # Strip common ONNX suffixes to get base model name
        base_name = name
        for suffix in ["-onnx", "_onnx"]:
            if base_name.lower().endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break
        
        # Xenova exports: Xenova/{base_name}
        candidates.append(f"Xenova/{base_name}")
        # Original PyTorch repo (fallback)
        candidates.append(f"{org}/{base_name}")
        
        return candidates

    def _retry_download(
        self,
        key: str,
        repo_candidates: List[str],
        target: Path,
        spinner_id: str,
    ) -> Path:
        """Try each candidate repo, with per‑repo retries."""
        errors = []

        for repo in repo_candidates:
            for attempt in range(1, self.max_retries + 1):
                try:
                    darwin.log(f"Trying {repo} (attempt {attempt}/{self.max_retries})")
                    # Update spinner message
                    darwin.spinner_start(spinner_id, f"Downloading from {repo} …")

                    # Fetch only ONNX files and essential JSON files
                    snapshot_download(
                        repo_id=repo,
                        local_dir=target,
                        local_dir_use_symlinks=False,
                        allow_patterns=[
                            "*.onnx",
                            "tokenizer.json",
                            "config.json",
                            "tokenizer_config.json",
                            "special_tokens_map.json",
                            "vocab.txt",
                            "preprocessor_config.json",
                            "*.txt",
                        ],
                        ignore_patterns=["*.safetensors", "*.bin", "*.pt"],  # safety
                    )

                    # Ensure model.onnx exists (snapshot_download may have put it in a subfolder)
                    onnx_files = list(target.rglob("*.onnx"))
                    if not onnx_files:
                        raise FileNotFoundError(
                            f"No .onnx file found after snapshot from {repo}"
                        )

                    # Move model.onnx to the root of target if it's nested
                    onnx_file = onnx_files[0]
                    if onnx_file.parent != target:
                        shutil.move(str(onnx_file), target / "model.onnx")
                        # Remove leftover empty dir
                        if onnx_file.parent != target:
                            shutil.rmtree(onnx_file.parent, ignore_errors=True)

                    return target

                except (RepositoryNotFoundError, EntryNotFoundError) as e:
                    darwin.log( f"Repo {repo} not found/missing: {e}",LogType.WARNING,
                    )
                    errors.append(e)
                    break  # no point retrying this repo

                except (HfHubHTTPError, ConnectionError, TimeoutError, OSError) as e:
                    darwin.log(f"Transient error from {repo}: {e}. "
                        f"Retrying in {self.retry_delay}s...",LogType.WARNING)
                    errors.append(e)
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    continue

                except Exception as e:
                    darwin.log( f"Unexpected error from {repo}: {e}",LogType.ERROR,)
                    errors.append(e)
                    break

        # All candidates exhausted
        raise OnnxModelDownloadError(key, repo_candidates, errors)

    def _validate_onnx_model(self, onnx_path: Path, key: str) -> None:
        """Try to create an ONNX Runtime session to verify the model."""
        try:
            import onnxruntime as ort
        except ImportError:
            darwin.log("onnxruntime not installed, skipping validation.",
                log_type=LogType.WARNING,)
            return

        try:
            session = ort.InferenceSession(str(onnx_path))
            darwin.log(f"ONNX model loaded successfully. "
                        f"Inputs: {[i.name for i in session.get_inputs()]}", log_type=LogType.SUCCESS)
        except Exception as e:
            darwin.log(f"ONNX validation failed: {e}", log_type=LogType.ERROR,)
            raise RuntimeError(f"Downloaded ONNX model is invalid: {e}") from e