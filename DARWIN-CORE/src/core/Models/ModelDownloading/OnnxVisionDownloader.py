# src/core/Models/ModelDownloading/SmolVLMOnnxDownloader.py
"""
SmolVLM ONNX Downloader – handles download, caching, integrity, cleanup.
No inference. Pure file management.
Includes internet connectivity waiting with user feedback.
NO EXCEPTIONS – returns None/False on failure.
"""

from __future__ import annotations

import shutil
import hashlib
import time
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from huggingface_hub import hf_hub_download

from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.InternetChecker import wait_for_internet, check_internet_robust


@dataclass
class SmolVLMConfig:
    """Configuration for SmolVLM ONNX – shared between downloader and inference."""
    model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    model_dir: Path = field(default_factory=lambda: Path("models/smolvlm"))
    revision: str = "9181b0e"

    decoder_precision: str = "int8"
    vision_precision: str = "fp32"
    embed_precision: str = "int8"
    precision: Optional[str] = None

    auto_download: bool = True
    expected_hashes: Optional[Dict[str, str]] = None

    def __post_init__(self):
        valid = {"fp32", "fp16", "int8", "uint8", "bnb4", "q4", "q4f16"}
        if self.precision is not None:
            if self.precision not in valid:
                darwin.log(f"Invalid precision {self.precision}, using default.", LogType.ERROR)
                self.decoder_precision = "int8"
                self.vision_precision = "fp32"
                self.embed_precision = "int8"
            else:
                self.decoder_precision = self.precision
                self.vision_precision = self.precision
                self.embed_precision = self.precision
        else:
            for name, val in [("decoder_precision", self.decoder_precision),
                              ("vision_precision", self.vision_precision),
                              ("embed_precision", self.embed_precision)]:
                if val not in valid:
                    darwin.log(f"Invalid {name}={val}, resetting to default.", LogType.ERROR)
                    if name == "decoder_precision":
                        self.decoder_precision = "int8"
                    elif name == "vision_precision":
                        self.vision_precision = "fp32"
                    else:
                        self.embed_precision = "int8"

        self.model_dir = Path(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _suffix(self, precision: str) -> str:
        return "" if precision == "fp32" else f"_{precision}"

    @property
    def decoder_path(self) -> Path:
        return self.model_dir / f"decoder_model_merged{self._suffix(self.decoder_precision)}.onnx"

    @property
    def vision_path(self) -> Path:
        return self.model_dir / f"vision_encoder{self._suffix(self.vision_precision)}.onnx"

    @property
    def embed_path(self) -> Path:
        return self.model_dir / f"embed_tokens{self._suffix(self.embed_precision)}.onnx"

    @property
    def tokenizer_path(self) -> Path:
        return self.model_dir / "tokenizer.json"

    @property
    def config_path(self) -> Path:
        return self.model_dir / "config.json"

    @property
    def preprocessor_config_path(self) -> Path:
        return self.model_dir / "preprocessor_config.json"

    @property
    def tokenizer_config_path(self) -> Path:
        return self.model_dir / "tokenizer_config.json"

    @property
    def all_onnx_paths(self) -> List[Path]:
        return [self.decoder_path, self.vision_path, self.embed_path]

    @property
    def all_config_paths(self) -> List[Path]:
        return [self.tokenizer_path, self.config_path, self.preprocessor_config_path, self.tokenizer_config_path]

    @property
    def all_cached(self) -> bool:
        return all(p.exists() for p in self.all_onnx_paths + self.all_config_paths)


class SmolVLMOnnxDownloader:
    """Handles downloading ONNX models and config files from HuggingFace Hub.
    Never raises exceptions – returns None on failure, True/False on success.
    """

    def __init__(self, config: SmolVLMConfig):
        self.cfg = config

    def _log(self, msg: str, level: LogType = LogType.INFO) -> None:
        darwin.log(f"[SmolVLM Downloader] {msg}", level)

    def download(self, force: bool = False, recache: bool = False) -> Optional[Dict[str, Path]]:
        """
        Download all required files.
        Returns dict mapping logical name -> local Path on success, None on failure.
        """
        should_force = force or recache

        if should_force:
            self._log("Force/recache mode: clearing existing ONNX files...")
            self._clear_onnx_cache()

        results = {}
        onnx_files = [
            ("decoder", self.cfg.decoder_path),
            ("vision", self.cfg.vision_path),
            ("embed", self.cfg.embed_path),
        ]

        spinner_id = f"dl_smolvlm_{self.cfg.model_id.split('/')[-1]}"
        darwin.spinner_start(spinner_id, f"Downloading SmolVLM ONNX models")

        for name, target_path in onnx_files:
            if target_path.exists() and not should_force:
                self._log(f"Cached {name}: {target_path.name}")
                results[name] = target_path
                continue

            self._log(f"Downloading {name}: {target_path.name}")
            try:
                cached_path = self._download_with_retry(
                    repo_filename=f"onnx/{target_path.name}",
                    dest_path=target_path,
                    min_size=10_000_000,  # 10 MB minimum for ONNX
                )
                if cached_path is None:
                    darwin.spinner_stop(spinner_id, success=False)
                    return None
                results[name] = target_path

            except Exception as e:
                self._log(f"Download failed for {name}: {e}", LogType.ERROR)
                darwin.spinner_stop(spinner_id, success=False)
                return None

        darwin.spinner_stop(spinner_id, success=True)

        # Download config files
        config_files = [
            ("tokenizer.json", self.cfg.tokenizer_path),
            ("config.json", self.cfg.config_path),
            ("preprocessor_config.json", self.cfg.preprocessor_config_path),
            ("tokenizer_config.json", self.cfg.tokenizer_config_path),
        ]

        for filename, target_path in config_files:
            if target_path.exists() and not should_force:
                self._log(f"Cached config: {filename}")
                continue

            self._log(f"Downloading config: {filename}")
            try:
                cached_path = self._download_with_retry(
                    repo_filename=filename,
                    dest_path=target_path,
                    min_size=100,  # 100 bytes minimum for config
                )
                if cached_path is None:
                    return None

            except Exception as e:
                self._log(f"Download failed for {filename}: {e}", LogType.ERROR)
                return None

        return results

    def verify_integrity(self) -> bool:
        """Return True if all expected files exist and hashes match (if provided)."""
        for p in self.cfg.all_onnx_paths + self.cfg.all_config_paths:
            if not p.exists():
                self._log(f"Missing file: {p}", LogType.ERROR)
                return False
            if p.suffix == ".onnx" and p.stat().st_size < 10_000_000:
                self._log(f"ONNX file too small ({p.stat().st_size} bytes): {p.name}", LogType.ERROR)
                return False
            if p.suffix == ".json" and p.stat().st_size < 100:
                self._log(f"Config file too small ({p.stat().st_size} bytes): {p.name}", LogType.ERROR)
                return False
            if self.cfg.expected_hashes and p.name in self.cfg.expected_hashes:
                try:
                    sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
                    if sha256 != self.cfg.expected_hashes[p.name]:
                        self._log(f"Hash mismatch for {p.name}", LogType.ERROR)
                        return False
                except Exception as e:
                    self._log(f"Hash check failed for {p.name}: {e}", LogType.ERROR)
                    return False
        return True

    def ensure(self, force: bool = False, recache: bool = False, wait_internet: bool = True, max_wait: int = 60) -> bool:
        """
        Ensure model files are present; download if missing.
        Returns True if files are ready (cached or newly downloaded), False otherwise.
        """
        should_force = force or recache

        if not self.cfg.auto_download:
            if self.cfg.all_cached and not should_force:
                return True
            self._log("auto_download=False but files missing.", LogType.ERROR)
            return False

        if self.cfg.all_cached and not should_force:
            self._log("All model files already cached.")
            return True

        # Internet wait
        if wait_internet:
            self._log(f"Waiting up to {max_wait} seconds for internet connection...")
            connected = wait_for_internet(
                pause=3,
                max_wait=max_wait,
                verbose=True,
            )
            if not connected:
                self._log(f"No internet after {max_wait}s. Cannot download.", LogType.ERROR)
                return False
        else:
            if not check_internet_robust():
                self._log("No internet and wait_internet=False.", LogType.ERROR)
                return False

        self._log("Starting download...")
        result = self.download(force=should_force)
        if result is None:
            return False

        if not self.verify_integrity():
            self._log("Integrity check failed after download.", LogType.ERROR)
            return False

        self._log("SmolVLM ONNX model download complete.", LogType.SUCCESS)
        return True

    def _clear_onnx_cache(self) -> None:
        """Remove all cached ONNX and config files to force fresh download."""
        for p in self.cfg.all_onnx_paths + self.cfg.all_config_paths:
            if p.exists():
                try:
                    p.unlink()
                    self._log(f"Cleared: {p.name}")
                except Exception as e:
                    self._log(f"Failed to clear {p.name}: {e}", LogType.WARNING)

    def _download_with_retry(
        self,
        repo_filename: str,
        dest_path: Path,
        min_size: int,
        max_retries: int = 3,
    ) -> Optional[Path]:
        """Download a single file from HF Hub cache to dest_path with retry and size validation."""
        for attempt in range(max_retries):
            try:
                self._log(f"  → Attempt {attempt + 1}/{max_retries}: {repo_filename}")

                # Download to HF cache (returns cached path)
                cached_path = Path(
                    hf_hub_download(
                        repo_id=self.cfg.model_id,
                        filename=repo_filename,
                        revision=self.cfg.revision,
                    )
                )

                # Validate cached file
                cached_size = cached_path.stat().st_size
                if cached_size < min_size:
                    raise RuntimeError(f"File too small ({cached_size} bytes)")

                self._log(f"  → Downloaded {cached_size:,} bytes to cache")

                # Atomic copy to target directory
                self._atomic_copy(cached_path, dest_path)

                # Verify destination
                if not dest_path.exists() or dest_path.stat().st_size < min_size:
                    raise RuntimeError("Copy failed or destination too small")

                self._log(f"  → Copied to {dest_path}")
                return dest_path

            except Exception as e:
                self._log(f"Attempt {attempt + 1}/{max_retries} failed: {e}", LogType.WARNING)
                if attempt < max_retries - 1:
                    backoff = 2 ** attempt
                    self._log(f"  → Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    self._log(f"Download failed after {max_retries} attempts: {e}", LogType.ERROR)
                    return None
        return None

    def _atomic_copy(self, src: Path, dst: Path) -> None:
        """Atomically copy src to dst using a temp file."""
        if dst.exists():
            dst.unlink(missing_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        try:
            shutil.copy2(str(src), str(tmp))
            tmp.rename(dst)
        except Exception as e:
            self._log(f"Atomic copy failed ({e}), falling back to direct copy", LogType.WARNING)
            shutil.copy2(str(src), str(dst))