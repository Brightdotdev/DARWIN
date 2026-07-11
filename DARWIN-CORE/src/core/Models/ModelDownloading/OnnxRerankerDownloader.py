# src/core/Models/ModelDownloading/OnnxRerankerDownloader.py
"""
ONNX Cross-Encoder Reranker Downloader — MS MARCO MiniLM-L6-v2
Downloads ONNX model + tokenizer files from HuggingFace Hub with:
  - Retry logic with exponential backoff
  - File size validation
  - Atomic file operations
  - Clean cache-to-target copy (no subdir flattening hacks)
  - Force/recache mode for corrupted downloads
"""

import shutil
import time
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

from src.core.Logger.DarwinLogger import darwin
from src.lib.InternetChecker import wait_for_internet, check_internet_robust


# ═══════════════════════════════════════════════════════════════════════════════
# Defaults (overridable via constructor)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_REPO_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_ONNX_FILENAME = "onnx/model_qint8_avx512_vnni.onnx"  # ~23.2 MB, quantized
# Alternative: "onnx/model.onnx"                    (~91 MB)
# Alternative: "onnx/model_O4.onnx"                 (~45.5 MB, optimized)

DEFAULT_TOKENIZER_FILES = [
    "tokenizer.json",
    "vocab.txt",
    "special_tokens_map.json",
]

MIN_ONNX_SIZE = 10_000_000   # 10 MB
MIN_TOKENIZER_SIZE = 100     # 100 bytes


class OnnxRerankerDownloader:
    """Downloads MS MARCO cross-encoder ONNX reranker from HuggingFace Hub."""

    def __init__(
        self,
        target_dir: Path,
        repo_id: str = DEFAULT_REPO_ID,
        onnx_filename: str = DEFAULT_ONNX_FILENAME,
        tokenizer_files: Optional[list[str]] = None,
    ):
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.onnx_filename = onnx_filename
        self.tokenizer_files = tokenizer_files or DEFAULT_TOKENIZER_FILES.copy()

    # ── Public API ───────────────────────────────────────────────────────────

    def ensure(
        self,
        force: bool = False,
        recache: bool = False,
        wait_internet: bool = True,
        max_wait: int = 60,
    ) -> bool:
        """
        One-call lifecycle: check → wait for net → download → verify.
        
        Args:
            force: If True, redownload even if files exist (corruption recovery)
            recache: Alias for force — clearer naming for "delete and re-download"
            wait_internet: Block until internet is available
            max_wait: Max seconds to wait for internet
        """
        should_force = force or recache

        if not should_force and self.verify_integrity():
            darwin.info("Model already cached and valid.")
            return True

        if should_force:
            darwin.info("Force/recache mode: clearing existing files...")
            self._clear_cache()

        if wait_internet:
            if not wait_for_internet(pause=3, max_wait=max_wait, verbose=True):
                darwin.error("No internet connection available.")
                return False
        else:
            if not check_internet_robust():
                darwin.error("No internet connection available.")
                return False

        if not self.download(force=should_force):
            darwin.error("Download failed.")
            return False

        if not self.verify_integrity():
            darwin.error("Integrity check failed after download.")
            return False

        darwin.success("Reranker ONNX model ready.")
        return True

    def download(self, force: bool = False) -> bool:
        """Download ONNX model + tokenizer files. Returns True on success."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            darwin.error("huggingface_hub not installed. pip install huggingface_hub")
            return False

        spinner_id = f"dl_reranker_{self.repo_id.split('/')[-1]}"
        darwin.spinner_start(spinner_id, f"Downloading {self.repo_id}")

        # ── ONNX model ──────────────────────────────────────────────────────
        dest_onnx = self.target_dir / "model.onnx"
        if not dest_onnx.exists() or force:
            darwin.info(f"Downloading ONNX: {self.onnx_filename}")
            if not self._download_file(
                repo_filename=self.onnx_filename,
                dest_path=dest_onnx,
                min_size=MIN_ONNX_SIZE,
            ):
                darwin.spinner_stop(spinner_id, success=False)
                return False
            darwin.info(f"ONNX model cached: {dest_onnx.name}")
        else:
            darwin.info(f"Cached ONNX: {dest_onnx.name}")

        # ── Tokenizer files ─────────────────────────────────────────────────
        for filename in self.tokenizer_files:
            dest = self.target_dir / filename
            if dest.exists() and not force:
                darwin.info(f"Cached tokenizer: {filename}")
                continue
            darwin.info(f"Downloading tokenizer: {filename}")
            if not self._download_file(
                repo_filename=filename,
                dest_path=dest,
                min_size=MIN_TOKENIZER_SIZE,
            ):
                darwin.spinner_stop(spinner_id, success=False)
                return False
            darwin.info(f"Tokenizer cached: {filename}")

        darwin.spinner_stop(spinner_id, success=True)
        return True

    def verify_integrity(self) -> bool:
        """Check all required files exist and have reasonable size."""
        required = [self.target_dir / "model.onnx"]
        required += [self.target_dir / f for f in self.tokenizer_files]

        for p in required:
            if not p.exists():
                darwin.error(f"Missing file: {p}")
                return False
            size = p.stat().st_size
            if p.name.endswith(".onnx") and size < MIN_ONNX_SIZE:
                darwin.error(f"ONNX file too small ({size} bytes): {p}")
                return False
            if size < MIN_TOKENIZER_SIZE:
                darwin.error(f"File too small ({size} bytes): {p}")
                return False
        return True

    def _clear_cache(self) -> None:
        """Remove all cached files to force fresh download."""
        files_to_remove = [self.target_dir / "model.onnx"]
        files_to_remove += [self.target_dir / f for f in self.tokenizer_files]

        for p in files_to_remove:
            if p.exists():
                try:
                    p.unlink()
                    darwin.info(f"Cleared: {p.name}")
                except Exception as e:
                    darwin.warning(f"Failed to clear {p}: {e}")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _download_file(
        self,
        repo_filename: str,
        dest_path: Path,
        min_size: int,
        max_retries: int = 3,
    ) -> bool:
        """Download a single file from HF Hub cache to dest_path with retry and size validation."""
        for attempt in range(max_retries):
            try:
                darwin.info(f"  → Attempt {attempt + 1}/{max_retries}: {repo_filename}")

                # Download to HF cache (returns cached path)
                cached_path = Path(
                    hf_hub_download(
                        repo_id=self.repo_id,
                        filename=repo_filename,
                    )
                )

                # Validate cached file
                cached_size = cached_path.stat().st_size
                if cached_size < min_size:
                    darwin.error(f"File too small ({cached_size} bytes)")
                    return False

                darwin.info(f"  → Downloaded {cached_size:,} bytes to cache")

                # Atomic copy to target directory
                self._atomic_copy(cached_path, dest_path)

                # Verify destination
                if not dest_path.exists() or dest_path.stat().st_size < min_size:
                    darwin.error("Copy failed or destination too small")
                    return False

                darwin.info(f"  → Copied to {dest_path}")
                return True

            except Exception as e:
                darwin.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    backoff = 2 ** attempt
                    darwin.info(f"  → Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    darwin.error(f"Download failed after {max_retries} attempts: {e}")
                    return False
        return False

    def _atomic_copy(self, src: Path, dst: Path) -> None:
        """Atomically copy src to dst using a temp file."""
        if dst.exists():
            dst.unlink(missing_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        try:
            shutil.copy2(str(src), str(tmp))
            tmp.rename(dst)
        except Exception as e:
            darwin.warning(f"Atomic copy failed ({e}), falling back to direct copy")
            shutil.copy2(str(src), str(dst))