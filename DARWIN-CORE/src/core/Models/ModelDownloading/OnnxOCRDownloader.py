# src/core/Models/ModelDownloading/OnnxOCRDownloader.py
"""
ONNX OCR Model Downloader — RapidOCR PP-OCRv4
Downloads detection + recognition ONNX models from HuggingFace Hub.
Classification (cls) is optional and will be downloaded if present.
"""

import time
from pathlib import Path
from typing import Optional, List
import shutil
from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.InternetChecker import wait_for_internet, check_internet_robust

DEFAULT_REPO_ID = "SWHL/RapidOCR"

# NOTE: cls is optional and not required for basic inference.
# If you want orientation correction, you can manually add cls later.
DEFAULT_ALLOW_PATTERNS = [
    "PP-OCRv4/ch_PP-OCRv4_det_infer.onnx",
    "PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx",
    # Optional: include cls if it exists (uncomment if you know it's there)
    # "PP-OCRv4/ch_PP-OCRv4_cls_infer.onnx",
]

DEFAULT_IGNORE_PATTERNS = [
    "*_server_*",
    "*_train_*",
    "PP-OCRv1/*",
    "PP-OCRv2/*",
    "PP-OCRv3/*",
    "*.pth",
    "*.pdparams",
    "*.pdopt",
    "*.nb",
    "*.dygraph*",
    "*.tar",
    "*.zip",
    "*.txt",
    "*.md",
]

MIN_MODEL_SIZE = 1_000_000   # 1 MB minimum


class OnnxOCRDownloader:
    def __init__(
        self,
        target_dir: Path,
        repo_id: str = DEFAULT_REPO_ID,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
    ):
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.allow_patterns = allow_patterns or DEFAULT_ALLOW_PATTERNS.copy()
        self.ignore_patterns = ignore_patterns or DEFAULT_IGNORE_PATTERNS.copy()

    def _log(self, msg: str, level: LogType = LogType.INFO):
        darwin.log(f"[OCR Downloader] {msg}", level)

    def ensure(self, force: bool = False, wait_internet: bool = True, max_wait: int = 60) -> bool:
        if self.verify_integrity() and not force:
            self._log("OCR models already cached and valid.")
            return True

        if wait_internet:
            if not wait_for_internet(pause=3, max_wait=max_wait, verbose=True):
                self._log("No internet connection available.", LogType.ERROR)
                return False
        else:
            if not check_internet_robust():
                self._log("No internet connection available.", LogType.ERROR)
                return False

        if not self.download(force=force):
            self._log("Download failed.", LogType.ERROR)
            return False

        if not self.verify_integrity():
            self._log("Integrity check failed after download.", LogType.ERROR)
            return False

        self._log("OCR models ready.", LogType.SUCCESS)
        return True

    def download(self, force: bool = False) -> bool:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            self._log("huggingface_hub not installed. pip install huggingface_hub", LogType.ERROR)
            return False

        if force:
            self._log("Force mode: clearing target directory...")
            for f in self.target_dir.iterdir():
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)

        spinner_id = f"dl_ocr_{self.repo_id.split('/')[-1]}"
        darwin.spinner_start(spinner_id, f"Downloading OCR models from {self.repo_id}")

        for attempt in range(3):
            try:
                self._log(f"Downloading OCR models...")
                snapshot_download(
                    repo_id=self.repo_id,
                    local_dir=str(self.target_dir),
                    local_dir_use_symlinks=False,
                    allow_patterns=self.allow_patterns,
                    ignore_patterns=self.ignore_patterns,
                )
                self._log("Snapshot download complete.")
                darwin.spinner_stop(spinner_id, success=True)
                return True
            except Exception as e:
                self._log(f"Attempt {attempt + 1}/3 failed: {e}", LogType.WARNING)
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    self._log(f"Download failed after 3 attempts: {e}", LogType.ERROR)
                    darwin.spinner_stop(spinner_id, success=False)
                    return False

        darwin.spinner_stop(spinner_id, success=False)
        return False

    def verify_integrity(self) -> bool:
        """Check that required ONNX models exist and have reasonable size."""
        all_ok = True
        for pattern in self.allow_patterns:
            p = self.target_dir / pattern
            if not p.exists():
                self._log(f"Missing required model: {pattern}", LogType.ERROR)
                all_ok = False
                continue
            size = p.stat().st_size
            if size < MIN_MODEL_SIZE:
                self._log(f"Model too small ({size} bytes): {pattern}", LogType.ERROR)
                all_ok = False
        return all_ok

    def get_model_paths(self) -> dict:
        paths = {}
        for pattern in self.allow_patterns:
            p = self.target_dir / pattern
            if p.exists():
                name = p.stem
                if "_det_" in name:
                    paths["detection"] = p
                elif "_cls_" in name:
                    paths["classification"] = p
                elif "_rec_" in name:
                    paths["recognition"] = p
        return paths