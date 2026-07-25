import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter

from src.core.Logger.DarwinLogger import darwin
from src.core.Models.ModeInterfaces import OCRInferenceInterface



MODELS = {
    "det": {
        "file": "en_PP-OCRv3_det_infer.onnx",
        "hf_repo": "SWHL/RapidOCR",
        "hf_path": "PP-OCRv4/en_PP-OCRv3_det_infer.onnx",
    },
    "rec": {
        "file": "ch_PP-OCRv4_rec_server_infer.onnx",
        "hf_repo": "SWHL/RapidOCR",
        "hf_path": "PP-OCRv4/ch_PP-OCRv4_rec_server_infer.onnx",
    },
    "cls": {
        "file": "ch_ppocr_mobile_v2.0_cls_train.onnx",
        "hf_repo": "SWHL/RapidOCR",
        "hf_path": "PP-OCRv3/ch_ppocr_mobile_v2.0_cls_train.onnx",
    },
}

# Expected size ranges (bytes) — catches truncated or corrupted downloads
VERIFY = {
    "det": {"min_bytes": 1_500_000, "max_bytes": 5_000_000},      # Actual: ~2.3 MB
    "rec": {"min_bytes": 80_000_000, "max_bytes": 100_000_000},   # Actual: ~86.3 MB
    "cls": {"min_bytes": 500_000, "max_bytes": 1_000_000},         # Actual: ~0.6 MB
}


# ---------------------------------------------------------------------------
# RapidOCR Inference
# ---------------------------------------------------------------------------

class RapidOCRInference(OCRInferenceInterface):
    """
    RapidOCR provider with automatic model download, integrity verification,
    and synthetic-image recovery.
    """

    def __init__(
        self,
        model_dir: Path,
        confidence_threshold: float = 0.7,
        det_db_thresh: float = 0.2,
        det_db_box_thresh: float = 0.4,
        max_workers: int = 1,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.confidence_threshold = confidence_threshold
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh

        self._engine = None
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    # -----------------------------------------------------------------------
    # Integrity verification (static — can be used before instantiation)
    # -----------------------------------------------------------------------

    @staticmethod
    def verify_integrity(model_dir: Path) -> Tuple[bool, str]:
        """
        Check that all required ONNX files exist and are within expected size ranges.
        Returns (ok, error_message). Safe to call before load().
        """
        model_dir = Path(model_dir)
        errors = []

        for key, info in MODELS.items():
            file_path = model_dir / info["file"]
            if not file_path.exists():
                errors.append(f"{key}: missing '{info['file']}'")
                continue

            size = file_path.stat().st_size
            bounds = VERIFY[key]
            if size < bounds["min_bytes"]:
                errors.append(
                    f"{key}: '{info['file']}' too small ({size / 1_048_576:.1f} MB < "
                    f"{bounds['min_bytes'] / 1_048_576:.1f} MB) — likely truncated"
                )
            elif size > bounds["max_bytes"]:
                errors.append(
                    f"{key}: '{info['file']}' unexpectedly large ({size / 1_048_576:.1f} MB > "
                    f"{bounds['max_bytes'] / 1_048_576:.1f} MB)"
                )

        if errors:
            return False, "; ".join(errors)
        return True, ""

    # -----------------------------------------------------------------------
    # Model lifecycle
    # -----------------------------------------------------------------------

    def load(self) -> None:
        """One-time setup: download ONNX files if missing, verify integrity, init engine."""
        # 1. Verify current state
        ok, err = self.verify_integrity(self.model_dir)
        if not ok:
            darwin.warning(f"OCR models missing or corrupt: {err}")
            self._ensure_models()

        # 2. Verify after download
        ok, err = self.verify_integrity(self.model_dir)
        if not ok:
            darwin.error(f"OCR model verification failed after download: {err}")
            return

        # 3. Initialize engine
        self._engine = self._init_engine()
        darwin.info("RapidOCR engine loaded and verified")

    def _ensure_models(self) -> None:
        """Download missing models from HuggingFace SWHL/RapidOCR mirror."""
        from huggingface_hub import hf_hub_download
        import shutil

        self.model_dir.mkdir(parents=True, exist_ok=True)

        for key, info in MODELS.items():
            file_path = self.model_dir / info["file"]
            if file_path.exists():
                continue

            darwin.info(f"Downloading OCR {key} model → {file_path}")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=info["hf_repo"],
                    filename=info["hf_path"],
                    local_dir=str(self.model_dir),
                )

                darwin.info(f"Downloaded to: {downloaded_path}")

                downloaded_path_obj = Path(downloaded_path)
                if downloaded_path_obj.parent != self.model_dir:
                    target_path = self.model_dir / info["file"]
                    darwin.info(f"Moving {downloaded_path} to {target_path}")
                    shutil.move(str(downloaded_path), str(target_path))

                    try:
                        downloaded_path_obj.parent.rmdir()
                    except OSError:
                        pass

                    file_path = target_path
                else:
                    file_path = downloaded_path_obj

                if not file_path.exists():
                    darwin.error(f"File not found after download: {file_path}")
                    return

                size_mb = file_path.stat().st_size / 1_048_576
                darwin.info(f"Downloaded {key} model ({size_mb:.1f} MB)")

            except Exception as e:
                darwin.error(f"Failed to download {key} model: {e}")
                return

    def _init_engine(self):
        """Initialize RapidOCR v3.x with dot-notation params dict."""
        from rapidocr import RapidOCR

        params = {
            "Global.text_score": self.confidence_threshold,
            "Det.model_path": str(self.model_dir / MODELS["det"]["file"]),
            "Det.thresh": self.det_db_thresh,
            "Det.box_thresh": self.det_db_box_thresh,
            "Det.unclip_ratio": 1.5,
            "Cls.model_path": str(self.model_dir / MODELS["cls"]["file"]),
            "Rec.model_path": str(self.model_dir / MODELS["rec"]["file"]),
        }
        return RapidOCR(params=params)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def extract_text(
        self,
        image: Union[str, Path, Image.Image],
        preserve_layout: bool = True,
    ) -> str:
        """
        Dual-pass extraction with post-processing fixes for English text.

        Args:
            image: Input image (path, Path, or PIL Image)
            preserve_layout: If True, keep newlines from multi-line detection.
                             If False, collapse all whitespace to single spaces.
                             Use False for simple English text extraction.

        Returns:
            Concatenated text or empty string.
        """
        pil_image = self._to_pil(image)

        # Pass 1 — raw
        text, confidence = await self._run_ocr(pil_image)
        if confidence >= self.confidence_threshold and text:
            return self._post_process(text, preserve_layout)

        # Pass 2 — synthetic-image preprocessing
        darwin.info(f"OCR pass 1 weak (conf={confidence:.2f}), attempting preprocessing...")
        preprocessed = self._preprocess_for_synthetic(pil_image)
        text, confidence = await self._run_ocr(preprocessed)

        if confidence >= self.confidence_threshold and text:
            return self._post_process(text, preserve_layout)

        darwin.warning(f"OCR failed both passes (best conf={confidence:.2f})")
        return ""

    # -----------------------------------------------------------------------
    # Post-processing
    # -----------------------------------------------------------------------

    def _post_process(self, text: str, preserve_layout: bool) -> str:
        """
        Clean up OCR output for English text.

        The ch_PP-OCRv4_rec model (Chinese) does not preserve inter-word spaces
        well. We apply heuristics to recover them.
        """
        if not text:
            return ""

        if preserve_layout:
            # Keep newlines, but normalize line-internal whitespace
            lines = text.split("\n")
            cleaned_lines = [self._recover_spaces(line) for line in lines]
            return "\n".join(cleaned_lines)
        else:
            # Collapse everything to single spaces
            flat = text.replace("\n", " ")
            flat = re.sub(r"\s+", " ", flat)
            return self._recover_spaces(flat).strip()

    def _recover_spaces(self, text: str) -> str:
        """
        Heuristic: insert spaces between camelCase/PascalCase transitions
        and around punctuation to recover English word boundaries.

        Examples:
            "HelloWorld"   → "Hello World"
            "DarkBackground" → "Dark Background"
            "Text123!@#"   → "Text 123 ! @ #" (handled by caller)
        """
        # Insert space before uppercase letters that follow lowercase (camelCase boundary)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        # Insert space after punctuation if followed by a letter/number
        text = re.sub(r"([.,;:!?])([A-Za-z0-9])", r"\1 \2", text)
        # Insert space before opening brackets/parens
        text = re.sub(r"([A-Za-z0-9])([\[({])", r"\1 \2", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    async def _run_ocr(self, image: Image.Image) -> Tuple[str, float]:
        """Run RapidOCR in a thread and return (text, mean_confidence)."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._thread_pool, self._sync_ocr, image)

        if not result or not result.txts:
            return "", 0.0

        texts = result.txts
        scores = result.scores

        full_text = "\n".join(texts)
        mean_conf = float(sum(scores) / len(scores)) if scores else 0.0
        return full_text, mean_conf

    def _sync_ocr(self, image: Image.Image):
        """Synchronous RapidOCR call — pass numpy array for v3.x compatibility."""
        img_array = np.array(image.convert("RGB"))
        return self._engine(img_array)

    def _to_pil(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Normalize input to PIL RGB."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        darwin.error(f"Unsupported image type: {type(image)}")
        return None

    def _preprocess_for_synthetic(self, image: Image.Image) -> Image.Image:
        """
        Degrade a perfect synthetic image so the DB detector sees 'real' text.
        Pipeline: slight blur → gaussian noise → adaptive threshold → back to RGB.
        """
        img = image.filter(ImageFilter.GaussianBlur(radius=0.5))

        arr = np.array(img).astype(np.int16)
        noise = np.random.normal(0, 4, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    def __del__(self):
        self._thread_pool.shutdown(wait=False)