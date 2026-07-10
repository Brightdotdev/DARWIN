# src/features/Images/ImageHandler.py
"""
Unified image processing using dedicated OCR model for text extraction
and vision model for captioning/description.
Supports custom prompts for OCR and vision stages.
"""

from __future__ import annotations

import time
import io
import hashlib
import re
import asyncio
import atexit
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Optional, Callable, Literal, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
from PIL import Image, ImageFilter, ImageStat, ImageEnhance

from src.core.Logger.DarwinLogger import darwin, LogType
from src.core.Models.ModelManager import modelManager
from src.core.Models.Lib.ModelRegistry import ModelCategory


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_MAX_DIMENSION = 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_BATCH_WORKERS = 4

DEFAULT_CAPTION_PROMPT = (
    "List all visible objects and any text exactly as written. "
    "Do not add details not present in the image."
)

DEFAULT_OCR_PROMPT = (
    "Extract all visible text from this image. Respond with the text only, "
    "in markdown format. Do not add any explanation or commentary."
)

MIN_OCR_DIMENSION = 960
REPETITION_THRESHOLD = 3

SUPPORTED_FORMATS = {
    "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "ppm", "pgm", "pbm",
    "gif", "ico", "pcx", "tga", "sgi", "sun", "ras", "xbm", "xpm", "psd",
}
PROBLEMATIC_FORMATS = {"gif", "ico", "psd"}
UNSUPPORTED_FORMATS = {
    "svg", "pdf", "eps", "ai", "cdr", "raw", "cr2", "nef", "arw", "dng",
    "heic", "heif", "avif", "jxl", "jp2", "j2k", "exr", "hdr", "pic",
}


# ──────────────────────────────────────────────────────────────────────────
# Unified Result Dataclass
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ImageProcessResult:
    text: Optional[str] = None          # OCR result
    description: Optional[str] = None   # caption result
    confidence: float = 0.0
    tokens_used: int = 0
    processing_time_ms: int = 0
    was_resized: bool = False
    image_size: Tuple[int, int] = (0, 0)
    model_id: str = "unknown"
    backend: str = "unknown"
    stages_executed: List[str] = field(default_factory=list)
    stage_results: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    suggested_mode: str = "both"
    format_warning: Optional[str] = None
    error: Optional[str] = None

    @property
    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())

    @property
    def has_description(self) -> bool:
        return bool(self.description and self.description.strip())


# ──────────────────────────────────────────────────────────────────────────
# ImageHandler Class
# ──────────────────────────────────────────────────────────────────────────

class ImageHandler:
    def __init__(
        self,
        max_dimension: int = DEFAULT_MAX_DIMENSION,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        batch_workers: int = DEFAULT_BATCH_WORKERS,
    ):
        self.max_dimension = max_dimension
        self.timeout_seconds = timeout_seconds
        self.batch_workers = batch_workers
        self._executor = ThreadPoolExecutor(max_workers=batch_workers)
        self._result_cache: Dict[str, ImageProcessResult] = {}
        atexit.register(self._shutdown)

    def _shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    # ─── Format Validation ──────────────────────────────────────────────────

    @staticmethod
    def is_supported(image_input: Union[str, Path, Image.Image, bytes]) -> Tuple[bool, Optional[str]]:
        if isinstance(image_input, Image.Image):
            return True, None
        if isinstance(image_input, bytes):
            try:
                img = Image.open(io.BytesIO(image_input))
                fmt = (img.format or "").lower()
                if fmt in UNSUPPORTED_FORMATS:
                    return False, f"Format '{fmt}' is not supported"
                if fmt in PROBLEMATIC_FORMATS:
                    return True, f"Format '{fmt}' may have limited support"
                return True, None
            except Exception as e:
                return False, f"Could not detect image format: {e}"
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return False, f"File not found: {path}"
            ext = path.suffix.lower().lstrip(".")
            if ext in UNSUPPORTED_FORMATS:
                return False, f"Format '{ext}' is not supported. Convert to PNG/JPEG."
            if ext in PROBLEMATIC_FORMATS:
                return True, f"Format '{ext}' may have limited support"
            if ext not in SUPPORTED_FORMATS:
                try:
                    with Image.open(path) as img:
                        img.verify()
                    return True, f"Format '{ext}' is readable by PIL"
                except Exception:
                    return False, f"Format '{ext}' is not supported"
            return True, None
        return False, f"Unsupported input type: {type(image_input)}"

    @staticmethod
    def get_supported_formats() -> List[str]:
        return sorted(SUPPORTED_FORMATS)

    @staticmethod
    def get_unsupported_formats() -> List[str]:
        return sorted(UNSUPPORTED_FORMATS)

    # ─── Model Ensuring ─────────────────────────────────────────────────────

    def ensure_ocr(self, auto_download: bool = True, force_download: bool = False) -> bool:
        spinner_id = f"ensure_ocr_{id(self)}"
        darwin.spinner_start(spinner_id, "Ensuring OCR model...")
        try:
            if modelManager.is_loaded(ModelCategory.OCR):
                darwin.spinner_stop(spinner_id, success=True)
                return True
            model = modelManager.ensure(ModelCategory.OCR, auto_download=auto_download)
            if model is not None:
                darwin.spinner_stop(spinner_id, success=True)
                return True
            if force_download:
                result = modelManager.setup(ModelCategory.OCR, auto_download=True, force_download=True)
                if result.success:
                    darwin.spinner_stop(spinner_id, success=True)
                    return True
            darwin.spinner_stop(spinner_id, success=False)
            return False
        except Exception as e:
            darwin.spinner_stop(spinner_id, success=False)
            darwin.error(f"Failed to load OCR model: {e}")
            return False

    def ensure_vision(self, auto_download: bool = True, force_download: bool = False) -> bool:
        spinner_id = f"ensure_vision_{id(self)}"
        darwin.spinner_start(spinner_id, "Ensuring vision model...")
        try:
            if modelManager.is_loaded(ModelCategory.VISION):
                darwin.spinner_stop(spinner_id, success=True)
                return True
            model = modelManager.ensure(ModelCategory.VISION, auto_download=auto_download)
            if model is not None:
                darwin.spinner_stop(spinner_id, success=True)
                return True
            if force_download:
                result = modelManager.setup(ModelCategory.VISION, auto_download=True, force_download=True)
                if result.success:
                    darwin.spinner_stop(spinner_id, success=True)
                    return True
            darwin.spinner_stop(spinner_id, success=False)
            return False
        except Exception as e:
            darwin.spinner_stop(spinner_id, success=False)
            darwin.error(f"Failed to load vision model: {e}")
            return False

    def ensure_for_stages(
        self,
        stages: Literal["ocr", "vision", "both"],
        auto_download: bool = True,
        force_download: bool = False,
    ) -> bool:
        ok = True
        if stages in ("ocr", "both"):
            ok = self.ensure_ocr(auto_download, force_download) and ok
        if stages in ("vision", "both"):
            ok = self.ensure_vision(auto_download, force_download) and ok
        return ok

    # ─── Main Process ──────────────────────────────────────────────────────

    def process(
        self,
        image_input: Union[str, Path, Image.Image, bytes],
        *,
        stages: Literal["ocr", "vision", "both"] = "both",
        timeout_seconds: Optional[float] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        auto_ensure: bool = True,
        use_cache: bool = True,
        ocr_prompt: Optional[str] = None,
        vision_prompt: Optional[str] = None,
    ) -> ImageProcessResult:
        start_ms = int(time.time() * 1000)
        timeout = timeout_seconds or self.timeout_seconds

        ocr_prompt = ocr_prompt or DEFAULT_OCR_PROMPT
        vision_prompt = vision_prompt or DEFAULT_CAPTION_PROMPT

        # Format validation
        is_valid, format_warning = self.is_supported(image_input)
        if not is_valid:
            return ImageProcessResult(
                error=format_warning or "Unsupported image format",
                processing_time_ms=int(time.time() * 1000) - start_ms,
            )

        # Cache check
        if use_cache:
            cached = self._get_cached(image_input)
            if cached is not None:
                _notify(stream_callback, "Using cached result")
                return cached

        # Load image
        _notify(stream_callback, "Loading image...")
        try:
            image = self._load_image(image_input)
        except Exception as e:
            return ImageProcessResult(
                error=f"Failed to load image: {e}",
                processing_time_ms=int(time.time() * 1000) - start_ms,
            )

        original_size = image.size
        _notify(stream_callback, f"Image loaded: {original_size[0]}x{original_size[1]}")

        # Quality assessment
        quality_score = self._assess_quality(image)
        suggested_mode = self._suggest_mode(image, quality_score)
        _notify(stream_callback, f"Quality: {quality_score:.2f}, Suggested: {suggested_mode}")

        # Ensure model(s)
        if auto_ensure:
            if not self.ensure_for_stages(stages):
                return ImageProcessResult(
                    error="Required model(s) not available",
                    processing_time_ms=int(time.time() * 1000) - start_ms,
                    image_size=original_size,
                )

        # Resize for vision (if needed)
        vision_image, was_resized = self._resize(image)
        if was_resized:
            _notify(stream_callback, f"Resized for vision to {vision_image.size[0]}x{vision_image.size[1]}")

        # Resolve stages
        stages_to_run = self._resolve_stages(stages, suggested_mode)
        _notify(stream_callback, f"Running stages: {', '.join(stages_to_run)}")

        # Run inference
        text: Optional[str] = None
        description: Optional[str] = None
        tokens_used = 0
        confidence = 0.0
        stage_results: Dict[str, Any] = {}
        stages_executed: List[str] = []
        error: Optional[str] = None
        model_id = "unknown"
        backend = "unknown"

        try:
            # OCR stage — uses dedicated OCR model
            if "ocr" in stages_to_run:
                _notify(stream_callback, "Running OCR...")
                ocr_result = self._run_ocr(vision_image, timeout)
                if ocr_result.get("success"):
                    text = ocr_result.get("description")
                    tokens_used += ocr_result.get("tokens", 0)
                    confidence = max(confidence, ocr_result.get("confidence", 0.0))
                    model_id = ocr_result.get("model_id", model_id)
                    backend = ocr_result.get("backend", backend)
                    stages_executed.append("ocr")
                    stage_results["ocr"] = ocr_result
                    _notify(stream_callback, f"OCR complete: {len(text or '')} chars")
                else:
                    stage_results["ocr"] = ocr_result
                    _notify(stream_callback, f"OCR failed: {ocr_result.get('error')}")

            # Caption stage — uses vision model
            if "vision" in stages_to_run:
                _notify(stream_callback, "Generating caption...")
                vision_result = self._run_vision(vision_image, timeout, prompt=vision_prompt)
                if vision_result.get("success"):
                    description = vision_result.get("description")
                    tokens_used += vision_result.get("tokens", 0)
                    v_conf = vision_result.get("confidence", 0.0)
                    confidence = v_conf if confidence == 0 else (confidence + v_conf) / 2
                    model_id = vision_result.get("model_id", model_id)
                    backend = vision_result.get("backend", backend)
                    stages_executed.append("vision")
                    stage_results["vision"] = vision_result
                    _notify(stream_callback, "Caption complete")
                else:
                    stage_results["vision"] = vision_result
                    _notify(stream_callback, f"Vision failed: {vision_result.get('error')}")

        except FutureTimeoutError:
            error = f"Inference timed out after {timeout}s"
            _notify(stream_callback, error)
        except Exception as e:
            error = f"Inference failed: {e}"
            _notify(stream_callback, error)
            darwin.error(error)

        elapsed = int(time.time() * 1000) - start_ms

        result = ImageProcessResult(
            text=self._clean_text(text),
            description=self._clean_text(description),
            confidence=round(confidence, 3),
            tokens_used=tokens_used,
            processing_time_ms=elapsed,
            was_resized=was_resized,
            image_size=original_size,
            model_id=model_id,
            backend=backend,
            stages_executed=stages_executed,
            stage_results=stage_results,
            quality_score=round(quality_score, 3),
            suggested_mode=suggested_mode,
            format_warning=format_warning,
            error=error,
        )

        if use_cache and error is None:
            self._cache_result(image_input, result)

        return result

    # ─── Batch Processing ──────────────────────────────────────────────────

    def process_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image, bytes]],
        *,
        stages: Literal["ocr", "vision", "both"] = "both",
        max_workers: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        stream_callback: Optional[Callable[[str, int, int], None]] = None,
        auto_ensure: bool = True,
        unload_after: bool = False,
        use_cache: bool = True,
        ocr_prompt: Optional[str] = None,
        vision_prompt: Optional[str] = None,
    ) -> List[ImageProcessResult]:
        if not image_inputs:
            return []

        total = len(image_inputs)
        workers = max_workers or self.batch_workers
        timeout = timeout_seconds or self.timeout_seconds

        _notify_batch(stream_callback, f"Starting batch ({total} images, {workers} workers)...", 0, total)

        valid_inputs: List[Tuple[int, Union[str, Path, Image.Image, bytes]]] = []
        for i, inp in enumerate(image_inputs):
            is_valid, warning = self.is_supported(inp)
            if is_valid:
                valid_inputs.append((i, inp))
            else:
                _notify_batch(stream_callback, f"Skipping image {i+1}: {warning}", i+1, total)

        if not valid_inputs:
            return [ImageProcessResult(error="No valid images to process") for _ in image_inputs]

        if auto_ensure:
            if not self.ensure_for_stages(stages):
                return [ImageProcessResult(error="Required model(s) not available") for _ in image_inputs]

        results: List[ImageProcessResult] = [ImageProcessResult(error="Not processed")] * total

        def process_one(idx: int, inp: Union[str, Path, Image.Image, bytes]) -> Tuple[int, ImageProcessResult]:
            _notify_batch(stream_callback, f"Processing {idx+1}/{total}...", idx+1, total)
            result = self.process(
                inp,
                stages=stages,
                timeout_seconds=timeout,
                auto_ensure=False,
                use_cache=use_cache,
                ocr_prompt=ocr_prompt,
                vision_prompt=vision_prompt,
            )
            return idx, result

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_one, idx, inp) for idx, inp in valid_inputs]
            for future in futures:
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = next((i for i, f in enumerate(futures) if f == future), 0)
                    results[idx] = ImageProcessResult(error=f"Batch error: {e}")

        _notify_batch(stream_callback, f"Batch complete: {total} images", total, total)

        if unload_after:
            _notify_batch(stream_callback, "Unloading models...", total, total)
            modelManager.unload(ModelCategory.OCR)
            modelManager.unload(ModelCategory.VISION)

        return results

    # ─── Internal OCR Runner ───────────────────────────────────────────────

    def _run_ocr(self, image: Image.Image, timeout: float) -> Dict[str, Any]:
        """Run the dedicated OCR model (RapidOCR) for text extraction."""
        start = time.perf_counter()
        try:
            ocr_model = modelManager.get(ModelCategory.OCR)
            if ocr_model is None:
                return {
                    "success": False,
                    "error": "OCR model not available",
                    "time_ms": (time.perf_counter() - start) * 1000,
                }

            # OCR extract_text is async — run it in a fresh event loop via executor
            async def _extract():
                return await asyncio.wait_for(
                    ocr_model.extract_text(image),
                    timeout=timeout,
                )

            def _run_async():
                return asyncio.run(_extract())

            future = self._executor.submit(_run_async)
            extracted_text = future.result(timeout=timeout + 1)
            elapsed = (time.perf_counter() - start) * 1000

            if not extracted_text or not extracted_text.strip():
                return {
                    "success": False,
                    "error": "No text detected",
                    "description": extracted_text,
                    "time_ms": elapsed,
                }

            return {
                "success": True,
                "description": extracted_text,
                "tokens": self._estimate_tokens(extracted_text),
                "confidence": 0.85,
                "model_id": getattr(ocr_model, "model_id", "rapidocr"),
                "backend": "rapidocr",
                "time_ms": elapsed,
            }
        except FutureTimeoutError:
            return {
                "success": False,
                "error": f"OCR timed out after {timeout}s",
                "time_ms": (time.perf_counter() - start) * 1000,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"OCR inference failed: {e}",
                "time_ms": (time.perf_counter() - start) * 1000,
            }

    # ─── Internal Vision Runner ────────────────────────────────────────────

    def _run_vision(self, image: Image.Image, timeout: float, prompt: str) -> Dict[str, Any]:
        """Run the vision model for captioning/description."""
        start = time.perf_counter()
        try:
            vision_model = modelManager.get(ModelCategory.VISION)
            if vision_model is None:
                return {
                    "success": False,
                    "error": "Vision model not available",
                    "time_ms": (time.perf_counter() - start) * 1000,
                }

            result = vision_model.generate(image=image, prompt=prompt)
            description = result.text if hasattr(result, "text") else str(result)
            elapsed = (time.perf_counter() - start) * 1000

            if self._is_repetitive(description):
                return {
                    "success": False,
                    "error": "Vision output is repetitive (hallucination)",
                    "description": description,
                    "tokens": 0,
                    "time_ms": elapsed,
                }

            if len(description.strip()) < 5:
                return {
                    "success": False,
                    "error": "Vision output too short",
                    "description": description,
                    "tokens": 0,
                    "time_ms": elapsed,
                }

            metadata = getattr(result, "metadata", None)
            backend = getattr(metadata, "backend", "unknown") if metadata else "unknown"
            model_id = getattr(metadata, "model_id", "unknown") if metadata else "unknown"
            tokens = getattr(metadata, "num_tokens_generated", 0) if metadata else 0

            return {
                "success": True,
                "description": description,
                "tokens": tokens or self._estimate_tokens(description),
                "confidence": 0.75,
                "model_id": model_id,
                "backend": backend,
                "time_ms": elapsed,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "time_ms": (time.perf_counter() - start) * 1000,
            }

    # ─── Utility Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _is_repetitive(text: str) -> bool:
        if not text:
            return False
        tokens = re.findall(r'\b\w+\b', text.lower())
        if len(tokens) < 3:
            return False
        count = 1
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i-1]:
                count += 1
                if count >= REPETITION_THRESHOLD:
                    return True
            else:
                count = 1
        from collections import Counter
        counter = Counter(tokens)
        for token, cnt in counter.items():
            if cnt > len(tokens) * 0.6:
                return True
        return False

    @staticmethod
    def _load_image(image_input: Union[str, Path, Image.Image, bytes]) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                darwin.error(f"Image not found: {path}")
                return None
            return Image.open(path).convert("RGB")
        elif isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            darwin.error(f"Unsupported image type: {type(image_input)}")
            return None

    def _resize(self, image: Image.Image) -> Tuple[Image.Image, bool]:
        w, h = image.size
        if max(w, h) <= self.max_dimension:
            return image, False
        scale = self.max_dimension / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size, Image.Resampling.LANCZOS), True

    @staticmethod
    def _assess_quality(image: Image.Image) -> float:
        try:
            gray = image.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_var = np.var(np.array(edges))
            blur_score = min(edge_var / 500.0, 1.0)
            stat = ImageStat.Stat(gray)
            contrast = stat.stddev[0]
            contrast_score = min(contrast / 50.0, 1.0)
            return (blur_score * 0.6 + contrast_score * 0.4)
        except Exception:
            return 0.5

    @staticmethod
    def _suggest_mode(image: Image.Image, quality: float) -> Literal["ocr", "vision", "both"]:
        try:
            gray = image.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_density = np.mean(np.array(edges) > 10)
            h_lines = gray.filter(ImageFilter.Kernel((3, 3), [-1, -1, -1, 2, 2, 2, -1, -1, -1], 1))
            line_density = np.mean(np.array(h_lines) > 20)
            if line_density > 0.15 and quality > 0.5:
                return "ocr"
            if edge_density < 0.05 and line_density < 0.05:
                return "vision"
            return "both"
        except Exception:
            return "both"

    @staticmethod
    def _resolve_stages(requested: str, suggested: str) -> List[str]:
        if requested == "ocr":
            return ["ocr"]
        if requested == "vision":
            return ["vision"]
        stages = ["ocr"]
        if suggested != "ocr":
            stages.append("vision")
        return stages

    @staticmethod
    def _clean_text(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        if not isinstance(text, str):
            text = str(text)
        cleaned = " ".join(text.split())
        return cleaned if cleaned else None

    @staticmethod
    def _estimate_tokens(text: Optional[str]) -> int:
        if not text:
            return 0
        if not isinstance(text, str):
            text = str(text)
        return int(len(text.split()) * 1.3)

    # ─── Cache ──────────────────────────────────────────────────────────────

    def _get_cached(self, image_input: Union[str, Path, Image.Image, bytes]) -> Optional[ImageProcessResult]:
        try:
            image = self._load_image(image_input)
            img_hash = hashlib.sha256(np.array(image).tobytes()).hexdigest()[:16]
            return self._result_cache.get(img_hash)
        except Exception:
            return None

    def _cache_result(self, image_input: Union[str, Path, Image.Image, bytes], result: ImageProcessResult) -> None:
        try:
            image = self._load_image(image_input)
            img_hash = hashlib.sha256(np.array(image).tobytes()).hexdigest()[:16]
            self._result_cache[img_hash] = result
        except Exception:
            pass

    def clear_cache(self) -> None:
        self._result_cache.clear()
        darwin.info("Image result cache cleared")

    def get_cache_size(self) -> int:
        return len(self._result_cache)


# ──────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────

def _notify(cb: Optional[Callable[[str], None]], msg: str) -> None:
    if cb:
        try:
            cb(msg)
        except Exception:
            pass

def _notify_batch(
    cb: Optional[Callable[[str, int, int], None]],
    msg: str,
    current: int,
    total: int,
) -> None:
    if cb:
        try:
            cb(msg, current, total)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────
# Module‑level convenience API
# ──────────────────────────────────────────────────────────────────────────

_default_handler: Optional[ImageHandler] = None

def _get_default_handler() -> ImageHandler:
    global _default_handler
    if _default_handler is None:
        _default_handler = ImageHandler()
    return _default_handler

def process(
    image_input: Union[str, Path, Image.Image, bytes],
    stages: Literal["ocr", "vision", "both"] = "both",
    **kwargs,
) -> ImageProcessResult:
    return _get_default_handler().process(image_input, stages=stages, **kwargs)

def process_batch(
    image_inputs: List[Union[str, Path, Image.Image, bytes]],
    stages: Literal["ocr", "vision", "both"] = "both",
    **kwargs,
) -> List[ImageProcessResult]:
    return _get_default_handler().process_batch(image_inputs, stages=stages, **kwargs)

def ensure_all(auto_download: bool = True, force_download: bool = False) -> bool:
    return _get_default_handler().ensure_for_stages("both", auto_download, force_download)

def is_supported(image_input: Union[str, Path, Image.Image, bytes]) -> Tuple[bool, Optional[str]]:
    return ImageHandler.is_supported(image_input)

def get_supported_formats() -> List[str]:
    return ImageHandler.get_supported_formats()