# src/core/Models/Infrence/Onnx/RapidOCRInference.py
"""
RapidOCR ONNX Inference Engine — PP-OCRv4
Runs detection → classification → recognition pipeline locally.
"""

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from PIL import Image

from src.core.Logger.DarwinLogger import darwin, LogType


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OCRMetadata:
    """Metadata about the OCR inference run."""
    model_name: str
    repo_id: str
    device: str
    onnx_runtime_version: str
    inference_time_ms: float
    num_text_regions: int
    image_size: Tuple[int, int]
    det_time_ms: float
    cls_time_ms: float
    rec_time_ms: float
    det_model: str
    cls_model: str
    rec_model: str


@dataclass
class TextRegion:
    """Single detected text region."""
    text: str
    score: float
    box: np.ndarray  # (4, 2) float32 — corner points
    cls_score: float
    cls_label: int   # 0 = normal, 1 = rotated 180°


@dataclass
class OCRResult:
    """Full OCR output."""
    regions: List[TextRegion]
    metadata: OCRMetadata
    annotated_image: Optional[np.ndarray] = None  # debug viz


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Detection
DET_INPUT_SIZE = 960
DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
DET_THRESH = 0.3
DET_BOX_THRESH = 0.5
DET_MAX_CANDIDATES = 1000
DET_UNCLIP_RATIO = 1.5
DET_MIN_SIZE = 3

# Classification
CLS_INPUT_SHAPE = (192, 48)  # W, H
CLS_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape((1, 1, 3))
CLS_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape((1, 1, 3))
CLS_THRESH = 0.9

# Recognition
REC_INPUT_SHAPE = (320, 48)  # W, H
REC_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape((1, 1, 3))
REC_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape((1, 1, 3))

# CTC blank index
CTC_BLANK_IDX = 0

# PP-OCRv4 rec dictionary (simplified Latin + digits + symbols)
REC_CHAR_DICT = (
    "0123456789abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
)


# ═══════════════════════════════════════════════════════════════════════════════
# Inference Engine
# ═══════════════════════════════════════════════════════════════════════════════

class RapidOCRInference:
    """
    RapidOCR PP-OCRv4 ONNX inference pipeline.
    Loads det + cls + rec models and runs full OCR.
    """

    def __init__(
        self,
        model_dir: Path,
        repo_id: str = "SWHL/RapidOCR",
        char_dict: Optional[str] = None,
        det_limit_side: int = DET_INPUT_SIZE,
        rec_batch_size: int = 6,
    ):
        self.model_dir = Path(model_dir)
        self.repo_id = repo_id
        self.char_dict = char_dict or REC_CHAR_DICT
        self.det_limit_side = det_limit_side
        self.rec_batch_size = rec_batch_size

        self._det_session = None
        self._cls_session = None
        self._rec_session = None
        self._device = "CPU"
        self._onnx_runtime_version = "unknown"

    # ── Darwin logging ───────────────────────────────────────────────────────

    def _log(self, msg: str, level: LogType = LogType.INFO):
        darwin.log(f"[RapidOCR] {msg}", level)

    # ── Context Manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "RapidOCRInference":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> "RapidOCRInference":
        """Load all three ONNX sessions. Chainable."""
        if self.is_loaded():
            self._log("Already loaded, skipping.")
            return self

        self._log(f"Loading RapidOCR from {self.model_dir}")
        self._load_sessions()
        self._log("RapidOCR loaded.", LogType.SUCCESS)
        return self

    def unload(self) -> None:
        """Unload all sessions and free memory."""
        if not self.is_loaded():
            self._log("Not loaded, nothing to unload.")
            return

        self._log("Unloading RapidOCR...")
        self._det_session = None
        self._cls_session = None
        self._rec_session = None
        gc.collect()
        self._log("Unloaded.", LogType.SUCCESS)

    def is_loaded(self) -> bool:
        # Only det and rec are mandatory; cls is optional
        return self._det_session is not None and self._rec_session is not None

    # ── Public inference API ─────────────────────────────────────────────────

    def infer(
        self,
        image: Union[str, Path, bytes, np.ndarray, Image.Image],
        return_annotated: bool = False,
    ) -> OCRResult:
        """
        Run full OCR pipeline on an image.
        Returns detected text regions with metadata.
        """
        if not self.is_loaded():
            self._log("Model not loaded. Call .load() first.", LogType.ERROR)
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Load and preprocess image
        img = self._load_image(image)
        h, w = img.shape[:2]
        self._log(f"Image: {w}×{h}")

        spinner_id = f"ocr_{id(img)}"
        darwin.spinner_start(spinner_id, "Running OCR pipeline")

        t_start = time.perf_counter()

        # ── Stage 1: Detection ──────────────────────────────────────────────
        t0 = time.perf_counter()
        dt_boxes = self._detect(img)
        t_det = (time.perf_counter() - t0) * 1000
        self._log(f"Detection: {len(dt_boxes)} regions in {t_det:.1f}ms")

        if len(dt_boxes) == 0:
            darwin.spinner_stop(spinner_id, success=True)
            meta = self._build_metadata(
                img, t_start, t_det, 0.0, 0.0, len(dt_boxes)
            )
            return OCRResult(regions=[], metadata=meta)

        # ── Stage 2: Classification + Recognition ─────────────────────────────
        t0_cls = time.perf_counter()
        t0_rec = time.perf_counter()

        regions: List[TextRegion] = []
        rec_inputs = []
        rec_boxes = []

        # Batch recognition
        for i, box in enumerate(dt_boxes):
            # Crop and align each box
            crop = self._crop_box(img, box)

            # Classification (orientation)
            cls_label, cls_score = self._classify(crop)
            if cls_label == 1 and cls_score > CLS_THRESH:
                crop = cv2.rotate(crop, cv2.ROTATE_180)

            # Prepare for recognition batching
            rec_input = self._preprocess_rec(crop)
            rec_inputs.append(rec_input)
            rec_boxes.append(box)

            # Run recognition batch when full or last
            if len(rec_inputs) >= self.rec_batch_size or i == len(dt_boxes) - 1:
                texts, scores = self._recognize_batch(rec_inputs)
                for text, score, b in zip(texts, scores, rec_boxes[-len(rec_inputs):]):
                    regions.append(TextRegion(
                        text=text,
                        score=score,
                        box=b,
                        cls_score=cls_score,
                        cls_label=cls_label,
                    ))
                rec_inputs = []
                rec_boxes = []

        t_cls = (time.perf_counter() - t0_cls) * 1000
        t_rec = (time.perf_counter() - t0_rec) * 1000

        # Sort top-to-bottom, left-to-right
        regions.sort(key=lambda r: (r.box[:, 1].min(), r.box[:, 0].min()))

        total_ms = (time.perf_counter() - t_start) * 1000
        darwin.spinner_stop(spinner_id, success=True)

        self._log(
            f"OCR complete: {len(regions)} regions in {total_ms:.1f}ms "
            f"(det={t_det:.1f} cls={t_cls:.1f} rec={t_rec:.1f})",
            LogType.SUCCESS,
        )

        meta = self._build_metadata(img, t_start, t_det, t_cls, t_rec, len(regions))
        result = OCRResult(regions=regions, metadata=meta)

        if return_annotated:
            result.annotated_image = self._draw_boxes(img, regions)

        return result

    # ── Internal: Session loading (FLEXIBLE PATH RESOLUTION) ───────────────

    def _load_sessions(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            self._log("onnxruntime required. pip install onnxruntime", LogType.ERROR)
            raise

        self._onnx_runtime_version = ort.__version__

        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]
        self._device = providers[0].replace("ExecutionProvider", "")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        def resolve_path(name: str, patterns: List[str]) -> Path:
            """
            Try multiple patterns (with or without subfolder).
            Patterns are tried in order; the first existing file is used.
            """
            for pattern in patterns:
                p = self.model_dir / pattern
                if p.exists():
                    return p
                # Also try without any subfolder (flat)
                flat = Path(pattern).name
                p_flat = self.model_dir / flat
                if p_flat.exists():
                    return p_flat
            raise FileNotFoundError(f"Missing model {name}: tried {patterns}")

        # Detection and recognition are required
        det_path = resolve_path(
            "detection",
            ["PP-OCRv4/ch_PP-OCRv4_det_infer.onnx", "ch_PP-OCRv4_det_infer.onnx"]
        )
        rec_path = resolve_path(
            "recognition",
            ["PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx", "ch_PP-OCRv4_rec_infer.onnx"]
        )

        self._log(f"Loading detection: {det_path}")
        self._det_session = ort.InferenceSession(str(det_path), sess_options=sess_opts, providers=providers)

        self._log(f"Loading recognition: {rec_path}")
        self._rec_session = ort.InferenceSession(str(rec_path), sess_options=sess_opts, providers=providers)

        # Classification is optional – try to load it, but if not found, warn and continue
        try:
            cls_path = resolve_path(
                "classification",
                ["PP-OCRv4/ch_PP-OCRv4_cls_infer.onnx", "ch_PP-OCRv4_cls_infer.onnx"]
            )
            self._log(f"Loading classification: {cls_path}")
            self._cls_session = ort.InferenceSession(str(cls_path), sess_options=sess_opts, providers=providers)
            self._log("Classification model loaded – orientation correction enabled.", LogType.SUCCESS)
        except FileNotFoundError:
            self._cls_session = None
            self._log("Classification model not found – orientation correction disabled.", LogType.WARNING)

        self._log(f"Providers: {providers}")

    # ── Internal: Image loading ──────────────────────────────────────────────

    @staticmethod
    def _load_image(image: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """Load image to BGR numpy array."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        elif isinstance(image, bytes):
            arr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            img = image.copy()
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif img.shape[2] == 3:
                # Assume RGB, convert to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        return img

    # ── Internal: Detection ──────────────────────────────────────────────────

    def _detect(self, img: np.ndarray) -> List[np.ndarray]:
        h, w = img.shape[:2]
        ratio = self.det_limit_side / max(h, w)
        if ratio < 1.0:
            new_h, new_w = int(h * ratio), int(w * ratio)
        else:
            new_h, new_w = h, w

        pad_h = (32 - new_h % 32) % 32
        pad_w = (32 - new_w % 32) % 32
        resize_h, resize_w = new_h + pad_h, new_w + pad_w

        resized = cv2.resize(img, (resize_w, resize_h))
        pad = np.zeros((resize_h, resize_w, 3), dtype=np.uint8)
        pad[:new_h, :new_w] = resized[:new_h, :new_w]   # ← FIXED

        # Normalize
        norm = pad.astype(np.float32) / 255.0
        norm = (norm - DET_MEAN) / DET_STD
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        # Inference
        outputs = self._det_session.run(None, {"x": tensor})
        pred = outputs[0][0, 0]  # (H, W)

        # Postprocess: threshold + contours
        binary = (pred > DET_THRESH).astype(np.uint8) * 255

        # Scale back to original image
        scale_h = resize_h / h
        scale_w = resize_w / w

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours[:DET_MAX_CANDIDATES]:
            if len(cnt) < 4:
                continue
            # Min area rect
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)  # (4, 2)
            # Scale back
            box[:, 0] /= scale_w
            box[:, 1] /= scale_h
            # Unclip / expand
            box = self._unclip(box, DET_UNCLIP_RATIO)
            # Check size
            if self._box_size(box) < DET_MIN_SIZE * DET_MIN_SIZE:
                continue
            boxes.append(box.astype(np.float32))

        # NMS-like deduplication (simplified: remove heavily overlapping)
        boxes = self._filter_boxes(boxes)
        # Sort top-to-bottom
        boxes.sort(key=lambda b: (b[:, 1].min(), b[:, 0].min()))
        return boxes

    @staticmethod
    def _unclip(box: np.ndarray, ratio: float) -> np.ndarray:
        """Expand box outward by ratio."""
        poly = box.copy()
        area = cv2.contourArea(poly)
        if area < 1:
            return poly
        # Simple expansion: offset from center
        center = poly.mean(axis=0)
        poly = center + (poly - center) * ratio
        return poly

    @staticmethod
    def _box_size(box: np.ndarray) -> float:
        """Approximate box area via shoelace."""
        x, y = box[:, 0], box[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    
    
    @staticmethod
    def _filter_boxes(boxes: List[np.ndarray], threshold: float = 0.7) -> List[np.ndarray]:
        """Simple overlap filter. Remove boxes that are mostly inside another."""
        if len(boxes) < 2:
            return boxes
        keep = []
        for i, b in enumerate(boxes):
            area_b = RapidOCRInference._box_size(b)
            if area_b < 1:
                continue
            suppressed = False
            for j, other in enumerate(boxes):
                if i == j:
                    continue
                if b.shape != (4, 2) or other.shape != (4, 2):
                    continue
                try:
                    # intersectConvexConvex returns (area, polygon)
                    _, inter = cv2.intersectConvexConvex(np.array(b), np.array(other))
                except cv2.error:
                    continue
                # inter is the intersection polygon (or None if no intersection)
                if inter is not None and inter.size >= 6:  # at least 3 points -> 6 coordinates
                    inter_area = cv2.contourArea(inter)
                    if inter_area / area_b > threshold:
                        if area_b < RapidOCRInference._box_size(other):
                            suppressed = True
                            break
            if not suppressed:
                keep.append(b)
        return keep
    # ── Internal: Crop and align ─────────────────────────────────────────────

    @staticmethod
    def _crop_box(img: np.ndarray, box: np.ndarray, height: int = 48) -> np.ndarray:
        """Perspective crop of a rotated box to a straight rectangle."""
        w = int(np.linalg.norm(box[0] - box[1]))
        h = int(np.linalg.norm(box[1] - box[2]))
        if h > w * 1.5:
            # Vertical text: reorder so width is the long side
            dst = np.array([[0, 0], [h, 0], [h, height], [0, height]], dtype=np.float32)
        else:
            dst = np.array([[0, 0], [w, 0], [w, height], [0, height]], dtype=np.float32)

        # Ensure box is ordered tl, tr, br, bl
        box = RapidOCRInference._order_points(box)
        M = cv2.getPerspectiveTransform(box.astype(np.float32), dst)
        crop = cv2.warpPerspective(img, M, (int(dst[2][0]), int(dst[2][1])))
        return crop

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order rect points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    # ── Internal: Classification ─────────────────────────────────────────────

    def _classify(self, crop: np.ndarray) -> Tuple[int, float]:
        # If cls session is not available, assume normal orientation
        if self._cls_session is None:
            return 0, 1.0

        """Run orientation classifier. Returns (label, score)."""
        # Resize to cls input
        h, w = crop.shape[:2]
        ratio = min(CLS_INPUT_SHAPE[0] / w, CLS_INPUT_SHAPE[1] / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(crop, (new_w, new_h))

        pad = np.zeros((CLS_INPUT_SHAPE[1], CLS_INPUT_SHAPE[0], 3), dtype=np.uint8)
        pad[:new_h, :new_w] = resized

        norm = pad.astype(np.float32) / 255.0
        norm = (norm - CLS_MEAN) / CLS_STD
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        outputs = self._cls_session.run(None, {"x": tensor})
        probs = self._softmax(outputs[0][0])
        label = int(np.argmax(probs))
        score = float(probs[label])
        return label, score

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    # ── Internal: Recognition ──────────────────────────────────────────────

    def _preprocess_rec(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess crop for recognition. Returns normalized tensor."""
        h, w = crop.shape[:2]
        ratio = min(REC_INPUT_SHAPE[0] / w, REC_INPUT_SHAPE[1] / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(crop, (new_w, new_h))

        pad = np.zeros((REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0], 3), dtype=np.uint8)
        pad[:new_h, :new_w] = resized

        norm = pad.astype(np.float32) / 255.0
        norm = (norm - REC_MEAN) / REC_STD
        tensor = np.transpose(norm, (2, 0, 1)).astype(np.float32)
        return tensor

    def _recognize_batch(self, inputs: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        """Run recognition on a batch of preprocessed crops."""
        if not inputs:
            return [], []

        # Stack batch
        batch = np.stack(inputs, axis=0).astype(np.float32)  # (B, 3, H, W)
        outputs = self._rec_session.run(None, {"x": batch})
        preds = outputs[0]  # (B, T, C)

        texts = []
        scores = []
        for pred in preds:
            text, score = self._ctc_decode(pred)
            texts.append(text)
            scores.append(score)
        return texts, scores

    def _ctc_decode(self, pred: np.ndarray) -> Tuple[str, float]:
        """CTC greedy decode."""
        char_list = []
        score_list = []
        pred_indices = np.argmax(pred, axis=1)

        for i, idx in enumerate(pred_indices):
            if idx != CTC_BLANK_IDX and (i == 0 or idx != pred_indices[i - 1]):
                if idx < len(self.char_dict):
                    char_list.append(self.char_dict[idx])
                score_list.append(float(pred[i, idx]))

        text = "".join(char_list)
        score = np.mean(score_list) if score_list else 0.0
        return text, score

    # ── Internal: Visualization ──────────────────────────────────────────────

    @staticmethod
    def _draw_boxes(img: np.ndarray, regions: List[TextRegion]) -> np.ndarray:
        """Draw bounding boxes and text on image for debugging."""
        vis = img.copy()
        for i, r in enumerate(regions):
            box = r.box.astype(np.int32)
            cv2.polylines(vis, [box], True, (0, 255, 0), 2)
            label = f"{i}: {r.text} ({r.score:.2f})"
            cv2.putText(vis, label, tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return vis

    # ── Internal: Metadata ───────────────────────────────────────────────────

    def _build_metadata(
        self,
        img: np.ndarray,
        t_start: float,
        t_det: float,
        t_cls: float,
        t_rec: float,
        num_regions: int,
    ) -> OCRMetadata:
        h, w = img.shape[:2]
        total_ms = (time.perf_counter() - t_start) * 1000
        return OCRMetadata(
            model_name="RapidOCR PP-OCRv4",
            repo_id=self.repo_id,
            device=self._device,
            onnx_runtime_version=self._onnx_runtime_version,
            inference_time_ms=round(total_ms, 3),
            num_text_regions=num_regions,
            image_size=(w, h),
            det_time_ms=round(t_det, 3),
            cls_time_ms=round(t_cls, 3),
            rec_time_ms=round(t_rec, 3),
            det_model="ch_PP-OCRv4_det_infer.onnx",
            cls_model="ch_PP-OCRv4_cls_infer.onnx",
            rec_model="ch_PP-OCRv4_rec_infer.onnx",
        )

    # ── Info ─────────────────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        """Return static info about the loaded models."""
        if not self.is_loaded():
            return {"status": "not_loaded", "model_dir": str(self.model_dir)}

        has_cls = self._cls_session is not None
        return {
            "status": "loaded",
            "model_dir": str(self.model_dir),
            "repo_id": self.repo_id,
            "device": self._device,
            "onnx_runtime_version": self._onnx_runtime_version,
            "det_limit_side": self.det_limit_side,
            "rec_batch_size": self.rec_batch_size,
            "char_dict_size": len(self.char_dict),
            "orientation_correction": has_cls,
        }