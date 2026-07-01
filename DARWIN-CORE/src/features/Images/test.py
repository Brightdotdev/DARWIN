"""
Battle Test Script for Vision Models (OCR + Captioning)
─────────────────────────────────────────────────────────
Tests both OCR and vision models against a directory of images.
Includes supported, unsupported, edge-case, and corrupted files.

Usage:
    python battle_test_vision.py --images ./mock_images --output ./test_results.json
    python battle_test_vision.py --images ./mock_images --stages both --verbose

Outputs:
    - Console logs (extensive, every step)
    - JSON file with all results, timings, errors, and statistics
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
from PIL import Image

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.Logger.DarwinLogger import darwin, LogType
from src.features.Images.ImageHandler import ImageHandler, is_supported
from src.core.Models.ModelManager import modelManager
from src.core.Models.Lib.ModelRegistry import ModelCategory


# ═══════════════════════════════════════════════════════════════════════
# Test Result Dataclass
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ImageTestResult:
    """Result of testing a single image."""
    filename: str
    filepath: str
    format: str
    size_bytes: int
    dimensions: Optional[Tuple[int, int]]

    # Format validation
    is_supported: bool
    support_warning: Optional[str]

    # Processing
    stages_requested: str
    stages_executed: List[str]

    # OCR results
    ocr_text: Optional[str]
    ocr_tokens: int
    ocr_time_ms: int
    ocr_success: bool
    ocr_error: Optional[str]

    # Vision results
    vision_text: Optional[str]
    vision_tokens: int
    vision_time_ms: int
    vision_success: bool
    vision_error: Optional[str]

    # Overall
    total_time_ms: int
    confidence: float
    quality_score: float
    suggested_mode: str

    # Errors
    error: Optional[str]
    traceback: Optional[str]


@dataclass
class BattleTestReport:
    """Complete test report."""
    timestamp: str
    total_images: int
    supported_images: int
    unsupported_images: int
    ocr_success_count: int
    vision_success_count: int
    total_ocr_tokens: int
    total_vision_tokens: int
    total_time_ms: int

    # Per-format statistics
    format_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Individual results
    results: List[Dict[str, Any]] = field(default_factory=list)

    # Error summary
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Model info
    vision_model_info: Optional[Dict[str, Any]] = None
    ocr_model_info: Optional[Dict[str, Any]] = None


# ═══════════════════════════════════════════════════════════════════════
# Logger Setup
# ═══════════════════════════════════════════════════════════════════════

class StepLogger:
    """Extensive step-by-step logger."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.steps: List[Dict[str, Any]] = []

    def log(self, step: str, message: str, level: str = "INFO", data: Optional[Dict] = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "message": message,
            "level": level,
            "data": data or {},
        }
        self.steps.append(entry)

        if self.verbose:
            prefix = f"[{step}]"
            if level == "ERROR":
                print(f"    {prefix} {message}")
            elif level == "WARNING":
                print(f"    {prefix} {message}")
            elif level == "SUCCESS":
                print(f"   {prefix} {message}")
            else:
                print(f"  ℹ️  {prefix} {message}")

        # Also log to darwin
        log_type = getattr(LogType, level, LogType.INFO)
        darwin.log(f"{step}: {message}", log_type)

    def get_steps(self) -> List[Dict[str, Any]]:
        return self.steps


# ═══════════════════════════════════════════════════════════════════════
# Main Test Runner
# ═══════════════════════════════════════════════════════════════════════

def run_battle_test(
    images_dir: Path,
    output_path: Path,
    stages: str = "both",
    verbose: bool = True,
    max_images: Optional[int] = None,
    timeout: float = 30.0,
) -> BattleTestReport:
    """
    Run comprehensive battle test against all images in directory.
    """
    logger = StepLogger(verbose)
    start_time = int(time.time() * 1000)

    logger.log("INIT", f"Battle test starting", "INFO", {
        "images_dir": str(images_dir),
        "output_path": str(output_path),
        "stages": stages,
        "max_images": max_images,
        "timeout": timeout,
    })

    # ── Step 1: Discover images ───────────────────────────────────────
    logger.log("DISCOVER", "Scanning for images...", "INFO")

    image_files = []
    for ext in ["jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp", "svg", "pdf", "ico", "psd", "raw", "heic"]:
        image_files.extend(images_dir.glob(f"*.{ext}"))
        image_files.extend(images_dir.glob(f"*.{ext.upper()}"))

    # Also grab files with no extension or weird extensions
    all_files = list(images_dir.iterdir())
    for f in all_files:
        if f.is_file() and f not in image_files:
            image_files.append(f)

    if max_images:
        image_files = image_files[:max_images]

    total = len(image_files)
    logger.log("DISCOVER", f"Found {total} files to test", "INFO", {"files": [f.name for f in image_files]})

    # ── Step 2: Initialize handler ──────────────────────────────────────
    logger.log("INIT", "Creating ImageHandler...", "INFO")
    handler = ImageHandler(timeout_seconds=timeout)
    logger.log("INIT", "ImageHandler created", "SUCCESS")

    # ── Step 3: Ensure models ──────────────────────────────────────────
    logger.log("MODELS", f"Ensuring models for stages: {stages}...", "INFO")

    if stages == "vision":
        vision_ready = handler.ensure_vision()
        ocr_ready = False
        logger.log("MODELS", f"Vision ready: {vision_ready}, OCR not needed", "INFO" if vision_ready else "WARNING")
    elif stages == "ocr":
        ocr_ready = handler.ensure_ocr()
        vision_ready = handler.ensure_vision()  # Fallback
        logger.log("MODELS", f"OCR ready: {ocr_ready}, Vision fallback: {vision_ready}", "INFO")
    else:  # both
        vision_ready = handler.ensure_vision()
        ocr_ready = handler.ensure_ocr()
        logger.log("MODELS", f"Vision ready: {vision_ready}, OCR ready: {ocr_ready}", 
                   "SUCCESS" if vision_ready else "ERROR")

    # Get model info
    vision_info = None
    ocr_info = None
    try:
        if vision_ready:
            vision_model = modelManager.get(ModelCategory.VISION)
            if vision_model and hasattr(vision_model, 'get_model_info'):
                vision_info = vision_model.get_model_info()
                logger.log("MODELS", f"Vision model: {vision_info.get('model_id', 'unknown')}", "INFO")
    except Exception as e:
        logger.log("MODELS", f"Could not get vision model info: {e}", "WARNING")

    try:
        if ocr_ready:
            ocr_model = modelManager.get(ModelCategory.OCR)
            if ocr_model and hasattr(ocr_model, 'get_model_info'):
                ocr_info = ocr_model.get_model_info()
                logger.log("MODELS", f"OCR model: {ocr_info.get('model_id', 'unknown')}", "INFO")
    except Exception as e:
        logger.log("MODELS", f"Could not get OCR model info: {e}", "WARNING")

    # ── Step 4: Test each image ─────────────────────────────────────────
    results: List[ImageTestResult] = []
    errors: List[Dict[str, Any]] = []
    format_stats: Dict[str, Dict[str, Any]] = {}

    for idx, image_path in enumerate(image_files, 1):
        logger.log(f"TEST_{idx}", f"Testing {image_path.name} ({idx}/{total})...", "INFO")

        result = test_single_image(handler, image_path, stages, logger, idx, total)
        results.append(result)

        # Update format stats
        fmt = result.format or "unknown"
        if fmt not in format_stats:
            format_stats[fmt] = {
                "count": 0,
                "supported": 0,
                "unsupported": 0,
                "ocr_success": 0,
                "vision_success": 0,
                "total_time_ms": 0,
            }
        format_stats[fmt]["count"] += 1
        if result.is_supported:
            format_stats[fmt]["supported"] += 1
        else:
            format_stats[fmt]["unsupported"] += 1
        if result.ocr_success:
            format_stats[fmt]["ocr_success"] += 1
        if result.vision_success:
            format_stats[fmt]["vision_success"] += 1
        format_stats[fmt]["total_time_ms"] += result.total_time_ms

        # Log result
        if result.error:
            logger.log(f"TEST_{idx}", f"FAILED: {result.error}", "ERROR")
            errors.append({
                "filename": result.filename,
                "error": result.error,
                "traceback": result.traceback,
            })
        elif not result.is_supported:
            logger.log(f"TEST_{idx}", f"REJECTED (unsupported format): {result.support_warning}", "WARNING")
        else:
            logger.log(f"TEST_{idx}", 
                      f"SUCCESS: OCR={result.ocr_success}, Vision={result.vision_success}, "
                      f"Time={result.total_time_ms}ms", "SUCCESS")

    # ── Step 5: Compile report ──────────────────────────────────────────
    total_time = int(time.time() * 1000) - start_time

    supported_count = sum(1 for r in results if r.is_supported)
    unsupported_count = total - supported_count
    ocr_success_count = sum(1 for r in results if r.ocr_success)
    vision_success_count = sum(1 for r in results if r.vision_success)
    total_ocr_tokens = sum(r.ocr_tokens for r in results)
    total_vision_tokens = sum(r.vision_tokens for r in results)

    report = BattleTestReport(
        timestamp=datetime.now().isoformat(),
        total_images=total,
        supported_images=supported_count,
        unsupported_images=unsupported_count,
        ocr_success_count=ocr_success_count,
        vision_success_count=vision_success_count,
        total_ocr_tokens=total_ocr_tokens,
        total_vision_tokens=total_vision_tokens,
        total_time_ms=total_time,
        format_stats=format_stats,
        results=[asdict(r) for r in results],
        errors=errors,
        vision_model_info=vision_info,
        ocr_model_info=ocr_info,
    )

    # ── Step 6: Save report ─────────────────────────────────────────────
    logger.log("SAVE", f"Saving report to {output_path}...", "INFO")
    try:
        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        logger.log("SAVE", f"Report saved: {output_path}", "SUCCESS")
    except Exception as e:
        logger.log("SAVE", f"Failed to save report: {e}", "ERROR")

    # ── Step 7: Summary ─────────────────────────────────────────────────
    logger.log("SUMMARY", "=" * 60, "INFO")
    logger.log("SUMMARY", f"Total images tested: {total}", "INFO")
    logger.log("SUMMARY", f"Supported: {supported_count} | Unsupported: {unsupported_count}", "INFO")
    logger.log("SUMMARY", f"OCR success: {ocr_success_count}/{supported_count}", "INFO")
    logger.log("SUMMARY", f"Vision success: {vision_success_count}/{supported_count}", "INFO")
    logger.log("SUMMARY", f"Total OCR tokens: {total_ocr_tokens}", "INFO")
    logger.log("SUMMARY", f"Total vision tokens: {total_vision_tokens}", "INFO")
    logger.log("SUMMARY", f"Total time: {total_time}ms", "INFO")
    logger.log("SUMMARY", f"Errors: {len(errors)}", "WARNING" if errors else "INFO")
    logger.log("SUMMARY", "=" * 60, "INFO")

    return report


def test_single_image(
    handler: ImageHandler,
    image_path: Path,
    stages: str,
    logger: StepLogger,
    idx: int,
    total: int,
) -> ImageTestResult:
    """Test a single image and return detailed result."""

    filename = image_path.name
    ext = image_path.suffix.lower().lstrip(".")
    size_bytes = image_path.stat().st_size if image_path.exists() else 0

    logger.log(f"TEST_{idx}", f"File: {filename}, Size: {size_bytes} bytes, Format: {ext}", "INFO")

    # Default result (will be filled in)
    result = ImageTestResult(
        filename=filename,
        filepath=str(image_path),
        format=ext,
        size_bytes=size_bytes,
        dimensions=None,
        is_supported=False,
        support_warning=None,
        stages_requested=stages,
        stages_executed=[],
        ocr_text=None,
        ocr_tokens=0,
        ocr_time_ms=0,
        ocr_success=False,
        ocr_error=None,
        vision_text=None,
        vision_tokens=0,
        vision_time_ms=0,
        vision_success=False,
        vision_error=None,
        total_time_ms=0,
        confidence=0.0,
        quality_score=0.0,
        suggested_mode="both",
        error=None,
        traceback=None,
    )

    start_ms = int(time.time() * 1000)

    try:
        # ── Step 1: Format validation ─────────────────────────────────
        logger.log(f"TEST_{idx}", "Checking format support...", "INFO")
        is_valid, warning = handler.is_supported(image_path)
        result.is_supported = is_valid
        result.support_warning = warning

        if not is_valid:
            logger.log(f"TEST_{idx}", f"Format rejected: {warning}", "WARNING")
            result.total_time_ms = int(time.time() * 1000) - start_ms
            return result

        if warning:
            logger.log(f"TEST_{idx}", f"Format warning: {warning}", "WARNING")

        # ── Step 2: Load image ────────────────────────────────────────
        logger.log(f"TEST_{idx}", "Loading image...", "INFO")
        try:
            image = handler._load_image(image_path)
            result.dimensions = image.size
            logger.log(f"TEST_{idx}", f"Loaded: {image.size[0]}x{image.size[1]}", "INFO")
        except Exception as e:
            result.error = f"Failed to load image: {e}"
            result.traceback = traceback.format_exc()
            logger.log(f"TEST_{idx}", f"Load failed: {e}", "ERROR")
            result.total_time_ms = int(time.time() * 1000) - start_ms
            return result

        # ── Step 3: Quality assessment ──────────────────────────────────
        logger.log(f"TEST_{idx}", "Assessing image quality...", "INFO")
        result.quality_score = handler._assess_quality(image)
        result.suggested_mode = handler._suggest_mode(image, result.quality_score)
        logger.log(f"TEST_{idx}", 
                  f"Quality: {result.quality_score:.2f}, Suggested: {result.suggested_mode}", "INFO")

        # ── Step 4: Process ────────────────────────────────────────────
        logger.log(f"TEST_{idx}", f"Processing with stages={stages}...", "INFO")

        process_result = handler.process(
            image_path,
            stages=stages,  # type: ignore
            auto_ensure=True,  # FIXED: allow handler to load models per-image
            stream_callback=lambda msg: logger.log(f"TEST_{idx}", f"  -> {msg}", "INFO"),
        )

        # ── Step 5: Extract results ───────────────────────────────────
        result.stages_executed = process_result.stages_executed
        result.confidence = process_result.confidence

        # OCR results
        if "ocr" in process_result.stages_executed:
            result.ocr_text = process_result.text
            result.ocr_tokens = process_result.tokens_used  # Approximate
            result.ocr_success = process_result.has_text
            result.ocr_time_ms = process_result.processing_time_ms
            if process_result.stage_results.get("ocr"):
                ocr_data = process_result.stage_results["ocr"]
                result.ocr_time_ms = ocr_data.get("time_ms", result.ocr_time_ms)
                if not ocr_data.get("success"):
                    result.ocr_error = ocr_data.get("error")
            logger.log(f"TEST_{idx}", 
                      f"OCR: {len(result.ocr_text or '')} chars, "
                      f"Success: {result.ocr_success}", 
                      "SUCCESS" if result.ocr_success else "WARNING")

        # Vision results
        if "vision" in process_result.stages_executed:
            result.vision_text = process_result.description
            result.vision_tokens = process_result.tokens_used
            result.vision_success = process_result.has_description
            result.vision_time_ms = process_result.processing_time_ms
            if process_result.stage_results.get("vision"):
                vision_data = process_result.stage_results["vision"]
                result.vision_time_ms = vision_data.get("time_ms", result.vision_time_ms)
                if not vision_data.get("success"):
                    result.vision_error = vision_data.get("error")
            logger.log(f"TEST_{idx}", 
                      f"Vision: {len(result.vision_text or '')} chars, "
                      f"Success: {result.vision_success}",
                      "SUCCESS" if result.vision_success else "WARNING")

        result.total_time_ms = int(time.time() * 1000) - start_ms
        logger.log(f"TEST_{idx}", f"Total time: {result.total_time_ms}ms", "INFO")

    except Exception as e:
        result.error = f"Unexpected error: {e}"
        result.traceback = traceback.format_exc()
        logger.log(f"TEST_{idx}", f"CRASH: {e}", "ERROR")
        result.total_time_ms = int(time.time() * 1000) - start_ms

    return result


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Battle test vision models")
    parser.add_argument("--images", "-i", type=Path, default=Path("./mock_images"),
                       help="Directory containing test images")
    parser.add_argument("--output", "-o", type=Path, default=Path("./vision_battle_test_results.json"),
                       help="Output JSON file path")
    parser.add_argument("--stages", "-s", choices=["ocr", "vision", "both"], default="both",
                       help="Processing stages to test")
    parser.add_argument("--max-images", "-m", type=int, default=None,
                       help="Maximum number of images to test")
    parser.add_argument("--timeout", "-t", type=float, default=30.0,
                       help="Timeout per image in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", default=True,
                       help="Verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Minimal logging (overrides --verbose)")

    args = parser.parse_args()

    verbose = not args.quiet

    print("=" * 70)
    print("VISION MODEL BATTLE TEST")
    print("=" * 70)
    print(f"Images directory: {args.images}")
    print(f"Output file: {args.output}")
    print(f"Stages: {args.stages}")
    print(f"Max images: {args.max_images or 'all'}")
    print(f"Timeout: {args.timeout}s")
    print("=" * 70)
    print()

    if not args.images.exists():
        print(f"ERROR: Images directory not found: {args.images}")
        sys.exit(1)

    darwin.configure()
    report = run_battle_test(
        images_dir=args.images,
        output_path=args.output,
        stages=args.stages,
        verbose=verbose,
        max_images=args.max_images,
        timeout=args.timeout,
    )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {args.output}")
    print(f"Total images: {report.total_images}")
    print(f"Supported: {report.supported_images}")
    print(f"Unsupported: {report.unsupported_images}")
    print(f"OCR success: {report.ocr_success_count}")
    print(f"Vision success: {report.vision_success_count}")
    print(f"Total time: {report.total_time_ms}ms")
    print("=" * 70)


if __name__ == "__main__":
    main()