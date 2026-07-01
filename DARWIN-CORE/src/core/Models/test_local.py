"""
Simple OCR Test Script
Reads test images from /mock_images and runs them through the OCR model.
"""

import os
import sys
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import asyncio

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent  # Adjust based on script location
sys.path.insert(0, str(project_root))

from src.core.Models.ModelManager import modelManager, ModelCategory
from src.core.Logger.DarwinLogger import darwin, LogType


def normalize_text(t):
    """Normalize text for fuzzy comparison."""
    return re.sub(r"\s+", " ", t.strip().lower())


async def test_ocr():
    """Test OCR on images from /mock_images."""

    # Initialize OCR
    darwin.log("Initializing OCR...", LogType.INFO)

    ocr = modelManager.get(ModelCategory.OCR)
    if ocr is None:
        darwin.log("Failed to initialize OCR", LogType.ERROR)
        return

    darwin.log("OCR initialized successfully", LogType.INFO)

    # Discover images from /mock_images
    mock_dir = Path("mock_images")
    if not mock_dir.exists():
        darwin.log(f"Mock images directory not found: {mock_dir}", LogType.ERROR)
        return

    # Supported image extensions
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    image_files = sorted([
        f for f in mock_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if not image_files:
        darwin.log(f"No images found in {mock_dir}", LogType.WARNING)
        return

    darwin.log(f"Found {len(image_files)} image(s) in {mock_dir}", LogType.INFO)
    darwin.log("=" * 70, LogType.INFO)

    # Test each image
    results = []
    for filepath in image_files:
        # Read image
        image = Image.open(filepath)

        # Run OCR (layout preserved for document-like images)
        try:
            extracted_text = await ocr.extract_text(image, preserve_layout=True)

            # Log result
            log_msg = f"📄 {filepath.name}:"
            log_msg += f"\n  Extracted:\n{extracted_text}"
            darwin.log(log_msg, LogType.INFO)

            results.append({
                "file": filepath.name,
                "extracted": extracted_text,
                "success": True
            })
        except Exception as e:
            darwin.log(f"  Error on {filepath.name}: {e}", LogType.ERROR)
            results.append({
                "file": filepath.name,
                "extracted": f"ERROR: {e}",
                "success": False
            })

    # Summary
    darwin.log("=" * 70, LogType.INFO)
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    darwin.log(f"SUMMARY: {successful}/{total} images processed successfully", LogType.INFO)

    # Save results to file
    result_file = Path("ocr_test_results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("OCR TEST RESULTS\n")
        f.write("=" * 70 + "\n")
        for r in results:
            status = "PASS" if r["success"] else "FAIL"
            f.write(f"\n{status}: {r['file']}\n")
            f.write(f"  Extracted:\n{r['extracted']}\n")
    darwin.log(f"\nDetailed results saved to: {result_file}", LogType.INFO)


def main():
    """Main entry point."""
    darwin.configure()
    darwin.log("Starting OCR Test Script", LogType.INFO)
    darwin.log("=" * 70, LogType.INFO)

    try:
        asyncio.run(test_ocr())
    except KeyboardInterrupt:
        darwin.log("Test interrupted by user", LogType.WARNING)
    except Exception as e:
        darwin.log(f"Test failed with error: {e}", LogType.ERROR)
        import traceback
        traceback.print_exc()

    darwin.log("\nOCR Test Complete", LogType.INFO)


if __name__ == "__main__":
    main()