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
    darwin.info("Initializing OCR...")

    ocr = modelManager.get(ModelCategory.OCR)
    if ocr is None:
        darwin.error("Failed to initialize OCR")
        return

    darwin.info("OCR initialized successfully")

    # Discover images from /mock_images
    mock_dir = Path("mock_images")
    if not mock_dir.exists():
        darwin.error(f"Mock images directory not found: {mock_dir}")
        return

    # Supported image extensions
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    image_files = sorted([
        f for f in mock_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if not image_files:
        darwin.warning(f"No images found in {mock_dir}")
        return

    darwin.info(f"Found {len(image_files)} image(s) in {mock_dir}")
    darwin.info("=" * 70)

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
            darwin.info(log_msg)

            results.append({
                "file": filepath.name,
                "extracted": extracted_text,
                "success": True
            })
        except Exception as e:
            darwin.error(f"Error on {filepath.name}: {e}")
            results.append({
                "file": filepath.name,
                "extracted": f"ERROR: {e}",
                "success": False
            })

    # Summary
    darwin.info("=" * 70)
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    darwin.info(f"SUMMARY: {successful}/{total} images processed successfully")

    # Save results to file
    result_file = Path("ocr_test_results.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("OCR TEST RESULTS\n")
        f.write("=" * 70 + "\n")
        for r in results:
            status = "PASS" if r["success"] else "FAIL"
            f.write(f"\n{status}: {r['file']}\n")
            f.write(f"  Extracted:\n{r['extracted']}\n")
    darwin.info(f"Detailed results saved to: {result_file}")


def main():
    """Main entry point."""
    darwin.configure()
    darwin.info("Starting OCR Test Script")
    darwin.info("=" * 70)

    try:
        asyncio.run(test_ocr())
    except KeyboardInterrupt:
        darwin.warning("Test interrupted by user")
    except Exception as e:
        darwin.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    darwin.info("OCR Test Complete")


if __name__ == "__main__":
    main()