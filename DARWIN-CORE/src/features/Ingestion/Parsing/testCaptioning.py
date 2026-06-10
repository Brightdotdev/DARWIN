#!/usr/bin/env python3
"""
test_captioning.py
Standalone test harness for the Darwin image captioning pipeline.

Loops over an input directory of images, feeds them through the production
_run_captioning() path (including model loading, caching, and slot replacement),
and writes the resulting markdown + caption manifest to an output directory.

Usage:
    python test_captioning.py -i ./my_images -o ./captioned_output
    python test_captioning.py -i ./my_images -o ./captioned_output --no-caption
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# ── Ensure project root is importable when run standalone ──────────────────────
if __name__ == "__main__":
    # If script lives at repo root, project root is here.
    # If script lives in tests/, project root is parent.
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir, script_dir.parent):
        if (candidate / "src").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break

from PIL import Image as PILImage

from src.features.Ingestion.Parsing.PdfExtractors import _run_captioning
from src.features.Ingestion.Parsing.lib import ExtractedImage
from src.core.Logger.DarwinLogger import darwin, LogType, Verbosity


# ── Helpers ────────────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def discover_images(input_dir: Path) -> List[Path]:
    """Find all image files in input directory, sorted alphabetically."""
    images = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    return images


def build_extracted_imgs(images: List[Path]) -> Tuple[List[Dict], str, str]:
    """
    Build extracted_imgs dicts and dummy display/embedding text with slots.
    """
    extracted: List[Dict] = []
    display_parts: List[str] = []
    embed_parts: List[str] = []

    for idx, img_path in enumerate(images):
        try:
            with PILImage.open(img_path) as pil_img:
                width, height = pil_img.size
        except Exception as e:
            print(f"  ⚠️  Could not read {img_path.name}: {e}")
            width, height = 0, 0

        extracted.append({
            "path":    str(img_path),
            "page":    0,
            "index":   idx,
            "slot":    idx,
            "caption": "",
            "width":   width,
            "height":  height,
        })

        slot = f"[Image:{idx}]"
        display_parts.append(slot)
        embed_parts.append(slot)

    display_text = "\n\n".join(display_parts)
    embedding_text = "\n\n".join(embed_parts)

    return extracted, display_text, embedding_text


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test Darwin image captioning pipeline against a folder of images."
    )
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        type=Path,
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        type=Path,
        help="Directory to write markdown, JSON manifest, and copied images.",
    )
    parser.add_argument(
        "--no-caption",
        action="store_true",
        help="Skip model captioning (test slot replacement / path resolution only).",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"❌ Input directory not found: {args.input_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Minimal logger config so _run_captioning logs are visible in terminal
    darwin.configure(
        enable_websocket=False,
        enable_history=False,
        verbosity_default=Verbosity.NORMAL,
        show_banner=False,
    )

    

    from src.core.Models import ModelManager
    ModelManager.setup()


    images = discover_images(args.input_dir)
    if not images:
        print(f"⚠️  No images found in {args.input_dir} (supported: {', '.join(sorted(SUPPORTED_EXTS))})")
        return 1

    print(f"📁 Found {len(images)} image(s) in {args.input_dir}:")
    for img in images:
        print(f"   • {img.name}")

    extracted_imgs, display_text, embedding_text = build_extracted_imgs(images)

    # We pass output_dir as images_dir so:
    #   1) .captions_cache.json is written there
    #   2) normalize_image_path falls back to images/<name> (good because we copy images below)
    print("\n🚀 Running _run_captioning()…")
    try:
        result_display, result_embed, all_images = _run_captioning(
            display_text=display_text,
            embedding_text=embedding_text,
            extracted_imgs=extracted_imgs,
            images_dir=args.output_dir,
            do_caption=not args.no_caption,
        )
    except Exception as e:
        print(f"\n❌ Captioning failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # ── Write outputs ──────────────────────────────────────────────────────
    display_path = args.output_dir / "output_display.md"
    embed_path   = args.output_dir / "output_embed.md"
    json_path    = args.output_dir / "captions.json"

    display_path.write_text(result_display, encoding="utf-8")
    embed_path.write_text(result_embed, encoding="utf-8")

    manifest = {
        "meta": {
            "input_dir": str(args.input_dir),
            "image_count": len(images),
            "captioned": not args.no_caption,
        },
        "captions": {
            img.path.name: img.caption for img in all_images
        },
    }
    json_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Copy images into output_dir/images/ so markdown relative paths resolve
    images_subdir = args.output_dir / "images"
    images_subdir.mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(img, images_subdir / img.name)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n  Results written to {args.output_dir}:")
    print(f"   • {display_path.name}   (display markdown)")
    print(f"   • {embed_path.name}     (embedding markdown)")
    print(f"   • {json_path.name}      (caption manifest)")
    print(f"   • images/               ({len(images)} copied images)")

    if all_images:
        print(f"\n📊 Caption summary:")
        for img in all_images:
            short = img.caption[:70] + "…" if len(img.caption) > 70 else img.caption
            status = " " if img.caption else "❌ (empty / failed)"
            print(f"   {status} {img.path.name:<30} → {short}")

    cache_file = args.output_dir / ".captions_cache.json"
    if cache_file.exists():
        print(f"\n💾 Cache file: {cache_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())