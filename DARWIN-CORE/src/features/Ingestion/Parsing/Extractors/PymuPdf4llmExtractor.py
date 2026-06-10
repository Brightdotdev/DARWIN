# ============================================================
# src/core/Parsing/Extractors/pymupdf4llm_extractor.py
# ============================================================

import pymupdf4llm
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from src.features.Ingestion.Parsing.Extractors.ExtractorUtils import strip_pymupdf_markers,convert_markdown_images_to_slots
from src.features.Ingestion.Parsing.Extractors.Constants import MIN_IMAGE_SIZE_PX, MAX_IMAGES_PER_DOC




def pymupdf4llm_extract(
    pdf_path: str,
    pages: List[int],
    images_dir: Optional[str] = None,
) -> Tuple[str, str, List[Dict]]:
    """
    Extract using pymupdf4llm (default backend). Images are written to disk and
    mapped to slots.
    """
    img_dir = Path(images_dir) if images_dir else None
    do_images = bool(img_dir)

    raw = pymupdf4llm.to_markdown(
        pdf_path,
        pages=pages,
        write_images=do_images,
        image_path=str(img_dir) if do_images else None,
        image_format="png",
        page_chunks=False,
        show_progress=False,
    )

    display_text = strip_pymupdf_markers(raw, keep_images=do_images)
    embedding_text = strip_pymupdf_markers(raw, keep_images=do_images)

    extracted_imgs: List[Dict] = []
    if img_dir and img_dir.exists():
        from PIL import Image as PILImage
        img_files = sorted(img_dir.glob("*.png"))
        slot = 0
        for img_path in img_files:
            if slot >= MAX_IMAGES_PER_DOC:
                break
            try:
                with PILImage.open(img_path) as pil_img:
                    w, h = pil_img.size
                if w < MIN_IMAGE_SIZE_PX or h < MIN_IMAGE_SIZE_PX:
                    continue
                extracted_imgs.append({
                    "path": str(img_path),
                    "page": pages[0] if pages else 0,
                    "index": slot,
                    "slot": slot,
                    "caption": "",
                    "width": w,
                    "height": h,
                })
                slot += 1
            except Exception:
                continue

    if do_images and extracted_imgs:
        display_text, embedding_text = convert_markdown_images_to_slots(
            display_text, embedding_text, extracted_imgs
        )

    return display_text, embedding_text, extracted_imgs