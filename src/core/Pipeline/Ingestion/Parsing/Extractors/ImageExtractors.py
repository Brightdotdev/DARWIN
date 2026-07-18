import fitz
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

from src.core.Logger.DarwinLogger import darwin
from src.core.Pipeline.Ingestion.Parsing.Extractors.Constants import (
    MIN_IMAGE_SIZE_PX, MAX_IMAGES_PER_PAGE, MAX_IMAGES_PER_DOC,
)
from src.core.Pipeline.Ingestion.Parsing.Extractors.ExtractorUtils import make_image_slot


def _is_image_xref(doc: fitz.Document, xref: int) -> bool:
    try:
        obj = doc.xref_object(xref)
        if "/Subtype /Image" not in obj:
            return False
        if "/ImageMask true" in obj:
            return False
        return True
    except Exception as e:
        return False


def extract_page_images(
    doc: fitz.Document,
    page: fitz.Page,
    page_num: int,
    label: str,
    img_dir: Optional[Path],
    seen_xrefs: Set[int],
    global_img_count: int,
) -> Tuple[List[Dict], int, Dict[int, int]]:
    """
    Extract all valid, unique images from a page.

    Returns:
      - List of image metadata dicts (newly saved images).
      - Updated global image count.
      - Mapping from xref -> slot index.
    """
    extracted: List[Dict] = []
    xref_to_slot: Dict[int, int] = {}

    # FIX #1: Early return when no output directory requested
    if not img_dir:
        return extracted, global_img_count, xref_to_slot

    img_infos = page.get_image_info()
    img_infos.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    for info in img_infos:
        xref = info["number"]

        if xref in seen_xrefs:
            continue

        if not _is_image_xref(doc, xref):
            continue

        if global_img_count >= MAX_IMAGES_PER_DOC or len(extracted) >= MAX_IMAGES_PER_PAGE:
            break

        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            # FIX #2: Capture dimensions BEFORE any potential reassignment
            img_w, img_h = pix.width, pix.height

            if img_w < MIN_IMAGE_SIZE_PX or img_h < MIN_IMAGE_SIZE_PX:
                pix = None
                continue

            slot = global_img_count
            img_path = img_dir / f"page{page_num}_img{slot}.png"
            pix.save(str(img_path))
            pix = None  # release after save

            extracted.append({
                "path":    str(img_path),
                "page":    page_num,
                "label":   label,
                "index":   slot,
                "caption": "",
                "width":   img_w,
                "height":  img_h,
                "xref":    xref,
                "slot":    slot,
            })

            seen_xrefs.add(xref)
            xref_to_slot[xref] = slot
            global_img_count += 1

        except Exception as e:
            darwin.warning(f"Failed to extract image xref {xref} on page {page_num}: {e}")
            continue

    return extracted, global_img_count, xref_to_slot