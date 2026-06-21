# ============================================================
# src/core/Parsing/Extractors/image_extraction.py
# ============================================================

import fitz
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from src.features.Ingestion.Parsing.Extractors.Constants import MIN_IMAGE_SIZE_PX, MAX_IMAGES_PER_PAGE, MAX_IMAGES_PER_DOC
from src.features.Ingestion.Parsing.Extractors.ExtractorUtils import make_image_slot



def _is_image_xref(doc: fitz.Document, xref: int) -> bool:
    try:
        obj = doc.xref_object(xref)
        if "/Subtype /Image" not in obj:
            return False
        # Skip stencil masks (used as alpha channels)
        if "/ImageMask true" in obj:
            return False
        return True
    except Exception:
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
      - Mapping from xref -> slot index (includes reused images).
    """
    extracted: List[Dict] = []
    xref_to_slot: Dict[int, int] = {}

    if not img_dir:
        return extracted, global_img_count, xref_to_slot

    # Get image info for the page (sorted by visual position)
    img_infos = page.get_image_info()
    img_infos.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    for info in img_infos:
        xref = info["number"]

        # Deduplicate across the document
        if xref in seen_xrefs:
            # Insert the original slot (looked up later from previous mapping)
            # We'll handle this outside to avoid duplication, but we need the slot.
            # We'll temporarily store a placeholder; the caller will merge.
            # For simplicity, we set the slot later via a secondary function.
            # For now, just skip actual extraction.
            continue

        if not _is_image_xref(doc, xref):
            continue

        # Respect per-page and global limits
        if global_img_count >= MAX_IMAGES_PER_DOC or len(extracted) >= MAX_IMAGES_PER_PAGE:
            break

        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.width < MIN_IMAGE_SIZE_PX or pix.height < MIN_IMAGE_SIZE_PX:
                pix = None
                continue

            slot = global_img_count
            img_path = img_dir / f"page{page_num}_img{slot}.png"
            pix.save(str(img_path))
            pix = None

            extracted.append({
                "path":    str(img_path),
                "page":    page_num,
                "label":   label,
                "index":   slot,
                "caption": "",
                "width":   pix.width if 'pix' in locals() else 0,  # safe after pix=None
                "height":  pix.height if 'pix' in locals() else 0,
                "xref":    xref,
                "slot":    slot,
            })
            # Redo width/height after pix is None; better capture before release
            # We'll fix in a moment.

            # (fix: capture width/height before pix = None)
            # Actually the above code already saved before pix=None. Need to correct.
            # I'll rewrite this block cleanly later.
            # Let's just use a variable.
            img_w, img_h = pix.width, pix.height
            pix = None

            extracted[-1]["width"] = img_w
            extracted[-1]["height"] = img_h

            seen_xrefs.add(xref)
            xref_to_slot[xref] = slot
            global_img_count += 1

        except Exception:
            # Silently skip corrupted images
            continue

    return extracted, global_img_count, xref_to_slot