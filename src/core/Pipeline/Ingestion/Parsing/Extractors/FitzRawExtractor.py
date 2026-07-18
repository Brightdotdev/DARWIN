

import fitz
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from src.core.Logger.DarwinLogger import darwin
from src.core.Pipeline.Ingestion.Parsing.Extractors.ExtractorUtils import make_image_slot, make_page_header
from src.core.Pipeline.Ingestion.Parsing.Extractors.ImageExtractors import extract_page_images
from src.core.Pipeline.Ingestion.Parsing.Extractors.Constants import MAX_IMAGES_PER_DOC


def fitz_raw_extract(
    pdf_path: str,
    pages: List[int],
    images_dir: Optional[str] = None,
) -> Tuple[ str, List[Dict]]:
    """
    Raw text extraction using PyMuPDF's plain text mode.
    Injects [Page N] headers before each page's text to maintain chunker compatibility.

    Returns:
        ( embedding_text, extracted_images)
    """
    doc = fitz.open(pdf_path)
    img_dir = Path(images_dir) if images_dir else None
    do_images = bool(img_dir)
    global_img_count = 0
    seen_xrefs: set = set()

    page_texts: List[str] = []
    all_extracted_images: List[Dict] = []

    # Build page label map (same logic as fitz_extract)
    label_map: Dict[int, str] = {}
    try:
        page_labels = doc.get_page_labels() or []
        for idx, entry in enumerate(page_labels):
            start = entry["startpage"]
            end = page_labels[idx + 1]["startpage"] if idx + 1 < len(page_labels) else doc.page_count
            for i in range(end - start):
                label_map[start + i] = str(entry.get("prefix", "") or (start + i + 1))
    except Exception as e:
        darwin.warning(f"Page label extraction failed, using simple numbering: {e}")

    def _page_label(idx: int) -> str:
        if label_map and idx in label_map:
            return label_map[idx]
        return str(idx + 1)

    for page_num in pages:
        if page_num >= doc.page_count:
            continue

        page = doc[page_num]
        label = _page_label(page_num)

        # Extract images only if requested
        xref_to_slot: Dict[int, int] = {}
        if do_images and global_img_count < MAX_IMAGES_PER_DOC:
            page_extracted, global_img_count, xref_to_slot = extract_page_images(
                doc, page, page_num, label, img_dir, seen_xrefs, global_img_count
            )
            all_extracted_images.extend(page_extracted)

        # Get plain text
        text = page.get_text("text").strip()

        # Insert image slots at end of page text (fast path: no position matching)
        if xref_to_slot:
            for slot in sorted(xref_to_slot.values()):
                text += f"\n\n{make_image_slot(slot)}\n\n"

        # FIX: Inject [Page N] header to maintain chunker compatibility
        if text:
            page_header = make_page_header(label)
            page_texts.append(f"{page_header}\n\n{text}")

    doc.close()

    full_text = "\n\n---\n\n".join(page_texts)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

    return full_text, all_extracted_images