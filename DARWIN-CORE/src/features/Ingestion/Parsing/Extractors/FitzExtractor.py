# ============================================================
# src/core/Parsing/Extractors/fitz_extractor.py
# ============================================================

import fitz
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from src.core.Logger.DarwinLogger import darwin, LogType


from src.features.Ingestion.Parsing.Extractors.Constants import(
    COLUMN_BUCKETS_FALLBACK, HEADING_SIZE_RATIO, CAPTION_SIZE_RATIO,
    LARGE_HEADING_RATIO,
)
from src.features.Ingestion.Parsing.Extractors.ExtractorUtils import make_image_slot,make_page_header
from src.features.Ingestion.Parsing.Extractors.ImageExtractors import extract_page_images
from src.features.Ingestion.Parsing.Extractors.TableDetection import detect_and_format_table



def _page_modal_font_size(page_dict: dict) -> float:
    """Compute the modal (most common) font size from a page's text."""
    size_votes = Counter()
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sz = round(span.get("size", 12), 1)
                text = span.get("text", "").strip()
                if text:
                    size_votes[sz] += len(text)
    if size_votes:
        return size_votes.most_common(1)[0][0]
    return 12.0


def _detect_columns(blocks: List[dict], page_w: float) -> int:
    """
    Heuristic to detect the number of columns.
    Returns 1 if single column, 2 for two-column, etc.
    Based on large vertical gaps between x-coordinate clusters.
    """
    x_midpoints = []
    for b in blocks:
        if b.get("type") == 0:
            bbox = b["bbox"]
            x_midpoints.append((bbox[0] + bbox[2]) / 2)
    if not x_midpoints:
        return 1

    # Cluster midpoints into columns if gap > 20% of page width
    gap_threshold = page_w * 0.20
    x_midpoints.sort()
    cols = 1
    last_x = x_midpoints[0]
    for x in x_midpoints[1:]:
        if x - last_x > gap_threshold:
            cols += 1
            last_x = x
        # else same column
    return max(1, cols)


def _sort_blocks_for_reading(blocks: List[dict], columns: int, page_w: float) -> List[dict]:
    """Sort text blocks in reading order for a given column layout."""
    if columns <= 1:
        return sorted([b for b in blocks if b.get("type") == 0],
                      key=lambda b: b["bbox"][1])
    # Multi-column: group by column then sort by y
    bucket_w = page_w / columns
    col_blocks = [[] for _ in range(columns)]
    for b in blocks:
        if b.get("type") == 0:
            col_idx = min(int(b["bbox"][0] / bucket_w), columns - 1)
            col_blocks[col_idx].append(b)
    # Each column sorted by y
    for col in col_blocks:
        col.sort(key=lambda b: b["bbox"][1])
    # Interleave columns? Usually, reading order is top of left column, then top of right, etc.
    # A common approach: sort all blocks by (y // some step, x) to keep reading flow.
    # We'll use a simple approach: blocks from top to bottom, left to right.
    # That means sorting by y, then x.
    all_text = [b for b in blocks if b.get("type") == 0]
    all_text.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return all_text


def _render_text_block(
    block: dict,
    modal_size: float,
    prev_block_was_image: bool,
    page: fitz.Page,
    page_w: float,
) -> Tuple[str, str, bool]:
    """
    Render a text block into display and embedding text.
    Returns (display_str, embedding_str, is_image_caption).
    """
    lines = block.get("lines", [])
    all_spans = [s for ln in lines for s in ln.get("spans", [])]
    if not all_spans:
        return "", "", False

    # Skip single-number blocks (likely page numbers)
    if len(all_spans) == 1 and all_spans[0].get("text", "").strip().isdigit():
        return "", "", False

    # Table detection
    table_md = detect_and_format_table(page, all_spans, page_w)
    if table_md:
        # Tables are not captions
        return table_md, table_md, False

    block_display_lines = []
    block_embedding_lines = []
    is_caption_block = False

    for line in lines:
        spans = line.get("spans", [])
        if not spans:
            continue
        line_size = max(s.get("size", modal_size) for s in spans)
        line_display = ""
        line_embedding = ""

        for span in spans:
            text = span.get("text", "")
            if not text.strip():
                line_display += text
                line_embedding += text
                continue
            sz = span.get("size", modal_size)
            flags = span.get("flags", 0)
            is_bold = bool(flags & (1 << 4))

            # Heading detection
            if sz >= modal_size * LARGE_HEADING_RATIO:
                line_display += f"## {text}"
                line_embedding += f"## {text}"
            elif sz >= modal_size * HEADING_SIZE_RATIO:
                line_display += f"### {text}"
                line_embedding += f"### {text}"
            elif is_bold:
                line_display += f"**{text}**"
                line_embedding += text
            else:
                line_display += text
                line_embedding += text

        line_display = line_display.strip()
        line_embedding = line_embedding.strip()
        if not line_display:
            continue

        # Caption detection (small font after image)
        if line_size <= modal_size * CAPTION_SIZE_RATIO and prev_block_was_image:
            line_display = f"*{line_display}*"
            line_embedding = f"[Caption: {line_embedding}]"
            is_caption_block = True

        block_display_lines.append(line_display)
        block_embedding_lines.append(line_embedding)

    display = "\n".join(block_display_lines)
    embedding = "\n".join(block_embedding_lines)
    return display, embedding, is_caption_block


def fitz_extract(
    pdf_path: str,
    pages: List[int],
    images_dir: Optional[str] = None,
) -> Tuple[str, str, List[Dict]]:
    """
    Extract text and images from a PDF using PyMuPDF (fitz) with advanced layout handling.
    """
    doc = fitz.open(pdf_path)
    img_dir = Path(images_dir) if images_dir else None
    global_img_count = 0

    display_pages: List[str] = []
    embedding_pages: List[str] = []
    all_extracted_images: List[Dict] = []
    seen_xrefs: Set[int] = set()

    # Page label mapping
    label_map: Dict[int, str] = {}
    try:
        page_labels = doc.get_page_labels() or []
        for idx, entry in enumerate(page_labels):
            start = entry["startpage"]
            end = page_labels[idx + 1]["startpage"] if idx + 1 < len(page_labels) else doc.page_count
            for i in range(end - start):
                label_map[start + i] = str(entry.get("prefix", "") or (start + i + 1))
    except Exception:
        pass

    def _page_label(idx: int) -> str:
        if label_map and idx in label_map:
            return label_map[idx]
        return str(idx + 1)

    for page_num in pages:
        if page_num >= doc.page_count:
            darwin.log(f"Page index {page_num} out of range (doc has {doc.page_count} pages).", LogType.WARNING)
            continue

        page = doc[page_num]
        label = _page_label(page_num)

        # Extract images first (to have xref->slot mapping)
        page_extracted, global_img_count, xref_to_slot = extract_page_images(
            doc, page, page_num, label, img_dir, seen_xrefs, global_img_count
        )
        all_extracted_images.extend(page_extracted)

        # Get page dictionary for text
        try:
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_DEHYPHENATE)
        except Exception as e:
            darwin.log(f"Failed to extract text from page {page_num}: {e}", LogType.ERROR)
            continue

        page_w = page.rect.width or 1.0
        modal_size = _page_modal_font_size(page_dict)

        blocks = page_dict.get("blocks", [])
        text_blocks = [b for b in blocks if b.get("type") == 0]
        image_blocks = [b for b in blocks if b.get("type") == 1]

        # Detect column layout
        columns = _detect_columns(blocks, page_w)
        text_blocks = _sort_blocks_for_reading(blocks, columns, page_w)

        # Sort image blocks by visual position
        image_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

        # Interleave text and image blocks
        all_blocks = []
        ti, ii = 0, 0
        while ti < len(text_blocks) and ii < len(image_blocks):
            ty = text_blocks[ti]["bbox"][1]
            iy = image_blocks[ii]["bbox"][1]
            if ty <= iy:
                all_blocks.append(("text", text_blocks[ti])); ti += 1
            else:
                all_blocks.append(("image", image_blocks[ii])); ii += 1
        for t in text_blocks[ti:]:
            all_blocks.append(("text", t))
        for img in image_blocks[ii:]:
            all_blocks.append(("image", img))

        display_parts = []
        embedding_parts = []
        prev_block_was_image = False

        for block_type, block in all_blocks:
            if block_type == "image":
                xref = block.get("number", -1)
                slot = xref_to_slot.get(xref)
                if slot is not None:
                    slot_str = make_image_slot(slot)
                    display_parts.append(slot_str)
                    embedding_parts.append(slot_str)
                    prev_block_was_image = True
                continue

            # text block
            disp, emb, is_caption = _render_text_block(
                block, modal_size, prev_block_was_image, page, page_w
            )
            if disp:
                display_parts.append(disp)
            if emb:
                embedding_parts.append(emb)
            prev_block_was_image = is_caption  # captions don't count as image for next block

        # Ensure all images have placeholders even if not linked to a block
        for xref, slot in xref_to_slot.items():
            slot_str = make_image_slot(slot)
            if slot_str not in display_parts:
                display_parts.append(slot_str)
            if slot_str not in embedding_parts:
                embedding_parts.append(slot_str)

        page_header = make_page_header(label)
        disp_body = "\n\n".join(p.strip() for p in display_parts if p.strip())
        emb_body = "\n\n".join(p.strip() for p in embedding_parts if p.strip())

        display_pages.append(f"{page_header}\n\n{disp_body}" if disp_body else page_header)
        embedding_pages.append(f"{page_header}\n\n{emb_body}" if emb_body else page_header)

    doc.close()

    sep = "\n\n---\n\n"
    display_text = re.sub(r"\n{3,}", "\n\n", sep.join(display_pages)).strip()
    embedding_text = re.sub(r"\n{3,}", "\n\n", sep.join(embedding_pages)).strip()
    return display_text, embedding_text, all_extracted_images