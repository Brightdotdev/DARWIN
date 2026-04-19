import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from src.lib.Terminal import TerminalUI






MIN_IMAGE_SIZE_PX = 100
MAX_IMAGES_PER_DOC = 50

HEADING_SIZE_RATIO  = 1.20
CAPTION_SIZE_RATIO  = 0.85
LARGE_HEADING_RATIO = 1.50
TABLE_X_SPREAD      = 0.35
TABLE_MIN_SPANS     = 3
COLUMN_BUCKETS      = 2
TABLE_Y_GAP_RATIO   = 2.5


def _strip_pymupdf_markers(text: str, keep_images: bool = False) -> str:
    """Clean up pymupdf4llm output.
    
    Args:
        text: Raw markdown text
        keep_images: If True, preserve ![](image.png) references for later processing
    """
    # Always remove debug markers
    text = re.sub(r"\*\*==>.*?<==\*\*", "", text, flags=re.DOTALL)
    
    # Only strip empty images if we don't need them
    if not keep_images:
        text = re.sub(r"!\[\]\([^)]*\)\s*\n", "", text)
    
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_image_slot(index: int) -> str:
    """Unique placeholder that survives concatenation and post-processing."""
    return f"[Image:{index}]"

def _convert_markdown_images_to_slots(
    display_text: str,
    embedding_text: str,
    extracted_imgs: List[Dict],
) -> Tuple[str, str]:
    """Replace ![](image.png) with [Image:N] slots for consistent captioning."""
    
    # Build mapping from filename to slot index
    filename_to_slot = {}
    for img in extracted_imgs:
        img_path = Path(img["path"])
        filename_to_slot[img_path.name] = img["slot"]
    
    # Pattern to match markdown images: ![](filename.png) or ![alt](filename.png)
    img_pattern = re.compile(r'!\[(.*?)\]\(([^)]+)\)')
       
    def replace_with_slot(match):
        alt_text = match.group(1)
        img_ref = match.group(2)
        
        # Extract just the filename from the path
        filename = Path(img_ref).name
        
        if filename in filename_to_slot:
            slot_idx = filename_to_slot[filename]
            
            return _make_image_slot(slot_idx)
        else:
            return match.group(0)
    
    display_text = img_pattern.sub(replace_with_slot, display_text)
    embedding_text = img_pattern.sub(replace_with_slot, embedding_text)
    
    return display_text, embedding_text




# ── Table detection helpers ────────────────────────────────────────────────────

def _is_table_block(all_spans: List[dict], page_w: float) -> bool:
    if len(all_spans) < TABLE_MIN_SPANS:
        return False

    spans = [s for s in all_spans if s.get("text", "").strip()]
    if len(spans) < TABLE_MIN_SPANS:
        return False

    xs = [s.get("bbox", [0, 0, 0, 0])[0] for s in spans]
    ys = [s.get("bbox", [0, 0, 0, 0])[1] for s in spans]

    x_range = (max(xs) - min(xs)) / max(page_w, 1.0)
    rows = _cluster_rows(spans)
    row_count = len(rows)

    if row_count < 2:
        return False

    y_gaps = []
    for i in range(1, len(rows)):
        prev_row_max_y = max(s.get("bbox", [0, 0, 0, 0])[3] for s in rows[i - 1])
        curr_row_min_y = min(s.get("bbox", [0, 0, 0, 0])[1] for s in rows[i])
        gap = curr_row_min_y - prev_row_max_y
        if gap > 0:
            y_gaps.append(gap)

    if y_gaps:
        avg_gap = sum(y_gaps) / len(y_gaps)
        max_gap = max(y_gaps)
        if max_gap > avg_gap * 2.5:
            return False

    spans_per_row = [len(row) for row in rows]
    if spans_per_row:
        avg_spans = sum(spans_per_row) / len(spans_per_row)
        variance = sum((s - avg_spans) ** 2 for s in spans_per_row) / len(spans_per_row)
        if variance > avg_spans * 2:
            return False

    x_positions = sorted(set(round(x, 0) for x in xs))
    if len(x_positions) > 20:
        return False

    row_x_sets = []
    for row in rows:
        row_xs = sorted(s.get("bbox", [0, 0, 0, 0])[0] for s in row)
        row_x_sets.append(tuple(round(x, 0) for x in row_xs))

    pattern_counts = Counter(row_x_sets)
    most_common_count = pattern_counts.most_common(1)[0][1] if pattern_counts else 0

    if most_common_count < row_count * 0.5:
  
  
        return False

    col_count = len(set(round(x, 1) for x in xs))

    result = x_range > TABLE_X_SPREAD and row_count >= 2 and col_count >= 3
    if not result:
        return False
    return result


def _cluster_rows(all_spans: List[dict], y_threshold: float = 3.0) -> List[List[dict]]:
    spans = [s for s in all_spans if s.get("text", "").strip()]
    if not spans:
        return []

    spans.sort(key=lambda s: s.get("bbox", [0, 0, 0, 0])[1])

    heights = []
    for s in spans:
        bbox = s.get("bbox", [0, 0, 0, 0])
        heights.append(bbox[3] - bbox[1])

    avg_height = sum(heights) / len(heights) if heights else 10
    dynamic_threshold = avg_height * 0.4

    rows: List[List[dict]] = []

    for span in spans:
        y = span.get("bbox", [0, 0, 0, 0])[1]
        placed = False

        for row in rows:
            row_y = sum(s.get("bbox", [0, 0, 0, 0])[1] for s in row) / len(row)
            if abs(y - row_y) <= max(y_threshold, dynamic_threshold):
                row.append(span)
                placed = True
                break

        if not placed:
            rows.append([span])

    return rows


def _format_table(rows: List[List[dict]]) -> str:
    formatted_rows = []

    for row in rows:
        row = sorted(row, key=lambda s: s.get("bbox", [0, 0, 0, 0])[0])
        cells = [
            s.get("text", "").strip()
            for s in row
            if s.get("text", "").strip()
        ]
        if cells:
            formatted_rows.append(" ".join(cells))

    return "\n".join(formatted_rows)



# ── Core fitz extractor ────────────────────────────────────────────────────────
def fitz_extract(
    pdf_path: str,
    pages: List[int],
    images_dir: Optional[str] = None,
) -> Tuple[str, str, List[Dict]]:
    """
    Returns (display_text, embedding_text, extracted_imgs).
    Image placeholders in the text use positional [Image:N] slots.
    """
    import fitz

    doc     = fitz.open(pdf_path)
    img_dir = Path(images_dir) if images_dir else None
    img_index = 0

    display_pages:  List[str]  = []
    embedding_pages: List[str] = []
    extracted_imgs: List[Dict] = []

    for page_num in pages:
        if page_num >= doc.page_count:
            continue

        page      = doc[page_num]
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        page_w    = page.rect.width or 1.0

        blocks = page_dict.get("blocks", [])

        # ── modal font size ──────────────────────────────────────────────────
        size_votes: Counter = Counter()
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sz  = round(span.get("size", 12), 1)
                    cnt = len(span.get("text", "").strip())
                    if cnt > 0:
                        size_votes[sz] += cnt

        modal_size: float = size_votes.most_common(1)[0][0] if size_votes else 12.0

        text_blocks  = [b for b in blocks if b.get("type") == 0]
        image_blocks = [b for b in blocks if b.get("type") == 1]

        if COLUMN_BUCKETS > 1 and text_blocks:
            bucket_w    = page_w / COLUMN_BUCKETS
            text_blocks = sorted(
                text_blocks,
                key=lambda b: (
                    min(int(b.get("bbox", [0])[0] / bucket_w), COLUMN_BUCKETS - 1),
                    b.get("bbox", [0, 0])[1],
                )
            )
        else:
            text_blocks = sorted(text_blocks, key=lambda b: b.get("bbox", [0, 0])[1])

        image_blocks = sorted(image_blocks, key=lambda b: b.get("bbox", [0, 0])[1])

        # merge text + image blocks in reading order
        all_blocks: List[dict] = []
        ti = ii = 0
        while ti < len(text_blocks) and ii < len(image_blocks):
            ty = text_blocks[ti].get("bbox",  [0, 0, 0, 0])[1]
            iy = image_blocks[ii].get("bbox", [0, 0, 0, 0])[1]
            if ty <= iy:
                all_blocks.append(text_blocks[ti]); ti += 1
            else:
                all_blocks.append(image_blocks[ii]); ii += 1
        all_blocks.extend(text_blocks[ti:])
        all_blocks.extend(image_blocks[ii:])

        # ── image extraction ─────────────────────────────────────────────────
        # FIX: Save the starting index for this page's images
        page_img_start_index = img_index
        page_img_xrefs: List[int] = []
        
        if img_dir:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if img_index >= MAX_IMAGES_PER_DOC:
                    break
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    if pix.width < MIN_IMAGE_SIZE_PX or pix.height < MIN_IMAGE_SIZE_PX:
                        pix = None
                        continue
                    img_path = img_dir / f"page{page_num}_img{img_index}.png"
                    pix.save(str(img_path))
                    extracted_imgs.append({
                        "path":    str(img_path),
                        "page":    page_num,
                        "index":   img_index,
                        "caption": "",
                        "width":   pix.width,
                        "height":  pix.height,
                        "xref":    xref,
                        "slot":    img_index,   # Global slot index
                    })
                    page_img_xrefs.append(xref)
                    pix = None
                    img_index += 1
                except Exception as e:
                    TerminalUI.warn(f"Image extraction failed page {page_num} xref {xref}: {e}")

        # ── render page content ──────────────────────────────────────────────
        display_parts:  List[str] = []
        embedding_parts: List[str] = []
        prev_was_image = False
        page_img_cursor = 0

        for block in all_blocks:
            btype = block.get("type")

            if btype == 1:
                # Use positional slot so captioning can replace exactly this slot later
                if img_dir and page_img_cursor < len(page_img_xrefs):
                    # FIX: Use the saved starting index to calculate correct slot
                    slot_index = page_img_start_index + page_img_cursor
                    slot = _make_image_slot(slot_index)
                    display_parts.append(f"\n\n{slot}\n\n")
                    embedding_parts.append(f"\n\n{slot}\n\n")
                    page_img_cursor += 1
                else:
                    display_parts.append("\n\n[Image]\n\n")
                    embedding_parts.append("\n\n[Image]\n\n")
                prev_was_image = True
                continue

            if btype != 0:
                continue

            lines     = block.get("lines", [])
            all_spans = [s for ln in lines for s in ln.get("spans", [])]
            is_table  = _is_table_block(all_spans, page_w)

            if is_table:
                rows       = _cluster_rows(sorted(all_spans, key=lambda s: s.get("bbox", [0, 0, 0, 0])[1]))
                table_text = _format_table(rows)
                if table_text.strip():
                    display_parts.append(f"\n\n[TABLE]\n{table_text}\n\n[/TABLE]\n")
                    embedding_parts.append(f"\n\n[Table]\n{table_text}\n\n")
                prev_was_image = False
                continue

            block_display:   List[str] = []
            block_embedding: List[str] = []

            for line in lines:
                spans = line.get("spans", [])
                if not spans:
                    continue

                line_size = max(
                    (s.get("size", modal_size) for s in spans),
                    default=modal_size,
                )

                line_d: List[str] = []
                line_e: List[str] = []

                for span in spans:
                    text  = span.get("text", "")
                    if not text.strip():
                        line_d.append(text); line_e.append(text)
                        continue

                    sz       = span.get("size", modal_size)
                    flags    = span.get("flags", 0)
                    is_bold  = bool(flags & (1 << 4))
                    is_large = sz >= modal_size * HEADING_SIZE_RATIO
                    is_small = sz <= modal_size * CAPTION_SIZE_RATIO

                    if is_large or (is_small and prev_was_image):
                        line_d.append(text); line_e.append(text)
                    elif is_bold:
                        line_d.append(f"**{text}**"); line_e.append(text)
                    else:
                        line_d.append(text); line_e.append(text)

                line_display   = "".join(line_d).strip()
                line_embedding = "".join(line_e).strip()
                if not line_display:
                    continue

                if line_size >= modal_size * LARGE_HEADING_RATIO:
                    line_display   = f"## {line_display}"
                    line_embedding = f"## {line_embedding}"
                elif line_size >= modal_size * HEADING_SIZE_RATIO:
                    line_display   = f"### {line_display}"
                    line_embedding = f"### {line_embedding}"
                elif line_size <= modal_size * CAPTION_SIZE_RATIO and prev_was_image:
                    line_display   = f"*{line_display}*"
                    line_embedding = f"[Caption: {line_embedding}]"

                block_display.append(line_display)
                block_embedding.append(line_embedding)

            if block_display:
                display_parts.append("\n".join(block_display))
                embedding_parts.append("\n".join(block_embedding))
                prev_was_image = False

        # FIX: Handle any images that weren't placed in text blocks
        if page_img_cursor < len(page_img_xrefs):
            # Add remaining images at the end of the page
            for i in range(page_img_cursor, len(page_img_xrefs)):
                slot_index = page_img_start_index + i
                slot = _make_image_slot(slot_index)
                display_parts.append(f"\n\n{slot}\n\n")
                embedding_parts.append(f"\n\n{slot}\n\n")
                
        display_pages.append("\n\n".join(p for p in display_parts  if p.strip()))
        embedding_pages.append("\n\n".join(p for p in embedding_parts if p.strip()))

    doc.close()

    sep            = "\n\n---\n\n"
    display_text   = re.sub(r"\n{3,}", "\n\n", sep.join(display_pages)).strip()
    embedding_text = re.sub(r"\n{3,}", "\n\n", sep.join(embedding_pages)).strip()

    return display_text, embedding_text, extracted_imgs


# ── pymupdf4llm extractor ──────────────────────────────────────────────────────

def pymupdf4llm_extract(
    pdf_path: str,
    pages: List[int],
    images_dir: Optional[str] = None,
) -> Tuple[str, str, List[Dict]]:
    import pymupdf4llm

    img_dir   = Path(images_dir) if images_dir else None
    do_images = bool(img_dir)

    raw = pymupdf4llm.to_markdown(
        pdf_path,
        pages=pages,
        write_images=do_images,
        image_path=str(img_dir) if do_images else None,
        image_format="png",
        page_chunks=False,
        show_progress=True,
    )

    # FIX: Don't strip image references - we need them for captioning
    display_text   = _strip_pymupdf_markers(raw, keep_images=True)
    embedding_text = _strip_pymupdf_markers(raw, keep_images=True)

    extracted_imgs: List[Dict] = []
    if img_dir and img_dir.exists():
        for idx, img_path in enumerate(sorted(img_dir.glob("*.png"))):
            if idx >= MAX_IMAGES_PER_DOC:
                break
            try:
                from PIL import Image as PILImage
                with PILImage.open(img_path) as pil_img:
                    width, height = pil_img.size
            except Exception:
                width, height = 0, 0

            extracted_imgs.append({
                "path":    str(img_path),
                "page":    pages[0] if pages else 0,
                "index":   idx,
                "slot":    idx,
                "caption": "",
                "width":   width,
                "height":  height,
            })
    
    # FIX: Convert markdown images to slots for consistent captioning
    if do_images and extracted_imgs:
        display_text, embedding_text = _convert_markdown_images_to_slots(
            display_text, embedding_text, extracted_imgs, img_dir
        )


    return display_text, embedding_text, extracted_imgs