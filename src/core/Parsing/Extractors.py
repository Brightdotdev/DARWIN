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

# ── Table detection thresholds (raised to reduce false positives) ──────────────
TABLE_X_SPREAD      = 0.45   # was 0.35 — require wider column spread
TABLE_MIN_SPANS     = 6      # was 3 — need more spans to call it a table
TABLE_MIN_COLS      = 3      # minimum distinct x-column positions
TABLE_MIN_ROWS      = 3      # was implicitly 2
TABLE_Y_GAP_RATIO   = 2.5
COLUMN_BUCKETS      = 2


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_pymupdf_markers(text: str, keep_images: bool = False) -> str:
    """Clean up pymupdf4llm output."""
    text = re.sub(r"\*\*==>.*?<==\*\*", "", text, flags=re.DOTALL)
    if not keep_images:
        text = re.sub(r"!\[\]\([^)]*\)\s*\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_image_slot(index: int) -> str:
    """Unique placeholder that survives concatenation and post-processing."""
    return f"[Image:{index}]"


def _make_page_header(page_label: str) -> str:
    """Explicit page boundary marker written by the extractor (not from PDF content)."""
    return f"[Page {page_label}]"


def _convert_markdown_images_to_slots(
    display_text: str,
    embedding_text: str,
    extracted_imgs: List[Dict],
) -> Tuple[str, str]:
    """Replace ![](image.png) with [Image:N] slots for consistent captioning."""
    filename_to_slot: Dict[str, int] = {}
    for img in extracted_imgs:
        img_path = Path(img["path"])
        filename_to_slot[img_path.name] = img["slot"]

    img_pattern = re.compile(r'!\[(.*?)\]\(([^)]+)\)')

    def replace_with_slot(match: re.Match) -> str:
        img_ref  = match.group(2)
        filename = Path(img_ref).name
        if filename in filename_to_slot:
            return _make_image_slot(filename_to_slot[filename])
        return match.group(0)  # leave unknown refs untouched

    display_text   = img_pattern.sub(replace_with_slot, display_text)
    embedding_text = img_pattern.sub(replace_with_slot, embedding_text)
    return display_text, embedding_text


# ─────────────────────────────────────────────────────────────────────────────
# Table detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_rows(all_spans: List[dict], y_threshold: float = 3.0) -> List[List[dict]]:
    spans = [s for s in all_spans if s.get("text", "").strip()]
    if not spans:
        return []

    spans.sort(key=lambda s: s.get("bbox", [0, 0, 0, 0])[1])

    heights = [s.get("bbox", [0, 0, 0, 0])[3] - s.get("bbox", [0, 0, 0, 0])[1] for s in spans]
    avg_height = sum(heights) / len(heights) if heights else 10.0
    dynamic_threshold = max(y_threshold, avg_height * 0.4)

    rows: List[List[dict]] = []
    for span in spans:
        y = span.get("bbox", [0, 0, 0, 0])[1]
        placed = False
        for row in rows:
            row_y = sum(s.get("bbox", [0, 0, 0, 0])[1] for s in row) / len(row)
            if abs(y - row_y) <= dynamic_threshold:
                row.append(span)
                placed = True
                break
        if not placed:
            rows.append([span])

    return rows


def _is_table_block(all_spans: List[dict], page_w: float) -> bool:
    """
    Heuristic table detector. Tuned conservatively to avoid flagging
    multi-column body text as tables.
    """
    if len(all_spans) < TABLE_MIN_SPANS:
        return False

    spans = [s for s in all_spans if s.get("text", "").strip()]
    if len(spans) < TABLE_MIN_SPANS:
        return False

    xs = [s.get("bbox", [0, 0, 0, 0])[0] for s in spans]
    ys = [s.get("bbox", [0, 0, 0, 0])[1] for s in spans]

    x_range = (max(xs) - min(xs)) / max(page_w, 1.0)
    if x_range < TABLE_X_SPREAD:
        return False  # ← early exit, not wide enough

    rows = _cluster_rows(spans)
    row_count = len(rows)
    if row_count < TABLE_MIN_ROWS:
        return False

    # Reject if row gaps are inconsistent (indicates paragraph breaks, not table rows)
    y_gaps = []
    for i in range(1, len(rows)):
        prev_max_y = max(s.get("bbox", [0, 0, 0, 0])[3] for s in rows[i - 1])
        curr_min_y = min(s.get("bbox", [0, 0, 0, 0])[1] for s in rows[i])
        gap = curr_min_y - prev_max_y
        if gap > 0:
            y_gaps.append(gap)

    if y_gaps:
        avg_gap = sum(y_gaps) / len(y_gaps)
        max_gap = max(y_gaps)
        if max_gap > avg_gap * TABLE_Y_GAP_RATIO:
            return False  # large gap → paragraph break, not table row

    # Reject if span count per row varies wildly (prose lines vary; table rows don't)
    spans_per_row = [len(row) for row in rows]
    avg_spans = sum(spans_per_row) / len(spans_per_row)
    if avg_spans < 2:
        return False  # single-column content isn't a table
    variance = sum((s - avg_spans) ** 2 for s in spans_per_row) / len(spans_per_row)
    if variance > avg_spans * 2:
        return False

    # Require consistent x-column alignment across rows
    # (bucket x positions to account for minor jitter)
    BUCKET = 5.0  # points
    def bucket_x(x: float) -> int:
        return int(x / BUCKET)

    row_x_sets = [
        tuple(sorted(bucket_x(s.get("bbox", [0, 0, 0, 0])[0]) for s in row))
        for row in rows
    ]
    pattern_counts = Counter(row_x_sets)
    most_common_count = pattern_counts.most_common(1)[0][1] if pattern_counts else 0
    if most_common_count < row_count * 0.5:
        return False  # rows don't share a column pattern → not a table

    # Final: require enough distinct columns
    col_positions = set(bucket_x(x) for x in xs)
    if len(col_positions) < TABLE_MIN_COLS:
        return False

    return True


def _format_table(rows: List[List[dict]]) -> str:
    formatted_rows = []
    for row in rows:
        row_sorted = sorted(row, key=lambda s: s.get("bbox", [0, 0, 0, 0])[0])
        cells = [s.get("text", "").strip() for s in row_sorted if s.get("text", "").strip()]
        if cells:
            formatted_rows.append(" | ".join(cells))  # pipe-separated for clarity
    return "\n".join(formatted_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Core fitz extractor
# ─────────────────────────────────────────────────────────────────────────────

def fitz_extract(
    pdf_path: str,
    pages: List[int],
    images_dir: Optional[str] = None,
) -> Tuple[str, str, List[Dict]]:
    """
    Extract text and images from a PDF using PyMuPDF (fitz).

    Args:
        pdf_path:   Path to the PDF file.
        pages:      0-based page indices to extract.
        images_dir: Directory to save extracted images. If None, images are skipped.

    Returns:
        (display_text, embedding_text, extracted_imgs)
        Image placeholders in both texts use [Image:N] slots.
        Page boundaries use [Page N] headers based on PDF page labels.
    """
    import fitz

    doc     = fitz.open(pdf_path)
    img_dir = Path(images_dir) if images_dir else None
    img_index = 0

    display_pages:   List[str] = []
    embedding_pages: List[str] = []
    extracted_imgs:  List[Dict] = []

    # Build a label map: page index → PDF page label (e.g. "iv", "1", "A-3")
    # fitz exposes this via doc.get_page_labels() when available, otherwise fall back to 1-based index.
    try:
        label_map: Dict[int, str] = {
            entry["startpage"] + i: str(entry.get("prefix", "") or (entry["startpage"] + i + 1))
            for entry in (doc.get_page_labels() or [])
            for i in range(
                0,
                (
                    next(
                        (e["startpage"] for e in doc.get_page_labels()
                         if e["startpage"] > entry["startpage"]),
                        doc.page_count,
                    ) - entry["startpage"]
                ),
            )
        }
    except Exception:
        label_map = {}

    def _page_label(idx: int) -> str:
        """Return a human-readable page label for page index `idx`."""
        if label_map and idx in label_map:
            return label_map[idx]
        # Fallback: use 1-based index so "page 1" matches what readers see
        return str(idx + 1)

    seen_xrefs: set = set()  # global dedup across pages

    for page_num in pages:
        if page_num >= doc.page_count:
            TerminalUI.warn(f"Page index {page_num} out of range (doc has {doc.page_count} pages), skipping.")
            continue

        page      = doc[page_num]
        # TEXT_DEHYPHENATE helps with hyphenated words split across lines;
        # drop TEXT_PRESERVE_WHITESPACE which can corrupt spacing in some PDFs.
        page_dict = page.get_text(
            "dict",
            flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_DEHYPHENATE,
        )
        page_w = page.rect.width or 1.0
        label  = _page_label(page_num)

        blocks = page_dict.get("blocks", [])

        # ── Modal (body) font size for this page ─────────────────────────────
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

        # Sort text blocks: column-first then top-to-bottom
        if COLUMN_BUCKETS > 1 and text_blocks:
            bucket_w    = page_w / COLUMN_BUCKETS
            text_blocks = sorted(
                text_blocks,
                key=lambda b: (
                    min(int(b.get("bbox", [0])[0] / bucket_w), COLUMN_BUCKETS - 1),
                    b.get("bbox", [0, 0])[1],
                ),
            )
        else:
            text_blocks = sorted(text_blocks, key=lambda b: b.get("bbox", [0, 0])[1])

        image_blocks = sorted(image_blocks, key=lambda b: b.get("bbox", [0, 0])[1])

        # Merge text + image blocks in reading order
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

        # ── Image extraction ─────────────────────────────────────────────────
        page_img_start_index = img_index
        page_img_xrefs: List[int] = []   # xrefs actually saved this page

        if img_dir:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue  # skip duplicate embedded image
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
                        "label":   label,
                        "index":   img_index,
                        "caption": "",
                        "width":   pix.width,
                        "height":  pix.height,
                        "xref":    xref,
                        "slot":    img_index,
                    })
                    page_img_xrefs.append(xref)
                    seen_xrefs.add(xref)
                    pix = None
                    img_index += 1
                except Exception as e:
                    TerminalUI.warn(f"Image extraction failed page {page_num} xref {xref}: {e}")

        # ── Render page content ──────────────────────────────────────────────
        display_parts:   List[str] = []
        embedding_parts: List[str] = []
        page_img_cursor = 0

        # Track whether the *immediately preceding rendered block* was an image,
        # so we can style the next line as a caption. Reset per-block.
        prev_block_was_image = False

        for block in all_blocks:
            btype = block.get("type")

            # ── Image block ──────────────────────────────────────────────────
            if btype == 1:
                if img_dir and page_img_cursor < len(page_img_xrefs):
                    slot_index = page_img_start_index + page_img_cursor
                    slot = _make_image_slot(slot_index)
                    display_parts.append(slot)
                    embedding_parts.append(slot)
                    page_img_cursor += 1
                    prev_block_was_image = True
                else:
                    # Image skipped (no dir, or over limit) — don't emit a slot
                    pass
                continue

            if btype != 0:
                continue

            # ── Text block ───────────────────────────────────────────────────
            lines     = block.get("lines", [])
            all_spans = [s for ln in lines for s in ln.get("spans", [])]

            # Skip blocks that are clearly just a lone page number printed by the PDF.
            # These are single short numeric spans near the page top or bottom.
            if (
                len(all_spans) == 1
                and all_spans[0].get("text", "").strip().isdigit()
            ):
                # Suppress: the extractor emits its own [Page N] header instead.
                prev_block_was_image = False
                continue

            is_table = _is_table_block(all_spans, page_w)

            if is_table:
                rows       = _cluster_rows(all_spans)
                table_text = _format_table(rows)
                if table_text.strip():
                    display_parts.append(f"[TABLE]\n{table_text}\n[/TABLE]")
                    embedding_parts.append(f"[Table]\n{table_text}")
                prev_block_was_image = False
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
                    text = span.get("text", "")
                    if not text.strip():
                        line_d.append(text)
                        line_e.append(text)
                        continue

                    sz      = span.get("size", modal_size)
                    flags   = span.get("flags", 0)
                    is_bold = bool(flags & (1 << 4))

                    if sz >= modal_size * HEADING_SIZE_RATIO:
                        # Headings: no inline markdown — handled at line level below
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
                elif line_size <= modal_size * CAPTION_SIZE_RATIO and prev_block_was_image:
                    # Caption: only applies to the first text block after an image
                    line_display   = f"*{line_display}*"
                    line_embedding = f"[Caption: {line_embedding}]"

                block_display.append(line_display)
                block_embedding.append(line_embedding)

            if block_display:
                display_parts.append("\n".join(block_display))
                embedding_parts.append("\n".join(block_embedding))
                # Only the very first text block after an image gets caption treatment
                prev_block_was_image = False

        # Append any images not interleaved in text (e.g. full-page images)
        for i in range(page_img_cursor, len(page_img_xrefs)):
            slot_index = page_img_start_index + i
            slot = _make_image_slot(slot_index)
            display_parts.append(slot)
            embedding_parts.append(slot)

        # ── Assemble page with explicit header ───────────────────────────────
        page_header = _make_page_header(label)

        # Join parts with a single blank line; parts themselves don't carry
        # leading/trailing newlines any more — we normalise here.
        def _join_parts(parts: List[str]) -> str:
            cleaned = [p.strip() for p in parts if p.strip()]
            return "\n\n".join(cleaned)

        display_body   = _join_parts(display_parts)
        embedding_body = _join_parts(embedding_parts)

        display_pages.append(f"{page_header}\n\n{display_body}" if display_body else page_header)
        embedding_pages.append(f"{page_header}\n\n{embedding_body}" if embedding_body else page_header)

    doc.close()

    sep            = "\n\n---\n\n"
    display_text   = re.sub(r"\n{3,}", "\n\n", sep.join(display_pages)).strip()
    embedding_text = re.sub(r"\n{3,}", "\n\n", sep.join(embedding_pages)).strip()

    return display_text, embedding_text, extracted_imgs


# ─────────────────────────────────────────────────────────────────────────────
# pymupdf4llm extractor
# ─────────────────────────────────────────────────────────────────────────────

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

    display_text   = _strip_pymupdf_markers(raw, keep_images=do_images)
    embedding_text = _strip_pymupdf_markers(raw, keep_images=do_images)

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

    if do_images and extracted_imgs:
        display_text, embedding_text = _convert_markdown_images_to_slots(
            display_text,
            embedding_text,
            extracted_imgs,
        )

    return display_text, embedding_text, extracted_imgs