"""
ParserUtils.py

Two extraction modes
--------------------
--fast    (default) _fitz_extract()
          Uses fitz get_text("dict") — raw span data we post-process.
          Heading/bold/table/caption detection via relative font sizes.
          Image extraction via fitz page.get_images().
          Speed: seconds for any PDF, including 1000+ page documents.

--precise           _pymupdf4llm_extract()
          Uses pymupdf4llm.to_markdown() — full layout reconstruction.
          Perfect column order, table formatting, image placement.
          Speed: 100-160s per 80-page chunk on CPU. Use for documents
          where layout fidelity matters more than indexing speed.

The `precise` flag threads from CLI → ParsingArgs → parse_pdf()
→ _parse_pdf_single() / _parse_pdf_chunked() → _parse_pdf_chunk().
Nothing else changes between modes.

Large PDF strategy
------------------
PDFs over LARGE_PDF_PAGE_THRESHOLD pages split into chunks processed
in parallel via ProcessPoolExecutor, glued back in page order.

Key design decisions
--------------------
- display_text  → safe to write to .md as-is
- embedding_text → clean text for vectorization, semantic tokens only
- Caption cache keys are bare filenames — portable across machines
"""

import os
import re
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable, Union, Tuple, Dict

from src.core.Parsing.Captioning import (
    caption_image,
    load_caption_model,
    caption_model_available,
    load_caption_cache,
    save_caption_cache,
)
from src.lib.Terminal import TerminalUI, Timer

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

SUPPORTED_TYPES: Dict[str, str] = {
    ".pdf":  "pdf",
    ".docx": "markitdown",
    ".pptx": "markitdown",
    ".txt":  "markitdown",
    ".md":   "markitdown",
}

CPU_COUNT          = os.cpu_count() or 4
MAX_FILE_SIZE_MB   = 500
MIN_IMAGE_SIZE_PX  = 100
MAX_IMAGES_PER_DOC = 50

PARSE_TIMEOUT_SECONDS   = 300
CAPTION_TIMEOUT_SECONDS = 30
CACHE_CAPTIONS          = True

LARGE_PDF_PAGE_THRESHOLD = 50
LARGE_PDF_CHUNK_PAGES    = 80

# _fitz_extract tuning — all ratios are relative to per-page modal font size
HEADING_SIZE_RATIO  = 1.20
CAPTION_SIZE_RATIO  = 0.85
LARGE_HEADING_RATIO = 1.50
TABLE_X_SPREAD      = 0.35
TABLE_MIN_SPANS     = 3
COLUMN_BUCKETS      = 2
TABLE_Y_GAP_RATIO   = 2.5 


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class FileWithImages:
    file:      Union[str, Path]
    image_dir: Optional[Path] = None


@dataclass
class ExtractedImage:
    path:    Path
    page:    int
    index:   int
    caption: str
    width:   int
    height:  int


@dataclass
class ParsingArgs:
    target:      str
    project_dir: Optional[str] = None
    images:      bool = False
    caption:     bool = False
    recursive:   bool = False
    rerun:       bool = False
    file:        Optional[str] = None
    precise:     bool = False         # ← new: use pymupdf4llm instead of fitz
    save : bool = True
    inspect : bool = False


@dataclass
class ParsedDocument:
    display_text:   str
    embedding_text: str
    filename:       str
    page_count:     int
    file_type:      str
    metadata:       Dict                 = field(default_factory=dict)
    error:          Optional[str]        = None
    images:         List[ExtractedImage] = field(default_factory=list)


# ============================================================================
# File helpers
# ============================================================================

def detect_type(path: Path) -> Optional[str]:
    return SUPPORTED_TYPES.get(path.suffix.lower())

def is_supported(path: Path) -> bool:
    return detect_type(path) is not None

def is_good_enough(text: str) -> bool:
    if not text or len(text.strip()) < 100:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) > 0.6

def check_file_size(path: Path) -> Optional[str]:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return (
            f"file too large: {size_mb:.0f}MB "
            f"(max {MAX_FILE_SIZE_MB}MB) — "
            f"consider splitting the document"
        )
    return None


# ============================================================================
# Caption cache
# ============================================================================

def get_caption_cache_path(images_dir: Path) -> Path:
    return images_dir / ".captions_cache.json"

def migrate_caption_cache(images_dir: Path) -> int:
    cache    = load_caption_cache(images_dir)
    migrated = {Path(k).name: v for k, v in cache.items()}
    count    = sum(1 for k in cache if Path(k).name != k)
    if count > 0:
        save_caption_cache(images_dir, migrated)
        logger.info(f"Migrated {count} caption cache keys to bare filenames")
    return count


# ============================================================================
# Path normalization
# ============================================================================

def normalize_image_path(img_path: Path, project_dir: Path) -> str:
    try:
        return img_path.relative_to(project_dir).as_posix()
    except ValueError:
        return img_path.name




def _is_table_block(all_spans: List[dict], page_w: float) -> bool:
    """
    Distinguish actual tables from multi-column text layouts.
    
    Key differences:
    - Tables: tight vertical clustering, grid-like structure, similar line counts per cell
    - Columns: continuous flowing text, variable line counts, paragraph breaks
    """
    if len(all_spans) < TABLE_MIN_SPANS:
        return False
    
    spans = [s for s in all_spans if s.get("text", "").strip()]
    if len(spans) < TABLE_MIN_SPANS:
        return False
    
    # Get bounding boxes
    xs = [s.get("bbox", [0, 0, 0, 0])[0] for s in spans]
    ys = [s.get("bbox", [0, 0, 0, 0])[1] for s in spans]
    
    x_range = (max(xs) - min(xs)) / max(page_w, 1.0)
    
    # Cluster into rows
    rows = _cluster_rows(spans)
    row_count = len(rows)
    
    if row_count < 2:
        return False
    
    # ─────────────────────────────────────────────────────────────
    # KEY DISTINCTION: Column detection vs Table detection
    # ─────────────────────────────────────────────────────────────
    
    # 1. Check for vertical gaps (paragraph breaks) - columns have them, tables don't
    y_gaps = []
    for i in range(1, len(rows)):
        prev_row_max_y = max(s.get("bbox", [0, 0, 0, 0])[3] for s in rows[i-1])
        curr_row_min_y = min(s.get("bbox", [0, 0, 0, 0])[1] for s in rows[i])
        gap = curr_row_min_y - prev_row_max_y
        if gap > 0:
            y_gaps.append(gap)
    
    # Columns have irregular large gaps (paragraphs). Tables have uniform small gaps.
    if y_gaps:
        avg_gap = sum(y_gaps) / len(y_gaps)
        max_gap = max(y_gaps)
        # If there's a gap > 2x average, it's probably column text with paragraphs
        if max_gap > avg_gap * 2.5:
            return False
    
    # 2. Check span count per row - tables are uniform, columns vary
    spans_per_row = [len(row) for row in rows]
    if spans_per_row:
        avg_spans = sum(spans_per_row) / len(spans_per_row)
        # High variance in spans per row = flowing text, not tabular data
        variance = sum((s - avg_spans) ** 2 for s in spans_per_row) / len(spans_per_row)
        if variance > avg_spans * 2:  # Too irregular to be a table
            return False
    
    # 3. Check X-position clustering - tables have clear column boundaries
    x_positions = sorted(set(round(x, 0) for x in xs))
    if len(x_positions) > 20:  # Too many X positions = scattered, not tabular
        return False
    
    # 4. Check for table-like row alignment
    # Tables: each row has spans at similar X positions
    # Columns: X positions drift as text flows
    row_x_sets = []
    for row in rows:
        row_xs = sorted(s.get("bbox", [0, 0, 0, 0])[0] for s in row)
        row_x_sets.append(tuple(round(x, 0) for x in row_xs))
    
    # Count how many rows share the same X pattern
    from collections import Counter
    pattern_counts = Counter(row_x_sets)
    most_common_count = pattern_counts.most_common(1)[0][1] if pattern_counts else 0
    
    # If < 50% of rows share the same column pattern, it's not a table
    if most_common_count < row_count * 0.5:
        return False
    
    # 5. Original checks (still useful)
    col_count = len(set(round(x, 1) for x in xs))
    
    return (
        x_range > TABLE_X_SPREAD
        and row_count >= 2
        and col_count >= 3
    )


def _cluster_rows(all_spans: List[dict], y_threshold: float = 3.0) -> List[List[dict]]:
    """
    Groups spans into rows using Y-axis proximity.
    Improved: uses span height to determine threshold dynamically.
    """
    spans = [s for s in all_spans if s.get("text", "").strip()]
    if not spans:
        return []
    
    # Sort by Y position
    spans.sort(key=lambda s: s.get("bbox", [0, 0, 0, 0])[1])
    
    # Calculate dynamic threshold based on average span height
    heights = []
    for s in spans:
        bbox = s.get("bbox", [0, 0, 0, 0])
        heights.append(bbox[3] - bbox[1])
    
    avg_height = sum(heights) / len(heights) if heights else 10
    dynamic_threshold = avg_height * 0.4  # 40% of line height
    
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
    """
    Converts clustered rows into clean structured text.
    """

    formatted_rows = []

    for row in rows:
        # sort left → right inside row
        row = sorted(row, key=lambda s: s.get("bbox", [0, 0, 0, 0])[0])

        cells = [
            s.get("text", "").strip()
            for s in row
            if s.get("text", "").strip()
        ]

        if cells:
            formatted_rows.append(" ".join(cells))

    return "\n".join(formatted_rows)
# ============================================================================
# _fitz_extract — FAST MODE
# get_text("dict") + our own post-processor. Seconds for any PDF.
# Safe in worker subprocesses — opens and closes doc itself.
# ============================================================================

def _fitz_extract(
    pdf_path:   str,
    pages:      List[int],
    images_dir: Optional[str] = None,
) -> Tuple[str, str, List[Dict]]:
    """
    Fast extraction using fitz get_text("dict").

    Returns (display_text, embedding_text, extracted_images).
    extracted_images is a list of dicts with path/page/index/width/height
    — populated only when images_dir is provided.

    Image bytes extracted via page.get_images() + fitz.Pixmap.
    CMYK pixmaps converted to RGB before saving.
    """
    import fitz

    doc        = fitz.open(pdf_path)
    img_dir    = Path(images_dir) if images_dir else None
    img_index  = 0

    display_pages:   List[str]  = []
    embedding_pages: List[str]  = []
    extracted_imgs:  List[Dict] = []

    for page_num in pages:
        if page_num >= doc.page_count:
            continue

        page      = doc[page_num]
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        page_w    = page.rect.width or 1.0

        blocks = page_dict.get("blocks", [])

        # ── 1. modal font size (body text baseline) ───────────────────────────
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

        # ── 2. reading order — multi-column aware ─────────────────────────────
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

        # interleave by y0
        all_blocks: List[dict] = []
        ti = ii = 0
        while ti < len(text_blocks) and ii < len(image_blocks):
            ty = text_blocks[ti].get("bbox",  [0, 0, 0, 0])[1]
            iy = image_blocks[ii].get("bbox", [0, 0, 0, 0])[1]
            if ty <= iy:
                all_blocks.append(text_blocks[ti]);  ti += 1
            else:
                all_blocks.append(image_blocks[ii]); ii += 1
        all_blocks.extend(text_blocks[ti:])
        all_blocks.extend(image_blocks[ii:])

        # ── 3. extract images for this page if requested ──────────────────────
        # We do this before block processing so placeholders match order.
        page_img_xrefs: List[int] = []
        if img_dir:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if img_index >= MAX_IMAGES_PER_DOC:
                    break
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:                       # CMYK → RGB
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
                    })
                    page_img_xrefs.append(xref)
                    pix = None                          # free memory
                    img_index += 1
                except Exception as e:
                    logger.warning(f"Image extraction failed page {page_num} xref {xref}: {e}")

        # ── 4. process blocks ─────────────────────────────────────────────────
        display_parts:   List[str] = []
        embedding_parts: List[str] = []
        prev_was_image             = False
        page_img_cursor            = 0     # tracks which extracted img we're on

        for block in all_blocks:
            btype = block.get("type")

            if btype == 1:
                # image block — use actual path if we extracted it
                if img_dir and page_img_cursor < len(page_img_xrefs):
                    img_info = next(
                        (i for i in extracted_imgs
                         if i.get("xref") == page_img_xrefs[page_img_cursor]),
                        None,
                    )
                    if img_info:
                        rel = img_info["path"]
                        display_parts.append(f"\n\n![image]({rel})\n\n")
                        embedding_parts.append("\n\n[Image]\n\n")
                        page_img_cursor += 1
                    else:
                        display_parts.append("\n\n[Image]\n\n")
                        embedding_parts.append("\n\n[Image]\n\n")
                else:
                    display_parts.append("\n\n[Image]\n\n")
                    embedding_parts.append("\n\n[Image]\n\n")
                prev_was_image = True
                continue

            if btype != 0:
                continue

            lines = block.get("lines", [])

            # ── table detection ───────────────────────────────────────────────
            all_spans = [s for ln in lines for s in ln.get("spans", [])]
            is_table = _is_table_block(all_spans, page_w)



            if is_table:

                rows = _cluster_rows(
                sorted(all_spans, key=lambda s: s.get("bbox", [0, 0, 0, 0])[1]))


                table_text = _format_table(rows)

                if table_text.strip():
                    display_parts.append(f"\n\n[TABLE]\n{table_text}\n\n[/TABLE]\n")
                    embedding_parts.append(f"\n\n[Table]\n{table_text}\n\n")

                prev_was_image = False
                continue
           

            # ── regular text ──────────────────────────────────────────────────
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
                    is_large = sz >= modal_size * HEADING_SIZE_RATIO
                    is_small = sz <= modal_size * CAPTION_SIZE_RATIO

                    if is_large or (is_small and prev_was_image):
                        line_d.append(text)
                        line_e.append(text)
                    elif is_bold:
                        line_d.append(f"**{text}**")
                        line_e.append(text)
                    else:
                        line_d.append(text)
                        line_e.append(text)

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

        display_pages.append("\n\n".join(p for p in display_parts   if p.strip()))
        embedding_pages.append("\n\n".join(p for p in embedding_parts if p.strip()))

    doc.close()

    sep            = "\n\n---\n\n"
    display_text   = re.sub(r"\n{3,}", "\n\n", sep.join(display_pages)).strip()
    embedding_text = re.sub(r"\n{3,}", "\n\n", sep.join(embedding_pages)).strip()

    return display_text, embedding_text, extracted_imgs


# ============================================================================
# _pymupdf4llm_extract — PRECISE MODE
# Full layout reconstruction. Slow but accurate.
# Safe in worker subprocesses.
# ============================================================================

def _pymupdf4llm_extract(
    pdf_path:   str,
    pages:      List[int],
    images_dir: Optional[str] = None,
) -> Tuple[str, str, List[Dict]]:
    """
    Precise extraction using pymupdf4llm.to_markdown().
    Handles complex layouts, multi-column, tables, precise image placement.

    Returns (display_text, embedding_text, extracted_images).
    extracted_images populated only when images_dir is provided.
    """
    import pymupdf4llm
    import re as _re

    img_dir   = Path(images_dir) if images_dir else None
    do_images = bool(img_dir)

    raw = pymupdf4llm.to_markdown(
        pdf_path,
        pages        = pages,
        write_images = do_images,
        image_path   = str(img_dir) if do_images else None,
        image_format = "png",
        page_chunks  = False,
    )

    # strip pymupdf4llm image markers from both outputs
    display_text   = _strip_pymupdf_markers(raw)
    embedding_text = _strip_pymupdf_markers(raw)

    # collect extracted images
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
                "caption": "",
                "width":   width,
                "height":  height,
            })

    return display_text, embedding_text, extracted_imgs


def _strip_pymupdf_markers(text: str) -> str:
    text = re.sub(r"\*\*==>.*?<==\*\*", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[\]\([^)]*\)\s*\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================================
# _run_captioning — shared by both modes
# Takes extracted image dicts, runs SmolVLM, updates captions in place,
# returns (display_text, embedding_text) with image markdown injected.
# ============================================================================

def _run_captioning(
    display_text:   str,
    embedding_text: str,
    extracted_imgs: List[Dict],
    images_dir:     Path,
    project_dir:    Optional[Path],
    do_caption:     bool,
    progress_callback: Optional[Callable] = None,
) -> Tuple[str, str, List[ExtractedImage]]:
    """
    Given extracted image dicts and raw display/embedding text,
    optionally caption images and inject markdown into both texts.

    For fast mode: replaces [Image] placeholders (left by _fitz_extract).
    For precise mode: appends image markdown after text (pymupdf4llm
    already placed images inline — we just add captions).
    """
    processor = model = None
    cache: Dict[str, str] = {}

    if do_caption and caption_model_available() and extracted_imgs:
        try:
            if progress_callback:
                progress_callback("Loading caption model...")
            processor, model, _ = load_caption_model()
            cache = load_caption_cache(images_dir)
        except Exception as e:
            logger.error(f"Caption model failed: {e}")
            do_caption = False

    all_images: List[ExtractedImage] = []

    for img_dict in extracted_imgs:
        img_path    = Path(img_dict["path"])
        img_caption = ""

        if do_caption and processor and model:
            img_caption = caption_image(img_path, processor, model, cache, progress_callback)
            img_dict["caption"] = img_caption

        all_images.append(ExtractedImage(
            path    = img_path,
            page    = img_dict.get("page", 0),
            index   = img_dict.get("index", 0),
            caption = img_caption,
            width   = img_dict.get("width", 0),
            height  = img_dict.get("height", 0),
        ))

        rel_path = (
            normalize_image_path(img_path, project_dir)
            if project_dir else img_path.name
        )

        display_md  = (
            f"\n\n![{img_caption}]({rel_path})\n*Figure: {img_caption}*\n\n"
            if img_caption else f"\n\n![image]({rel_path})\n\n"
        )
        embed_token = (
            f"\n\n[Image: {img_caption}]\n\n"
            if img_caption else "\n\n[Image]\n\n"
        )

        # replace placeholder if present, otherwise append
        if "[Image]" in display_text:
            display_text   = display_text.replace("[Image]",   display_md,  1)
            embedding_text = embedding_text.replace("[Image]", embed_token, 1)
        else:
            display_text   += display_md
            embedding_text += embed_token

    if do_caption and CACHE_CAPTIONS and cache:
        save_caption_cache(images_dir, cache)

    return display_text, embedding_text, all_images


# ============================================================================
# PDF chunk worker
# Top-level function — required for pickle/multiprocessing.
# ============================================================================

def _parse_pdf_chunk(
    pdf_path:    str,
    images_dir:  Optional[str],
    project_dir: Optional[str],
    pages:       List[int],
    do_images:   bool,
    do_caption:  bool,
    chunk_index: int,
    precise:     bool = False,         # ← new
) -> Dict:
    """
    Parse one page-range chunk in a worker process.
    Routes to _fitz_extract (fast) or _pymupdf4llm_extract (precise).
    Returns plain dict — must be picklable.
    """
    
    t = TerminalUI.timer(f"chunk-{chunk_index}")

    img_dir  = Path(images_dir)  if images_dir  else None
    proj_dir = Path(project_dir) if project_dir else None

    extractor = _pymupdf4llm_extract if precise else _fitz_extract

    try:
        display_text, embedding_text, extracted_imgs = extractor(
            pdf_path,
            pages,
            images_dir if do_images else None,
        )
    except Exception as e:
        return {"display_text": "", "embedding_text": "", "images": [], "error": str(e)}

    mode = "pymupdf4llm" if precise else "fitz_extract"
    t.split(mode)

    if not do_images or not img_dir:
        t.done()
        return {
            "display_text":   display_text,
            "embedding_text": embedding_text,
            "images":         [],
            "error":          None,
        }

    display_text, embedding_text, all_images = _run_captioning(
        display_text, embedding_text, extracted_imgs,
        img_dir, proj_dir, do_caption,
    )

    t.split("captioning")
    t.done()

    return {
        "display_text":   display_text,
        "embedding_text": embedding_text,
        "images":         [
            {
                "path":    str(img.path),
                "page":    img.page,
                "index":   img.index,
                "caption": img.caption,
                "width":   img.width,
                "height":  img.height,
            }
            for img in all_images
        ],
        "error": None,
    }


# ============================================================================
# parse_pdf — public entry point
# ============================================================================

def parse_pdf(
    path:              Path,
    images_dir:        Optional[Path] = None,
    caption:           bool = False,
    project_dir:       Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    precise:           bool = False,   # ← new
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    """
    Parse a PDF. Routes to fast (fitz) or precise (pymupdf4llm) mode.
    Returns (display_text, embedding_text, page_count, metadata, images)
    """
    import fitz

    t = TerminalUI.timer(f"parse_pdf:{path.name}")

    doc        = fitz.open(str(path))
    page_count = doc.page_count
    metadata   = dict(doc.metadata or {})
    doc.close()

    t.split("fitz_open")

    mode_label = "precise" if precise else "fast"
    if progress_callback:
        progress_callback(f"Parsing {path.name} ({page_count} pages) [{mode_label} mode]...")

    if page_count <= LARGE_PDF_PAGE_THRESHOLD:
        result = _parse_pdf_single(
            path, images_dir, caption, project_dir,
            progress_callback, page_count, metadata, t, precise,
        )
    else:
        n_chunks = -(-page_count // LARGE_PDF_CHUNK_PAGES)
        if progress_callback:
            progress_callback(
                f"Large PDF — {n_chunks} parallel chunks [{mode_label} mode]..."
            )
        result = _parse_pdf_chunked(
            path, images_dir, caption, project_dir,
            progress_callback, page_count, metadata, t, precise,
        )

    t.done()
    return result


def _parse_pdf_single(
    path:              Path,
    images_dir:        Optional[Path],
    caption:           bool,
    project_dir:       Optional[Path],
    progress_callback: Optional[Callable],
    page_count:        int,
    metadata:          Dict,
    t:                 Timer,
    precise:           bool = False,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    """Parse a small PDF in the current process."""

    extractor = _pymupdf4llm_extract if precise else _fitz_extract

    display_text, embedding_text, extracted_imgs = extractor(
        str(path),
        list(range(page_count)),
        str(images_dir) if images_dir else None,
    )

    mode = "pymupdf4llm" if precise else "fitz_extract"
    t.split(mode)

    if not images_dir:
        return display_text, embedding_text, page_count, metadata, []

    display_text, embedding_text, all_images = _run_captioning(
        display_text, embedding_text, extracted_imgs,
        images_dir, project_dir, caption, progress_callback,
    )

    t.split("captioning")
    return display_text, embedding_text, page_count, metadata, all_images


def _parse_pdf_chunked(
    path:              Path,
    images_dir:        Optional[Path],
    caption:           bool,
    project_dir:       Optional[Path],
    progress_callback: Optional[Callable],
    page_count:        int,
    metadata:          Dict,
    t:                 Timer,
    precise:           bool = False,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    """Split large PDF into chunks, parse in parallel, glue results."""

    all_pages = list(range(page_count))
    chunks    = [
        all_pages[i : i + LARGE_PDF_CHUNK_PAGES]
        for i in range(0, page_count, LARGE_PDF_CHUNK_PAGES)
    ]
    n_chunks  = len(chunks)
    n_workers = min(n_chunks, max(1, CPU_COUNT // 2) if caption else CPU_COUNT)

    if progress_callback:
        progress_callback(f"Using {n_workers} workers for {n_chunks} chunks...")

    chunk_results: Dict[int, Dict] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(
                _parse_pdf_chunk,
                str(path),
                str(images_dir) if images_dir else None,
                str(project_dir) if project_dir else None,
                chunk_pages,
                bool(images_dir),
                caption,
                chunk_idx,
                precise,               # ← thread through
            ): chunk_idx
            for chunk_idx, chunk_pages in enumerate(chunks)
        }

        for future in as_completed(future_to_idx):
            chunk_idx = future_to_idx[future]
            try:
                chunk_results[chunk_idx] = future.result(timeout=PARSE_TIMEOUT_SECONDS)
            except Exception as e:
                chunk_results[chunk_idx] = {
                    "display_text": "", "embedding_text": "",
                    "images": [], "error": f"chunk {chunk_idx} failed: {e}",
                }
            if progress_callback:
                progress_callback(f"Chunks complete: {len(chunk_results)}/{n_chunks}")

    t.split("parallel_parsing")

    display_parts:   List[str]            = []
    embedding_parts: List[str]            = []
    all_images:      List[ExtractedImage] = []

    for chunk_idx in range(n_chunks):
        result = chunk_results.get(chunk_idx, {})
        if result.get("error"):
            logger.warning(f"Chunk {chunk_idx} error: {result['error']}")
            continue
        display_parts.append(result.get("display_text",   ""))
        embedding_parts.append(result.get("embedding_text", ""))
        for img_dict in result.get("images", []):
            all_images.append(ExtractedImage(
                path    = Path(img_dict["path"]),
                page    = img_dict["page"],
                index   = img_dict["index"],
                caption = img_dict["caption"],
                width   = img_dict["width"],
                height  = img_dict["height"],
            ))

    display_text   = re.sub(r"\n{3,}", "\n\n", "\n\n".join(display_parts)).strip()
    embedding_text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(embedding_parts)).strip()

    t.split("Chunking Merge")

    if progress_callback:
        progress_callback(f"Glued {n_chunks} chunks — {page_count} pages complete")

    return display_text, embedding_text, page_count, metadata, all_images


# ============================================================================
# markitdown (non-PDF)
# ============================================================================

def parse_markitdown(path: Path) -> Tuple[str, int, Dict, List]:
    from markitdown import MarkItDown
    t = TerminalUI.timer(f"markitdown:{path.name}")
    result = MarkItDown().convert(str(path))
    text   = result.text_content or ""
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    word_count = len(text.split())
    page_count = max(1, round(word_count / 275))
    t.done()
    return text.strip(), page_count, {}, []