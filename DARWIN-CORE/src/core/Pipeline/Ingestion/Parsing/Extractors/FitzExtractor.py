import fitz
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image as PILImage  # imported once

import pymupdf4llm

from src.core.Pipeline.Ingestion.Parsing.Extractors.ExtractorUtils import (
    make_page_header,
    strip_pymupdf_markers,
    convert_markdown_images_to_slots,
)
from src.core.Pipeline.Ingestion.Parsing.Extractors.Constants import (
    MAX_IMAGES_PER_DOC,
    MIN_IMAGE_SIZE_PX,
)

# ── Caption heuristic ────────────────────────────────────────────────────────
_CAPTION_RE = re.compile(r'^(Figure|Fig\.?|Table|Tbl\.?)\b', re.IGNORECASE)


def _apply_rag_semantics(md_text: str) -> Tuple[str, str]:
    """
    Split markdown into display and embedding text with custom RAG semantics.

    Rules
    -----
    - Headings (# …)          → identical in both
    - Bold (**text**)         → keep in display, strip in embedding
    - Italic (*text*) right after an [Image:N] on its own line
                               → display: *text*  |  embedding: [Caption: text]
    - Text starting with Figure/Table right after an [Image:N]
                               → same caption treatment
    - Tables (|…|)             → identical in both
    - Image slots [Image:N]    → identical in both
    - Everything else          → identical in both
    """
    lines = md_text.split('\n')
    display_lines: List[str] = []
    embedding_lines: List[str] = []

    prev_was_image = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Preserve empty lines
        if not stripped:
            display_lines.append(line)
            embedding_lines.append(line)
            i += 1
            continue

        # Headings
        if re.match(r'^#{1,6}\s', stripped):
            display_lines.append(line)
            embedding_lines.append(line)
            prev_was_image = False
            i += 1
            continue

        # Image slot on its own line → triggers caption hunting on next line
        if re.match(r'^\[Image:\d+\]$', stripped):
            display_lines.append(line)
            embedding_lines.append(line)
            prev_was_image = True
            i += 1
            continue

        # Inline image slot → do NOT trigger caption detection
        if '[Image:' in stripped:
            display_lines.append(line)
            embedding_lines.append(line)
            prev_was_image = False
            i += 1
            continue

        # Table rows / separators
        if stripped.startswith('|'):
            display_lines.append(line)
            embedding_lines.append(line)
            prev_was_image = False
            i += 1
            continue

        # Caption detection: the first non-empty line after a standalone image slot
        if prev_was_image:
            is_italic = (
                stripped.startswith('*') and stripped.endswith('*')
                and not stripped.startswith('**') and not stripped.endswith('**')
            )
            looks_like_caption = bool(_CAPTION_RE.match(stripped))

            if is_italic or looks_like_caption:
                caption_text = stripped[1:-1] if is_italic else stripped
                display_lines.append(f"*{caption_text}*")
                embedding_lines.append(f"[Caption: {caption_text}]")
                prev_was_image = False
                i += 1
                continue

        # Bold handling
        if '**' in line:
            display_lines.append(line)
            embedding_lines.append(re.sub(r'\*\*(.*?)\*\*', r'\1', line))
            prev_was_image = False
            i += 1
            continue

        # Default pass-through
        display_lines.append(line)
        embedding_lines.append(line)
        prev_was_image = False
        i += 1

    return '\n'.join(display_lines), '\n'.join(embedding_lines)


def fitz_extract(
    pdf_path: str,
    pages: List[int],
    images_dir: Optional[str] = None,
) -> Tuple[str, str, List[Dict]]:
    """
    UNIFIED precise extraction.

    Uses pymupdf4llm for layout-aware markdown (tables, columns, reading order),
    then applies custom RAG post-processing:
      - Display vs Embedding text split
      - [Image:N] slot injection
      - Caption detection
      - Bold preservation in display, stripping in embedding
      - Page headers for chunker compatibility

    Backward‑compatible fallback: if the installed pymupdf4llm does not support
    `page_chunks`, we process each requested page individually and then merge
    the results.

    Returns
    -------
    (display_text, embedding_text, extracted_images)
    """
    img_dir = Path(images_dir) if images_dir else None
    do_images = bool(img_dir)

    # ── 1. Extract structured markdown per page ──────────────────────────────
    # Try with page_chunks first (newer versions)
    try:
        raw_result = pymupdf4llm.to_markdown(
            pdf_path,
            pages=pages,
            write_images=do_images,
            image_path=str(img_dir) if do_images else None,
            image_format="png",
            page_chunks=True,
            show_progress=False,
        )
        is_chunked = True
    except Exception:
        # Fallback: older versions that do not accept page_chunks
        # We will call to_markdown for each page individually
        raw_result = []
        for p in pages:
            # We need to extract one page at a time; pymupdf4llm accepts pages as list of ints
            # but if we pass a single page, we get a string for that page.
            # However, we want to keep images per page – we need separate image directories?
            # We'll use a temporary subfolder per page? Simpler: extract all pages at once
            # without page_chunks, but then we lose page separation.
            # Better: extract each page separately with a unique image subfolder.
            # For simplicity and speed, we fall back to extracting the whole document
            # as a single chunk but we will use page numbers to create headers later.
            # But we cannot split the markdown by page reliably.
            # Instead, we use a more robust fallback: extract each page one by one
            # with a temporary image directory per page, then concatenate.
            # To avoid excessive I/O, we'll extract all pages in one go without page_chunks
            # and then treat the whole as a single page, accepting loss of per-page headers.
            # However, that would break the page header logic.
            # We'll do the per‑page extraction, which is correct but slower.
            # We'll create a temporary subdirectory for each page's images.
            import tempfile
            import shutil
            page_results = []
            for page_num in pages:
                with tempfile.TemporaryDirectory() as tmpdir:
                    page_md = pymupdf4llm.to_markdown(
                        pdf_path,
                        pages=[page_num],
                        write_images=do_images,
                        image_path=tmpdir if do_images else None,
                        image_format="png",
                        page_chunks=False,
                        show_progress=False,
                    )
                    # Collect images from tmpdir
                    page_images = []
                    if do_images:
                        for img_path in Path(tmpdir).glob("*.png"):
                            page_images.append(str(img_path))
                    page_results.append({
                        "text": page_md,
                        "images": page_images,
                        "page": page_num,
                    })
            # Now we have a list of dicts similar to page_chunks structure
            # We'll convert to the expected format for downstream processing
            raw_result = [
                {
                    "text": r["text"],
                    "images": [{"path": p} for p in r["images"]],
                    "metadata": {"page": r["page"]}
                }
                for r in page_results
            ]
            is_chunked = True  # we have per-page chunks now

    # Normalise return type: if it's a string (old version without page_chunks and we didn't catch),
    # we convert to a single chunk.
    if isinstance(raw_result, str):
        page_chunks = [{"text": raw_result, "images": [], "metadata": {}}]
        is_chunked = False
    else:
        page_chunks = raw_result
        # Ensure each entry has the expected keys
        for chunk in page_chunks:
            if "images" not in chunk:
                chunk["images"] = []
            if "metadata" not in chunk:
                chunk["metadata"] = {}

    # ── 2. Page label map (for headers) ──────────────────────────────────────
    label_map: Dict[int, str] = {}
    try:
        with fitz.open(pdf_path) as doc:
            page_labels = doc.get_page_labels() or []
            for idx, entry in enumerate(page_labels):
                start = entry["startpage"]
                end = page_labels[idx + 1]["startpage"] if idx + 1 < len(page_labels) else doc.page_count
                for i in range(end - start):
                    label_map[start + i] = str(entry.get("prefix", "") or (start + i + 1))
            # If page_labels is empty, fallback to page numbers
            if not label_map:
                for i in range(doc.page_count):
                    label_map[i] = str(i + 1)
    except Exception:
        # If anything fails, use simple numbering
        label_map = {i: str(i + 1) for i in range(len(pages))}

    def _page_label(idx: int) -> str:
        # If we have a label for this index, use it; otherwise fallback to idx+1
        return label_map.get(idx, str(idx + 1))

    # ── 3. Pre-scan images directory for fallback correlation ──────────────────
    dir_images_by_page: Dict[int, List[Path]] = {}
    if do_images and img_dir and img_dir.exists():
        for img_path in sorted(img_dir.glob("*.png")):
            m = re.search(r'page[_-]?(\d+)', img_path.name, re.IGNORECASE)
            if m:
                dir_images_by_page.setdefault(int(m.group(1)), []).append(img_path)

    # ── 4. Process each page chunk ─────────────────────────────────────────────
    display_pages: List[str] = []
    embedding_pages: List[str] = []
    all_extracted_images: List[Dict] = []
    global_slot = 0

    # Determine page numbers: either from the pages list or from metadata
    # We'll iterate over chunks and try to get page from metadata, else fallback to pages list.
    for chunk_idx, chunk in enumerate(page_chunks):
        # Get page number from metadata if available, else from pages list
        meta_page = chunk.get("metadata", {}).get("page")
        if meta_page is not None:
            page_num = meta_page
        elif chunk_idx < len(pages):
            page_num = pages[chunk_idx]
        else:
            page_num = chunk_idx  # fallback

        label = _page_label(page_num)

        raw_md = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        raw_md = strip_pymupdf_markers(raw_md, keep_images=do_images)

        # ── Collect images for this page ─────────────────────────────────────
        page_images: List[Dict] = []

        # 4a. Try chunk metadata first
        chunk_imgs = chunk.get("images", []) if isinstance(chunk, dict) else []
        for img_info in chunk_imgs:
            if global_slot >= MAX_IMAGES_PER_DOC:
                break
            if isinstance(img_info, dict):
                img_path = Path(img_info.get("path", ""))
            else:
                img_path = Path(str(img_info))
            if not img_path.exists() and img_dir:
                img_path = img_dir / img_path.name
            if not img_path.exists():
                continue

            try:
                with PILImage.open(img_path) as pil_img:
                    w, h = pil_img.size
            except Exception:
                w, h = 0, 0

            if w < MIN_IMAGE_SIZE_PX or h < MIN_IMAGE_SIZE_PX:
                continue

            page_images.append({
                "path": str(img_path),
                "page": page_num,
                "index": global_slot,
                "slot": global_slot,
                "caption": "",
                "width": w,
                "height": h,
            })
            global_slot += 1

        # 4b. Fallback: directory glob if chunk metadata was empty
        if not page_images and page_num in dir_images_by_page:
            for img_path in dir_images_by_page[page_num]:
                if global_slot >= MAX_IMAGES_PER_DOC:
                    break
                try:
                    with PILImage.open(img_path) as pil_img:
                        w, h = pil_img.size
                except Exception:
                    continue
                if w < MIN_IMAGE_SIZE_PX or h < MIN_IMAGE_SIZE_PX:
                    continue

                page_images.append({
                    "path": str(img_path),
                    "page": page_num,
                    "index": global_slot,
                    "slot": global_slot,
                    "caption": "",
                    "width": w,
                    "height": h,
                })
                global_slot += 1

        # ── Convert markdown images → [Image:N] slots ────────────────────────
        if do_images and page_images:
            md_with_slots, _ = convert_markdown_images_to_slots(raw_md, raw_md, page_images)
        else:
            md_with_slots = raw_md

        # ── Apply display / embedding semantic split ─────────────────────────
        page_display, page_embedding = _apply_rag_semantics(md_with_slots)

        # ── Inject page header ─────────────────────────────────────────────────
        page_header = make_page_header(label)
        display_pages.append(
            f"{page_header}\n\n{page_display.strip()}" if page_display.strip() else page_header
        )
        embedding_pages.append(
            f"{page_header}\n\n{page_embedding.strip()}" if page_embedding.strip() else page_header
        )

        all_extracted_images.extend(page_images)

    # ── 5. Final assembly ──────────────────────────────────────────────────────
    sep = "\n\n---\n\n"
    display_text = re.sub(r"\n{3,}", "\n\n", sep.join(display_pages)).strip()
    embedding_text = re.sub(r"\n{3,}", "\n\n", sep.join(embedding_pages)).strip()

    return display_text, embedding_text, all_extracted_images  # fixed variable name