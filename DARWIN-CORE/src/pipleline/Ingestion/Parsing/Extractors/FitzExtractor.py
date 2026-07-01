import fitz
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pymupdf4llm

from src.pipleline.Ingestion.Parsing.Extractors.ExtractorUtils import (
    make_page_header,
    strip_pymupdf_markers,
    convert_markdown_images_to_slots,
)
from src.pipleline.Ingestion.Parsing.Extractors.Constants import (
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

    Returns
    -------
    (display_text, embedding_text, extracted_images)
    """
    img_dir = Path(images_dir) if images_dir else None
    do_images = bool(img_dir)

    # ── 1. Extract structured markdown per page ──────────────────────────────
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
    except Exception:
        # Fallback for older pymupdf4llm versions that don't support page_chunks
        raw_result = pymupdf4llm.to_markdown(
            pdf_path,
            pages=pages,
            write_images=do_images,
            image_path=str(img_dir) if do_images else None,
            image_format="png",
            page_chunks=False,
            show_progress=False,
        )

    # Normalise return type
    if isinstance(raw_result, str):
        page_chunks = [{"text": raw_result, "images": [], "metadata": {}}]
    else:
        page_chunks = raw_result

    # ── 2. Page label map (for headers) ──────────────────────────────────────
    doc = fitz.open(pdf_path)
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
    doc.close()

    def _page_label(idx: int) -> str:
        return label_map[idx] if label_map and idx in label_map else str(idx + 1)

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

    for chunk_idx, chunk in enumerate(page_chunks):
        page_num = pages[chunk_idx] if chunk_idx < len(pages) else chunk_idx
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
            img_path = Path(img_info.get("path", "")) if isinstance(img_info, dict) else Path(str(img_info))
            if not img_path.exists() and img_dir:
                img_path = img_dir / img_path.name
            if not img_path.exists():
                continue

            try:
                from PIL import Image as PILImage
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
                    from PIL import Image as PILImage
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

    return display_text, embedding_text, all_extracted_imagess