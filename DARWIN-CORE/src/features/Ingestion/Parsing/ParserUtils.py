"""
ParserUtils.py — PDF extraction utilities with fast and precise modes.

Data-flow contract
──────────────────
Extractor  →  [Image:N] slots + [Page N] headers in raw text
ParserUtils →  NO captioning here; only extraction.
Captioning is done later in the main process.
"""

import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from src.core.Logger.DarwinLogger import darwin, LogType

from src.features.Ingestion.Parsing.lib import (
    ExtractedImage,
    MIN_TEXT_LENGTH,
    ASCII_RATIO_THRESHOLD,
    MAX_FILE_SIZE_MB,
    NON_LATIN_ASCII_FLOOR,
    IMAGE_SLOT_RE,
    LARGE_PDF_CHUNK_PAGES,
    LARGE_PDF_PAGE_THRESHOLD
)
from src.features.Ingestion.Parsing.PDFExtractors import fitz_extract, pymupdf4llm_extract

# ── Helpers ────────────────────────────────────────────────────────────────────

def is_good_enough(text: str) -> bool:
    """Returns True if the extracted text looks usable."""
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False

    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_chars / len(text)

    if ascii_ratio < NON_LATIN_ASCII_FLOOR:
        printable = sum(1 for c in text if c.isprintable() and not c.isspace())
        ok = printable > MIN_TEXT_LENGTH
        if not ok:
            darwin.log(f"is_good_enough: non-latin doc rejected (printable={printable})")
        return ok

    ok = ascii_ratio > ASCII_RATIO_THRESHOLD
    if not ok:
        darwin.log(f"is_good_enough: rejected (ascii_ratio={ascii_ratio:.2f}, threshold={ASCII_RATIO_THRESHOLD})")
    return ok


def normalize_image_path(img_path: Path, images_dir: Path = None) -> str:
    """Return a relative path for display from the markdown file's perspective."""
    if images_dir:
        markdown_dir = images_dir.parent
        try:
            return img_path.relative_to(markdown_dir).as_posix()
        except ValueError:
            pass
        return f"images/{img_path.name}"
    return img_path.name


def _sanitize_caption(caption: str) -> str:
    """Collapse whitespace and strip characters that break markdown image syntax."""
    if not caption:
        return ""
    caption = " ".join(caption.split())
    caption = caption.replace("[", "").replace("]", "")
    return caption.strip()


def check_file_size(path: Path) -> Optional[str]:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return f"file too large: {size_mb:.0f}MB (max {MAX_FILE_SIZE_MB}MB)"
    return None


# ── Image slot replacement (used in main process after captioning) ─────────────

def replace_image_slots(
    display_text:   str,
    embedding_text: str,
    slot_map:       Dict[int, Dict],
) -> Tuple[str, str]:
    """Replace [Image:N] placeholders with final display/embedding tokens."""
    def _display(m: re.Match) -> str:
        idx  = int(m.group(1))
        info = slot_map.get(idx)
        if not info:
            return "[Image]"
        caption  = info.get("caption", "")
        rel_path = info.get("rel_path", info.get("path", ""))
        clean = _sanitize_caption(caption)
        if clean:
            return f"\n\n![{clean}]({rel_path})\n\n"
        return f"\n\n![image]({rel_path})\n\n"

    
    def _embed(m: re.Match) -> str:
        idx  = int(m.group(1))
        info = slot_map.get(idx)
        if not info:
            return "[Image]"
        caption = info.get("caption", "")
        clean = _sanitize_caption(caption)
        if clean:
            return f"\n\n[Image: {clean}]\n\n"   # ← marker added
        return "\n\n[Image]\n\n"

    display_text   = IMAGE_SLOT_RE.sub(_display,   display_text)
    embedding_text = IMAGE_SLOT_RE.sub(_embed, embedding_text)
    return display_text, embedding_text


def _remap_image_slots(
    display_text:   str,
    embedding_text: str,
    offset:         int,
) -> Tuple[str, str]:
    """Shift all [Image:N] slot indices in text by `offset`."""
    if offset == 0:
        return display_text, embedding_text

    def _shift(m: re.Match) -> str:
        return f"[Image:{int(m.group(1)) + offset}]"

    display_text   = IMAGE_SLOT_RE.sub(_shift, display_text)
    embedding_text = IMAGE_SLOT_RE.sub(_shift, embedding_text)
    return display_text, embedding_text


# ── PDF chunk worker (no captioning) ──────────────────────────────────────────

def _parse_pdf_chunk(
    pdf_path:    str,
    images_dir:  Optional[str],
    pages:       List[int],
    do_images:   bool,
    precise:     bool = False,
) -> Dict[str, Any]:
    """
    Extract one contiguous page range from a PDF.
    No captioning – only text and raw image extraction.
    """
    extractor = pymupdf4llm_extract if precise else fitz_extract

    try:
        display_text, embedding_text, extracted_imgs = extractor(
            pdf_path,
            pages,
            images_dir if do_images else None,
        )
    except Exception as e:
        return {
            "display_text":   "",
            "embedding_text": "",
            "images":         [],
            "img_count":      0,
            "error":          str(e),
        }

    return {
        "display_text":   display_text,
        "embedding_text": embedding_text,
        "images":         extracted_imgs,
        "img_count":      len(extracted_imgs),
        "error":          None,
    }


def _parse_pdf_single(
    path:        Path,
    images_dir:  Optional[Path],
    caption:     bool,          # ignored – kept for compatibility
    page_count:  int,
    metadata:    Dict,
    precise:     bool = False,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    """
    Parse a small PDF (single process, no chunking).
    Captioning is NOT performed here.
    """
    extractor = pymupdf4llm_extract if precise else fitz_extract

    display_text, embedding_text, extracted_imgs = extractor(
        str(path),
        list(range(page_count)),
        str(images_dir) if images_dir else None,
    )

    mode = "pymupdf4llm" if precise else "fitz_extract"
    darwin.log(f"Extraction mode: {mode}")

    all_images: List[ExtractedImage] = []
    if images_dir and extracted_imgs:
        # Keep images for later, don't caption yet
        for img_dict in extracted_imgs:
            img_path = Path(img_dict["path"])
            all_images.append(ExtractedImage(
                path=img_path,
                page=img_dict.get("page", 0),
                index=img_dict.get("index", 0),
                caption="",   # will be filled later
                width=img_dict.get("width", 0),
                height=img_dict.get("height", 0),
            ))

    return display_text, embedding_text, page_count, metadata, all_images


def _parse_pdf_chunked(
    path:        Path,
    images_dir:  Optional[Path],
    caption:     bool,          # ignored
    page_count:  int,
    metadata:    Dict,
    precise:     bool = False,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    """
    Extract a large PDF sequentially (no inner pool).
    Captioning NOT performed.
    """
    all_pages   = list(range(page_count))
    page_chunks = [
        all_pages[i : i + LARGE_PDF_CHUNK_PAGES]
        for i in range(0, page_count, LARGE_PDF_CHUNK_PAGES)
    ]

    n_chunks = len(page_chunks)
    darwin.log(f"Processing {n_chunks} chunks sequentially...")

    display_parts   = []
    embedding_parts = []
    all_raw_imgs    = []
    slot_offset = 0

    for chunk_idx, chunk_pages in enumerate(page_chunks):
        result = _parse_pdf_chunk(
            str(path),
            str(images_dir) if images_dir else None,
            chunk_pages,
            bool(images_dir),
            precise,
        )
        if result.get("error"):
            darwin.log(f"Chunk {chunk_idx} error: {result['error']}", LogType.WARNING)
            continue

        d_text = result.get("display_text", "")
        e_text = result.get("embedding_text", "")
        imgs   = result.get("images", [])

        # Remap slots to avoid index collisions across chunks
        if slot_offset > 0 and imgs:
            d_text, e_text = _remap_image_slots(d_text, e_text, slot_offset)

        for img_dict in imgs:
            adjusted = dict(img_dict)
            if "slot" in adjusted:
                adjusted["slot"]  = adjusted["slot"]  + slot_offset
            if "index" in adjusted:
                adjusted["index"] = adjusted["index"] + slot_offset
            all_raw_imgs.append(adjusted)

        slot_offset += result.get("img_count", 0)

        display_parts.append(d_text)
        embedding_parts.append(e_text)

        darwin.log(f"Chunks complete: {chunk_idx+1}/{n_chunks}")

    display_text   = re.sub(r"\n{3,}", "\n\n", "\n\n".join(display_parts)).strip()
    embedding_text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(embedding_parts)).strip()

    # Build ExtractedImage list
    all_images = []
    if images_dir:
        for img_dict in all_raw_imgs:
            img_path = Path(img_dict["path"])
            all_images.append(ExtractedImage(
                path=img_path,
                page=img_dict.get("page", 0),
                index=img_dict.get("index", 0),
                caption="",
                width=img_dict.get("width", 0),
                height=img_dict.get("height", 0),
            ))

    darwin.log(f"Merged {n_chunks} chunks — {page_count} pages, {len(all_images)} images")
    return display_text, embedding_text, page_count, metadata, all_images


# ── Main entry for PDF parsing ────────────────────────────────────────────────

def parse_pdf(
    path:        Path,
    images_dir:  Optional[Path] = None,
    caption:     bool = False,        # still accepted but ignored
    precise:     bool = False,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    """
    Parse a PDF and return (display_text, embedding_text, page_count, metadata, images).
    Images are returned WITHOUT captions — captioning must be done later.
    """
    import fitz

    doc        = fitz.open(str(path))
    page_count = doc.page_count
    metadata   = dict(doc.metadata or {})
    doc.close()

    mode_label = "precise" if precise else "fast"
    darwin.log(f"Parsing {path.name} ({page_count} pages) [{mode_label} mode]...")

    if page_count <= LARGE_PDF_PAGE_THRESHOLD:
        return _parse_pdf_single(
            path=path, images_dir=images_dir, caption=False,
            page_count=page_count, metadata=metadata, precise=precise
        )
    else:
        n_chunks = -(-page_count // LARGE_PDF_CHUNK_PAGES)
        darwin.log(f"Large PDF — {n_chunks} chunks [{mode_label} mode]...")
        return _parse_pdf_chunked(
            path=path, images_dir=images_dir, caption=False,
            page_count=page_count, metadata=metadata, precise=precise
        )


# ── Non-PDF parsing (unchanged) ───────────────────────────────────────────────

def parse_markitdown(path: Path) -> Tuple[str, str, int, Dict, List]:
    """Parse non-PDF documents via MarkItDown."""
    from markitdown import MarkItDown

    result = MarkItDown().convert(str(path))
    text   = result.text_content or ""
    text   = re.sub(r"\n{3,}", "\n\n", text).strip()

    word_count = len(text.split())
    page_count = max(1, round(word_count / 275))

    return text, text, page_count, {}, []