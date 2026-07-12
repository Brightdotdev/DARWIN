import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from src.core.Logger.DarwinLogger import darwin, LogType

from src.core.Pipeline.Ingestion.Parsing.lib import (
    ExtractedImage,
    MAX_FILE_SIZE_MB,
    IMAGE_SLOT_RE,
    LARGE_PDF_CHUNK_PAGES,
    LARGE_PDF_PAGE_THRESHOLD,
)

from src.core.Pipeline.Ingestion.Parsing.Extractors.FitzExtractor import fitz_extract
from src.core.Pipeline.Ingestion.Parsing.Extractors.FitzRawExtractor import fitz_raw_extract


# ── Helpers ────────────────────────────────────────────────────────────────────


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
            return f"\n\n[Image: {clean}]\n\n"
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
    # Choose extractor based on precision tier
    if precise:
        extractor = fitz_extract
    else:
        extractor = fitz_raw_extract  # FAST path

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
    extractor = fitz_extract if precise else fitz_raw_extract

    display_text, embedding_text, extracted_imgs = extractor(
        str(path),
        list(range(page_count)),
        str(images_dir) if images_dir else None,
    )

    mode = "fitz_extract (precise)" if precise else "fitz_raw_extract (fast)"
    darwin.info(f"Extraction mode: {mode}")

    all_images: List[ExtractedImage] = []
    if images_dir and extracted_imgs:
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
    Extract a large PDF using parallel chunk processing.
    Captioning NOT performed.

    FIX: Uses ThreadPoolExecutor for intra-PDF parallelism (zero spawn overhead).
    """
    from concurrent.futures import ThreadPoolExecutor

    all_pages   = list(range(page_count))
    page_chunks = [
        all_pages[i : i + LARGE_PDF_CHUNK_PAGES]
        for i in range(0, page_count, LARGE_PDF_CHUNK_PAGES)
    ]

    n_chunks = len(page_chunks)
    darwin.info(f"Processing {n_chunks} chunks in parallel...")

    # FIX: Use threads for chunk parallelism (fitz releases GIL, so this works)
    # Zero spawn overhead vs ProcessPool, and threads share memory.
    chunk_results = []
    do_images = bool(images_dir)

    with ThreadPoolExecutor(max_workers=min(4, n_chunks)) as executor:
        futures = {
            executor.submit(
                _parse_pdf_chunk,
                str(path),
                str(images_dir) if images_dir else None,
                chunk,
                do_images,
                precise,
            ): chunk_idx
            for chunk_idx, chunk in enumerate(page_chunks)
        }

        for future in futures:
            chunk_idx = futures[future]
            try:
                result = future.result()
                chunk_results.append((chunk_idx, result))
                darwin.info(f"Chunk {chunk_idx + 1}/{n_chunks} complete")
            except Exception as e:
                darwin.warning(f"Chunk {chunk_idx + 1}/{n_chunks} failed: {e}")
                chunk_results.append((chunk_idx, {
                    "display_text": "", "embedding_text": "",
                    "images": [], "img_count": 0, "error": str(e),
                }))

    # Sort by chunk index to maintain page order
    chunk_results.sort(key=lambda x: x[0])

    display_parts   = []
    embedding_parts = []
    all_raw_imgs    = []
    slot_offset = 0

    for _, result in chunk_results:
        if result.get("error"):
            continue

        d_text = result.get("display_text", "")
        e_text = result.get("embedding_text", "")
        imgs   = result.get("images", [])

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

    display_text   = re.sub(r"\n{3,}", "\n\n", "\n\n".join(display_parts)).strip()
    embedding_text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(embedding_parts)).strip()

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

    darwin.info(f"Merged {n_chunks} chunks: {page_count} pages, {len(all_images)} images")
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

    SPEED TIERS:
        precise=False  → fitz_raw_extract  (blazing fast, plain text)
        precise=True   → fitz_extract      (full layout analysis)
    """
    import fitz

    doc        = fitz.open(str(path))
    page_count = doc.page_count
    metadata   = dict(doc.metadata or {})
    doc.close()

    mode_label = "precise" if precise else "fast"
    darwin.info(f"Parsing {path.name} ({page_count} pages) [{mode_label} mode]...")

    if page_count <= LARGE_PDF_PAGE_THRESHOLD:
        return _parse_pdf_single(
            path=path, images_dir=images_dir, caption=False,
            page_count=page_count, metadata=metadata, precise=precise
        )
    else:
        n_chunks = -(-page_count // LARGE_PDF_CHUNK_PAGES)
        darwin.info(f"Large PDF: {n_chunks} chunks [{mode_label} mode]...")
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