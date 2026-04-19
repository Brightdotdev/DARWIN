"""
ParserUtils.py - PDF extraction utilities with fast and precise modes
"""

import os
import re
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict, Any

from src.core.Parsing.Captioning import (
    caption_image,
    load_caption_model,
    caption_model_available,
    load_caption_cache,
    save_caption_cache,
)
from src.lib.Terminal import TerminalUI, Timer
from src.core.Parsing.Extractors import (fitz_extract,pymupdf4llm_extract)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SUPPORTED_TYPES: Dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "markitdown",
    ".pptx": "markitdown",
    ".txt": "markitdown",
    ".md": "markitdown",
}

CPU_COUNT = os.cpu_count() or 4
MAX_FILE_SIZE_MB = 500
MIN_IMAGE_SIZE_PX = 100
MAX_IMAGES_PER_DOC = 50

PARSE_TIMEOUT_SECONDS = 300
CAPTION_TIMEOUT_SECONDS = 30
CACHE_CAPTIONS = True

# FIX: threshold must be >= chunk size so a just-over-threshold PDF uses chunks
# Old values were inverted (threshold=50, chunk=80 meant 51-page PDFs chunked unnecessarily)
LARGE_PDF_PAGE_THRESHOLD = 80
LARGE_PDF_CHUNK_PAGES    = 80

# Extraction tuning
HEADING_SIZE_RATIO  = 1.20
CAPTION_SIZE_RATIO  = 0.85
LARGE_HEADING_RATIO = 1.50
TABLE_X_SPREAD      = 0.35
TABLE_MIN_SPANS     = 3
COLUMN_BUCKETS      = 2
TABLE_Y_GAP_RATIO   = 2.5

# is_good_enough tuning
MIN_TEXT_LENGTH       = 100
ASCII_RATIO_THRESHOLD = 0.60   # used only for latin-script docs
# If fewer than this fraction of chars are ASCII we assume a non-latin script
# and fall back to a simpler length check instead of rejecting the doc.
NON_LATIN_ASCII_FLOOR = 0.20


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class FileWithImages:
    file: Union[str, Path]
    image_dir: Optional[Path] = None


@dataclass
class ExtractedImage:
    path: Path
    page: int
    index: int
    caption: str
    width: int
    height: int



@dataclass
class ParsingArgs:
    target: str
    project_dir: Optional[str] = None
    images: bool = False
    caption: bool = False
    recursive: bool = False
    rerun: bool = False
    file: Optional[str] = None
    precise: bool = False
    save: bool = True
    timeout: int = PARSE_TIMEOUT_SECONDS
    cleanup: bool = True


@dataclass
class ParsedDocument:
    display_text: str
    embedding_text: str
    filename: str
    page_count: int
    file_type: str
    metadata: Dict = field(default_factory=dict)
    error: Optional[str] = None
    images: List[ExtractedImage] = field(default_factory=list)
    


# ── Helpers ────────────────────────────────────────────────────────────────────

def detect_type(path: Path) -> Optional[str]:
    return SUPPORTED_TYPES.get(path.suffix.lower())


def is_supported(path: Path) -> bool:
    return detect_type(path) is not None


def is_good_enough(text: str) -> bool:
    """
    Returns True if the extracted text looks usable.

    Handles three cases:
    - Too short → reject
    - Latin-script doc (high ASCII ratio) → apply the old ASCII threshold
    - Non-latin script (low ASCII ratio, e.g. Arabic / Chinese / heavy math) →
      accept as long as there is meaningful content; rejecting on ASCII ratio
      would silently drop legitimate documents.
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False

    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_chars / len(text)

    # Non-latin script: ASCII ratio is naturally low — don't penalise it.
    # We just need there to be *some* printable content.
    if ascii_ratio < NON_LATIN_ASCII_FLOOR:
        printable = sum(1 for c in text if c.isprintable() and not c.isspace())
        is_ok = printable > MIN_TEXT_LENGTH
        if not is_ok:
            logger.debug("is_good_enough: non-latin doc rejected (printable=%d)", printable)
        return is_ok

    # Latin-script: apply the original ASCII ratio gate.
    is_ok = ascii_ratio > ASCII_RATIO_THRESHOLD
    if not is_ok:
        logger.debug(
            "is_good_enough: rejected (ascii_ratio=%.2f, threshold=%.2f)",
            ascii_ratio, ASCII_RATIO_THRESHOLD,
        )
    return is_ok


def normalize_image_path(img_path: Path, project_dir: Path) -> str:
    try:
        return img_path.relative_to(project_dir).as_posix()
    except ValueError:
        return img_path.name
    

def check_file_size(path: Path) -> Optional[str]:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return f"file too large: {size_mb:.0f}MB (max {MAX_FILE_SIZE_MB}MB)"
    return None



# ── Positional image slot helpers ──────────────────────────────────────────────

def _replace_image_slots(
    display_text: str,
    embedding_text: str,
    slot_map: Dict[int, Dict],  # index → img_info dict
) -> Tuple[str, str]:
    """
    Replace positional [Image:N] slots with their final markdown / embed tokens.
    Slots with no matching image are replaced with a generic [Image] marker so
    the document never contains dangling slot tokens.
    """
    # Find every slot token present in the text
    slot_pattern = re.compile(r"\[Image:(\d+)\]")

    def _display_replacement(m: re.Match) -> str:
        idx = int(m.group(1))
        info = slot_map.get(idx)
        if not info:
            logger.debug(f"_replace_image_slots: No info for slot {idx}, using [Image] fallback")
            return "[Image]"
        caption  = info.get("caption", "")
        rel_path = info.get("rel_path", info.get("path", ""))
        if caption:
            return f"\n\n![{caption}]({rel_path})\n*Figure: {caption}*\n\n"
        return f"\n\n![image]({rel_path})\n\n"

    def _embed_replacement(m: re.Match) -> str:
        idx = int(m.group(1))
        info = slot_map.get(idx)
        if not info:
            return "[Image]"
        caption = info.get("caption", "")
        return f"\n\n[Image: {caption}]\n\n" if caption else "\n\n[Image]\n\n"

    display_text   = slot_pattern.sub(_display_replacement, display_text)
    embedding_text = slot_pattern.sub(_embed_replacement, embedding_text)
    return display_text, embedding_text






# ── Captioning ─────────────────────────────────────────────────────────────────

def _run_captioning(
    display_text: str,
    embedding_text: str,
    extracted_imgs: List[Dict],
    images_dir: Path,
    project_dir: Optional[Path],
    do_caption: bool,
) -> Tuple[str, str, List[ExtractedImage]]:
    processor = model = None
    cache: Dict[str, str] = {}

    if do_caption and caption_model_available() and extracted_imgs:
        try:
            TerminalUI.dim("Loading caption model...")
            processor, model, _ = load_caption_model()
            cache = load_caption_cache(images_dir)
        except Exception as e:
            TerminalUI.error(f"Caption model failed: {e}")
            do_caption = False

    all_images: List[ExtractedImage] = []

    # Build a slot→info map so we can do a single regex pass at the end
    slot_map: Dict[int, Dict] = {}

    for img_dict in extracted_imgs:
        img_path   = Path(img_dict["path"])
        img_caption = ""

        if do_caption and processor and model:
            img_caption = caption_image(img_path, processor, model, cache)
            img_dict["caption"] = img_caption

        rel_path = (
            normalize_image_path(img_path, project_dir)
            if project_dir else img_path.name
        )

        slot_index = img_dict.get("slot", img_dict.get("index", 0))
        slot_map[slot_index] = {
            "caption":  img_caption,
            "rel_path": rel_path,
            "path":     str(img_path),
        }

        all_images.append(ExtractedImage(
            path=img_path,
            page=img_dict.get("page", 0),
            index=img_dict.get("index", 0),
            caption=img_caption,
            width=img_dict.get("width", 0),
            height=img_dict.get("height", 0),
        ))

    # Single regex pass — positional, never misaligns
    display_text, embedding_text = _replace_image_slots(display_text, embedding_text, slot_map)

    if do_caption and CACHE_CAPTIONS and cache:
        save_caption_cache(images_dir, cache)

    return display_text, embedding_text, all_images


# ── PDF chunk worker ───────────────────────────────────────────────────────────

def _parse_pdf_chunk(
    pdf_path: str,
    images_dir: Optional[str],
    project_dir: Optional[str],
    pages: List[int],
    do_images: bool,
    do_caption: bool,
    chunk_index: int,
    precise: bool = False,
) -> Dict[str, Any]:
    t = TerminalUI.timer(f"chunk-{chunk_index}")

    img_dir  = Path(images_dir)  if images_dir  else None
    proj_dir = Path(project_dir) if project_dir else None

    extractor = pymupdf4llm_extract if precise else fitz_extract

    try:
        display_text, embedding_text, extracted_imgs = extractor(
            pdf_path,
            pages,
            images_dir if do_images else None,
        )
    except Exception as e:
        return {"display_text": "", "embedding_text": "", "images": [],  "error": str(e)}

    mode = "pymupdf4llm" if precise else "fitz_extract"
    t.split(mode)

    if not do_images or not img_dir:
        t.done()
        return {
            "display_text":  display_text,
            "embedding_text": embedding_text,
            "images": [],
            "error":  None,
        }

    display_text, embedding_text, all_images = _run_captioning(
        display_text, embedding_text, extracted_imgs,
        img_dir, proj_dir, do_caption,
    )

    t.split("captioning")
    TerminalUI.dim(f"Captioned {len(all_images)} images")
    t.done()

    return {
        "display_text":  display_text,
        "embedding_text": embedding_text,
        "images": [
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


# ── parse_pdf ──────────────────────────────────────────────────────────────────

def parse_pdf(
    path: Path,
    images_dir: Optional[Path] = None,
    caption: bool = False,
    project_dir: Optional[Path] = None,
    precise: bool = False,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    """
    Returns (display_text, embedding_text, page_count, metadata, images).
    """
    import fitz

    t = TerminalUI.timer(f"parse_pdf:{path.name}")

    doc        = fitz.open(str(path))
    page_count = doc.page_count
    metadata   = dict(doc.metadata or {})
    doc.close()

    t.split("fitz_open")

    mode_label = "precise" if precise else "fast"
    TerminalUI.dim(f"Parsing {path.name} ({page_count} pages) [{mode_label} mode]...")

    if page_count <= LARGE_PDF_PAGE_THRESHOLD:
        result = _parse_pdf_single(
            path, images_dir, caption, project_dir,
            page_count, metadata, t, precise,
        )
    else:
        n_chunks = -(-page_count // LARGE_PDF_CHUNK_PAGES)
        TerminalUI.dim(f"Large PDF found — {n_chunks} parallel chunks [{mode_label} mode]...")
        result = _parse_pdf_chunked(
            path, images_dir, caption, project_dir, page_count, metadata, t, precise,
        )

    t.done()
    return result


def _parse_pdf_single(
    path: Path,
    images_dir: Optional[Path],
    caption: bool,
    project_dir: Optional[Path],
    page_count: int,
    metadata: Dict,
    t: Timer,
    precise: bool = False,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    extractor = pymupdf4llm_extract if precise else fitz_extract

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
        images_dir, project_dir, caption,
    )

    t.split("captioning")
    return display_text, embedding_text, page_count, metadata, all_images


def _parse_pdf_chunked(
    path: Path,
    images_dir: Optional[Path],
    caption: bool,
    project_dir: Optional[Path],
    page_count: int,
    metadata: Dict,
    t: Timer,
    precise: bool = False,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    all_pages = list(range(page_count))
    chunks    = [
        all_pages[i:i + LARGE_PDF_CHUNK_PAGES]
        for i in range(0, page_count, LARGE_PDF_CHUNK_PAGES)
    ]
    n_chunks  = len(chunks)
    n_workers = min(n_chunks, max(1, CPU_COUNT // 2) if caption else CPU_COUNT)

    TerminalUI.dim(f"Using {n_workers} workers for {n_chunks} chunks...")

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
                precise,
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
                    "images": [],  "error": f"chunk {chunk_idx} failed: {e}",
                }
            TerminalUI.dim(f"Chunks complete: {len(chunk_results)}/{n_chunks}")

    t.split("parallel_parsing")

    display_parts:  List[str]          = []
    embedding_parts: List[str]         = []
    all_images: List[ExtractedImage]   = []
    

    for chunk_idx in range(n_chunks):
        result = chunk_results.get(chunk_idx, {})
        if result.get("error"):
            TerminalUI.warn(f"Chunk {chunk_idx} error: {result['error']}")
            continue
        display_parts.append(result.get("display_text", ""))
        embedding_parts.append(result.get("embedding_text", ""))
        for img_dict in result.get("images", []):
            all_images.append(ExtractedImage(
                path=Path(img_dict["path"]),
                page=img_dict["page"],
                index=img_dict["index"],
                caption=img_dict["caption"],
                width=img_dict["width"],
                height=img_dict["height"],
            ))
            

    display_text   = re.sub(r"\n{3,}", "\n\n", "\n\n".join(display_parts)).strip()
    embedding_text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(embedding_parts)).strip()

    t.split("Chunking Merge")
    TerminalUI.dim(f"Merged {n_chunks} chunks — {page_count} pages complete")

    return display_text, embedding_text, page_count, metadata, all_images


# ── markitdown ─────────────────────────────────────────────────────────────────

def parse_markitdown(path: Path) -> Tuple[str, int, Dict, List]:
    from markitdown import MarkItDown
    t      = TerminalUI.timer(f"markitdown:{path.name}")
    result = MarkItDown().convert(str(path))
    text   = result.text_content or ""
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    word_count = len(text.split())
    page_count = max(1, round(word_count / 275))
    t.done()
    return text.strip(), page_count, {}, []