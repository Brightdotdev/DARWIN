"""
ParserUtils.py — PDF extraction utilities with fast and precise modes.

Data-flow contract
──────────────────
Extractor  →  [Image:N] slots + [Page N] headers in raw text
ParserUtils →  _run_captioning resolves [Image:N] → final markdown/embed tokens
Chunker    →  receives clean text; ChunkCleaners strips any residual artefacts

IMPORTANT: [Image:N] slots must be resolved by _replace_image_slots() here,
before text reaches ChunkCleaners. ChunkCleaners._remove_artefacts() will
strip any *unresolved* slots as a safety net, but relying on that means
images are silently dropped rather than captioned.
"""

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict, Any
from src.core.Parsing.Captioning import (
    caption_image,
    load_caption_cache,
    save_caption_cache,
)
from src.lib.Terminal import TerminalUI, Timer
from src.core.Parsing.Extractors import fitz_extract, pymupdf4llm_extract



# ── Constants ──────────────────────────────────────────────────────────────────

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

LARGE_PDF_PAGE_THRESHOLD = 80
LARGE_PDF_CHUNK_PAGES    = 80

# is_good_enough tuning
MIN_TEXT_LENGTH       = 100
ASCII_RATIO_THRESHOLD = 0.60
NON_LATIN_ASCII_FLOOR = 0.20
CHECKPOINT_PAGE_THRESHOLD = 300  # Only checkpoint files with 300+ pages

# Compiled once at module level — used in hot paths
_IMAGE_SLOT_RE = re.compile(r"\[Image:(\d+)\]")


_MIN_DIMENSION   = 100   # px — skip images smaller than this in either axis
_MAX_CAPTION_DIM = 512   # px — resize before sending to model
_MAX_NEW_TOKENS  = 40    # tokens — captions don't need to be long

def _should_caption_image(img_dict: dict) -> tuple[bool, str]:
    """
    Returns (should_caption, reason_if_skipped).
    Checks dimensions from the dict — no file I/O needed.
    """
    w = img_dict.get("width", 0)
    h = img_dict.get("height", 0)

    if w < _MIN_DIMENSION or h < _MIN_DIMENSION:
        return False, f"too small ({w}x{h}px)"

    return True, ""


def _prepare_image(image_path: Path) :
    """Open, convert to RGB, and resize to model's expected input size."""
    from PIL import Image as PILImage

    image = PILImage.open(image_path).convert("RGB")

    if max(image.size) > _MAX_CAPTION_DIM:
        image.thumbnail((_MAX_CAPTION_DIM, _MAX_CAPTION_DIM), PILImage.LANCZOS)

    return image

# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class FileWithImages:
    file: Union[str, Path]
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
    images:      bool = False
    caption:     bool = False
    recursive:   bool = False
    rerun:       bool = False
    file:        Optional[str] = None
    precise:     bool = False
    save:        bool = True
    timeout:     int  = PARSE_TIMEOUT_SECONDS
    cleanup:     bool = True


@dataclass
class ParsedDocument:
    display_text:   str
    embedding_text: str
    filename:       str
    page_count:     int
    file_type:      str
    metadata:       Dict                  = field(default_factory=dict)
    error:          Optional[str]         = None
    images:         List[ExtractedImage]  = field(default_factory=list)


# ── Helpers ────────────────────────────────────────────────────────────────────

def detect_type(path: Path) -> Optional[str]:
    return SUPPORTED_TYPES.get(path.suffix.lower())


def _should_checkpoint(page_count: int) -> bool:
    """Checkpoint only for very large documents."""
    return page_count >= CHECKPOINT_PAGE_THRESHOLD

def is_supported(path: Path) -> bool:
    return detect_type(path) is not None


def is_good_enough(text: str) -> bool:
    """
    Returns True if the extracted text looks usable.

    Three cases:
    - Too short                → reject
    - Latin-script (high ASCII ratio) → apply ASCII threshold
    - Non-latin (low ASCII ratio, e.g. Arabic / Chinese / heavy math notation)
      → accept as long as there is meaningful printable content; rejecting on
        ASCII ratio would silently drop legitimate documents.
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False

    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_chars / len(text)

    if ascii_ratio < NON_LATIN_ASCII_FLOOR:
        printable = sum(1 for c in text if c.isprintable() and not c.isspace())
        ok = printable > MIN_TEXT_LENGTH
        if not ok:
            TerminalUI.dim("is_good_enough: non-latin doc rejected (printable=%d)", printable)
        return ok

    ok = ascii_ratio > ASCII_RATIO_THRESHOLD
    if not ok:
        TerminalUI.dim(
            "is_good_enough: rejected (ascii_ratio=%.2f, threshold=%.2f)",
            ascii_ratio, ASCII_RATIO_THRESHOLD,
        )
    return ok


def normalize_image_path(img_path: Path, base_dir: Path = None) -> str:
    """Return a relative path for display. Falls back to filename."""
    if base_dir:
        try:
            return img_path.relative_to(base_dir).as_posix()
        except ValueError:
            pass
    return img_path.name


def check_file_size(path: Path) -> Optional[str]:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return f"file too large: {size_mb:.0f}MB (max {MAX_FILE_SIZE_MB}MB)"
    return None


# ── Image slot helpers ─────────────────────────────────────────────────────────

def _replace_image_slots(
    display_text:   str,
    embedding_text: str,
    slot_map:       Dict[int, Dict],  # slot_index → {caption, rel_path, path}
) -> Tuple[str, str]:
    """
    Replace [Image:N] placeholders with their final display / embedding tokens.

    Called once per document after all images have been captioned.
    Slots with no entry in slot_map are replaced with a plain [Image] marker
    so no dangling tokens reach the chunker.
    """
    def _display(m: re.Match) -> str:
        idx  = int(m.group(1))
        info = slot_map.get(idx)
        if not info:
            TerminalUI.dim("_replace_image_slots: no info for slot %d → [Image] fallback", idx)
            return "[Image]"
        caption  = info.get("caption", "")
        rel_path = info.get("rel_path", info.get("path", ""))
        if caption:
            return f"\n\n![{caption}]({rel_path})\n*Figure: {caption}*\n\n"
        return f"\n\n![image]({rel_path})\n\n"

    def _embed(m: re.Match) -> str:
        idx  = int(m.group(1))
        info = slot_map.get(idx)
        if not info:
            return "[Image]"
        caption = info.get("caption", "")
        return f"\n\n[Image: {caption}]\n\n" if caption else "\n\n[Image]\n\n"

    display_text   = _IMAGE_SLOT_RE.sub(_display,   display_text)
    embedding_text = _IMAGE_SLOT_RE.sub(_embed, embedding_text)
    return display_text, embedding_text


def _remap_image_slots(
    display_text:   str,
    embedding_text: str,
    offset:         int,
) -> Tuple[str, str]:
    """
    Shift all [Image:N] slot indices in text by `offset`.

    Used when assembling chunked PDF results so each chunk's slots have a
    globally unique index rather than all starting from 0.
    """
    if offset == 0:
        return display_text, embedding_text

    def _shift(m: re.Match) -> str:
        return f"[Image:{int(m.group(1)) + offset}]"

    display_text   = _IMAGE_SLOT_RE.sub(_shift, display_text)
    embedding_text = _IMAGE_SLOT_RE.sub(_shift, embedding_text)
    return display_text, embedding_text


# ── Captioning ─────────────────────────────────────────────────────────────────


def _run_captioning(
    display_text:   str,
    embedding_text: str,
    extracted_imgs: List[Dict],
    images_dir:     Path,
    do_caption:     bool,
) -> Tuple[str, str, List[ExtractedImage]]:
    """
    Caption all extracted images and resolve [Image:N] slots to final tokens.
    Does NOT unload the vision model — caller handles that.
    """

    processor = model = device = None
    cache: Dict[str, str] = {}

    # FIX: add diagnostic log so it's immediately visible whether captioning
    # is attempted or skipped, and why.
    TerminalUI.dim(
        f"_run_captioning: do_caption={do_caption}, "
        f"img_count={len(extracted_imgs)}, "
        f"images_dir={images_dir}"
    )

    should_caption = bool(do_caption and extracted_imgs)

    if should_caption:
        try:
            from src.core.Models import get_loader, ModelCategory

            loader = get_loader()

            # FIX: guard against missing vision preference before attempting load,
            # so the error message is actionable rather than a generic exception.
            vision_key = loader.get_model_key(ModelCategory.VISION)
            if vision_key == "not set":
                TerminalUI.error(
                    "_run_captioning: no vision model configured — "
                    "run onboarding to set one. Captioning skipped."
                )
                should_caption = False
            else:
                TerminalUI.dim(f"Loading vision model '{vision_key}'...")
                processor, model, device = loader.get(ModelCategory.VISION)
                TerminalUI.dim(f"Vision model loaded on {device}")
                cache = load_caption_cache(images_dir)

        except Exception as e:
            TerminalUI.error(
                f"_run_captioning: failed to load vision model — "
                f"{type(e).__name__}: {e}. Captioning skipped."
            )
            should_caption = False

    all_images: List[ExtractedImage] = []
    slot_map:   Dict[int, Dict]      = {}


    for img_dict in extracted_imgs:
        img_path  = Path(img_dict["path"])
        rel_path  = normalize_image_path(img_path, images_dir)
        img_caption = ""

        # ── resolve slot index early so we can still register skipped images ──
        slot_index = img_dict.get("slot") if img_dict.get("slot") is not None else img_dict.get("index")
        if slot_index is None:
            TerminalUI.error(
                "_run_captioning: img_dict missing 'slot' and 'index' keys for %s",
                img_path.name,
            )
            continue

        # ── caption if enabled ───────────────────────────────────────────────
        if should_caption:
            cache_hit = cache.get(img_path.name)

            if cache_hit:
                img_caption = cache_hit
                TerminalUI.dim(f"  [cache] {img_path.name}")

            else:
                ok, reason = _should_caption_image(img_dict)

                if not ok:
                    TerminalUI.dim(f"  [skip] {img_path.name} — {reason}")
                else:
                    try:
                        from PIL import Image as PILImage
                        image = _prepare_image(img_path)
                        img_caption = caption_image(
                            img_path, processor, model, device,
                            cache,
                            preloaded_image=image,       # ← pass resized image in
                            max_new_tokens=_MAX_NEW_TOKENS,
                        )
                    except Exception as e:
                        TerminalUI.warn(f"Captioning failed for {img_path.name}: {e}")

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

    display_text, embedding_text = _replace_image_slots(
        display_text, embedding_text, slot_map
    )

    if should_caption and CACHE_CAPTIONS and cache:
        save_caption_cache(images_dir, cache)

    TerminalUI.dim(
        f"_run_captioning: resolved {len(all_images)} images, "
        f"captioned={should_caption}"
    )

    return display_text, embedding_text, all_images



# ── PDF chunk worker (runs in subprocess) ─────────────────────────────────────

def _parse_pdf_chunk(
    pdf_path:    str,
    images_dir:  Optional[str],
    pages:       List[int],
    do_images:   bool,
    precise:     bool = False,
) -> Dict[str, Any]:
    """
    Extract one contiguous page range from a PDF.

    Returns a plain dict (picklable for IPC) containing:
      display_text, embedding_text, images list, error string, img_count.

    Slot indices emitted by the extractor always start from 0 within each
    chunk. The parent process is responsible for remapping them to globally
    unique indices via _remap_image_slots() before joining chunk texts.

    Caption model is NOT loaded here even if do_caption=True: loading a CUDA
    model inside a subprocess forked from a CUDA parent is unreliable.
    Captioning is deferred to the parent after merging.
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
    caption:     bool,
    page_count:  int,
    metadata:    Dict,
    t:           Timer,
    precise:     bool = False,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    extractor = pymupdf4llm_extract if precise else fitz_extract

    display_text, embedding_text, extracted_imgs = extractor(
        str(path),
        list(range(page_count)),
        str(images_dir) if images_dir else None,
    )

    mode = "pymupdf4llm" if precise else "fitz_extract"
    t.split(mode)

    all_images: List[ExtractedImage] = []

    if not images_dir:
        return display_text, embedding_text, page_count, metadata, all_images

    if extracted_imgs:
        TerminalUI.dim(f"Running captioning on {len(extracted_imgs)} images...")
        display_text, embedding_text, all_images = _run_captioning(
            display_text,
            embedding_text,
            extracted_imgs,
            images_dir,
            caption,
        )

        # Unload vision model after captioning this file
        if caption:
            try:
                from src.core.Models import get_loader, ModelCategory
                # FIX: pass the enum — unload() now handles both enum and str
                get_loader().unload(ModelCategory.VISION)
                TerminalUI.dim("Vision model unloaded from memory")
            except Exception:
                pass
    else:
        TerminalUI.dim("No images found, skipping captioning")

    t.split("captioning")
    return display_text, embedding_text, page_count, metadata, all_images



def _parse_pdf_chunked(
    path:        Path,
    images_dir:  Optional[Path],
    caption:     bool,
    page_count:  int,
    metadata:    Dict,
    t:           Timer,
    precise:     bool = False,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    """
    Extract a large PDF in parallel page-range chunks.

    Slot-index remapping
    ────────────────────
    Each subprocess worker calls fitz_extract / pymupdf4llm_extract on its own
    page range starting from page index 0 within that range. This means every
    worker emits [Image:0], [Image:1], … which would collide when texts are
    joined. We fix this by:

      1. Recording how many images each chunk produced (img_count).
      2. Walking chunks in order and computing a running offset.
      3. Calling _remap_image_slots(offset) on each chunk's text before joining.
      4. Adjusting slot indices in each chunk's raw image dicts by the same offset.

    After joining, captioning runs once in the parent process on the full
    merged slot_map — avoiding the CUDA-in-subprocess reliability problem.
    """
    all_pages   = list(range(page_count))
    page_chunks = [
        all_pages[i : i + LARGE_PDF_CHUNK_PAGES]
        for i in range(0, page_count, LARGE_PDF_CHUNK_PAGES)
    ]

    n_chunks  = len(page_chunks)
    n_workers = min(n_chunks, max(1, CPU_COUNT // 2) if caption else CPU_COUNT)

    TerminalUI.dim(f"Using {n_workers} workers for {n_chunks} chunks...")

    chunk_results: Dict[int, Dict] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(
                _parse_pdf_chunk,
                str(path),
                str(images_dir) if images_dir else None,
                chunk_pages,
                bool(images_dir),
                precise,
            ): chunk_idx
            for chunk_idx, chunk_pages in enumerate(page_chunks)
        }

        for future in as_completed(future_to_idx):
            chunk_idx = future_to_idx[future]
            try:
                chunk_results[chunk_idx] = future.result(timeout=PARSE_TIMEOUT_SECONDS)
            except Exception as e:
                chunk_results[chunk_idx] = {
                    "display_text":   "",
                    "embedding_text": "",
                    "images":         [],
                    "img_count":      0,
                    "error":          f"chunk {chunk_idx} failed: {e}",
                }
            TerminalUI.dim(f"Chunks complete: {len(chunk_results)}/{n_chunks}")

    t.split("parallel_parsing")

    # ── Merge chunks with slot remapping ──
    display_parts:   List[str]  = []
    embedding_parts: List[str]  = []
    all_raw_imgs:    List[Dict] = []
    slot_offset = 0

    for chunk_idx in range(n_chunks):
        result = chunk_results.get(chunk_idx, {})

        if result.get("error"):
            TerminalUI.warn(f"Chunk {chunk_idx} error: {result['error']}")
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

    t.split("chunk_merge")

    # ── Captioning pass ──
    if not images_dir or not all_raw_imgs:
        return display_text, embedding_text, page_count, metadata, []

    display_text, embedding_text, all_images = _run_captioning(
        display_text, embedding_text, all_raw_imgs,
        images_dir, caption,
    )

    # ── Unload vision model ──
    if caption:
        try:
            from src.core.Models import get_loader, ModelCategory
            # FIX: pass the enum — unload() now handles both enum and str
            get_loader().unload(ModelCategory.VISION)
            TerminalUI.dim("Vision model unloaded from memory")
        except Exception:
            pass

    t.split("captioning")
    TerminalUI.dim(
        f"Merged {n_chunks} chunks — {page_count} pages, "
        f"{len(all_images)} images complete"
    )

    return display_text, embedding_text, page_count, metadata, all_images


# ── parse_pdf main pdf parsing ─────────────────────────────────────────────────

def parse_pdf(
    path:        Path,
    images_dir:  Optional[Path] = None,
    caption:     bool = False,
    precise:     bool = False,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    """
    Parse a PDF and return (display_text, embedding_text, page_count, metadata, images).

    Routing:
    - page_count <= LARGE_PDF_PAGE_THRESHOLD → single-process extraction
    - page_count >  LARGE_PDF_PAGE_THRESHOLD → parallel chunked extraction
      with slot index remapping and deferred captioning
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

    if page_count <= LARGE_PDF_PAGE_THRESHOLD and page_count <= CHECKPOINT_PAGE_THRESHOLD:
        result = _parse_pdf_single(
            path=path, images_dir=images_dir, caption=caption,
            page_count=page_count, metadata=metadata, t=t, precise=precise
        )
    else:
        n_chunks = -(-page_count // LARGE_PDF_CHUNK_PAGES)
        TerminalUI.dim(
            f"Large PDF — splitting into {n_chunks} chunks "
            f"({LARGE_PDF_CHUNK_PAGES} pages each) [{mode_label} mode]..."
        )
        result = _parse_pdf_chunked(
            path=path, images_dir=images_dir, caption=caption,
            page_count=page_count, metadata=metadata, t=t, precise=precise
        )

    t.done()
    return result


# ── markitdown ─────────────────────────────────────────────────────────────────

def parse_markitdown(path: Path) -> Tuple[str, str, int, Dict, List]:
    """
    Parse non-PDF documents via MarkItDown.

    Returns (display_text, embedding_text, page_count, metadata, images).
    Both text fields are identical (markitdown produces a single text output).
    page_count is estimated from word count at 275 words/page.
    """
    from markitdown import MarkItDown

    t      = TerminalUI.timer(f"markitdown:{path.name}")
    result = MarkItDown().convert(str(path))
    text   = result.text_content or ""
    text   = re.sub(r"\n{3,}", "\n\n", text).strip()

    word_count = len(text.split())
    page_count = max(1, round(word_count / 275))

    t.done()
    return text, text, page_count, {}, []