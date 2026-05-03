"""
ParserUtils.py — PDF extraction utilities with fast and precise modes.

Data-flow contract
──────────────────
Extractor  →  [Image:N] slots + [Page N] headers in raw text
ParserUtils →  _run_captioning resolves [Image:N] → final markdown/embed tokens
Chunker    →  receives clean text; ChunkCleaners strips any residual artefacts
"""

import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from src.core.Logger.DarwinLogger import darwin, LogType
from src.Models import ModelManager, ModelCategory

from src.features.Parsing.Captioning import (
    caption_image,
    load_caption_cache,
    save_caption_cache,
)
from src.features.Parsing.lib import (
    should_caption_image,
    MAX_NEW_TOKENS,
    ExtractedImage,
    MIN_TEXT_LENGTH,
    ASCII_RATIO_THRESHOLD,
    MAX_FILE_SIZE_MB,
    NON_LATIN_ASCII_FLOOR,
    IMAGE_SLOT_RE,
    prepare_image,
    LARGE_PDF_CHUNK_PAGES,
    LARGE_PDF_PAGE_THRESHOLD,
    CPU_COUNT,
    PARSE_TIMEOUT_SECONDS,

    )
from src.features.Parsing.Extractors import fitz_extract, pymupdf4llm_extract

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
            darwin.log(f"is_good_enough: non-latin doc rejected (printable={printable})", LogType.INFO)
        return ok

    ok = ascii_ratio > ASCII_RATIO_THRESHOLD
    if not ok:
        darwin.log(f"is_good_enough: rejected (ascii_ratio={ascii_ratio:.2f}, threshold={ASCII_RATIO_THRESHOLD})", LogType.INFO)
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
    slot_map:       Dict[int, Dict],
) -> Tuple[str, str]:
    """Replace [Image:N] placeholders with final display/embedding tokens."""
    def _display(m: re.Match) -> str:
        idx  = int(m.group(1))
        info = slot_map.get(idx)
        if not info:
            darwin.log(f"_replace_image_slots: no info for slot {idx} → [Image] fallback", LogType.INFO)
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


# ═══════════════════════════════════════════════════════════════════════════
# Captioning
# ═══════════════════════════════════════════════════════════════════════════

def _run_captioning(
    display_text:   str,
    embedding_text: str,
    extracted_imgs: List[Dict],
    images_dir:     Path,
    do_caption:     bool,
) -> Tuple[str, str, List[ExtractedImage]]:
    """Caption all extracted images and resolve [Image:N] slots to final tokens."""

    processor = model = device = None
    is_api_model = False
    cache: Dict[str, str] = {}

    darwin.log(
        f"_run_captioning: do_caption={do_caption}, img_count={len(extracted_imgs)}, images_dir={images_dir}",
        LogType.INFO
    )

    should_caption = bool(do_caption and extracted_imgs)

    if should_caption:
        try:
            loader = ModelManager.get_loader()

            try:
                vision_key = loader.get_model_key(ModelCategory.VISION)
            except ValueError:
                vision_key = "not set"

            if vision_key == "not set":
                darwin.log(
                    "_run_captioning: no vision model configured — run onboarding to set one. Captioning skipped.",
                    LogType.ERROR
                )
                should_caption = False
            else:
                darwin.log(f"Loading vision model '{vision_key}'...", LogType.INFO)
                loaded = loader.get(ModelCategory.VISION)
                
                # API models return a wrapper, local models return (processor, model, device)
                if hasattr(loaded, 'caption'):
                    # API vision model (LiteLLM wrapper)
                    model = loaded
                    is_api_model = True
                    device = "api"
                    darwin.log(f"Vision model loaded via API ({model.model_id})", LogType.INFO)
                else:
                    # Local vision model
                    processor, model, device = loaded
                    darwin.log(f"Vision model loaded on {device}", LogType.INFO)
                
                cache = load_caption_cache(images_dir)

        except Exception as e:
            darwin.log(
                f"_run_captioning: failed to load vision model — {type(e).__name__}: {e}. Captioning skipped.",
                LogType.ERROR
            )
            should_caption = False

    all_images: List[ExtractedImage] = []
    slot_map:   Dict[int, Dict]      = {}

    for img_dict in extracted_imgs:
        img_path  = Path(img_dict["path"])
        rel_path  = normalize_image_path(img_path, images_dir)
        img_caption = ""

        slot_index = img_dict.get("slot") if img_dict.get("slot") is not None else img_dict.get("index")
        if slot_index is None:
            darwin.log(f"_run_captioning: img_dict missing 'slot' and 'index' keys for {img_path.name}", LogType.ERROR)
            continue

        if should_caption:
            cache_hit = cache.get(img_path.name)
            if cache_hit is not None:
                img_caption = cache_hit
                darwin.log(f"  [cache] {img_path.name}", LogType.INFO)
            else:
                ok, reason = should_caption_image(img_dict)
                if not ok:
                    darwin.log(f"  [skip] {img_path.name} — {reason}", LogType.INFO)
                    cache[img_path.name] = ""
                else:
                    try:
                        if is_api_model:
                            # API captioning — no processor/device needed
                            img_caption = model.caption(
                                img_path,
                                max_new_tokens=MAX_NEW_TOKENS,
                            )
                        else:
                            # Local captioning — existing pipeline
                            image = prepare_image(img_path)
                            img_caption = caption_image(
                                img_path, processor, model, device,
                                cache,
                                preloaded_image=image,
                                max_new_tokens=MAX_NEW_TOKENS,
                            )
                    except Exception as e:
                        darwin.log(f"Captioning failed for {img_path.name}: {e}", LogType.WARNING)
                        cache[img_path.name] = ""

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

    if images_dir:
        save_caption_cache(images_dir, cache)

    darwin.log(f"_run_captioning: resolved {len(all_images)} images, captioned={should_caption}", LogType.INFO)

    return display_text, embedding_text, all_images

# ── PDF chunk worker (runs in subprocess) ─────────────────────────────────────

def _parse_pdf_chunk(
    pdf_path:    str,
    images_dir:  Optional[str],
    pages:       List[int],
    do_images:   bool,
    precise:     bool = False,
) -> Dict[str, Any]:
    """Extract one contiguous page range from a PDF."""

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
    precise:     bool = False,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    extractor = pymupdf4llm_extract if precise else fitz_extract

    display_text, embedding_text, extracted_imgs = extractor(
        str(path),
        list(range(page_count)),
        str(images_dir) if images_dir else None,
    )

    mode = "pymupdf4llm" if precise else "fitz_extract"
    darwin.log(f"Extraction mode: {mode}", LogType.INFO)

    all_images: List[ExtractedImage] = []

    if not images_dir:
        return display_text, embedding_text, page_count, metadata, all_images

    if extracted_imgs:
        darwin.log(f"Running captioning on {len(extracted_imgs)} images...", LogType.INFO)
        display_text, embedding_text, all_images = _run_captioning(
            display_text,
            embedding_text,
            extracted_imgs,
            images_dir,
            caption,
        )

        if caption:
            try:
                ModelManager.get_loader().unload(ModelCategory.VISION)
                darwin.log("Vision model unloaded from memory", LogType.INFO)
            except Exception:
                pass
    else:
        darwin.log("No images found, skipping captioning", LogType.INFO)
        display_text, embedding_text = _replace_image_slots(
            display_text, embedding_text, {}
        )

    return display_text, embedding_text, page_count, metadata, all_images


def _parse_pdf_chunked(
    path:        Path,
    images_dir:  Optional[Path],
    caption:     bool,
    page_count:  int,
    metadata:    Dict,
    precise:     bool = False,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    """Extract a large PDF in parallel page-range chunks."""
    all_pages   = list(range(page_count))
    page_chunks = [
        all_pages[i : i + LARGE_PDF_CHUNK_PAGES]
        for i in range(0, page_count, LARGE_PDF_CHUNK_PAGES)
    ]

    n_chunks  = len(page_chunks)
    n_workers = min(n_chunks, max(1, CPU_COUNT // 2) if caption else CPU_COUNT)

    darwin.log(f"Using {n_workers} workers for {n_chunks} chunks...", LogType.INFO)

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
            darwin.log(f"Chunks complete: {len(chunk_results)}/{n_chunks}", LogType.INFO)

    # ── Merge chunks with slot remapping ──
    display_parts:   List[str]  = []
    embedding_parts: List[str]  = []
    all_raw_imgs:    List[Dict] = []
    slot_offset = 0

    for chunk_idx in range(n_chunks):
        result = chunk_results.get(chunk_idx, {})

        if result.get("error"):
            darwin.log(f"Chunk {chunk_idx} error: {result['error']}", LogType.WARNING)
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

    # ── Captioning pass ──
    if not images_dir:
        return display_text, embedding_text, page_count, metadata, []

    if not all_raw_imgs:
        darwin.log("No images extracted from any chunk — skipping captioning", LogType.INFO)
        return display_text, embedding_text, page_count, metadata, []

    display_text, embedding_text, all_images = _run_captioning(
        display_text, embedding_text, all_raw_imgs,
        images_dir, caption,
    )

    if caption:
        try:
            ModelManager.get_loader().unload(ModelCategory.VISION)
            darwin.log("Vision model unloaded from memory", LogType.INFO)
        except Exception:
            pass

    darwin.log(f"Merged {n_chunks} chunks — {page_count} pages, {len(all_images)} images complete", LogType.INFO)

    return display_text, embedding_text, page_count, metadata, all_images


# ═══════════════════════════════════════════════════════════════════════════
# parse_pdf main entry point
# ═══════════════════════════════════════════════════════════════════════════

def parse_pdf(
    path:        Path,
    images_dir:  Optional[Path] = None,
    caption:     bool = False,
    precise:     bool = False,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    """Parse a PDF and return (display_text, embedding_text, page_count, metadata, images)."""
    import fitz

    doc        = fitz.open(str(path))
    page_count = doc.page_count
    metadata   = dict(doc.metadata or {})
    doc.close()

    mode_label = "precise" if precise else "fast"
    darwin.log(f"Parsing {path.name} ({page_count} pages) [{mode_label} mode]...", LogType.INFO)

    if page_count <= LARGE_PDF_PAGE_THRESHOLD:
        result = _parse_pdf_single(
            path=path, images_dir=images_dir, caption=caption,
            page_count=page_count, metadata=metadata, precise=precise
        )
    else:
        n_chunks = -(-page_count // LARGE_PDF_CHUNK_PAGES)
        darwin.log(f"Large PDF — splitting into {n_chunks} chunks ({LARGE_PDF_CHUNK_PAGES} pages each) [{mode_label} mode]...", LogType.INFO)
        result = _parse_pdf_chunked(
            path=path, images_dir=images_dir, caption=caption,
            page_count=page_count, metadata=metadata, precise=precise
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# markitdown
# ═══════════════════════════════════════════════════════════════════════════

def parse_markitdown(path: Path) -> Tuple[str, str, int, Dict, List]:
    """Parse non-PDF documents via MarkItDown."""
    from markitdown import MarkItDown

    result = MarkItDown().convert(str(path))
    text   = result.text_content or ""
    text   = re.sub(r"\n{3,}", "\n\n", text).strip()

    word_count = len(text.split())
    page_count = max(1, round(word_count / 275))

    return text, text, page_count, {}, []