"""
ParserUtils.py

Utility functions and data models for the document parser.

Large PDF strategy
------------------
PDFs over LARGE_PDF_PAGE_THRESHOLD pages are split into page-range chunks,
each chunk is processed in parallel via ProcessPoolExecutor, then the
results are glued back together in page order.

1120-page Bible → 4 × 280-page jobs in parallel → ~4x speedup.

Key design decisions
--------------------
- parse_pdf() returns fully normalized paths in display_text
  (relative to project_dir, e.g. .darwin/file_pdf/extracted_images/img.png)
- embedding_text replaces image markdown with plain [Image: caption] tokens
- Caption cache keys are bare filenames only — portable across machines/moves
- ParsedDocument.display_text  → safe to write to .md as-is
- ParsedDocument.embedding_text → clean text for vectorization
"""

import os
import re
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable, Union, Tuple, Dict

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

CPU_COUNT        = os.cpu_count() or 4
MAX_FILE_SIZE_MB = 500
MIN_IMAGE_SIZE_PX  = 100
MAX_IMAGES_PER_DOC = 50

PARSE_TIMEOUT_SECONDS   = 300
CAPTION_TIMEOUT_SECONDS = 30
CACHE_CAPTIONS          = True

# Large PDF splitting — PDFs over this page count get chunked + parallelised
LARGE_PDF_PAGE_THRESHOLD = 100
LARGE_PDF_CHUNK_PAGES    = 280


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


@dataclass
class ParsedDocument:
    """
    display_text   — markdown with relative image paths, write directly to .md
    embedding_text — plain text for vectorization, no markdown image syntax
    """
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

def load_caption_cache(images_dir: Path) -> Dict[str, str]:
    """Keys are bare filenames — never absolute paths."""
    if not CACHE_CAPTIONS:
        return {}
    cache_path = get_caption_cache_path(images_dir)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load caption cache: {e}")
    return {}

def save_caption_cache(images_dir: Path, cache: Dict[str, str]) -> None:
    if not CACHE_CAPTIONS:
        return
    try:
        with open(get_caption_cache_path(images_dir), "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save caption cache: {e}")

def migrate_caption_cache(images_dir: Path) -> int:
    """Migrate absolute-path cache keys to bare filenames. Safe to call repeatedly."""
    cache    = load_caption_cache(images_dir)
    migrated = {Path(k).name: v for k, v in cache.items()}
    count    = sum(1 for k in cache if Path(k).name != k)
    if count > 0:
        save_caption_cache(images_dir, migrated)
        logger.info(f"Migrated {count} caption cache keys to bare filenames")
    return count


# ============================================================================
# Caption model
# ============================================================================

def caption_model_available() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False

def load_caption_model(model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"):
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelClass

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model     = ModelClass.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="auto"
    )
    return processor, model, device

def caption_image(
    image_path: Path,
    processor,
    model,
    cache: Dict[str, str],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Caption one image. Cache key is bare filename — never absolute."""
    import torch

    cache_key = image_path.name
    if CACHE_CAPTIONS and cache_key in cache:
        return cache[cache_key]

    try:
        from PIL import Image as PILImage
        if progress_callback:
            progress_callback(f"Captioning {image_path.name}")

        image    = PILImage.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text":
                    "Describe what this image shows in one or two sentences. "
                    "Focus on the content and any visible text. "
                    "If it is a diagram or chart, describe what it illustrates."
                },
            ],
        }]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=80, do_sample=False)

        decoded = processor.decode(output[0], skip_special_tokens=True)
        result  = decoded.split("Assistant:")[-1].strip() if "Assistant:" in decoded else decoded.strip()

        if CACHE_CAPTIONS:
            cache[cache_key] = result
        return result

    except Exception as e:
        logger.warning(f"Caption failed for {image_path.name}: {e}")
        return f"image (caption failed: {e})"


# ============================================================================
# Path normalization
# ============================================================================

def normalize_image_path(img_path: Path, project_dir: Path) -> str:
    """
    Absolute → project-relative POSIX string for markdown.

    C:/Users/.../studyagent/.darwin/file_pdf/extracted_images/img.png
    → .darwin/file_pdf/extracted_images/img.png
    """
    try:
        return img_path.relative_to(project_dir).as_posix()
    except ValueError:
        logger.warning(f"Image {img_path} is outside project_dir {project_dir}, using bare filename")
        return img_path.name


# ============================================================================
# Markdown utilities
# ============================================================================

def strip_image_markers(text: str) -> str:
    """Remove pymupdf4llm **==>...<==** markers and tidy whitespace."""
    text = re.sub(r"\*\*==>.*?<==\*\*", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[\]\([^)]*\)\s*\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================================
# PDF chunk worker
# Must be a top-level function — pickle requires it for multiprocessing.
# ============================================================================

def _parse_pdf_chunk(
    pdf_path:    str,
    images_dir:  Optional[str],
    project_dir: Optional[str],
    pages:       List[int],
    do_images:   bool,
    do_caption:  bool,
    chunk_index: int,
) -> Dict:
    """
    Parse one page-range chunk of a PDF in a worker process.
    Returns a plain dict (must be picklable).

    Keys: display_text, embedding_text, images (list of dicts), error
    """
    import pymupdf4llm

    path     = Path(pdf_path)
    img_dir  = Path(images_dir)  if images_dir  else None
    proj_dir = Path(project_dir) if project_dir else None

    try:
        raw = pymupdf4llm.to_markdown(
            str(path),
            pages        = pages,
            write_images = do_images,
            image_path   = str(img_dir) if do_images else None,
            image_format = "png",
            page_chunks  = False,
            margins      = 0,
        )
    except Exception as e:
        return {"display_text": "", "embedding_text": "", "images": [], "error": str(e)}

    if not do_images:
        clean = strip_image_markers(raw)
        return {"display_text": clean, "embedding_text": clean, "images": [], "error": None}

    # ── Process images ────────────────────────────────────────────────────────
    image_files = sorted(img_dir.glob("*.png")) if img_dir else []

    processor = model = None
    cache: Dict[str, str] = {}

    if do_caption and caption_model_available() and image_files:
        try:
            processor, model, _ = load_caption_model()
            cache = load_caption_cache(img_dir)
        except Exception as e:
            logger.error(f"Caption model failed in chunk {chunk_index}: {e}")
            do_caption = False

    images_out: List[Dict]   = []
    display_text   = raw
    embedding_text = raw

    for idx, img_path in enumerate(image_files):
        if idx >= MAX_IMAGES_PER_DOC:
            break

        try:
            from PIL import Image as PILImage
            with PILImage.open(img_path) as pil_img:
                width, height = pil_img.size
        except Exception:
            width, height = 0, 0

        img_caption = ""
        if do_caption and processor and model:
            img_caption = caption_image(img_path, processor, model, cache)

        rel_path = normalize_image_path(img_path, proj_dir) if proj_dir else img_path.name

        display_md  = (
            f"\n\n![{img_caption}]({rel_path})\n*Figure: {img_caption}*\n\n"
            if img_caption else f"\n\n![image]({rel_path})\n\n"
        )
        embed_token = f"\n\n[Image: {img_caption}]\n\n" if img_caption else "\n\n[Image]\n\n"

        marker_re = re.compile(
            r"\*\*==>.*?" + re.escape(img_path.name) + r".*?<==\*\*", re.DOTALL
        )
        if marker_re.search(display_text):
            display_text   = marker_re.sub(display_md,  display_text,   count=1)
            embedding_text = marker_re.sub(embed_token, embedding_text, count=1)
        else:
            display_text   += display_md
            embedding_text += embed_token

        images_out.append({
            "path":    str(img_path),
            "page":    pages[0] + 1 if pages else 0,
            "index":   idx,
            "caption": img_caption,
            "width":   width,
            "height":  height,
        })

    display_text   = strip_image_markers(display_text)
    embedding_text = strip_image_markers(embedding_text)
    embedding_text = re.sub(r"\n{3,}", "\n\n", embedding_text).strip()

    if do_caption and CACHE_CAPTIONS and cache:
        save_caption_cache(img_dir, cache)

    return {
        "display_text":   display_text,
        "embedding_text": embedding_text,
        "images":         images_out,
        "error":          None,
    }


# ============================================================================
# parse_pdf — public entry point, routes small vs large automatically
# ============================================================================

def parse_pdf(
    path:              Path,
    images_dir:        Optional[Path] = None,
    caption:           bool = False,
    project_dir:       Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    """
    Parse a PDF. Automatically routes to single or chunked strategy.

    Small (≤ LARGE_PDF_PAGE_THRESHOLD pages) → parsed in current process
    Large (> LARGE_PDF_PAGE_THRESHOLD pages) → split into chunks, parallel workers

    Returns (display_text, embedding_text, page_count, metadata, images)
    """
    import fitz

    doc        = fitz.open(str(path))
    page_count = doc.page_count
    metadata   = dict(doc.metadata or {})
    doc.close()

    if progress_callback:
        progress_callback(f"Parsing {path.name} ({page_count} pages)...")

    if page_count <= LARGE_PDF_PAGE_THRESHOLD:
        return _parse_pdf_single(
            path, images_dir, caption, project_dir,
            progress_callback, page_count, metadata,
        )
    else:
        if progress_callback:
            n_chunks = -(-page_count // LARGE_PDF_CHUNK_PAGES)  # ceiling div
            progress_callback(
                f"Large PDF — splitting {page_count} pages into "
                f"{n_chunks} parallel chunks..."
            )
        return _parse_pdf_chunked(
            path, images_dir, caption, project_dir,
            progress_callback, page_count, metadata,
        )


def _parse_pdf_single(
    path:              Path,
    images_dir:        Optional[Path],
    caption:           bool,
    project_dir:       Optional[Path],
    progress_callback: Optional[Callable],
    page_count:        int,
    metadata:          Dict,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    """Parse a small PDF in the current process."""
    import pymupdf4llm

    do_images = bool(images_dir)

    raw = pymupdf4llm.to_markdown(
        str(path),
        write_images = do_images,
        image_path   = str(images_dir) if do_images else None,
        image_format = "png",
        page_chunks  = False,
        margins      = 0,
    )

    if not do_images:
        clean = strip_image_markers(raw)
        return clean, clean, page_count, metadata, []

    image_files = sorted(images_dir.glob("*.png"))
    processor = model = None
    cache: Dict[str, str] = {}

    if caption and caption_model_available() and image_files:
        try:
            if progress_callback:
                progress_callback("Loading caption model...")
            processor, model, _ = load_caption_model()
            cache = load_caption_cache(images_dir)
        except Exception as e:
            logger.error(f"Caption model failed: {e}")
            caption = False

    all_images: List[ExtractedImage] = []
    display_text   = raw
    embedding_text = raw

    for idx, img_path in enumerate(image_files):
        if idx >= MAX_IMAGES_PER_DOC:
            break

        try:
            from PIL import Image as PILImage
            with PILImage.open(img_path) as pil_img:
                width, height = pil_img.size
        except Exception:
            width, height = 0, 0

        img_caption = ""
        if caption and processor and model:
            img_caption = caption_image(img_path, processor, model, cache, progress_callback)

        all_images.append(ExtractedImage(
            path=img_path, page=0, index=idx,
            caption=img_caption, width=width, height=height,
        ))

        rel_path = normalize_image_path(img_path, project_dir) if project_dir else img_path.name

        display_md  = (
            f"\n\n![{img_caption}]({rel_path})\n*Figure: {img_caption}*\n\n"
            if img_caption else f"\n\n![image]({rel_path})\n\n"
        )
        embed_token = f"\n\n[Image: {img_caption}]\n\n" if img_caption else "\n\n[Image]\n\n"

        marker_re = re.compile(
            r"\*\*==>.*?" + re.escape(img_path.name) + r".*?<==\*\*", re.DOTALL
        )
        if marker_re.search(display_text):
            display_text   = marker_re.sub(display_md,  display_text,   count=1)
            embedding_text = marker_re.sub(embed_token, embedding_text, count=1)
        else:
            display_text   += display_md
            embedding_text += embed_token

        if progress_callback and idx % 5 == 0:
            progress_callback(f"Processed {idx + 1}/{len(image_files)} images")

    display_text   = strip_image_markers(display_text)
    embedding_text = strip_image_markers(embedding_text)
    embedding_text = re.sub(r"\n{3,}", "\n\n", embedding_text).strip()

    if caption and CACHE_CAPTIONS and cache:
        save_caption_cache(images_dir, cache)

    return display_text, embedding_text, page_count, metadata, all_images


def _parse_pdf_chunked(
    path:              Path,
    images_dir:        Optional[Path],
    caption:           bool,
    project_dir:       Optional[Path],
    progress_callback: Optional[Callable],
    page_count:        int,
    metadata:          Dict,
) -> Tuple[str, str, int, Dict, List[ExtractedImage]]:
    """
    Split a large PDF into page-range chunks, parse in parallel, glue results.
    """
    all_pages = list(range(page_count))
    chunks    = [
        all_pages[i : i + LARGE_PDF_CHUNK_PAGES]
        for i in range(0, page_count, LARGE_PDF_CHUNK_PAGES)
    ]
    n_chunks  = len(chunks)
    n_workers = min(n_chunks, max(1, CPU_COUNT // 2) if caption else CPU_COUNT)

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

    # Glue in page order
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

    if progress_callback:
        progress_callback(f"Glued {n_chunks} chunks — {page_count} pages complete")

    return display_text, embedding_text, page_count, metadata, all_images


# ============================================================================
# markitdown (non-PDF)
# ============================================================================

def parse_markitdown(path: Path) -> Tuple[str, int, Dict, List]:
    from markitdown import MarkItDown
    result     = MarkItDown().convert(str(path))
    text       = result.text_content or ""
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    word_count = len(text.split())
    page_count = max(1, round(word_count / 275))
    return text.strip(), page_count, {}, []