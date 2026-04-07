"""
ParserUtils.py

Utility functions and data models for the document parser.
"""


from asyncio.log import logger
import os
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Optional, List, Callable
import math
import json

SUPPORTED_TYPES: dict[str, str] = {
    ".pdf":  "pdf",
    ".docx": "markitdown",
    ".pptx": "markitdown",
    ".txt":  "markitdown",
    ".md":   "markitdown",
}


CPU_COUNT     = os.cpu_count() or 4
LARGE_FILE_MB = 10

MAX_FILE_SIZE_MB   = 500
MIN_IMAGE_SIZE_PX  = 100    # skip tiny images — logos, bullets, decorations
MAX_IMAGES_PER_DOC = 50     # cap to avoid processing book-length image dumps


PARSE_TIMEOUT_SECONDS = 300  # 5 minutes per file
CAPTION_TIMEOUT_SECONDS = 30  # 30 seconds per image
CACHE_CAPTIONS = True  # Cache captions to avoid reprocessing




@dataclass
class ExtractedImage:
    path:    Path
    page:    int
    index:   int
    caption: str
    width:   int
    height:  int


@dataclass
class ParsedDocument:
    text:       str
    filename:   str
    page_count: int
    file_type:  str
    strategy:   str
    metadata:   dict = field(default_factory=dict)
    images:     list = field(default_factory=list)   # list[ExtractedImage]
    error:      str | None = None






def check_file_size(path: Path) -> str | None:
    """Check if file exceeds maximum size limit."""
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return (
            f"file too large: {size_mb:.0f}MB "
            f"(max {MAX_FILE_SIZE_MB}MB) — "
            f"consider splitting the document"
        )
    return None

# ── Caption model with caching ───────────────────────────────────────────────

def caption_model_available() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def get_caption_cache_path(images_dir: Path) -> Path:
    """Get path to caption cache file."""
    return images_dir / ".captions_cache.json"


def load_caption_cache(images_dir: Path) -> dict:
    """Load cached captions if they exist."""
    if not CACHE_CAPTIONS:
        return {}
    
    cache_path = get_caption_cache_path(images_dir)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load caption cache: {e}")
    return {}


def save_caption_cache(images_dir: Path, cache: dict):
    """Save captions to cache."""
    if not CACHE_CAPTIONS:
        return
    
    cache_path = get_caption_cache_path(images_dir)
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save caption cache: {e}")




# Todo :: make dynamic with llm config
def load_caption_model(model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    import torch

    # 1. Auto-detect the best device
    # Note: 'mps' is for Mac M1/M2/M3 chips!
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16 # GPUs love float16 (faster/less VRAM)
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32 # CPUs are safer with float32

    processor = AutoProcessor.from_pretrained(model_id)

    # 2. The Multi-Device Config
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" # Let Transformers handle the switching logic
    )
    
    return processor, model, device




def caption_image(
    image_path: Path,
    processor,
    model,
    cache: dict,
    progress_callback: Optional[Callable] = None
) -> str:
    """Generate a plain English caption for one image with caching."""
    # Check cache first
    if CACHE_CAPTIONS and str(image_path) in cache:
        logger.debug(f"Using cached caption for {image_path.name}")
        return cache[str(image_path)]
    
    try:
        from PIL import Image
        import torch

        if progress_callback:
            progress_callback(f"Captioning {image_path.name}")

        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text":
                    "Describe what this image shows in one or two sentences. "
                    "Focus on the content and any visible text. "
                    "If it is a diagram or chart, describe what it illustrates."},
            ],
        }]

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=80, do_sample=False)

        decoded = processor.decode(output[0], skip_special_tokens=True)
        if "Assistant:" in decoded:
            caption = decoded.split("Assistant:")[-1].strip()
        else:
            caption = decoded.strip()
        
        # Cache the result
        if CACHE_CAPTIONS:
            cache[str(image_path)] = caption
        
        return caption

    except Exception as e:
        error_msg = f"image (caption failed: {e})"
        logger.warning(f"Caption failed for {image_path.name}: {e}")
        return error_msg


# ── Image extraction with progress ───────────────────────────────────────────

def extract_pdf_images(
    path: Path,
    images_dir: Path,
    caption: bool,
    progress_callback: Optional[Callable] = None,
) -> tuple[list[ExtractedImage], dict[int, list[ExtractedImage]]]:
    """Extract images from PDF, optionally caption them."""
    import fitz

    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Load caption cache
    cache = load_caption_cache(images_dir) if caption else {}

    doc = fitz.open(str(path))
    all_images: list[ExtractedImage] = []
    page_to_imgs: dict[int, list[ExtractedImage]] = {}
    image_count = 0

    processor = model = None
    if caption and caption_model_available():
        try:
            processor, model = load_caption_model()
            logger.info(f"Caption model loaded for {path.name}")
        except Exception as e:
            logger.error(f"Caption model failed: {e} — skipping captions")
            caption = False

    for page_num in range(doc.page_count):
        if image_count >= MAX_IMAGES_PER_DOC:
            break

        page = doc[page_num]
        img_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(img_list):
            if image_count >= MAX_IMAGES_PER_DOC:
                break

            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_ext = base_image["ext"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                if width < MIN_IMAGE_SIZE_PX or height < MIN_IMAGE_SIZE_PX:
                    continue

                img_name = f"page{page_num + 1}_img{img_idx + 1}.{img_ext}"
                img_path = images_dir / img_name

                # Skip if already extracted and cached?
                if img_path.exists() and CACHE_CAPTIONS and str(img_path) in cache:
                    logger.debug(f"Using existing image {img_name}")
                else:
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                img_caption = ""
                if caption and processor and model:
                    img_caption = caption_image(
                        img_path, processor, model, cache, progress_callback
                    )

                extracted = ExtractedImage(
                    path=img_path,
                    page=page_num + 1,
                    index=img_idx + 1,
                    caption=img_caption,
                    width=width,
                    height=height,
                )
                all_images.append(extracted)
                page_to_imgs.setdefault(page_num + 1, []).append(extracted)
                image_count += 1

                if progress_callback:
                    progress_callback(f"Extracted {image_count}/{MAX_IMAGES_PER_DOC} images")

            except Exception as e:
                logger.debug(f"Failed to extract image {img_idx}: {e}")
                continue

    doc.close()
    
    # Save caption cache
    if caption and CACHE_CAPTIONS:
        save_caption_cache(images_dir, cache)
    
    return all_images, page_to_imgs


# ── Markdown image handling ────────────────────────────────────────────────────

def inject_captions(text: str, page_to_imgs: dict[int, list[ExtractedImage]]) -> str:
    """Replace pymupdf4llm image markers with caption text."""
    import re

    page_counters: dict[int, int] = {}
    pattern = re.compile(r"\*\*==>.*?<==\*\*", re.DOTALL)

    def replace_marker(match):
        pos = match.start()
        text_before = text[:pos]
        page_refs = re.findall(r"<!--\s*page[:\s]+(\d+)", text_before, re.IGNORECASE)
        page_num = int(page_refs[-1]) if page_refs else 1

        page_counters.setdefault(page_num, 0)
        page_counters[page_num] += 1
        img_idx = page_counters[page_num]

        imgs = page_to_imgs.get(page_num, [])
        img = next((i for i in imgs if i.index == img_idx), None)

        if img:
            caption = img.caption or "image"
            return (
                f"\n![image]({img.path.name})\n"
                f"*Figure: {caption}*\n"
            )
        return "\n*[image]*\n"

    return pattern.sub(replace_marker, text)





def strip_image_markers(text: str) -> str:
    """Remove all pymupdf4llm image markers when images are disabled."""
    import re
    text = re.sub(r"\*\*==>.*?<==\*\*", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Worker / page range helpers ────────────────────────────────────────────────

def workers_for_pages(page_count: int) -> int:
    if page_count <= 20: return 1
    if page_count <= 50: return 2
    if page_count <= 100: return 4
    return min(8, CPU_COUNT)


def split_page_ranges(page_count: int, n_workers: int) -> list[list[int]]:
    all_pages = list(range(page_count))
    chunk_size = math.ceil(page_count / n_workers)
    return [all_pages[i: i + chunk_size] for i in range(0, page_count, chunk_size)]


def parse_pdf_pages(file_path: str, pages: list[int]) -> str:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(file_path, pages=pages)


def parse_markitdown(path: Path) -> tuple[str, int, dict, list]:
    from markitdown import MarkItDown
    result = MarkItDown().convert(str(path))
    text = result.text_content or ""
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    word_count = len(text.split())
    page_count = max(1, round(word_count / 275))
    return text.strip(), page_count, {}, []


def parse_pdf(
    path: Path,
    images_dir: Path | None = None,
    caption: bool = False,
    progress_callback: Optional[Callable] = None,
) -> tuple[str, int, dict, list[ExtractedImage]]:
    import fitz
    import pymupdf4llm

    # 1. Metadata setup
    doc = fitz.open(str(path))
    page_count = doc.page_count
    metadata = dict(doc.metadata or {})
    doc.close()

    # Determine if we are even touching images
    do_images = bool(images_dir)

    # 2. THE FAST PARSE
    # If do_images is False, pymupdf4llm ignores them entirely.
    text = pymupdf4llm.to_markdown(
        str(path), 
        write_images=do_images, 
        image_path=str(images_dir) if do_images else None,
        image_format="png"
    )

    all_images: list[ExtractedImage] = []

    # 3. THE "DOPAMINE" LOGIC
    if not do_images:
        # Tier 1: Just clean up any stray markers and return
        text = strip_image_markers(text)
        return text, page_count, metadata, []

    # If we are here, we are Tier 2 or 3
    image_files = sorted(list(images_dir.glob("*.png")))
    
    # Load VLM ONLY if Tier 3 (captioning) is requested
    processor = model = None
    cache = {}
    if caption and caption_model_available() and image_files:
        if progress_callback: progress_callback("Loading Vision AI...")
        processor, model, device = load_caption_model()
        cache = load_caption_cache(images_dir)

    for idx, img_path in enumerate(image_files):
        if idx >= MAX_IMAGES_PER_DOC: break
        
        img_caption = ""
        # Only run the heavy VLM if explicitly asked
        if caption and processor and model:
            img_caption = caption_image(img_path, processor, model, cache, progress_callback)
        
        all_images.append(ExtractedImage(
            path=img_path, page=0, index=idx, caption=img_caption
        ))

        # Update the Markdown
        placeholder = f"![]({img_path.name})"
        if caption and img_caption:
            # Tier 3: Searchable
            replacement = f"\n\n> [VISUAL]: {img_caption}\n![{img_caption}]({img_path.name})\n\n"
        else:
            # Tier 2: Just show the image in the MD without descriptions
            replacement = f"\n![image]({img_path.name})\n"
        
        text = text.replace(placeholder, replacement)

    if caption and CACHE_CAPTIONS:
        save_caption_cache(images_dir, cache)

    return text, page_count, metadata, all_images