import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict


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

LARGE_PDF_PAGE_THRESHOLD = 80
LARGE_PDF_CHUNK_PAGES    = 80

# is_good_enough tuning
MIN_TEXT_LENGTH       = 100
ASCII_RATIO_THRESHOLD = 0.60
NON_LATIN_ASCII_FLOOR = 0.20

# Compiled once at module level — used in hot paths
IMAGE_SLOT_RE = re.compile(r"\[Image:(\d+)\]")

MIN_DIMENSION   = 100
MAX_CAPTION_DIM = 512
MAX_NEW_TOKENS  = 40


def should_caption_image(img_dict: dict) -> tuple[bool, str]:
    """Returns (should_caption, reason_if_skipped)."""
    w = img_dict.get("width", 0)
    h = img_dict.get("height", 0)

    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        return False, f"too small ({w}x{h}px)"

    return True, ""


def prepare_image(image_path: Path):
    """Open, convert to RGB, and resize to model's expected input size."""
    from PIL import Image as PILImage

    image = PILImage.open(image_path).convert("RGB")

    if max(image.size) > MAX_CAPTION_DIM:
        image.thumbnail((MAX_CAPTION_DIM, MAX_CAPTION_DIM), PILImage.LANCZOS)

    return image

def detect_type(path: Path) -> Optional[str]:
    return SUPPORTED_TYPES.get(path.suffix.lower())


def is_supported(path: Path) -> bool:
    return detect_type(path) is not None

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
    target:            str
    files: list[str] = field(default_factory=list)
    images:            bool = False
    caption:           bool = False
    recursive:         bool = False
    rerun:             bool = False
    precise:           bool = False
    save:              bool = True
    cleanup:           bool = True
    skip_if_markdown:  bool = False  

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

