import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict



URL_RE = re.compile(r"^https?://", re.IGNORECASE)

BLOCKED_HOSTS = frozenset({
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
})

PRIVATE_IP_PREFIXES = (
    "10.", "192.168.", "172.16.", "172.17.", "172.18.",
    "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
    "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
    "172.29.", "172.30.", "172.31.",
)

# ── Constants ──────────────────────────────────────────────────────────────────

SUPPORTED_TYPES: Dict[str, str] = {
    ".pdf":  "pdf",
    ".docx": "markitdown",
    ".pptx": "markitdown",
    ".txt":  "markitdown",
    ".md":   "markitdown",
    ".html": "html",
    ".htm":  "html",
    ".xml":  "xml",
    ".csv":  "csv",
    ".tsv":  "csv",
    ".xlsx": "spreadsheet",
    ".xls":  "spreadsheet",
    ".ods":  "spreadsheet",
    ".epub": "ebook",
    ".eml":  "email",
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


def is_good_enough(text: str,darwin) -> bool:
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



# ── BlobHandler ─────────────────────────────────────────────────────────────

@dataclass
class BlobInput:
    """Raw text content from clipboard, chat, or other non-file sources."""
    content: str                        # The raw text/HTML/markdown
    source: str                         # e.g. "clipboard", "chat", "web_clip"
    title: str = ""                     # Optional title/context
    mime_hint: str = "text/plain"       # text/plain, text/html, text/markdown
    turn_id: Optional[str] = None       # Conversation turn for unique filenames
    resolve_links: bool = True           # Immediately fetch linked URLs?
    link_timeout: float = 15.0          # Per-link fetch timeout
    image_dir: Optional[Path] = None
    @property
    def filename(self) -> str:
        """Generate unique filename from turn + source."""
        prefix = f"{self.turn_id}_" if self.turn_id else ""
        return f"{prefix}{self.source}"


@dataclass
class ParsingArgs:
    target:            str
    files: list[str] = field(default_factory=list)
    images:            bool = False
    caption:           bool = False
    ocr:           bool = False
    recursive:         bool = False
    rerun:             bool = False
    precise:           bool = False
    save:              bool = True
    cleanup:           bool = True
    skip_if_markdown:  bool = False
    blobs: List[BlobInput] = field(default_factory=list)  # ←




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

