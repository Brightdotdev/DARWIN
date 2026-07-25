import re
from pathlib import Path
from typing import List, Dict
def strip_pymupdf_markers(text: str, keep_images: bool = False) -> str:
    """Remove pymupdf4llm debug markers and optionally image tags."""
    text = re.sub(r"\*\*==>.*?<==\*\*", "", text, flags=re.DOTALL)
    if not keep_images:
        text = re.sub(r"!\[\]\([^)]*\)\s*\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_image_slot(index: int) -> str:
    """Create a stable image placeholder."""
    return f"[Image:{index}]"


def make_page_header(label: str) -> str:
    """Explicit page boundary marker."""
    return f"[Page {label}]"


def convert_markdown_images_to_slots(
    embedding_text: str,
    extracted_imgs: List[Dict],
) ->str:
    """
    Replace Markdown ![](img.png) with [Image:N] slots,
    matching filenames to the extracted image list.
    """
    filename_to_slot: Dict[str, int] = {}
    for img in extracted_imgs:
        img_path = Path(img["path"])
        filename_to_slot[img_path.name] = img["slot"]

    pattern = re.compile(r"!\[(.*?)\]\(([^)]+)\)")

    def replace(match: re.Match) -> str:
        img_ref = match.group(2)
        filename = Path(img_ref).name
        if filename in filename_to_slot:
            return make_image_slot(filename_to_slot[filename])
        return match.group(0)

    embedding_text = pattern.sub(replace, embedding_text)
    return  embedding_text