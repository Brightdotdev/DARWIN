"""
darwin.core.parser
-----------------
Parses PDF files into text using pymupdf4llm.

Supports two strategies:
  - light: fast simple text extraction
  - high: detailed extraction with layout preservation

Usage:
    doc = parsepdf("file.pdf")
    doc = parsepdf("file.pdf", strategy="high")
"""
import pymupdf4llm
import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ParsedDocument:
    text: str
    page_count: int
    metadata: dict


def parsepdf(
    file_path: str | Path,
    pages: list[int] | None = None,
    strategy: str = "light",
) -> ParsedDocument:

    path = Path(file_path)

    # --- Text extraction ---
    if strategy == "light":
        text = pymupdf4llm.to_markdown(str(path), pages=pages)
    elif strategy == "high":
        text = pymupdf4llm.to_markdown(
            str(path),
            pages=pages,
            embed_images=False,
            extract_images=False,
        )
    else:
        raise ValueError("strategy must be 'light' or 'high'")

    # --- Metadata + page count ---
    doc = fitz.open(str(path))
    page_count = doc.page_count
    metadata = doc.metadata or {}
    doc.close()

    return ParsedDocument(
        text=text,
        page_count=page_count,
        metadata=metadata,
    )