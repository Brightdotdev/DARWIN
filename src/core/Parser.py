"""
src/core/Parser.py
------------------
Parses documents into clean markdown text ready for chunking.

Uses:
    pymupdf4llm  → PDF (fast, great markdown output)
    markitdown   → DOCX, PPTX, TXT, MD (handles everything else)

Parallelism:
    single large PDF  → split by page ranges across cores
    multiple files    → one slot per file in ProcessPoolExecutor
    never both        → safe on user's machine

Usage:
    # single file
    doc = parse("biology.pdf")
    doc = parse("notes.docx")

    # multiple files at once
    docs = parse_many(["biology.pdf", "notes.docx", "lecture.pptx"])

    # scan a whole folder
    docs = scan_directory("/path/to/notes")

    # use the result
    doc.text        # clean markdown → feed straight to chunk_document()
    doc.page_count  # for citations
    doc.filename    # original filename
    doc.error       # None if success, message if failed
"""

import os
import math
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path


# ── Supported types ────────────────────────────────────────────────────────────
# extend this dict to support new formats — nothing else needs changing

SUPPORTED_TYPES: dict[str, str] = {
    ".pdf":  "pdf",
    ".docx": "markitdown",
    ".pptx": "markitdown",
    ".txt":  "markitdown",
    ".md":   "markitdown",
}

# ── Parallelism ────────────────────────────────────────────────────────────────

CPU_COUNT     = os.cpu_count() or 4
LARGE_FILE_MB = 10


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class ParsedDocument:
    text:       str                              # markdown → feed to chunk_document()
    filename:   str                              # original filename
    page_count: int                              # pages (1 for non-PDF)
    file_type:  str                              # pdf | markitdown | unsupported
    strategy:   str                              # light | high
    metadata:   dict = field(default_factory=dict)
    error:      str | None = None


# ── Worker count ───────────────────────────────────────────────────────────────

def _workers_for_pages(page_count: int) -> int:
    if page_count <= 20:  return 1
    if page_count <= 50:  return 2
    if page_count <= 100: return 4
    return min(8, CPU_COUNT)


def _split_page_ranges(page_count: int, n_workers: int) -> list[list[int]]:
    """Split page indices into n_workers equal ranges."""
    all_pages  = list(range(page_count))
    chunk_size = math.ceil(page_count / n_workers)
    return [
        all_pages[i : i + chunk_size]
        for i in range(0, page_count, chunk_size)
    ]


# ── PDF worker (runs in subprocess) ───────────────────────────────────────────

def _parse_pdf_pages(file_path: str, pages: list[int]) -> str:
    """
    Parse specific pages from a PDF.
    Called in a subprocess — only takes serializable args.
    Opens the file independently so concurrent reads work fine.
    """
    import pymupdf4llm
    return pymupdf4llm.to_markdown(file_path, pages=pages)


# ── Individual parsers ─────────────────────────────────────────────────────────

def _parse_pdf(path: Path) -> tuple[str, int, dict]:
    """
    Parse PDF with pymupdf4llm.
    Large files (>10MB) split across CPU cores by page ranges.
    Never splits the file itself — only the page list.
    """
    import fitz
    import pymupdf4llm

    doc        = fitz.open(str(path))
    page_count = doc.page_count
    metadata   = dict(doc.metadata or {})
    doc.close()

    file_size_mb = path.stat().st_size / (1024 * 1024)
    n_workers    = _workers_for_pages(page_count)

    # small file → parse directly, no subprocess overhead
    if n_workers == 1 or file_size_mb < LARGE_FILE_MB:
        text = pymupdf4llm.to_markdown(str(path))
        return text, page_count, metadata

    # large file → split page ranges across workers
    page_ranges = _split_page_ranges(page_count, n_workers)
    parts       = [""] * len(page_ranges)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(_parse_pdf_pages, str(path), rng): idx
            for idx, rng in enumerate(page_ranges)
        }
        for future in as_completed(future_to_idx):
            idx        = future_to_idx[future]
            parts[idx] = future.result()

    text = "\n\n".join(p for p in parts if p.strip())
    return text, page_count, metadata


def _parse_markitdown(path: Path) -> tuple[str, int, dict]:
    """
    Parse DOCX, PPTX, TXT, MD using markitdown.
    Returns markdown text. Page count approximated from word count.
    """
    from markitdown import MarkItDown

    result = MarkItDown().convert(str(path))
    text   = result.text_content or ""

    # normalize excessive blank lines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    # approximate page count — ~275 words per page
    word_count = len(text.split())
    page_count = max(1, round(word_count / 275))

    return text.strip(), page_count, {}


# ── Type detection ─────────────────────────────────────────────────────────────

def detect_type(path: Path) -> str | None:
    """Returns type string or None if unsupported."""
    return SUPPORTED_TYPES.get(path.suffix.lower())


def is_supported(path: Path) -> bool:
    return detect_type(path) is not None


# ── Quality check (from your original agent.py) ───────────────────────────────

def is_good_enough(text: str) -> bool:
    """
    Checks if parsed text is usable — not empty or garbled.
    Adapted from your original agent.py — keeping your logic.
    """
    if not text or len(text.strip()) < 100:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / len(text)) > 0.6


# ── Main parse function ────────────────────────────────────────────────────────

def parse(
    file_path: str | Path,
    strategy:  str = "light",
) -> ParsedDocument:
    """
    Parse a single file into a ParsedDocument.
    Never raises — errors go into .error field.

    Args:
        file_path: path to file
        strategy:  "light" or "high" (only affects PDF for now)

    Returns:
        ParsedDocument with .text ready for chunk_document()
    """
    path = Path(file_path).resolve()

    if not path.exists():
        return ParsedDocument(
            text       = "",
            filename   = path.name,
            page_count = 0,
            file_type  = "unknown",
            strategy   = strategy,
            error      = f"file not found: {path}",
        )

    file_type = detect_type(path)
    if file_type is None:
        supported = ", ".join(SUPPORTED_TYPES.keys())
        return ParsedDocument(
            text       = "",
            filename   = path.name,
            page_count = 0,
            file_type  = "unsupported",
            strategy   = strategy,
            error      = f"unsupported type '{path.suffix}' — supported: {supported}",
        )

    try:
        if file_type == "pdf":
            text, page_count, metadata = _parse_pdf(path)
        else:
            text, page_count, metadata = _parse_markitdown(path)

        # quality gate — your original logic from agent.py
        if not is_good_enough(text):
            return ParsedDocument(
                text       = "",
                filename   = path.name,
                page_count = page_count,
                file_type  = file_type,
                strategy   = strategy,
                metadata   = metadata,
                error      = "parsed text is empty or garbled",
            )

        return ParsedDocument(
            text       = text,
            filename   = path.name,
            page_count = page_count,
            file_type  = file_type,
            strategy   = strategy,
            metadata   = metadata,
            error      = None,
        )

    except Exception as e:
        return ParsedDocument(
            text       = "",
            filename   = path.name,
            page_count = 0,
            file_type  = file_type,
            strategy   = strategy,
            error      = f"{type(e).__name__}: {e}",
        )


# ── Multi-file parsing ─────────────────────────────────────────────────────────

def parse_many(
    file_paths:       list[str | Path],
    strategy:         str  = "light",
    skip_unsupported: bool = True,
) -> list[ParsedDocument]:
    """
    Parse multiple files in parallel.
    Results returned in same order as input — even if workers finish out of order.
    Failed files included with .error set — never silently dropped.
    """
    paths   = [Path(p).resolve() for p in file_paths]
    results: list[ParsedDocument | None] = [None] * len(paths)
    to_parse: list[tuple[int, Path]]     = []

    for idx, path in enumerate(paths):
        if not is_supported(path):
            if skip_unsupported:
                warnings.warn(
                    f"skipping unsupported file: {path.name}",
                    stacklevel=2,
                )
                results[idx] = ParsedDocument(
                    text       = "",
                    filename   = path.name,
                    page_count = 0,
                    file_type  = "unsupported",
                    strategy   = strategy,
                    error      = f"unsupported type: {path.suffix}",
                )
            else:
                to_parse.append((idx, path))
        else:
            to_parse.append((idx, path))

    if not to_parse:
        return [r for r in results if r is not None]

    # single file — skip pool overhead
    if len(to_parse) == 1:
        idx, path    = to_parse[0]
        results[idx] = parse(path, strategy=strategy)
        return results  # type: ignore

    # multiple files — one worker per file
    n_workers = min(len(to_parse), CPU_COUNT)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(parse, path, strategy): idx
            for idx, path in to_parse
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                path         = paths[idx]
                results[idx] = ParsedDocument(
                    text       = "",
                    filename   = path.name,
                    page_count = 0,
                    file_type  = detect_type(path) or "unknown",
                    strategy   = strategy,
                    error      = f"worker failed: {type(e).__name__}: {e}",
                )

    return results  # type: ignore


# ── Directory scanning ─────────────────────────────────────────────────────────

def scan_directory(
    directory: str | Path,
    recursive: bool = False,
    strategy:  str  = "light",
) -> list[ParsedDocument]:
    """
    Find and parse all supported files in a directory.
    Skips hidden files and .darwin / .agent folders automatically.
    """
    directory = Path(directory).resolve()

    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    pattern   = "**/*" if recursive else "*"
    all_files = [
        p for p in directory.glob(pattern)
        if p.is_file()
        and is_supported(p)
        and not any(part.startswith(".") for part in p.parts)
    ]

    if not all_files:
        return []

    return parse_many(all_files, strategy=strategy)