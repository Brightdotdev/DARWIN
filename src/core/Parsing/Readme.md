# Parsing Module

Converts raw document files (PDF, DOCX, PPTX, TXT, MD) into structured text
ready for chunking and embedding. Every parsed document produces two text
representations — one for display to humans, one optimised for embedding
models — plus optional extracted images, image captions, and hyperlinks.

---

## Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)
  - [1. Entry Point — `main.py`](#1-entry-point--mainpy)
  - [2. Orchestrator — `Parser.py`](#2-orchestrator--parserpy)
  - [3. Utilities — `ParserUtils.py`](#3-utilities--parserutilspy)
  - [4. Captioning — `Captioning.py`](#4-captioning--captioningpy)
- [Output Format](#output-format)
  - [ParsedDocument](#parseddocument)
  - [ExtractedImage](#extractedimage)
  - [ExtractedLink](#extractedlink)
- [Modes](#modes)
  - [Fast Mode (default)](#fast-mode-default)
  - [Precise Mode (`--precise`)](#precise-mode---precise)
- [Text Structure Signals](#text-structure-signals)
- [Link Extraction](#link-extraction)
- [Image Extraction and Captioning](#image-extraction-and-captioning)
- [Large PDF Handling](#large-pdf-handling)
- [Caching](#caching)
- [CLI Usage](#cli-usage)
- [Limitations](#limitations)

---

## Overview

```
Input file(s)
     │
     ▼
scan_directory()          ← finds supported files
     │
     ▼
parse_document()          ← subprocess worker, one per file
     │
     ▼
parse() → ParsedDocument  ← never raises; errors in .error field
     │
     ├── PDF  → _fitz_extract()  or  _pymupdf4llm_extract()
     │          + extract_links()
     │          + _run_captioning()  (optional)
     │
     └── Other → parse_markitdown()
```

The pipeline is parallel by default: `parse_documents()` spawns one subprocess
per file using `ProcessPoolExecutor`. Large PDFs (> 80 pages) are additionally
chunked internally and processed in parallel page-ranges.

---

## File Structure

```
src/core/Parsing/
├── main.py          Entry point, CLI, orchestration
├── Parser.py        parse() and scan_directory()
├── ParserUtils.py   PDF extraction, link extraction, captioning runner
└── Captioning.py    Local VLM caption model (SmolVLM)
```

---

## How It Works

### 1. Entry Point — `main.py`

`handle_parsing(args: ParsingArgs)` is the main callable. It:

1. Scans the target path for supported files via `scan_directory()`.
2. Checks the `DarwinStore` for already-parsed files and skips them unless
   `--rerun` is passed.
3. Wraps each file in a `FileWithImages` (which carries an optional image
   output directory) and dispatches to `parse_documents()`.
4. Saves successful results to the store via `store.save_markdown()`.

`parse_documents()` manages the `ProcessPoolExecutor` pool, progress
reporting, and error collection. Files are submitted sorted by size
(smallest first) so short files don't wait behind a giant PDF.

---

### 2. Orchestrator — `Parser.py`

`parse(file_path, ...)` is the single-file entry point:

1. Validates file existence, type, and size.
2. Routes to `parse_pdf()` or `parse_markitdown()` based on extension.
3. Runs `is_good_enough()` on the result — rejects empty or heavily
   garbled text.
4. Returns a `ParsedDocument`. **Never raises** — all errors land in
   `.error`.

`scan_directory(target, recursive)` walks a directory (or accepts a single
file path) and returns a list of `Path` objects for supported file types,
skipping hidden files and directories.

---

### 3. Utilities — `ParserUtils.py`

This is the heaviest file. It contains:

#### `_fitz_extract()` — fast PDF extraction

Uses PyMuPDF (`fitz`) directly for maximum speed. Per page it:

- Reads the text dict (`page.get_text("dict")`) to get individual spans
  with font size, bold flags, and bounding boxes.
- Computes a **modal font size** by counting characters at each size. This
  is the baseline for heading and caption detection.
- Sorts blocks into reading order. For multi-column documents it buckets
  blocks by horizontal position first (`COLUMN_BUCKETS = 2`).
- Detects table blocks using `_is_table_block()` — a heuristic that checks
  x-spread, row count, column alignment consistency, and span-count
  variance. Detected tables are wrapped in `[TABLE]...[/TABLE]`.
- Stamps heading markup (`## ` / `### `) on lines whose font size exceeds
  the modal size by `HEADING_SIZE_RATIO` / `LARGE_HEADING_RATIO`.
- Emits `**bold**` for bold spans in the `display_text` only (stripped in
  `embedding_text`).
- Assigns positional `[Image:N]` slots for extracted images so captioning
  can replace them accurately later.

#### `_pymupdf4llm_extract()` — precise PDF extraction

Delegates to `pymupdf4llm.to_markdown()` which uses a layout analysis
pipeline to handle complex column layouts, wrapped text, and academic
formatting more accurately. Slower than fitz direct mode.

#### `extract_links(pdf_path, pages)` — hyperlink extraction

For each requested page:

1. Calls `page.get_links()` to get all link annotations with their URI
   and bounding rect.
2. For each URI link, calls `page.get_text("words", clip=rect)` on a
   slightly expanded clip rect to retrieve anchor text.
3. Deduplicates on `(page, url)` — the same URL on the same page is only
   recorded once.
4. Returns `List[ExtractedLink]`.

Links are serialised across subprocess boundaries via `links_to_dict()` /
`links_from_dict()`.

#### `_run_captioning()` — image captioning

After extraction, if captioning is enabled and images were found:

1. Loads the SmolVLM caption model once (via `load_caption_model()`).
2. Captions each image that isn't already in the cache.
3. Builds a `slot_map: Dict[int, Dict]` keyed by the positional slot index.
4. Calls `_replace_image_slots()` which does a **single regex pass** to
   replace all `[Image:N]` tokens with their final markdown/embed form.
   This is the key fix over the original `str.replace()` approach —
   mismatched image counts produce `[Image]` placeholders instead of
   silently mis-mapping captions.

#### `parse_pdf()` — top-level PDF dispatcher

Opens the PDF to read page count and metadata, then routes to:

- `_parse_pdf_single()` for documents ≤ 80 pages
- `_parse_pdf_chunked()` for larger documents

`_parse_pdf_chunked()` splits pages into 80-page ranges, submits each as
a separate subprocess via `ProcessPoolExecutor`, then merges the results
back in order.

#### `is_good_enough(text)` — quality gate

Three-path check:

| Condition | Action |
|-----------|--------|
| Text shorter than 100 chars | Reject |
| ASCII ratio < 20% | Non-latin script — accept if printable char count > 100 |
| ASCII ratio ≥ 20% | Latin script — require ASCII ratio > 60% |

This allows Arabic, Chinese, Japanese, and math-heavy documents to pass
without being rejected for low ASCII content.

---

### 4. Captioning — `Captioning.py`

A thin wrapper around HuggingFace `transformers`. Loads
`HuggingFaceTB/SmolVLM-256M-Instruct` (256M parameters — fast on CPU).

- Uses CUDA → MPS → CPU in that priority order.
- Cache keys are **bare filenames**, never absolute paths. Migration from
  old absolute-path keys is handled by `migrate_caption_cache()`.
- The cache is a simple JSON file (`.captions_cache.json`) written to the
  images directory after each document is processed.

---

## Output Format

### ParsedDocument

```python
@dataclass
class ParsedDocument:
    display_text:   str                  # markdown with bold, headings, image refs
    embedding_text: str                  # clean text for embedding (no bold, rel paths stripped)
    filename:       str
    page_count:     int
    file_type:      str                  # "pdf", "markitdown", "unsupported"
    metadata:       Dict                 # PDF metadata dict (title, author, etc.)
    error:          Optional[str]        # None on success
    images:         List[ExtractedImage]
    links:          List[ExtractedLink]  # all hyperlinks in the document
```

### ExtractedImage

```python
@dataclass
class ExtractedImage:
    path:    Path   # absolute path to saved PNG
    page:    int    # 0-based page number
    index:   int    # extraction order
    caption: str    # VLM caption, or "" if captioning was disabled
    width:   int    # pixels
    height:  int    # pixels
```

### ExtractedLink

```python
@dataclass
class ExtractedLink:
    url:  str         # destination URI
    page: int         # 0-based page number
    text: str         # anchor text (may be empty for image links)
    rect: List[float] # [x0, y0, x1, y1] bounding box
```

---

## Modes

### Fast Mode (default)

Uses `_fitz_extract()` — PyMuPDF's raw span dict. Fastest option.
Best for: standard business documents, reports, books with simple layouts.

### Precise Mode (`--precise`)

Uses `pymupdf4llm.to_markdown()`. Layout-aware, handles multi-column
academic papers and complex formatting better. ~3–5× slower than fast mode.

---

## Text Structure Signals

Both `display_text` and `embedding_text` use a small set of tokens that
downstream systems (the Chunker, in particular) depend on:

| Token | Meaning |
|-------|---------|
| `## heading` | Major section boundary |
| `### heading` | Sub-section boundary |
| `[TABLE]...[/TABLE]` | Structured table block (display) |
| `[Table]...` | Table block (embedding) |
| `[Caption: ...]` | Figure caption |
| `[Image: ...]` | Image reference with caption (embedding) |
| `---` | Page separator |

Do not strip or modify these tokens before passing `embedding_text` to
the Chunker — it relies on them for section detection and page tracking.

---

## Link Extraction

Links are extracted for every PDF page processed and attached to
`ParsedDocument.links`. To pass them to the Chunker:

```python
from src.core.Parsing.ParserUtils import links_to_dict

doc = parse(path)
raw_links = links_to_dict(doc.links)   # serialisable list of dicts
chunks, meta = chunk_for_model(text=doc.embedding_text, raw_links=raw_links)
# each chunk.links now contains the hyperlinks scoped to its page range
```

---

## Image Extraction and Captioning

Image extraction requires `--images` flag. Captioning additionally requires
`--caption` (and `transformers` + `torch` installed).

Images are saved as PNGs under the project's image directory:

```
.darwin/
└── images/
    └── my_document.pdf/
        ├── page0_img0.png
        ├── page0_img1.png
        └── .captions_cache.json
```

Images smaller than `MIN_IMAGE_SIZE_PX = 100` pixels in either dimension
are silently skipped (decorative rules, icons, etc.).

---

## Large PDF Handling

PDFs over `LARGE_PDF_PAGE_THRESHOLD = 80` pages are split into chunks of
`LARGE_PDF_CHUNK_PAGES = 80` pages and processed in parallel. Worker count
is `CPU_COUNT` normally, halved when captioning is enabled (to avoid OOM
from multiple VLM model copies).

Results are merged back in page order after all workers complete.

---

## Caching

Two levels of caching:

**Parsed markdown** — `DarwinStore.has_markdown(filename)` returns True if
a file has already been parsed. Skipped unless `--rerun` is passed.

**Image captions** — `.captions_cache.json` in the image directory. Keyed
by bare filename. Survives across reruns so you don't re-caption images
when only the text changed.

---

## CLI Usage

```bash
# Parse all files in current directory
uv run src/core/Parsing/main.py

# Parse a single file
uv run src/core/Parsing/main.py --file report.pdf

# Parse recursively
uv run src/core/Parsing/main.py --recursive

# Extract images and generate captions
uv run src/core/Parsing/main.py --images --caption

# Use layout-accurate extraction (slower)
uv run src/core/Parsing/main.py --precise

# Reprocess already-parsed files
uv run src/core/Parsing/main.py --rerun

# Parse but don't save (preview)
uv run src/core/Parsing/main.py --no-save

# Custom project directory
uv run src/core/Parsing/main.py --project /path/to/project
```

---

## Limitations

**Table detection is heuristic.** `_is_table_block()` uses x-spread, row
consistency, and column alignment to detect tables. It works well on
clean business PDFs but can misclassify:

- Dense academic tables with irregular column widths
- Text blocks in two-column layouts (false positive)
- Tables without visible borders where whitespace is inconsistent

Enable `DEBUG` logging to see per-block rejection reasons.

**Non-latin and mixed-script documents.** The `is_good_enough()` quality
gate accepts non-latin documents based on a printable-character count
rather than the ASCII ratio. This is correct but means a heavily corrupted
CJK scan may pass the gate if it has enough characters. There is no OCR
fallback — scanned documents without a text layer will produce empty or
garbled output.

**Column detection is fixed at 2.** `COLUMN_BUCKETS = 2` assumes at most
two columns. Three-column layouts (common in academic papers) will have
their reading order scrambled.

**Image slot alignment.** The positional `[Image:N]` slot system requires
that `_fitz_extract()` and `_run_captioning()` agree on how many images
were extracted from a chunk. If an image fails to save mid-extraction,
subsequent slot indices will be off by one, causing captions to be
attached to the wrong image reference. Failures are logged at WARNING
level.

**Page numbers are estimates.** For non-PDF formats (DOCX, PPTX) page
count is derived from word count (`words / 275`). The chunker's page
tracking is based on counting `---` separators, not real page metadata.

**Caption model quality.** SmolVLM-256M is fast and small but produces
generic captions. For high-quality caption retrieval, swap in a larger
model via the `model_id` parameter in `load_caption_model()`.

**Link extraction is URI-only.** Internal PDF cross-references (page
jumps, named destinations) are not extracted — only external `uri` links.
Email links (`mailto:`) and relative paths are included as-is without
normalisation.

**Subprocess isolation.** Each file is parsed in its own subprocess. This
means any module-level state (e.g. a loaded caption model) is not shared
between files. For large batches with captioning enabled, the model is
loaded once per worker process, which can be memory-intensive.