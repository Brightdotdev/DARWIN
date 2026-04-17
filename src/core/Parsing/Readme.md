# Darwin — Parsing Module

The parsing module converts documents into two text representations: `display_text` (readable markdown saved to disk) and `embedding_text` (clean text for vectorization). It handles PDFs, Word docs, PowerPoints, plain text, and markdown files.

---

## How to run it

```bash
# parse everything in the current directory
uv run src/core/Parsing/main.py

# parse a specific file
uv run src/core/Parsing/main.py --file path/to/file.pdf

# parse a directory recursively
uv run src/core/Parsing/main.py path/to/folder --recursive

# extract images too
uv run src/core/Parsing/main.py --images

# extract images + generate captions with SmolVLM or your selected image captioning model
uv run src/core/Parsing/main.py --images --caption

# use precise mode (slower but better layout for complex PDFs)
uv run src/core/Parsing/main.py --precise

# reprocess already-cached files
uv run src/core/Parsing/main.py --rerun

# save markdown output to disk
uv run src/core/Parsing/main.py --save-md
```

All flags can be combined. Example — precise mode with images on a single file, force reprocess:

```bash
uv run src/core/Parsing/main.py --file lecture.pdf --precise --images --caption --rerun
```

---

## Supported file types

| Extension | Parser |
|-----------|--------|
| `.pdf` | fitz (fast) or pymupdf4llm (precise) |
| `.docx` | markitdown |
| `.pptx` | markitdown |
| `.txt` | markitdown |
| `.md` | markitdown |

Files over 500MB are rejected. Parsed output with less than 100 characters or more than 40% non-ASCII is considered garbled and rejected.

---

## Two extraction modes

### Fast mode (default)

Uses `fitz.get_text("dict")` — PyMuPDF reads the raw PDF content stream and returns block/span data with no layout reconstruction. We post-process that data ourselves.

Speed: **seconds for any PDF**, including 1000+ page documents.

What it detects, all relative to the per-page modal font size (most common font size weighted by character count = body text):

- **Headings** — font size ≥ 1.5× modal → `##`, font size ≥ 1.2× modal → `###`
- **Bold body text** — bold flag set but not large enough to be a heading → `**wrapped**` in `display_text` only, plain in `embedding_text`
- **Captions** — font size ≤ 0.85× modal immediately after an image block → `*italicised*` in display, `[Caption: ...]` in embedding
- **Tables** — spans spread across more than 35% of page width → `[Table] ...` token in embedding
- **Multi-column layouts** — blocks bucketed by x-position, sorted by y within each column so reading order is correct for two-column papers
- **Images** — fitz image blocks leave a `[Image]` placeholder, replaced by actual extracted image paths if `--images` is passed

### Precise mode (`--precise`)

Uses `pymupdf4llm.to_markdown()` — full layout reconstruction including precise column ordering, table formatting, and image placement. Use this when a document has complex layouts where the fast mode produces garbled reading order.

Speed: **100–160 seconds per 80-page chunk** on CPU. For a 400-page textbook, expect 5–15 minutes.

---

## How large PDFs are handled

PDFs over 50 pages are split into chunks of 80 pages and processed in parallel using `ProcessPoolExecutor`.

```
1120-page Bible
  → 14 chunks of 80 pages
  → 4 workers (CPU count)
  → chunks 0-3 run simultaneously
  → as each finishes, next queued chunk starts
  → results stored by chunk index (not arrival order)
  → glued back together in page order
```

The key detail: `as_completed()` returns futures as they finish — chunk 7 might finish before chunk 2. Results are stored in a dict keyed by `chunk_idx`, then the glue loop reads them in order `0 → n` so the final document is always in correct page order regardless of which worker finished first.

PDFs under 50 pages are parsed in the current process with no chunking.

**Important:** Do not run multiple large PDFs simultaneously — each large PDF already uses all available CPU cores for chunking. Running two large PDFs at the same time causes workers to compete for cores and slows everything down. The module processes one file at a time for large PDFs naturally.

---

## Image extraction and captioning

When `--images` is passed, images are extracted per page using `fitz.Pixmap`. CMYK images are converted to RGB. Images smaller than 100×100 pixels are skipped. Maximum 50 images per document.

Extracted images are saved as `.png` files in the project's image directory. Their paths are injected into `display_text` as standard markdown:

```markdown
![caption text](relative/path/to/image.png)
*Figure: caption text*
```

In `embedding_text`, images appear as semantic tokens:

```
[Image: caption text]
```

When `--caption` is also passed, SmolVLM-256M-Instruct(or your selected model) generates a one-to-two sentence description of each image. Captions are cached in `.captions_cache.json` in the image directory — rerunning without `--rerun` skips already-captioned images.

---

## Output

Parsed documents are stored in the `.darwin` project directory. Each file gets its own subdirectory:

```
.darwin/
  csb_pew_bible_2nd_printing_pdf/
    csb_pew_bible_2nd_printing_pdf.md   ← display_text
    extracted_images/
      page0_img0.png
      page0_img1.png
      .captions_cache.json
```

The markdown file is `display_text` — safe to open directly in any markdown viewer. The `embedding_text` is passed downstream to the chunking and embedding pipeline and not saved separately unless `--save-md` is passed.

---

## File structure

```
src/core/Parsing/
  main.py          entry point, CLI flags, parallel file dispatch
  Parser.py        single-file parse orchestration, error handling
  ParserUtils.py   all extraction logic, data models, constants
  Captioning.py    SmolVLM model loading, caption generation, cache

src/lib/
  Terminal.py      TerminalUI + Timer class
```

### Data flow

```
CLI args
  → handle_parsing()         main.py
    → scan_directory()       find supported files
    → parse_documents()      dispatch files to workers
      → parse_document()     runs in subprocess
        → parse()            Parser.py — error handling wrapper
          → parse_pdf()      ParserUtils.py — routes small vs large
            → _fitz_extract()            fast mode
            → _pymupdf4llm_extract()     precise mode
          → _run_captioning()            if --images + --caption
        → parse_markitdown() non-PDF files
    → DarwinStore.save_markdown()   persist to .darwin/
```

---

## Key data models

### `ParsedDocument`

The output of every parse operation.

```python
@dataclass
class ParsedDocument:
    display_text:   str          # markdown, safe to write to .md
    embedding_text: str          # clean text for vectorization
    filename:       str
    page_count:     int
    file_type:      str          # "pdf" | "markitdown" | "unsupported"
    metadata:       Dict         # fitz metadata (title, author, etc.)
    error:          Optional[str] # None if successful
    images:         List[ExtractedImage]
```

### `ParsingArgs`

All CLI parameters in one place, passed through the whole call chain.

```python
@dataclass
class ParsingArgs:
    target:      str             # directory or file path
    project_dir: Optional[str]   # .darwin location, default cwd
    images:      bool            # extract images
    caption:     bool            # run SmolVLM on images
    recursive:   bool            # scan subdirectories
    rerun:       bool            # reprocess cached files
    file:        Optional[str]   # single file override
    precise:     bool            # use pymupdf4llm instead of fitz
```

---

## Tuning constants

All in `ParserUtils.py`. Change these to adjust behaviour without touching logic.

| Constant | Default | Effect |
|----------|---------|--------|
| `LARGE_PDF_PAGE_THRESHOLD` | 50 | PDFs over this page count get chunked |
| `LARGE_PDF_CHUNK_PAGES` | 80 | Pages per chunk |
| `MAX_FILE_SIZE_MB` | 500 | Files larger than this are rejected |
| `MAX_IMAGES_PER_DOC` | 50 | Images extracted per document |
| `MIN_IMAGE_SIZE_PX` | 100 | Images smaller than this are skipped |
| `HEADING_SIZE_RATIO` | 1.20 | Font ≥ this × modal → heading |
| `LARGE_HEADING_RATIO` | 1.50 | Font ≥ this × modal → `##` (else `###`) |
| `CAPTION_SIZE_RATIO` | 0.85 | Font ≤ this × modal near image → caption |
| `TABLE_X_SPREAD` | 0.35 | Spans across > 35% page width → table |
| `COLUMN_BUCKETS` | 2 | Number of x-columns to detect |
| `CACHE_CAPTIONS` | True | Cache SmolVLM captions to disk |

---

## The `Timer` utility

Available anywhere in the codebase via `TerminalUI.timer()`. Prints through rich console so output looks native to the rest of Darwin's terminal UI.

```python
from src.lib.Terminal import TerminalUI

t = TerminalUI.timer("embedding")
embed_chunks(chunks)
t.split("ollama_embed")    # ⏱  [embedding] ollama_embed · 3.21s
save_npy(embeddings)
t.split("save_npy")        # ⏱  [embedding] save_npy · 0.04s
t.done()                   # ⏱  [embedding] ── total · 3.25s ──
```

Note: `Timer` inside chunk workers (`_parse_pdf_chunk`) runs in subprocesses. Those timings print to worker stdout which is not visible in the main terminal. Timings in `parse_pdf`, `_parse_pdf_single`, and `_parse_pdf_chunked` are in the main process and show up normally.

---

## Performance reference

Tested on a 4-core Windows machine, no GPU.

| Document | Pages | Mode | Time |
|----------|-------|------|------|
| CSB Pew Bible | 1120 | fast | ~45s |
| CSB Pew Bible | 1120 | precise | ~16 min |
| OpenClaw ebook | 41 | fast | ~2s |
| OpenClaw ebook | 41 | precise | ~24s |

The fast mode is 20–25× faster than precise for large documents...but ofc there is a trade off in accuracy compared to using pmdf4llm directly.