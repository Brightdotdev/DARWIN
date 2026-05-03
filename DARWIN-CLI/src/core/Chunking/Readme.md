# Chunking Module

Converts parser output (`embedding_text`) into overlapping, token-bounded
chunks ready for embedding and retrieval. Chunks preserve document structure
(headings, tables, captions), carry page and citation metadata, and — when
link data is supplied — each chunk knows which hyperlinks appear on its pages.

---

## Table of Contents

- [Overview](#overview)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)
  - [1. Entry Point — `main.py`](#1-entry-point--mainpy)
  - [2. Public API — `Chunker.py`](#2-public-api--chunkerpy)
  - [3. Utilities — `ChunkerUtils.py`](#3-utilities--chunkerutilspy)
- [Chunking Strategy](#chunking-strategy)
  - [Step 1: Section Splitting](#step-1-section-splitting)
  - [Step 2: Section to Chunks](#step-2-section-to-chunks)
  - [Step 3: Sentence Splitting with Overlap](#step-3-sentence-splitting-with-overlap)
  - [Step 4: Small Chunk Merging](#step-4-small-chunk-merging)
  - [Step 5: Link Assignment](#step-5-link-assignment)
- [Output Format](#output-format)
  - [Chunk](#chunk)
  - [ChunkLink](#chunklink)
- [Token Counting](#token-counting)
- [Model Presets](#model-presets)
- [Dynamic Sizing](#dynamic-sizing)
- [CLI Usage](#cli-usage)
- [Limitations](#limitations)

---

## Overview

```
embedding_text  (from ParsedDocument)
      │
      ▼
split_into_sections()     ← split at ## / ### heading boundaries
      │
      ▼
section_to_chunks()       ← one section at a time
      │
      ├── fits in budget?      → one Chunk
      ├── [Table] block?       → one Chunk, never split
      ├── [Caption:] block?    → appended to previous Chunk
      └── too large?
              │
              ├── split into paragraphs, each inherits heading
              └── paragraph still too large?
                      └── sentence split with overlap
      │
      ▼
merge_small()             ← merge undersized consecutive chunks
      │
      ▼
assign_links_to_chunks()  ← stamp per-chunk hyperlinks (optional)
      │
      ▼
List[Chunk]
```

---

## File Structure

```
src/core/Chunking/
├── main.py         Entry point, CLI, inspection output
├── Chunker.py      chunk_document() and chunk_for_model() — public API
└── ChunkerUtils.py All logic: splitting, cleaning, merging, link assignment
```

---

## How It Works

### 1. Entry Point — `main.py`

`handle_chunking(args: ChunkingArgs)` is the main callable. It:

1. Resolves which files to chunk (single file or all parsed files in the
   store).
2. Skips files with fresh chunks unless `--rerun` is passed.
3. Loads `embedding_text` from the store (or `args.embedding_text` if
   provided directly).
4. Loads `raw_links` from the store if `store.load_links()` is available.
5. Calls `chunk_for_model()` and saves the result via `store.save_chunks()`.
6. Prints a summary including total links distributed across chunks.

With `--inspect`, a detailed breakdown of chunk stats and a sample of
individual chunks (with link previews) is printed to the terminal.

---

### 2. Public API — `Chunker.py`

#### `chunk_document(text, source, max_tokens, min_tokens, overlap, raw_links=None)`

The core function. Accepts explicit token budget parameters (set by the
caller based on their embedding model) and an optional list of link dicts.

Returns `List[Chunk]`, re-indexed from 0, with `.links` populated if
`raw_links` was supplied.

#### `chunk_for_model(text, source, raw_links=None)`

Convenience wrapper. Calls `get_dynamic_chunk_params_for_text()` to
automatically select token budget parameters based on the document's
length and the configured embedding model, then calls `chunk_document()`.

Returns `(List[Chunk], chunk_metadata)` where `chunk_metadata` contains
the resolved parameters plus document stats — suitable for passing
directly to `store.save_chunks(..., chunk_meta=chunk_metadata)`.

---

### 3. Utilities — `ChunkerUtils.py`

#### `_clean(text)` — markdown stripping

Produces the `clean` field of each Chunk. Removes:

- Code blocks and inline code
- Image references
- Hyperlink syntax (keeps anchor text)
- Bold / italic markers
- HTML tags and custom markers
- Heading `#` prefixes (replaced with newline)

Converts bullet lists to `•` prefix, normalises whitespace, and
preserves paragraph separation (`\n\n`). The result is plain prose
that embedding models handle without markdown tokenization noise.

#### `split_into_sections(text)` — section detection

Splits `embedding_text` at Markdown heading boundaries (`#` through
`######`) using a single regex pass. Returns a list of
`(heading, body, level)` tuples.

Text before the first heading is emitted as a headingless section
`("", preamble, 0)`. If no headings exist, the entire document is
returned as a single section — the chunker falls back to paragraph
splitting in that case.

#### `section_to_chunks()` — section processing

Processes one `(heading, body)` pair:

1. If the whole body fits within `max_tokens` → emit one Chunk.
2. Otherwise split on `\n\n` into paragraphs and process each:
   - `[Table]` blocks are always their own Chunk (never split mid-table).
   - `[Caption:]` blocks are appended to the previous Chunk if they fit,
     otherwise they become their own Chunk.
   - Small paragraphs are accumulated into `pending` and flushed when
     the pending buffer reaches `min_tokens` or a large paragraph arrives.
   - Paragraphs exceeding `max_tokens` are sentence-split with overlap.

Page tracking: `current_page` is incremented by `count_page_seps(para)`
for every paragraph, so `page_end` is accurate even for multi-page tables
and long sections.

#### `_split_with_overlap()` — sentence-level splitting

Called when a paragraph exceeds `max_tokens`. Uses BlingFire (fast,
multilingual) or a regex fallback to split text into sentences.

Maintains two parallel windows:

- `window_clean` — cleaned sentences, used only for token budget checking
- `window_orig` — original sentences, used to build the final Chunk text

The budget check uses `window_clean` so token counting and final Chunk
construction are consistent. When a window overflows, the last `overlap`
sentences are carried into the next window as context.

#### `merge_small()` — post-processing merge

After all sections are processed, any chunks still below `min_tokens` are
merged with their neighbours. Two rules:

| Rule | Condition | Action |
|------|-----------|--------|
| Same-section merge | Both chunks < `min_tokens`, same heading, fits in `max_tokens` | Merge, keep heading |
| Cross-section tiny merge | Both chunks < `min_tokens / 2`, any heading, fits | Merge, keep first heading |

The cross-section rule prevents micro-chunk floods in documents with many
terse headings (legal documents, academic papers, slide-converted PDFs).

#### `assign_links_to_chunks()` — link distribution

Iterates over every chunk and every link. A link on page N is assigned to
a chunk if `chunk.page <= N <= chunk.page_end`. Duplicate URLs within a
single chunk are deduplicated. Links with empty URLs are skipped.

A link near a page boundary may end up on two adjacent chunks — this is
intentional. It is better to have a link in two retrieval candidates than
to miss it in both.

---

## Chunking Strategy

### Step 1: Section Splitting

The parser bakes heading structure into `embedding_text` using standard
Markdown `##` / `###` syntax. `split_into_sections()` uses this to divide
the document into logical units before any token-level decisions are made.

This is the key design choice: **split on meaning first, then on size**.
Paragraph-first strategies produce chunks that cross section boundaries,
which hurts retrieval precision because the chunk's heading no longer
describes its content.

### Step 2: Section to Chunks

Each section is evaluated against the token budget:

```
section fits whole   →  one chunk (best case — full section context)
section too large    →  paragraph loop
  paragraph fits     →  accumulate in pending buffer
  paragraph too big  →  sentence split
```

The heading of the section is inherited by every chunk produced from it.
This means every chunk can be cited with its section title even if the
section was split across many chunks.

### Step 3: Sentence Splitting with Overlap

Overlap is measured in **sentences**, not tokens. The `overlap` parameter
(typically 2–5 sentences depending on model) controls how many sentences
from the end of one chunk are prepended to the start of the next.

This preserves cross-boundary context for retrieval: a query about
something that straddles two chunks will find both candidates, and the
LLM will have the overlap sentences available to reconstruct the full
context.

### Step 4: Small Chunk Merging

After all sections are chunked, a single left-to-right merge pass reduces
fragment chunks. The cross-heading tiny merge is conservative (only when
both chunks are below half of `min_tokens`) to avoid collapsing distinct
sections.

### Step 5: Link Assignment

Links are assigned after final re-indexing so `page` and `page_end` on
each Chunk are stable. The assignment is O(chunks × links) — for typical
documents (< 1000 chunks, < 500 links) this is negligible.

---

## Output Format

### Chunk

```python
@dataclass
class Chunk:
    text:        str          # original text — send to LLM for generation
    clean:       str          # stripped plain text — feed to embedding model
    index:       int          # 0-based position in document
    source:      str          # filename — use for citations
    token_count: int          # estimated tokens of clean text
    heading:     str          # nearest section heading (may be "")
    page:        int          # first page this chunk touches (1-based)
    page_end:    int          # last page this chunk touches
    is_overlap:  bool         # True if chunk contains overlap from previous
    links:       List[ChunkLink]  # hyperlinks scoped to this chunk's pages
```

### ChunkLink

```python
@dataclass
class ChunkLink:
    url:  str          # destination URI
    page: int          # page where the link appears (0-based, from parser)
    text: str          # anchor text (may be empty)
    rect: List[float]  # [x0, y0, x1, y1] bounding box
```

**Passing links to your vector store:**

```python
# Serialise a chunk's links for storage in metadata
chunk_meta = {
    "source":   chunk.source,
    "heading":  chunk.heading,
    "page":     chunk.page,
    "page_end": chunk.page_end,
    "links": [
        {"url": lk.url, "text": lk.text, "page": lk.page}
        for lk in chunk.links
    ],
}
```

---

## Token Counting

`_tokens(text)` uses tiktoken (`cl100k_base`) when available, loaded once
per process via `@lru_cache`. This encoder is a good proxy for most modern
transformer tokenizers (OpenAI, BGE, E5, etc.).

When tiktoken is not installed the fallback is `len(text.split()) / 0.75`,
which is accurate for standard prose but **underestimates** for:

- CJK text (characters are not space-separated)
- Dense code blocks
- Documents with many numbers or symbols

Install tiktoken (`pip install tiktoken`) for accurate counting in
production.

---

## Model Presets

`get_model_preset(model_name)` returns `{max_tokens, min_tokens, overlap}`
for known embedding models:

| Model | max_tokens | min_tokens | overlap |
|-------|-----------|-----------|---------|
| e5-small-v2 / e5-base-v2 / e5-large-v2 | 480 | 120 | 3 |
| bge-base | 480 | 120 | 3 |
| bge-m3 | 8000 | 400 | 5 |
| gritlm | 1900 | 200 | 3 |
| gritlm-7b | 3800 | 300 | 4 |
| text-embedding-3-small/large | 8000 | 400 | 5 |
| text-embedding-ada-002 | 8000 | 400 | 5 |
| embed-english / embed-multilingual | 480 | 120 | 3 |
| voyage | 3800 | 300 | 4 |
| *(default)* | 350 | 80 | 2 |

Partial name matching is supported — `"intfloat/e5-small-v2"` matches
`"e5-small-v2"`.

---

## Dynamic Sizing

`get_dynamic_chunk_params_for_text()` scales the preset's `max_tokens`
down for longer documents:

| Document length | Chunk size ratio | Extra overlap sentences |
|----------------|-----------------|------------------------|
| < 2,000 tokens (1–2 pages) | 100% | 0 |
| 2,000–10,000 (5–10 pages) | 90% | 0 |
| 10,000–50,000 (25–50 pages) | 75% | 1 |
| 50,000–200,000 (100–200 pages) | 60% | 2 |
| > 200,000 (book-length) | 50% | 3 |

Rationale: longer documents need smaller, more precise chunks to avoid
retrieval candidates that are too broad. Short documents can use larger
chunks to preserve full paragraph context.

The minimum chunk size is always `max(150, max_capacity × 0.4)`.

---

## CLI Usage

```bash
# Chunk all parsed files in current directory
uv run src/core/Chunking/main.py

# Chunk a single file
uv run src/core/Chunking/main.py --file biology.pdf

# Inspect chunking output before embedding
uv run src/core/Chunking/main.py --inspect

# Force rechunk even if fresh chunks exist
uv run src/core/Chunking/main.py --rerun

# Custom project directory
uv run src/core/Chunking/main.py --project /path/to/project
```

---

## Limitations

**Heading detection requires parser cooperation.** `split_into_sections()`
relies entirely on `##` / `###` markers in `embedding_text`. If the parser
fails to detect headings (e.g. a scanned PDF with no font-size variation,
or a plain-text file with no markdown) the entire document becomes one
headingless section and falls back to paragraph splitting. All chunks will
have an empty `heading` field, which degrades citation quality.

**Page tracking is based on `---` separators, not real metadata.** Page
numbers in chunks are derived by counting `---` markers emitted by the
parser. If the parser emits extra or missing separators (e.g. in chunked
large-PDF mode where page boundary detection can drift slightly), chunk
page numbers will be off. For strict page-accurate citation, cross-check
against `ParsedDocument.metadata`.

**Overlap can slightly exceed `max_tokens` in edge cases.** When the last
`overlap` sentences of a previous window are long, the new window starts
slightly above budget before the next sentence triggers a flush. In
practice this is bounded by `overlap × (average_sentence_tokens)` and is
unlikely to exceed model limits, but callers with hard token limits should
add a safety margin.

**Cross-heading merge loses section context.** The tiny-chunk cross-heading
merge keeps the first chunk's heading. If the second chunk belongs to a
semantically distinct section, retrieval for that section's content may
return a chunk labelled with the wrong heading. This trade-off favours
embedding quality (avoiding near-empty chunks) over citation precision.
Disable by setting `min_tokens` low enough that no chunks fall below the
tiny threshold.

**`chunk_for_model()` ignores user config for model name.** The dynamic
parameter selection calls `get_dynamic_chunk_params_for_text(text)` with
the `DEFAULT_MODEL` constant (`intfloat/e5-small-v2`). If your deployment
uses a different embedding model, either pass `model_name` explicitly or
call `chunk_document()` with manually set parameters.

**BlingFire sentence splitting is not installed by default.** Without it,
the module falls back to a regex splitter that handles common English
abbreviations but will mis-split on domain-specific abbreviations (e.g.
`Fig. 3` → two sentences), long compound sentences in German, and some
CJK punctuation patterns. Install via `pip install blingfire`.

**`assign_links_to_chunks()` is O(chunks × links).** For very large
documents (> 5,000 chunks) with dense hyperlinks (> 2,000 links), this
can become a noticeable bottleneck. An interval-tree approach would reduce
this to O((chunks + links) log n) if needed.

**Links from non-PDF formats are not extracted.** `extract_links()` only
operates on PDFs via fitz. DOCX, PPTX, and plain-text files will always
have `ParsedDocument.links = []`, and consequently `chunk.links = []`.
DOCX hyperlink extraction via `python-docx` could be added to
`parse_markitdown()` as a future improvement.