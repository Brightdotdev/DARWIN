# Darwin — Chunking Module

Converts parser `embedding_text` into overlapping chunks ready for embedding. Each chunk carries its original text, stripped clean text, section heading, and page number.

---

## How to run it

```bash
# chunk everything parsed in current directory
uv run src/core/Chunking/main.py

# chunk a specific file
uv run src/core/Chunking/main.py --file biology.pdf

# chunk and print sample output so you can see exactly what's coming out
uv run src/core/Chunking/main.py --inspect

# force rechunk even if fresh chunks already exist
uv run src/core/Chunking/main.py --rerun

# set token budget based on embedding model
uv run src/core/Chunking/main.py --model bge-m3

# manual token budget override
uv run src/core/Chunking/main.py --max-tokens 512
```

---

## What `--inspect` shows you

Running with `--inspect` prints a table of stats plus sample chunks to the terminal so you can verify chunking quality before embedding. Example output:

```
┌─────────────────────────────────┬──────────┐
│ metric                          │ value    │
├─────────────────────────────────┼──────────┤
│ total chunks                    │ 423      │
│ avg tokens                      │ 287      │
│ min tokens                      │ 82       │
│ max tokens                      │ 348      │
│ with heading                    │ 418 (99%)│
│ table chunks                    │ 12       │
│ caption chunks                  │ 8        │
│ overlap chunks                  │ 67       │
└─────────────────────────────────┴──────────┘

index:   0
heading: The Light Reactions
page:    14
tokens:  231
overlap: no

  text:  ## The Light Reactions
         The light-dependent reactions of photosynthesis occur in the
         thylakoid membranes. Light energy is absorbed by chlorophyll...

  clean: The Light Reactions
         The light-dependent reactions of photosynthesis occur in the
         thylakoid membranes. Light energy is absorbed by chlorophyll...
```

The `text` field is what gets shown to the user and sent to the LLM. The `clean` field is what gets fed to the embedding model — markdown stripped, same content.

---

## Chunking strategy

The parser outputs `embedding_text` with structure signals already baked in:

```
## Chapter 6: Photosynthesis       ← heading boundary
### The Light Reactions             ← subheading boundary

Body paragraph text here.

Another paragraph.

[Table] Process: Light | Location: Thylakoid | Products: ATP
[Caption: Electron transport chain diagram]

---                                 ← page separator

## Chapter 7: Cellular Respiration
```

The chunker exploits these signals directly instead of blindly splitting on `\n\n`:

1. Split at `##` / `###` boundaries into sections
2. Each section = heading + its body paragraphs
3. Section fits in token budget → one chunk, heading attached
4. Section too big → split into paragraphs, each inherits the heading
5. Paragraph still too big → sentence split with overlap
6. Section too small → merge with next section of same heading level
7. `[Table]` blocks → always their own chunk, never split mid-table
8. `[Caption:]` blocks → attach to previous chunk if room, else own chunk
9. No headings found → fall back to paragraph splitting

Every chunk inherits the nearest heading above it automatically — no extra bookkeeping needed for citations.

---

## Two text fields

Every `Chunk` has two text fields:

| Field | Content | Used for |
|-------|---------|----------|
| `text` | original markdown from parser | shown to user, sent to LLM |
| `clean` | stripped plain text | fed to embedding model |

`clean` is `text` with markdown symbols removed — `**bold**` → `bold`, `## Heading` → `Heading`, `[Table]` prefix stripped. The actual content is identical, just without formatting syntax that doesn't help the embedding model.

---

## The `Chunk` dataclass

```python
@dataclass
class Chunk:
    text:        str    # original text — shown to user, sent to LLM
    clean:       str    # stripped plain text — fed to embedding model
    index:       int    # position in document (0-based)
    source:      str    # filename for citations
    token_count: int    # estimated tokens of clean text
    heading:     str    # nearest section heading above this chunk
    page:        int    # page number (0 if unknown)
    page_end:    int    # last page this chunk touches
    is_overlap:  bool   # True if chunk contains overlap from previous
```

Chunks are saved to `.darwin/filename/chunks.json` as a JSON array of these fields.

---

## Token budget

Token budget is set by the caller based on the embedding model. The chunker never decides — the model context window decides.

| Model | Window | Recommended `max_tokens` |
|-------|--------|--------------------------|
| e5-small-v2 | 512 | 350 |
| BGE-M3 | 8192 | 800 |
| GritLM-7B | 4096 | 512 |

The `--model` flag sets this automatically:

```bash
uv run src/core/Chunking/main.py --model bge-m3   # sets max_tokens=800
uv run src/core/Chunking/main.py --model e5-small  # sets max_tokens=350
```

Token counting is estimated via `word_count / 0.75` — no tokenizer dependency. Differs from actual subword token count by less than 5% for English academic text.

---

## Overlap

When a section or paragraph is too large and gets sentence-split, the last `overlap` sentences of the previous window are prepended to the next chunk. Default is 2 sentences.

This prevents a concept explanation that spans a chunk boundary from being split in a way where neither chunk has the full context. A student asking about the Krebs cycle shouldn't get half an explanation.

Overlap only happens within sentence-split chunks — sections that fit in the budget are never padded.

---

## What gets skipped

Paragraphs that produce fewer than 4 words after cleaning are dropped entirely. This catches:

- Orphan page separator lines (`---`)
- Lone image placeholders (`[Image]`)
- Empty sections with only a heading and no body

---

## Tuning constants

All in `Chunker.py`.

| Constant | Default | Effect |
|----------|---------|--------|
| `DEFAULT_MAX_TOKENS` | 350 | Max tokens per chunk |
| `DEFAULT_MIN_TOKENS` | 80 | Chunks below this get merged |
| `DEFAULT_OVERLAP` | 2 | Sentences of overlap on sentence-split chunks |

---

## File structure

```
src/core/
  Chunker.py          all chunking logic, Chunk dataclass
  Chunking/
    main.py           CLI entry point, --inspect output
```

### Data flow

```
DarwinStore.load_markdown(filename)
  → embedding_text (markdown string)
    → chunk_document()
        → _split_into_sections()    heading-based split
        → _section_to_chunks()      per-section processing
            → _split_with_overlap() if paragraph too big
        → _merge_small()            final merge pass
    → list[Chunk]
      → DarwinStore.save_chunks()
          → chunks.json
```