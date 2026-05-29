# Ingestion Pipeline

Unified pipeline that runs **parsing → chunking** in one command.

This is the main entry point for turning raw documents into chunked, model-ready data.

---

## Overview

The ingestion pipeline combines:

1. **Parsing**
   - Extracts text from documents
   - Converts to structured markdown

2. **Chunking**
   - Splits text into model-friendly chunks
   - Adds metadata (tokens, pages, links)

---

## Pipeline Flow

```

Files (.pdf, .docx, .pptx, .txt, .md, .json)
│
▼
┌──────────────┐
│   PARSING    │  → markdown + metadata
└──────┬───────┘
▼
┌──────────────┐
│  CHUNKING    │  → chunks + tokenized segments
└──────┬───────┘
▼
.darwin store

````

---

## Supported File Types

- `.pdf`
- `.docx`
- `.pptx`
- `.txt`
- `.md`
- `.json`

---

## Usage

### Basic

```bash
# Ingest everything in current directory
uv run src/features/Ingestion/main.py

# Ingest a directory
uv run src/features/Ingestion/main.py ./documents
````

---

### Single File

```bash
uv run src/features/Ingestion/main.py --file biology.pdf
```

---

### Multiple Files 

```bash
uv run src/features/Ingestion/main.py --file bio.pdf chem.pdf physics.pdf
```

---

## Parsing Options

```bash
--recursive   Scan subdirectories
--images      Extract images
--caption     Generate captions (requires --images)
--precise     Better layout (slower)
--rerun       Reprocess cached files
--no-save     Parse without saving
```

---

## Chunking Options

```bash
--inspect     Show chunk samples and stats
--rerun       Rechunk existing files
```

---

## Examples

### Full pipeline with inspection

```bash
uv run src/features/Ingestion/main.py --file book.pdf --inspect
```

---

### High-quality parsing + chunking

```bash
uv run src/features/Ingestion/main.py --file paper.pdf --precise --images --caption
```

---

### Batch ingestion

```bash
uv run src/features/Ingestion/main.py ./docs --recursive
```

---

## Output

After running, data is stored in `.darwin/`:

```
.darwin/
├── markdown/     # parsed documents
├── chunks/       # chunked outputs
├── registry.json
```

---

## Behavior Notes

* Chunking only runs on successfully parsed files
* Failed parses are skipped automatically
* Cached results are reused unless `--rerun` is used
* file args mode overrides automatic file discovery

---

## Programmatic Usage

```python
from src.features.Ingestion.Parsing.lib import ParsingArgs
from src.features.Ingestion.Chunking.lib import ChunkingArgs
from src.core.Ingestion.main import handle_ingestion

parsing_args = ParsingArgs(
    target=".",
    files=None,
    images=False,
    caption=False,
    recursive=False,
    rerun=False,
    precise=False,
    save=True,
)

chunking_args = ChunkingArgs(
    target=".",
    project=".",
    inspect=True,
    rerun=False,
)

handle_ingestion(parsing_args, chunking_args)
```

---

## Failure Handling

* If parsing fails → file is skipped
* If no valid files → pipeline aborts
* Errors are logged per file, not fatal to batch

---

## When to Use This

Use this instead of running parsing and chunking separately when you want:

* One command workflow
* Clean pipeline execution
* Automatic handoff between stages

---

## Summary

```
ingestion = parsing + chunking
```

Run this first before indexing or search.


