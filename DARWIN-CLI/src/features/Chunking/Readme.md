# 📦 Chunking Pipeline — README

## Overview

This module takes **already parsed documents** (from your parsing pipeline) and splits them into **LLM-ready chunks**.

Think of it like the second stage:

```
raw files → (parsing) → markdown → (THIS MODULE) → chunks
```

Each chunk is:

* token-aware
* context-preserving (via overlap)
* enriched with metadata (headings, links, pages)

---

## 🚀 How to Run

### Using Python

```bash
python main.py [path] [options]
```

### Using uv

```bash
uv run main.py [path] [options]
```

If no path is provided, it defaults to the indexes you havnt chunked yet.

---

## ⚠️ Prerequisite (Important)

This **will NOT work** unless parsing has already been run.

If you see:

```
No parsed files found. Run parsing first.
```

You need to run your parsing pipeline first.

---

## 📥 What It Reads

This module pulls from your `DarwinStore`:

* Previously parsed markdown
* File registry metadata

It does **NOT** read raw PDFs or docs directly.

---

## ⚙️ Command-Line Arguments

### Core

| Argument          | Description                                     |
| ----------------- | ----------------------------------------------- |
| `path`            | Project directory (default: `.`)                |
| `--file`, `-f`    | Chunk a single file instead of all parsed files |
| `--project`, `-p` | Override project directory                      |

---

### Behavior

| Argument          | Description                            |
| ----------------- | -------------------------------------- |
| `--rerun`, `-R`   | Re-chunk even if chunks already exist  |
| `--inspect`, `-i` | Print detailed chunk inspection output |

---

### WebSocket

| Argument    | Description                               |
| ----------- | ----------------------------------------- |
| `--ws-port` | Port for WebSocket server (default: 8765) |
| `--no-ws`   | Disable WebSocket server                  |

---

## 🧠 What the Pipeline Does

### 1. Load Store

```python
store = DarwinStore()
```

This gives access to:

* parsed markdown
* file registry
* existing chunks

---

### 2. Resolve Target Files

* If `--file` is used → only that file
* Otherwise → all parsed files in the store

---

### 3. Skip Logic

Files are skipped if:

* chunks already exist
* AND `--rerun` is NOT used

You’ll see:

```
file.md — already chunked (use --rerun to rechunk)
```

---

### 4. Load Markdown

```python
tobe_chunked = store.load_markdown(filename)
```

Handles:

* `ParsedDocument` objects
* raw strings

---

### 5. Chunking

```python
chunks, metadata = chunk_for_model(text, source=filename)
```

This step:

* splits text into token-sized chunks
* preserves headings when possible
* adds overlap between chunks
* extracts links
* detects tables/captions

---

### 6. Save Results

```python
store.save_chunks(...)
```

Each file gets:

* chunk list
* metadata (including total links)

---

### 7. Logging Output

Example:

```
chunk:report.pdf — 0.42s
report.pdf — 32 chunks · avg 420 tokens · 18 links distributed
```

---

## 🔍 Inspection Mode (`--inspect`)

This is where things get interesting.

Run:

```bash
python chunk.py --inspect
```

You’ll get:

### Metrics

```
total chunks        32
avg tokens          420
min tokens          120
max tokens          512
with heading        20 (62%)
table chunks        3
caption chunks      2
overlap chunks      10
chunks w/ links     12
total links         18
```

---

### Sample Chunk Output

```
index:     0
heading:   Introduction
page:      1–2
tokens:    410
overlap:   none
links:     2

text:  This document explains...
link:  https://example.com [source]
```

You’ll also see:

* first few chunks
* middle chunk (if large)
* last chunk

---

## 📦 Chunk Structure

Each `Chunk` contains:

```python
{
    text: str,
    clean: str,
    index: int,
    source: str,
    token_count: int,
    heading: str,
    page: int,
    page_end: int,
    overlap_direction: OverlapDirection,
    links: List[ChunkLink],
    is_overlap: bool  # derived property
}
```

---

## ⚡ Performance

* Chunking is CPU-light compared to parsing
* Usually completes in milliseconds per file
* Large documents scale linearly

---

## ❌ Failure Cases

You may see errors like:

```
file.pdf — no markdown found, skipping
file.pdf — unexpected markdown type: <class '...'>
```

Or:

```
file.pdf — produced 0 chunks
```

These usually mean:

* parsing didn’t run properly
* markdown is corrupted or empty

---

## 🧪 Example Commands

### Chunk everything

```bash
python chunk.py
```

---

### Chunk one file

```bash
python chunk.py --file report.pdf
```

---

### Force re-chunk

```bash
python chunk.py -R
```

---

### Inspect output

```bash
python chunk.py -i
```

---

### Disable WebSocket

```bash
python chunk.py --no-ws
```

---

## ⚠️ What to Expect

* Fast execution
* Clean chunk boundaries most of the time
* Occasional weird splits (depends on source text quality)
* Overlap chunks to preserve context
* Link extraction works if links exist in text

---

## 🧯 Common Pitfalls

* Running before parsing
* Forgetting `--rerun` and wondering why nothing happens
* Expecting perfect chunking from messy PDFs
* Misinterpreting overlap chunks as duplicates

---

## 🧩 Summary

This module:

* Converts parsed text into LLM-ready chunks
* Preserves structure where possible
* Adds metadata for smarter retrieval
* Is fast, deterministic, and easy to rerun

---

If parsing is about **extracting text**, this step is about **making that text usable**.

---

If you want next, I can help you:

* tune chunk sizes
* optimize for embeddings vs retrieval
* or visualize chunk distributions
