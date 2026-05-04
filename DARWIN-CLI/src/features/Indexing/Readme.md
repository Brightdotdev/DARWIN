# Indexing Pipeline

Builds searchable indexes from chunked documents.

This runs:
- **BM25 (sparse keyword index)**
- **Embeddings (dense vector index)**

---

## Overview

The indexing pipeline takes chunked documents from `.darwin` and produces:

- BM25 index → keyword search
- Embeddings → semantic search

---

## Pipeline Flow

```

Chunked Documents (.darwin/chunks)
│
▼
┌──────────────┐
│    BM25      │  → keyword index
└──────┬───────┘
▼
┌──────────────┐
│  Embeddings  │  → vector index
└──────────────┘

````

---

## Usage

### Basic

```bash
uv run src/core/Indexing/main.py
````

Indexes all eligible files in the store.

---

### Single File

```bash
uv run src/core/Indexing/main.py --file biology.pdf
```

---

### Multiple Files (Cluster)

```bash
uv run src/core/Indexing/main.py --cluster bio.pdf chem.pdf physics.pdf
```

---

## Parameters

### Core

| Flag        | Description                        |
| ----------- | ---------------------------------- |
| `path`      | Target project directory           |
| `--file`    | Index a single file                |
| `--cluster` | Index multiple specific files      |
| `--rerun`   | Rebuild indexes even if they exist |

---

### BM25 Options

| Flag              | Default                 | Description               |
| ----------------- | ----------------------- | ------------------------- |
| `--threshold`     | `DOMAIN_STOP_THRESHOLD` | Domain stopword filtering |
| `--idf-smoothing` | `DEFAULT_IDF_SMOOTHING` | IDF smoothing strength    |

---

### General

| Flag        | Description                  |
| ----------- | ---------------------------- |
| `--inspect` | Show index stats and samples |
| `--ws-port` | WebSocket port               |
| `--no-ws`   | Disable WebSocket            |

---

## Behavior

### Execution Modes

**1. Full Store Mode**

```bash
uv run src/core/Indexing/main.py
```

* Finds all chunked files
* Indexes everything

---

**2. File Mode**

```bash
--file document.pdf
```

* Only indexes that file

---

**3. Cluster Mode**

```bash
--cluster file1.pdf file2.pdf
```

* Processes files sequentially
* Each file runs:

  * BM25 → Embedding

---

## Output

Stored in `.darwin/`:

```
.darwin/
├── bm25/          # sparse indexes
├── embeddings/    # vector indexes
├── registry.json
```

---

## Execution Order

For each file:

```
1. BM25 indexing
2. Embedding generation
```

If no file/cluster is provided:

```
1. BM25 (all files)
2. Embeddings (all files)
```

---

## Requirements

* Files must already be:

  * parsed
  * chunked

If not:

```
No chunked files found. Run chunking first.
```

---

## Examples

### Full indexing

```bash
uv run src/core/Indexing/main.py
```

---

### Force rebuild

```bash
uv run src/core/Indexing/main.py --rerun
```

---

### Tuned BM25

```bash
uv run src/core/Indexing/main.py --threshold 0.25 --idf-smoothing 1.0
```

---

### Debug / inspection

```bash
uv run src/core/Indexing/main.py --inspect
```

---

## Failure Handling

* Errors are isolated per file
* Batch continues even if one fails

Example:

```
bio.pdf ✓ indexed
chem.pdf ✗ failed
physics.pdf ✓ indexed

Indexing complete — 2 indexed · 1 failed
```

---

## Notes

* `--cluster` and `--file` cannot be used together
* Cached indexes are reused unless `--rerun` is set
* Embedding model is managed internally

---

## Summary

```
indexing = BM25 + embeddings
```

Run this after ingestion, before search.

```
