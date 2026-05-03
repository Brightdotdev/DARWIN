# 📄 README — Indexing Module

Here’s a clean, real-world README you can drop in.

---

# 📦 Indexing Pipeline (BM25 + Embeddings)

## Overview

This module builds **search indexes** from already parsed and chunked documents.

It runs two systems:

1. **BM25 Indexing**

   * keyword-based retrieval
   * lexical scoring

2. **Embedding Indexing**

   * semantic search
   * vector-based similarity

Both run sequentially on the same dataset.

---

## 🚀 How to Run

```bash id="k8d1qz"
python main.py [path] [options]
```

---

## 📥 Input Requirements

This module expects:

* Parsed documents exist in `DarwinStore`
* Chunking has already been completed

If not, indexing will fail or produce empty results.

---

## ⚙️ CLI Arguments

### Core

| Argument       | Description                                    |
| -------------- | ---------------------------------------------- |
| `path`         | Project directory (default: current directory) |
| `--file`, `-f` | Index a single file                            |

---

### BM25 Configuration

| Argument                | Description               |
| ----------------------- | ------------------------- |
| `--threshold`, `-t`     | Domain stopword threshold |
| `--idf-smoothing`, `-s` | IDF smoothing factor      |

---

### Behavior Flags

| Argument          | Description                    |
| ----------------- | ------------------------------ |
| `--rerun`, `-R`   | Rebuild indexes even if cached |
| `--inspect`, `-i` | Show index stats and samples   |

---

### WebSocket

| Argument    | Description                    |
| ----------- | ------------------------------ |
| `--ws-port` | WebSocket port (default: 8765) |
| `--no-ws`   | Disable WebSocket server       |

---

## 🧠 What Happens During Execution

### 1. Input Validation

* checks file exists (if provided)
* ensures path exists
* aborts early if invalid

---

### 2. BM25 Indexing

```text id="q3p8jd"
[1/2] BM25 indexing
```

Builds:

* term frequency index
* document frequency stats
* inverted index structure

Used for:

* keyword search
* exact matching queries

---

### 3. Embedding Indexing

```text id="k1n9sa"
[2/2] Embedding indexing
```

Builds:

* vector representations
* semantic search index

Used for:

* meaning-based retrieval
* fuzzy queries

---

## 📊 Logging Output

Example:

```text id="m9zq1p"
Indexing
test.pdf | Threshold: 0.6 | IDF smoothing: 1.2 | Inspect: True | Rerun: False

[1/2] BM25 indexing
[2/2] Embedding indexing
Indexing complete
```

---

## ⚠️ Common Issues

### ❌ File not found

```
File not found: document.pdf
```

Fix:

* check path spelling
* ensure correct working directory

---

### ❌ Empty index output

Usually means:

* parsing not run
* chunking missing
* store not populated

---

### ❌ Slow indexing

Causes:

* large corpus
* embedding model latency
* rerun without cache

---

## 🧯 Design Notes

* BM25 and embeddings are independent
* They run sequentially, not parallel
* Both depend on chunked data
* Storage layer is the source of truth

---

## 🧩 Architecture Flow

```text id="v3x8qp"
Parsing → Chunking → Storage → Indexing
                                ├── BM25
                                └── Embeddings
```

---

## ⚡ Performance

* BM25: fast, CPU-based
* Embeddings: slower, model-dependent
* Scaling depends on corpus size

---

## 🧠 Key Insight

BM25 = precision keyword search
Embeddings = semantic understanding
Together = hybrid retrieval system

---

## ✅ Summary

This module:

* validates input paths
* builds two complementary search indexes
* depends entirely on prior parsing + chunking
* is safe to rerun (`--rerun`) for updates

---


