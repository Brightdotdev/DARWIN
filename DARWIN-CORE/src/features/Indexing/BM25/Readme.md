# 📦 BM25 Indexing Pipeline

## Overview

This module builds a **sparse lexical search index (BM25)** from pre-chunked documents stored in `DarwinStore`.

It is part of the retrieval stack:

```text id="b2x8kq"
Parsing → Chunking → BM25 Indexing → Search
```

BM25 is used for:

* keyword search
* exact term matching
* fast retrieval over large corpora

---

## 🚀 How to Run

```bash id="m9xw1a"
python main.py [path] [options]
```

Default behavior:

* processes all chunked files in the store

---

## 📥 Prerequisites

Before running BM25 indexing:

1. Documents must be parsed
2. Chunking must be completed
3. Chunks must exist in `DarwinStore`

If not, you will see:

```text id="k2l9pw"
No chunked files found. Run chunking first.
```

---

## ⚙️ CLI Arguments

### Core

| Argument       | Description                                    |
| -------------- | ---------------------------------------------- |
| `path`         | Project directory (default: current directory) |
| `--file`, `-f` | Index a single file                            |

---

### BM25 Tuning

| Argument                | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| `--threshold`, `-t`     | Domain stopword threshold (higher = stricter filtering) |
| `--idf-smoothing`, `-s` | Controls rarity boost for terms                         |

---

### Behavior

| Argument          | Description                                 |
| ----------------- | ------------------------------------------- |
| `--rerun`, `-R`   | Force re-index even if index already exists |
| `--inspect`, `-i` | Print vocabulary stats + sample tokens      |

---

### WebSocket

| Argument    | Description                    |
| ----------- | ------------------------------ |
| `--ws-port` | WebSocket port (default: 8765) |
| `--no-ws`   | Disable WebSocket server       |

---

## 🧠 What the Pipeline Does

### 1. Load Store

```python id="q7m3xk"
store = DarwinStore()
```

Reads:

* chunked documents
* file registry
* existing BM25 index state

---

### 2. File Selection

Two modes:

#### Single file

```bash id="p3n8wd"
--file biology.pdf
```

Only processes if chunks exist.

#### Full project

Scans store:

* selects all files marked as `is_chunked`

---

### 3. Skip Logic

If index already exists:

```text id="x8c2jv"
Skipping file.pdf (fresh index exists)
```

Unless:

```bash id="r5m2pq"
--rerun
```

---

### 4. Index Building

Core function:

```python id="t9k1xq"
run_bm25_indexing_pipeline(...)
```

It:

* tokenizes chunk text
* builds inverted index
* computes term frequencies
* calculates IDF scores
* applies smoothing + stopword filtering

---

### 5. Save Output

Index is persisted via `DarwinStore`.

---

### 6. Optional Inspection

With:

```bash id="w2q8mn"
--inspect
```

You get:

* vocabulary size
* token distribution
* sample terms
* index statistics

---

## 📊 Logging Output

Example:

```text id="h7x9pl"
BM25 indexing
Threshold: 0.6 | IDF smoothing: 1.2 | Rerun: False | Inspect: True

Indexing 3 file(s)

done — 3 indexed · 0 skipped · 0 failed
```

---

## 🔍 Inspection Mode Output

Shows internal index structure:

* token frequency distribution
* sample vocabulary terms
* document-term relationships

Useful for debugging retrieval quality.

---

## ⚠️ Common Issues

### ❌ No chunked files found

```text id="c9v2mp"
No chunked files found. Run chunking first.
```

Fix:

* run chunking pipeline first

---

### ❌ File skipped unexpectedly

Cause:

* index already exists
* `--rerun` not used

Fix:

```bash id="n3k8wd"
--rerun
```

---

### ❌ Empty index

Usually caused by:

* empty chunks
* failed parsing step
* missing markdown in store

---

## 🧯 Design Notes

* BM25 depends entirely on chunk output
* It does NOT parse raw documents
* It does NOT embed text
* It is purely lexical (keyword-based)

---

## 🧠 Architecture Flow

```text id="z8m2qp"
Parsed Docs → Chunks → BM25 Index → Search Layer
```

BM25 is the **fast retrieval layer**, not semantic understanding.

---

## ⚡ Performance

* Fast to build (CPU-based)
* Scales linearly with chunk count
* No GPU required
* Ideal for large corpora keyword search

---

## 🔥 Key Insight

BM25 is:

* precise
* deterministic
* fast
* but not semantic

It works best when combined with embeddings.

---

##   Summary

This module:

* builds lexical search indexes from chunked data
* supports reruns, inspection, and tuning
* depends on chunking stage only
* integrates with DarwinStore for persistence

---