# 📦 Embedding Pipeline

## Overview

This module converts **chunked documents** into **vector embeddings** and stores them as a persistent `.npz` index inside `DarwinStore`.

It is the **semantic layer** of your retrieval system.

Pipeline order:

```text id="x7k2mq"
Parsing → Chunking → Embedding → Search
```

---

## 🚀 How to Run

```bash id="q9m1wp"
python main.py [path] [options]
```

Default behavior:

* processes all chunked files in the store

---

## 📥 Prerequisites

Before running embeddings:

1. Documents must be parsed
2. Chunking must be completed
3. Chunks must exist in `DarwinStore`

If not:

```text id="k3n8vd"
No chunked files found. Run chunking first.
```

---

## ⚙️ CLI Arguments

### Core

| Argument       | Description                                    |
| -------------- | ---------------------------------------------- |
| `path`         | Project directory (default: current directory) |
| `--file`, `-f` | Embed a single file                            |

---

### Behavior

| Argument          | Description                             |
| ----------------- | --------------------------------------- |
| `--rerun`, `-R`   | Recompute embeddings even if cached     |
| `--inspect`, `-i` | Print vector statistics + sample chunks |

---

### WebSocket

| Argument    | Description                    |
| ----------- | ------------------------------ |
| `--ws-port` | WebSocket port (default: 8765) |
| `--no-ws`   | Disable WebSocket server       |

---

## 🧠 What This Pipeline Does

### 1. Load Store

```python id="m1x9qp"
store = DarwinStore()
```

Reads:

* chunked documents
* existing embedding state
* file registry metadata

---

### 2. File Selection

Two modes:

#### Single file

```bash id="c7p2ld"
--file biology.pdf
```

Only processed if chunks exist.

#### Batch mode

Processes all files where:

```python id="p8q3wx"
info.get("is_chunked", False)
```

---

### 3. Skip Logic

If embeddings already exist:

```text id="v4k9mz"
skipping file.pdf (fresh embeddings exist)
```

Unless:

```bash id="t2m8xq"
--rerun
```

---

### 4. Embedding Generation

Core function:

```python id="h3q8wd"
run_embedding_pipeline(...)
```

For each chunk:

* text is encoded by active model
* vector is generated
* stored as NumPy array

Model behavior:

* loaded once for batch
* reused across files
* optionally unloaded at end

---

### 5. Storage Format

Embeddings are saved as:

* `.npz` files
* aligned with chunk metadata

Each entry corresponds to:

* chunk text
* heading
* page
* vector

---

### 6. Model Lifecycle

```text id="z7x2mq"
load model → embed batch → unload model
```

Unload behavior:

* happens after last file OR explicitly via loader
* prevents GPU/CPU memory leaks

---

## 📊 Logging Output

Example:

```text id="k9p3wx"
Embedding
Rerun: False | Inspect: True

Embedding 4 file(s)

done — 4 embedded · 1 skipped · 0 failed
```

---

## 🔍 Inspection Mode (`--inspect`)

Shows detailed vector diagnostics:

### Metrics

* vector shape
* embedding dimension
* L2 norm distribution
* dot product sanity check
* page coverage
* heading coverage

Example:

```text id="m2x8qv"
vectors shape                   (128, 768)
embedding dim                  768
L2 norms (should be ~1.0)     min=0.998 max=1.002
chunks with heading           45 / 128
```

---

### Sample Output

* chunk text preview
* vector preview (first + last values)
* page + heading metadata

---

## ⚠️ Common Issues

### ❌ No chunked files found

```text id="n7q2wx"
Run chunking first.
```

Fix:

* run chunking pipeline before embedding

---

### ❌ Skipped files unexpectedly

Cause:

* embeddings already exist
* `--rerun` not used

Fix:

```bash id="t9m3xq"
--rerun
```

---

### ❌ Memory pressure

Cause:

* large embedding model
* too many chunks at once

Mitigation:

* model unload logic is already included
* batch processing is used

---

### ❌ Empty vectors or mismatch

Usually caused by:

* corrupted chunks
* missing text in store
* failed parsing upstream

---

## 🧯 Design Notes

* Embeddings depend strictly on chunk output
* No direct parsing or indexing happens here
* Model is loaded once per batch for efficiency
* Storage layer persists everything

---

## 🧠 Architecture Flow

```text id="x8q1md"
Parsed Docs → Chunks → Embedding Model → Vector Store (.npz)
```

This is the **semantic retrieval layer** of the system.

---

## ⚡ Performance

* GPU optional (depends on model)
* batch optimized embedding generation
* model reuse reduces overhead significantly
* scales with chunk count, not file count

---

## 🔥 Key Insight

Embeddings provide:

* semantic similarity
* fuzzy search capability
* meaning-based retrieval

Unlike BM25:

* not keyword-based
* not deterministic
* works on context

---

## 🧩 Relationship to BM25

| System     | Type     | Strength            |
| ---------- | -------- | ------------------- |
| BM25       | lexical  | exact keyword match |
| Embeddings | semantic | meaning similarity  |

Together they form a **hybrid retrieval system**.

---

## ✅ Summary

This module:

* converts chunks into vector embeddings
* stores them as `.npz` artifacts
* manages model lifecycle efficiently
* supports rerun, inspect, and batch processing
* depends entirely on chunking output

---

