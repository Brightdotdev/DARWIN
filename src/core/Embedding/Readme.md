# Darwin — Embedding Module

Converts chunks into dense vector embeddings using `intfloat/e5-small-v2`. Saves vectors and chunk metadata together in a single `.npz` bundle per document.

---

## How to run it

```bash
# embed everything chunked in current directory
uv run src/core/Embedding/main.py

# embed a specific file
uv run src/core/Embedding/main.py --file biology.pdf

# inspect output — print vector stats and verify what's in the .npz
uv run src/core/Embedding/main.py --inspect

# force re-embed even if fresh embeddings exist
uv run src/core/Embedding/main.py --rerun

# lower batch size if running out of RAM
uv run src/core/Embedding/main.py --batch-size 16
```

---

## What `--inspect` shows you

Running with `--inspect` loads the saved `.npz` and prints verification stats so you can confirm vectors are correct before running search.

```
┌──────────────────────────────────────┬──────────────────────┐
│ metric                               │ value                │
├──────────────────────────────────────┼──────────────────────┤
│ vectors shape                        │ (423, 384)           │
│ dtype                                │ float32              │
│ embedding dim                        │ 384                  │
│ L2 norms (should be ~1.0)            │ min=1.0000 max=1.0000│
│ v[0] · v[0] (should be ~1.0)         │ 1.0000               │
│ unique pages                         │ 47                   │
│ chunks with heading                  │ 418 / 423            │
└──────────────────────────────────────┴──────────────────────┘

index:   0
heading: The Light Reactions
page:    14

  text:  The light-dependent reactions of photosynthesis occur in the
         thylakoid membranes. Light energy is absorbed by chlorophyll...

  vector: [0.0312, -0.0187, ... 0.0241]
```

The L2 norms should all be exactly 1.0 — this confirms normalization worked and cosine similarity at search time is a clean dot product.

---

## Why e5-small

`intfloat/e5-small-v2` is 33MB, runs on CPU in seconds, and achieves retrieval quality that competes with models 10× its size on MTEB benchmarks for English semantic search. For clean academic text in English it's the right default.

Embedding dimension is 384. One document with 450 chunks produces a `(450, 384)` float32 matrix — about 700KB of vectors.

---

## The e5 prefix requirement

e5 models were trained with asymmetric task prefixes. Skipping them noticeably degrades retrieval quality.

| Context | Prefix | Applied by |
|---------|--------|------------|
| Indexing chunks | `passage: {text}` | `embed_chunks()` automatically |
| Search query | `query: {text}` | `embed_query()` automatically |

You never apply these manually — both functions handle it internally. The important thing is that you use `embed_query()` at search time, not raw `model.encode()`, otherwise the asymmetry breaks.

---

## The `.npz` bundle

Vectors and chunk metadata are stored together in a single compressed numpy archive. One file — if it exists, both vectors and metadata exist. No separate JSON that could drift out of sync.

```
.darwin/
  biology_pdf/
    e5_small_biology_pdf.npz     ← everything in one file
```

Bundle contents:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `vectors` | `(n, 384)` | float32 | L2-normalized embedding vectors |
| `text` | `(n,)` | object | original chunk text (shown to user) |
| `clean` | `(n,)` | object | stripped chunk text (used for embedding) |
| `headings` | `(n,)` | object | nearest section heading per chunk |
| `pages` | `(n,)` | int32 | page number per chunk |
| `source` | `(n,)` | object | filename repeated n times |

Loading is one call:

```python
data     = store.load_embeddings("biology.pdf")
vectors  = data["vectors"]    # np.ndarray shape (n, 384)
texts    = data["text"]       # list[str]
headings = data["headings"]   # list[str]
pages    = data["pages"]      # list[int]
```

The file is versioned by model name — switching embedding models creates a new `.npz` without overwriting the old one:

```
e5_small_biology_pdf.npz
bge_m3_biology_pdf.npz      ← if you switch models
```

---

## L2 normalization

All vectors are L2-normalized before saving. This means every vector has length 1.0.

The practical benefit: cosine similarity at search time becomes a dot product:

```python
# without normalization
score = np.dot(query_vec, chunk_vec) / (norm(query_vec) * norm(chunk_vec))

# with L2 normalization — same result, no division
score = np.dot(query_vec, chunk_vec)
```

Faster and simpler at search time. `embed_query()` also normalizes the query vector so the math is consistent.

---

## Batch size

Chunks are embedded in batches. Larger batches are faster but use more RAM.

| Batch size | RAM usage (approx) | Use when |
|------------|-------------------|---------|
| 16 | ~1GB | Low-end machine, 8GB RAM |
| 32 | ~2GB | Default, 8GB+ RAM |
| 64 | ~3GB | 16GB+ RAM |

The model itself (~300MB) loads once and stays in memory for the session.

---

## File structure

```
src/core/
  Embedder.py          embed_chunks(), embed_query(), run_embedding_pipeline()
  Embedding/
    main.py            CLI entry point, --inspect output
```

### Data flow

```
DarwinStore.load_chunks(filename)
  → list[Chunk]
    → embed_chunks()
        → prefix_passage(chunk.clean) per chunk
        → model.encode() in batches of batch_size
        → L2 normalize
    → np.ndarray shape (n_chunks, 384)
      → DarwinStore.save_embeddings(filename, vectors, chunks)
          → np.savez_compressed → e5_small_filename.npz
```

### Search time data flow

```
user query string
  → embed_query()
      → prefix_query(query)
      → model.encode()
      → L2 normalize
  → np.ndarray shape (384,)
    → np.dot(query_vec, vectors)   ← fast dot product (vectors already normalized)
    → top_k indices
      → texts[indices]             ← chunk text for those indices
      → headings[indices]          ← section context for citations
```

---

## Performance reference

Tested on a 4-core CPU machine, batch_size=32.

| Document | Chunks | Time |
|----------|--------|------|
| 400-page textbook | ~450 | ~45s |
| 40-page ebook | ~80 | ~8s |

Model load on first call: ~3s. Subsequent documents in the same session skip the load.