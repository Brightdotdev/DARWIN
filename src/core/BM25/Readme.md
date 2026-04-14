# Darwin — BM25 Indexing Module

The BM25 indexing module builds sparse retrieval indexes from chunked documents. It handles tokenization, domain-specific stopword detection, and term frequency calculation, saving everything to `.bm25` bundles for later keyword search.

---

## How to run it

```bash
# index everything chunked in current directory
uv run src/core/BM25/main.py

# index a specific file
uv run src/core/BM25/main.py --file biology.pdf

# adjust domain stopword threshold (lower = more aggressive filtering)
uv run src/core/BM25/main.py --threshold 0.25

# inspect output — show vocabulary stats and sample tokens
uv run src/core/BM25/main.py --inspect

# force re-index even if fresh index exists
uv run src/core/BM25/main.py --rerun
```

All flags can be combined. Example — index a single file with custom threshold and inspection:

```bash
uv run src/core/BM25/main.py --file bible.pdf --threshold 0.2 --inspect
```

---

## How it works

```
chunks → detect domain stopwords → tokenize → compute frequencies → save .bm25
```

1. **Domain stopword detection**: Terms appearing in >30% of chunks are flagged as noise. "photosynthesis" in a biology textbook or "lord" in a Bible — these appear everywhere and don't discriminate between chunks
2. **Tokenization**: BlingFire `text_to_words()` splits text, removes standard English stopwords + domain stopwords
3. **Frequency calculation**: Document frequencies (DF), chunk lengths, average document length (avgdl)
4. **Save**: All raw counts saved via `DarwinStore.save_bm25()` for later scoring

**Why save counts not scores?** BM25 scoring requires IDF and TF calculations with tunable `k1` and `b` parameters. Saving raw frequencies lets search-time code compute scores dynamically with any parameters.

---

## Core functions

| Function | Purpose |
|----------|---------|
| `detect_domain_stopwords()` | Find terms appearing in >threshold% of chunks |
| `tokenize_chunks()` | Convert chunk.clean → list of tokens |
| `index_chunks_for_bm25()` | Build complete BM25Data from chunks |
| `run_bm25_indexing_pipeline()` | Full pipeline + save to store |

---

## Output data model

```python
@dataclass
class BM25Data:
    tokens: List[List[str]]    # Tokenized chunks (for scoring)
    doc_freqs: dict            # Term → document frequency (for IDF)
    avgdl: float               # Average chunk length in tokens
    doc_lengths: List[int]     # Length of each chunk
    domain_stopwords: List[str]# Detected per-document stopwords
    threshold: float           # Threshold used for detection
    chunk_count: int           # Total chunks indexed
    vocab_size: int            # Unique terms in vocabulary
```

---

## Domain stopword detection

Standard stopword lists (the, and, of) don't know that "photosynthesis" in a biology textbook is noise — it appears in 400 chunks and tells you nothing about which chunk is relevant to a query. Domain stopword detection fixes this automatically.

| Threshold | Effect |
|-----------|--------|
| 0.30 (default) | Terms in >30% of chunks become stopwords |
| 0.20 (aggressive) | More terms filtered, smaller vocabulary, faster search |
| 0.50 (conservative) | Only ubiquitous terms filtered |

**Example from CSB Pew Bible (1120 pages):**
```
Domain stopwords detected: lord, god, israel, david, said, people, king, son, house, land...
```

These terms appear everywhere — removing them from BM25 indexing improves precision because only discriminating terms remain.

---

## Tokenization

Uses BlingFire `text_to_words()` — same library as sentence splitting, no new dependencies.

```python
# Input: "David, king of Israel, said to Nathan..."
# Output: ["david", "king", "israel", "said", "nathan"]
```

Filters applied:
- Minimum token length: 2 characters
- Removes standard English stopwords (the, and, of, etc.)
- Removes detected domain stopwords
- Strips punctuation and numbers

No lemmatization — BM25 + dense embeddings together cover semantic variation without the complexity.

---

## File structure

```
src/core/BM25/
├── __init__.py           Module exports
├── BM25Indexer.py        Core indexing logic, pipeline orchestration
├── BM25Utils.py          Tokenization, stopword detection, args
└── main.py               CLI entry point, inspection
```

### Data flow

```
CLI args
  → handle_bm25_indexing()              main.py
    → DarwinStore.load_chunks()         get chunks from .darwin
    → run_bm25_indexing_pipeline()      BM25Indexer.py
      → detect_domain_stopwords()       BM25Utils.py — term frequency across chunks
      → tokenize_chunks()               BlingFire + stopword filtering
      → compute doc_freqs, avgdl        Counter + statistics
      → BM25Data                        package for saving
    → DarwinStore.save_bm25()           persist to .darwin
```

---

## Constants

| Constant | Default | Effect |
|----------|---------|--------|
| `DOMAIN_STOP_THRESHOLD` | 0.30 | Terms in >30% of chunks → domain stopword |
| `MIN_TOKEN_LENGTH` | 2 | Skip single characters and empty tokens |
| `STANDARD_STOPS` | ~50 words | Hardcoded English stopwords |

---

## Integration with search

The `.bm25` bundle is loaded at search time for keyword retrieval:

```python
# Later, in KeywordSearcher.py
data = store.load_bm25(filename)

# BM25 scoring happens dynamically
bm25 = BM25Okapi(
    corpus=data.tokens,
    k1=1.5,   # Term frequency saturation
    b=0.75,   # Length normalization
)

scores = bm25.get_scores(tokenize_query("david covenant"))
```

---

## Performance reference

Tested on CSB Pew Bible (1120 pages, 1143 chunks).

| Operation | Time |
|-----------|------|
| Domain stopword detection | ~0.3s |
| Tokenization (1143 chunks) | ~0.5s |
| Frequency computation | ~0.1s |
| **Total indexing** | **~1.0s** |
| Output file size | ~2.8 MB (.pkl) |

BM25 indexing is extremely fast — ~1000 chunks per second on CPU Compared to the implemeentation of the sematic embedding done in src/core/Embedding which is ~100 chunks per second.

