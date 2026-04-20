# Darwin — Keyword Search Module

The keyword search module performs pure BM25 sparse retrieval over indexed document chunks. It loads `.pkl` bundles from DarwinStore and returns exact keyword matches ranked by term frequency and inverse document frequency.

---

## How to run it

```bash
# search all documents in current project
uv run src/core/Search/Keyword/main.py "photosynthesis"

# search a specific document
uv run src/core/Search/Keyword/main.py "cell division" --file biology.pdf

# get more results
uv run src/core/Search/Keyword/main.py "mitochondria" --top-k 10

# tune BM25 parameters (k1 controls term frequency saturation, b controls length normalization)
uv run src/core/Search/Keyword/main.py "ATP synthesis" --k1 1.2 --b 0.5

# inspect — show full chunks with metadata and query tokens
uv run src/core/Search/Keyword/main.py "DNA replication" --inspect

# export as JSON for scripting
uv run src/core/Search/Keyword/main.py "enzymes" --json
```

All flags can be combined. Example — search specific file with custom BM25 parameters and inspection:

```bash
uv run src/core/Search/Keyword/main.py "david covenant" --file bible.pdf --k1 1.2 --top-k 10 --inspect
```

---

## How it works

```
query → tokenize → BM25 scoring → top-k chunks
```

1. **Query tokenization**: User query is tokenized using BlingFire `text_to_words()` — same tokenizer used during BM25 indexing
2. **Standard stopword removal**: Common English words (the, and, of) are filtered out
3. **BM25 scoring**: Each chunk receives a score based on term frequency, inverse document frequency, and chunk length
4. **Ranking**: Results sorted by descending BM25 score, capped at `top_k`

**Why same tokenizer?** The query must use identical tokenization to the index — otherwise terms won't match the vocabulary and search returns nothing.

---

## BM25 formula

```
score(D, Q) = Σ IDF(qi) × TF(qi, D)

Where:
    IDF(qi) = log((N - df + 0.5) / (df + 0.5) + 1)
    TF(qi, D) = (freq × (k1 + 1)) / (freq + k1 × (1 - b + b × len/avgdl))
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `k1` | 1.5 | Term frequency saturation — higher values weight repeated terms more heavily |
| `b` | 0.75 | Length normalization — higher values penalize long chunks more |

---

## Core functions

| Function | Purpose |
|----------|---------|
| `keyword_search()` | Search within pre-loaded index data + chunks |
| `search_single_document_keyword()` | Search one file from DarwinStore |
| `search_across_documents_keyword()` | Search all indexed docs, merge and re-rank results |

All three return `List[KeywordSearchResult]` sorted by descending BM25 score.

---

## Output data model

```python
@dataclass
class KeywordSearchResult:
    text: str           # Original content → LLM context
    clean: str          # What was indexed
    score: float        # BM25 score (not normalized to 0-1)
    source: str         # Document filename
    page: int           # Page number
    heading: str        # Section title (empty if none)
    chunk_index: int    # Original position in document
```

**Note on scores:** BM25 scores are not bounded to 0-1 like cosine similarity. A score of 5.2 is better than 3.1, but there's no absolute "good" threshold. Scores are only comparable within the same search.

---

## Integration with RAG

```python
from src.core.KeywordSearch import search_across_documents_keyword, to_llm_context

# Search across all indexed documents
results = search_across_documents_keyword(
    query="photosynthesis light reactions",
    store=store,
    top_k=5
)

# Format for LLM consumption
context = to_llm_context(results)

# Send to LLM
response = llm.generate(f"{context}\n\nQuestion: {query}")
```

The `to_llm_context()` helper formats results identically to semantic search:

```
[Document: biology.pdf, Page: 42]
Heading: Light-Dependent Reactions
Content: The light-dependent reactions take place in the thylakoid membrane...
---
```

---

## Query tokenization

Queries undergo the same tokenization as chunks during indexing:

```python
# Input: "David, king of Israel, said to Nathan..."
# Output: ["david", "king", "israel", "said", "nathan"]
```

Filters applied:
- Minimum token length: 2 characters
- Removes standard English stopwords (the, and, of, etc.)
- Strips punctuation and numbers
- Lowercases all tokens

No lemmatization or stemming — exact token matching only.

---

## Performance characteristics

| Metric | Value |
|--------|-------|
| Search complexity | O(|Q| × N) where \|Q\| = query terms, N = chunks |
| 1k chunks, 3-term query | <5ms on CPU |
| Memory footprint | Index loaded from `.pkl` (~2-3MB per 1k chunks) |

BM25 search is extremely fast — scoring is simple arithmetic on pre-computed frequencies.

---

## File structure

```
src/core/KeywordSearch/
├── __init__.py           Module exports
├── KeywordSearcher.py    Core BM25 scoring, multi-document orchestration
├── KeywordUtils.py       Query tokenization, result formatting, args
└── main.py               CLI entry point, inspection, JSON export
```

### Data flow

```
CLI args
  → handle_keyword_search()                  main.py
    → search_single_document_keyword()       if --file passed
    → search_across_documents_keyword()      otherwise (all docs)
      → DarwinStore.load_bm25()              load index data (.pkl)
      → DarwinStore.load_chunks()            load chunks for metadata
      → keyword_search()                     KeywordSearcher.py
        → tokenize_query()                   BlingFire + stopword filtering
        → _compute_bm25_scores()             BM25 formula over all chunks
        → top-k indices                      argsort / argpartition
    → format_keyword_results()               terminal table or JSON
    → to_llm_context()                       preview for inspection
```

---

## Constants

| Constant | Default | Effect |
|----------|---------|--------|
| `DEFAULT_TOP_K` | 5 | Number of results returned |
| `DEFAULT_K1` | 1.5 | Term frequency saturation |
| `DEFAULT_B` | 0.75 | Length normalization |
| `STANDARD_STOPS` | ~50 words | Hardcoded English stopwords |
| `MIN_TOKEN_LENGTH` | 2 | Skip single characters |

---

## Comparison: Semantic vs Keyword Search

| Aspect | Semantic Search | Keyword Search |
|--------|-----------------|----------------|
| Matching | Conceptual similarity | Exact term matching |
| Query "car" finds | "automobile", "vehicle" | Only "car" |
| Query "photosynthesis" | All biology content | Chunks containing "photosynthesis" |
| Score range | 0-1 (cosine similarity) | Unbounded (BM25) |
| Speed | ~10ms per 10k chunks | ~5ms per 10k chunks |
| Best for | Broad topics, paraphrased content | Specific terms, proper nouns |


key note : when you want to search for a specific concpet or sometinhh....you can always single it out or specify it with "" marks e.g instead of searching "what are the core neurual network practices" search "what are the core "neural network practicees" "
---
