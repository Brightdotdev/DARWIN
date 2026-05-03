# Darwin — Semantic Search Module

The semantic search module performs pure dense vector retrieval over embedded document chunks using cosine similarity. It loads `.npz` bundles from DarwinStore and returns ranked results with full metadata for RAG contexts.

---

## How to run it

```bash
# search all documents in current project
uv run src/core/Search/Sematic/main.py "photosynthesis"

# search a specific document
uv run src/core/Search/Sematic/main.py "cell division" --file biology.pdf

# get more results
uv run src/core/Search/Sematic/main.py "mitochondria" --top-k 10

# set minimum similarity threshold
uv run src/core/Search/Sematic/main.py "ATP synthesis" --min-score 0.5

# export as JSON for scripting
uv run src/core/Search/Sematic/main.py "enzymes" --json

# inspect — show full chunks with metadata
uv run src/core/Search/Sematic/main.py "DNA replication" --inspect
```

All flags can be combined. Example — search specific file with high precision and inspect output:

```bash
uv run src/core/Search/Sematic/main.py "photosynthesis light reactions" --file biology.pdf --top-k 5 --min-score 0.4 --inspect
```

---

## How it works

```
query → embed → dot product → top-k chunks
```

1. **Query embedding**: User query gets `query: ` prefix and runs through the same e5-small model used for chunks
2. **Similarity calculation**: Dot product between query vector and all chunk vectors (L2-normalized, so dot product = cosine similarity)
3. **Ranking**: Results sorted by score, filtered by optional threshold, capped at `top_k`

The asymmetric prefixes (`passage: ` for chunks, `query: ` for search) are **critical** for e5 models. Using incorrect or missing prefixes significantly degrades retrieval quality.

---

## Core functions

| Function | Purpose |
|----------|---------|
| `semantic_search()` | Search within pre-loaded vectors + chunks array |
| `search_single_document()` | Search one file from DarwinStore |
| `search_across_documents()` | Search all embedded docs, merge and re-rank results |

All three return `List[SearchResult]` sorted by descending similarity.

---

## Output data model

```python
@dataclass
class SearchResult:
    text: str           # Original content → LLM context
    clean: str          # What was embedded (normalized)
    score: float        # Cosine similarity (0-1)
    source: str         # Document filename
    page: int           # Page number
    heading: str        # Section title (empty if none)
    chunk_index: int    # Original position in document
```

---

## Integration with RAG

```python
from src.features.Retrival import search_across_documents, to_llm_context

# Search across all embedded documents
results = search_across_documents(
    query="photosynthesis light reactions",
    store=store,
    top_k=5,
    min_score=0.3
)

# Format for LLM consumption
context = to_llm_context(results)

# Send to LLM
response = llm.generate(f"{context}\n\nQuestion: {query}")
```

The `to_llm_context()` helper formats results as:

```
[Document: biology.pdf, Page: 42]
Heading: Light-Dependent Reactions
Content: The light-dependent reactions take place in the thylakoid membrane...
---
[Document: biology.pdf, Page: 43]
Heading: Photosystems
Content: Photosystem II absorbs light at 680nm...
---
```

---

## Performance characteristics

| Metric | Value |
|--------|-------|
| Search complexity | O(N × D) where N = chunks, D = 384 |
| 10k chunks search time | <10ms on CPU |
| Memory footprint | Vectors loaded from `.npz` (~1.5MB per 1k chunks) |
| Index required | No — works directly from saved bundles |

Vectors are L2-normalized during embedding, so cosine similarity reduces to a simple dot product — no normalization step needed at search time.

---

## File structure

```
src/core/Search/Sematic/
├── __init__.py           Module exports
├── SematicSearch.py           Core search logic, multi-document orchestration
├── SematicSearchUtils.py        Argument handling, result formatting, LLM context
└── main.py               CLI entry point, inspection, JSON export
```

### Data flow

```
CLI args
  → handle_search()                    main.py
    → DarwinStore.load_config()        get embedding model name
    → search_single_document()         if --file passed
    → search_across_documents()        otherwise (all docs)
      → DarwinStore.load_embeddings()  load vectors + metadata per doc
      → semantic_search()              Searcher.py
        → embed_query()                Embedder.py — query vector
        → np.dot(vectors, query_vec)   cosine similarity
        → top-k indices                argpartition for speed
      → _reconstruct_chunks()          rebuild Chunk objects from stored data
    → format_results()                 terminal table or JSON
    → to_llm_context()                 preview for inspection
```

---

## Key utilities

### `format_results(results, json_output=False)`

Returns formatted string for terminal display or JSON export. Terminal output uses rich tables with score, source, page, and text preview.

### `to_llm_context(results, max_chunks=5)`

Converts search results to LLM-ready context string with document source, page number, heading, and full chunk text separated by `---` dividers.

### `SearchArgs`

All CLI parameters in one dataclass:

```python
@dataclass
class SearchArgs:
    query: str
    file: Optional[str] = None
    project: Optional[str] = None
    model: str = DEFAULT_MODEL
    top_k: int = 5
    min_score: float = 0.0
    inspect: bool = False
    json_output: bool = False
```

---

## Constants

| Constant | Default | Effect |
|----------|---------|--------|
| `DEFAULT_TOP_K` | 5 | Number of results returned |
| `DEFAULT_MIN_SCORE` | 0.0 | Minimum similarity threshold (0-1) |
| `DEFAULT_MODEL` | `"intfloat/e5-small-v2"` | Embedding model for queries |

---

## Limitations

- **Exhaustive search only** — no approximate nearest neighbor (ANN) index. Fine for <50k chunks, consider FAISS or LanceDB for larger collections
- **Single model assumption** — all searched documents must use the same embedding model (stored in config)
- **In-memory loading** — vectors for all searched documents loaded into RAM. For many large documents, batch or stream

