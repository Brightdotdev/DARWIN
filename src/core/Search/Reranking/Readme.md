# Darwin — Reranking Module

The reranking module improves search precision using cross-encoder models. It takes candidate chunks from any search method (semantic, keyword, or hybrid) and re-scores them by processing the query and chunk together with full transformer attention.

---

## How to run it

```bash
# Rerank across all documents
uv run src/core/Search/Reranking/main.py "photosynthesis" --search-type semantic

# Rerank within a specific file
uv run src/core/Search/Reranking/main.py "david covenant" --file bible.pdf --search-type hybrid

# Rerank with fast model on single file
uv run src/core/Search/Reranking/main.py "covenant" --file bible.pdf --model fast --inspect

# Compare before/after scores
uv run src/core/Search/Reranking/main.py "cell division" --inspect

# Use most accurate model (slower but better)
uv run src/core/Search/Reranking/main.py "ATP synthesis" --model accurate

# Fetch more candidates before reranking
uv run src/core/Search/Reranking/main.py "mitochondria" --pool-size 100 --top-k 10

# Export as JSON
uv run src/core/Search/Reranking/main.py "enzymes" --json
```

All flags can be combined. Example — rerank hybrid search within a specific file with inspection:

```bash
uv run src/core/Search/Reranking/main.py "covenant" --file bible.pdf --search-type hybrid --model balanced --pool-size 75 --inspect
```

---

## How it works

```
candidates (50-100) → cross-encoder scores → re-sort → top-k results
```

1. **Retrieve candidates**: Fast retrieval (semantic, keyword, or hybrid) fetches `pool_size` candidates from either all documents or a single file
2. **Cross-encoder scoring**: Each (query, chunk) pair is processed together through a transformer
3. **Score blending**: Original retrieval score is optionally blended with rerank score
4. **Truncation**: Top `top_k` results are returned

**Why cross-encoders?** Bi-encoders (semantic search) encode query and chunk separately—fast but misses nuance like negation ("not good"). Cross-encoders process both together with full attention—slower but understands context.

---

## Model options

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| `fast` | 15MB | ~10ms/chunk | Quick iteration, low-resource environments |
| `balanced` | 80MB | ~50ms/chunk | Default, best trade-off |
| `accurate` | 280MB | ~100ms/chunk | Maximum precision, critical queries |

**Model mappings:**
- `fast` → `cross-encoder/ms-marco-TinyBERT-L-2-v2`
- `balanced` → `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `accurate` → `BAAI/bge-reranker-base`

---

## Core functions

| Function | Purpose |
|----------|---------|
| `rerank_results()` | Generic reranking for any result type |
| `rerank_semantic()` | Rerank semantic search results |
| `rerank_keyword()` | Rerank keyword search results |
| `rerank_hybrid()` | Rerank hybrid search results |
| `semantic_search_with_rerank()` | Full pipeline: semantic → rerank |
| `hybrid_search_with_rerank()` | Full pipeline: hybrid → rerank |

All functions support both single-file and multi-document search via the `filename` parameter.

All return `List[RerankedResult]` sorted by descending `final_score`.

---

## Output data model

```python
@dataclass
class RerankedResult:
    text: str                     # Original content → LLM context
    clean: str                    # Cleaned text
    rerank_score: float           # Raw cross-encoder score
    original_score: Optional[float]  # Score from original retriever
    final_score: float            # Combined score (rerank + blended original)
    source: str                   # Document filename
    page: int                     # Page number
    heading: str                  # Section heading
    chunk_index: int              # Original position in document
```

---

## Integration with other search modules

```python
from src.core.Search.Reranking import hybrid_search_with_rerank, to_llm_context

# One-line hybrid search with automatic reranking (all documents)
results = hybrid_search_with_rerank(
    query="photosynthesis light reactions",
    store=store,
    top_k=5,
    pool_size=50,
    model_name="balanced"
)

# Search within specific file
results = hybrid_search_with_rerank(
    query="covenant",
    store=store,
    filename="bible.pdf",  # ← Single file only
    top_k=5
)

# Send to LLM
context = to_llm_context(results)
response = llm.generate(f"{context}\n\nQuestion: {query}")
```

For manual control:

```python
from src.core.Search.Reranking import rerank_hybrid
from src.core.Search.Hybrid import search_single_document_hybrid

# Get candidates from specific file
candidates = search_single_document_hybrid(
    query="david", 
    filename="bible.pdf", 
    store=store, 
    top_k=50
)

# Rerank
results = rerank_hybrid(query, candidates, top_k=5)
```

---

## CLI Arguments

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `query` | (positional) | Required | Search query string |
| `--file` | `-f` | None | Search within specific file |
| `--project` | `-p` | cwd | Project directory |
| `--model` | `-m` | balanced | Reranking model (fast/balanced/accurate) |
| `--search-type` | `-s` | hybrid | Search method (semantic/keyword/hybrid) |
| `--top-k` | `-k` | 5 | Final results to return |
| `--pool-size` | | 50 | Candidates fetched before reranking |
| `--inspect` | `-i` | False | Show detailed before/after comparison |
| `--json` | `-j` | False | Output as JSON |

---

## Performance characteristics

| Stage | Time (50 candidates) |
|-------|---------------------|
| Fast model | ~0.5s |
| Balanced model | ~2.5s |
| Accurate model | ~5.0s |

**Trade-off:** Reranking adds latency but significantly improves precision. Research shows hybrid + rerank outperforms either method alone by 15-25% on retrieval benchmarks.

---

## File structure

```
src/core/Search/Reranking/
├── __init__.py           Module exports
├── Reranker.py           Core cross-encoder logic, integration wrappers
├── RerankerUtils.py      Model loading, args, result formatting
└── main.py               CLI entry point, inspection, file filtering
```

### Data flow

```
CLI args
  → handle_reranking()                    main.py
    → if --file: search_single_*()        fetch from specific file
    → else: search_across_*()             fetch from all documents
    → rerank_*()                          Reranker.py
      → load_cross_encoder()              lazy-load model (cached)
      → model.predict()                   score all (query, chunk) pairs
      → score blending                    combine with original scores
    → format_reranked_results()           terminal table or JSON
```

---

## Constants

| Constant | Default | Effect |
|----------|---------|--------|
| `DEFAULT_MODEL` | `balanced` | Default cross-encoder |
| `DEFAULT_TOP_K` | 5 | Final results returned |
| `DEFAULT_POOL_SIZE` | 50 | Candidates fetched before reranking |
| `preserve_original_weight` | 0.3 | Blend weight for original score |

---

## When to use reranking

| Scenario | Recommendation |
|----------|---------------|
| Simple factoid queries ("What year...") | Skip—bi-encoder is sufficient |
| Complex queries with negation ("without", "except") | **Use reranking** |
| Critical precision (legal, medical) | **Use accurate model** |
| Interactive search (needs speed) | Skip or use fast model |
| Single document deep dive | **Use --file + rerank** |
| Batch processing | **Use reranking** |

---

## Example: Before vs After

**Query:** "verses about David that do NOT mention Goliath"
**File:** bible.pdf

| Without Reranking | With Reranking |
|-------------------|----------------|
| Chunk about David and Goliath (score 0.84) | Chunk about David and Bathsheba (rerank 0.91) |
| Chunk about David and Saul (score 0.81) | Chunk about David's psalms (rerank 0.87) |
| Chunk about Goliath's defeat (score 0.79) | Chunk about David's covenant (rerank 0.85) |

The cross-encoder understands negation and correctly deprioritizes Goliath content.

