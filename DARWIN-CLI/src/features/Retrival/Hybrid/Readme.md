# Darwin — Hybrid Search Module

The hybrid search module combines semantic (dense) and keyword (sparse) retrieval via Reciprocal Rank Fusion (RRF). It fuses results from both methods into a single ranked list, leveraging the strengths of each: semantic understanding from embeddings and exact term matching from BM25.

---

## How to run it

```bash
# search all documents in current project
uv run src/core/Search/Hybrid/HybridSearch/main.py "photosynthesis"

# search a specific document
uv run src/core/Search/Hybrid/HybridSearch/main.py "cell division" --file biology.pdf

# get more final results
uv run src/core/Search/Hybrid/HybridSearch/main.py "mitochondria" --top-k 10

# tune RRF dampening constant (lower = more weight on top ranks)
uv run src/core/Search/Hybrid/HybridSearch/main.py "ATP synthesis" --rrf-k 30

# adjust retriever weights (favor semantic over keyword)
uv run src/core/Search/Hybrid/HybridSearch/main.py "enzyme function" --semantic-weight 1.5 --keyword-weight 0.8

# fetch more candidates from each retriever before fusion
uv run src/core/Search/Hybrid/HybridSearch/main.py "cell metabolism" --pool-size 100

# inspect — show individual retriever scores and RRF calculation
uv run src/core/Search/Hybrid/HybridSearch/main.py "DNA replication" --inspect

# export as JSON for scripting
uv run src/core/Search/Hybrid/HybridSearch/main.py "proteins" --json
```

All flags can be combined. Example — search specific file with custom weights and inspection:

```bash
uv run src/core/Search/Hybrid/HybridSearch/main.py "david covenant" --file bible.pdf --semantic-weight 1.2 --pool-size 75 --inspect
```

---

## How it works

```
query
   │
   ├─→ Semantic Search (dense)  → Top N candidates
   │                                      ↘
   └─→ Keyword Search (sparse)  → Top N candidates → RRF Fusion → Top K results
```

1. **Candidate retrieval**: Each retriever independently fetches `pool_size` results
2. **RRF scoring**: Each unique chunk receives an RRF score based on its rank in each retriever's list
3. **Fusion**: Results are sorted by RRF score descending
4. **Truncation**: Top `top_k` results are returned

**Why RRF?** Semantic scores (cosine similarity 0-1) and keyword scores (unbounded BM25) cannot be directly compared. RRF solves this by using only rank position, making score normalization unnecessary.

---

## Reciprocal Rank Fusion formula

```
RRF_score(d) = Σ w_i / (k + rank_i(d))
```

Where:
- `w_i` = weight for retriever i (semantic_weight, keyword_weight)
- `k` = dampening constant (default 60)
- `rank_i(d)` = position of document d in retriever i's results (1-indexed)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `rrf_k` | 60 | Higher values reduce impact of rank differences |
| `semantic_weight` | 1.0 | Weight multiplier for semantic results |
| `keyword_weight` | 1.0 | Weight multiplier for keyword results |
| `pool_size` | 50 | Candidates fetched from each retriever |

---

## Core functions

| Function | Purpose |
|----------|---------|
| `hybrid_search()` | Fuse pre-retrieved semantic and keyword results |
| `search_single_document_hybrid()` | Hybrid search within one file |
| `search_across_documents_hybrid()` | Hybrid search across all indexed documents |

All three return `List[HybridSearchResult]` sorted by descending RRF score.

---

## Output data model

```python
@dataclass
class HybridSearchResult:
    text: str                     # Original content → LLM context
    clean: str                    # Cleaned text (what was indexed)
    rrf_score: float              # Combined RRF score
    semantic_score: Optional[float]  # Original semantic score (if present)
    keyword_score: Optional[float]   # Original BM25 score (if present)
    source: str                   # Document filename
    page: int                     # Page number
    heading: str                  # Section title
    chunk_index: int              # Original position in document
```

**Note on scores:** `rrf_score` is the fused ranking metric. Individual retriever scores are preserved for inspection but not used for final ranking.

---

## Integration with RAG

```python
from src.features.HybridSearch import search_across_documents_hybrid, to_llm_context

# Hybrid search across all documents
results = search_across_documents_hybrid(
    query="photosynthesis light reactions",
    store=store,
    top_k=5,
    semantic_weight=1.2,  # Favor semantic understanding
)

# Format for LLM
context = to_llm_context(results)

# Generate response
response = llm.generate(f"{context}\n\nQuestion: {query}")
```

The `to_llm_context()` helper formats results identically to other search modules:

```
[Document: biology.pdf, Page: 42]
Heading: Light-Dependent Reactions
Content: The light-dependent reactions take place in the thylakoid membrane...
---
```

---

## Performance characteristics

| Metric | Value |
|--------|-------|
| Semantic retrieval | ~10ms per 10k chunks |
| Keyword retrieval | ~5ms per 10k chunks |
| RRF fusion | <1ms for 100 candidates |
| **Total hybrid search** | **~15-20ms** |

Both retrievers run independently, so total time is roughly the slower of the two plus fusion overhead.

---


### Data flow

```
CLI args
  → handle_hybrid_search()                    main.py
    → search_single_document_hybrid()         if --file passed
    → search_across_documents_hybrid()        otherwise (all docs)
      → semantic_search()                     Searcher.py (dense)
      → search_keyword()                      KeywordSearcher.py (sparse)
      → hybrid_search()                       HybridSearcher.py
        → reciprocal_rank_fusion()            HybridUtils.py — RRF algorithm
        → deduplication                       Track by (source, chunk_index)
        → RRF scoring                         Weighted rank fusion
    → format_hybrid_results()                 Terminal table or JSON
    → to_llm_context()                        Preview for inspection
```

---

## Constants

| Constant | Default | Effect |
|----------|---------|--------|
| `DEFAULT_TOP_K` | 5 | Final results returned |
| `DEFAULT_RRF_K` | 60 | RRF dampening constant |
| `DEFAULT_SEMANTIC_WEIGHT` | 1.0 | Weight for semantic retriever |
| `DEFAULT_KEYWORD_WEIGHT` | 1.0 | Weight for keyword retriever |
| `DEFAULT_POOL_SIZE` | 50 | Candidates per retriever |

---

## Comparison: Single vs Hybrid Retrieval

| Method | Strengths | Weaknesses | Best For |
|--------|-----------|------------|----------|
| **Semantic** | Conceptual similarity, paraphrasing | Misses exact terms, proper nouns | Broad topics |
| **Keyword** | Exact term matching, proper nouns | Misses synonyms, context | Specific terminology |
| **Hybrid** | Combines both strengths | Slightly slower | General purpose |

**Example:** Query "cell death"
- Semantic finds "apoptosis", "necrosis", "programmed cell death"
- Keyword finds exact phrase "cell death"
- Hybrid surfaces both, with RRF ranking the most relevant across both sets

---

## Tuning recommendations

| Scenario | Recommended Settings |
|----------|---------------------|
| Academic papers (dense terminology) | `--semantic-weight 1.2 --keyword-weight 1.0` |
| Legal documents (exact phrasing matters) | `--semantic-weight 0.8 --keyword-weight 1.5` |
| General purpose | Defaults (1.0, 1.0) |
| Speed critical | `--pool-size 25` |
| Recall critical | `--pool-size 100 --top-k 20` |

---

