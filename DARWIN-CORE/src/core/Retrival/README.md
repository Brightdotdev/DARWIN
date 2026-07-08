# Retrival — Search Engine

**What it is:** The search and retrieval system supporting hybrid (BM25 + semantic), pure semantic, and pure keyword search with configurable reranking, merging, fallback, and structural penalties.

**Why it exists:** RAG needs multiple search strategies. Hybrid search combines keyword precision (BM25) with semantic understanding (embeddings). The orchestrator layer adds filtering, penalization, merging, and fallback logic for production-quality retrieval.

## Architecture

```
Query ──> handle_retrieval()
            ├── hybrid ──> RRF fusion of BM25 + semantic scores
            ├── semantic ──> USearch vector similarity
            └── keyword ──> BM25 OKapi scoring
                    │
                    ▼
         Reranking (optional) ──> ONNX cross-encoder
                    │
                    ▼
         RetrievalOrchestrator.retrieve_and_resolve()
            ├── Over-fetch + filter sources
            ├── Structural penalties (headers, tables)
            ├── Fallback detection (relax thresholds)
            ├── Merge adjacent passages + fill gaps
            └── Resolve → RetrievalContext
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Unified entry point `handle_retrieval()` |
| `lib.py` | Gold-standard defaults, arg dataclasses, result dataclasses |
| `RetrievalArgs.py` | Unified `RetrievalArgs` dataclass |
| `RetrievalResult.py` | `RetrievalResult` with `ResultType` enum |
| `OrchestratorTypes.py` | `OrchestratorConfig`, `MergeStrategy`, `FallbackStrategy`, `ResolvedPassage`, `RetrievalContext` |
| `Hybrid/` | `HybridSearch.py` — RRF fusion of BM25 + semantic |
| `Semantic/` | `SemanticSearch.py` — USearch vector similarity |
| `Keyword/` | `KeywordSearch.py` — BM25 OKapi scoring |
| `Reranking/` | `Reranker.py` — ONNX cross-encoder |

## Usage

### Via Orchestrator (recommended)

```python
from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
from src.core.Retrival.OrchestratorTypes import OrchestratorConfig, MergeStrategy

store = DarwinStore.get_default_store()
orch = RetrievalOrchestrator(store)

ctx = orch.retrieve_and_resolve(OrchestratorConfig(
    query="machine learning",
    method="hybrid",
    top_k=10,
    max_passages=5,
    merge_strategy=MergeStrategy.ADJACENT,
    rerank=True,
))

for p in ctx.passages:
    print(f"[{p.source}] {p.text[:200]}")
```

### Direct search methods

```python
from src.core.Retrival.main import handle_retrieval
from src.core.Retrival import RetrievalArgs

args = RetrievalArgs(query="AI safety", method="hybrid", top_k=10)
results = handle_retrieval(store, args)
```

### Via CLI

```bash
darwin search "your query"
darwin search "your query" --method semantic --top-k 20
```

### Via MCP tool

```python
# From an AI agent via FastMCP
result = await call_tool("search_darwin", {
    "query": "your query",
    "method": "hybrid",
    "top_k": 10,
})
```

## Key Methods

| Method | Description |
|---|---|
| `hybrid_search` | RRF fusion of BM25 + embedding scores |
| `semantic_search` | USearch cosine similarity of embeddings |
| `keyword_search` | BM25 OKapi TF-IDF scoring |
| `rerank_passages` | ONNX cross-encoder re-scoring |
| `retrieve_and_resolve` | Full pipeline: fetch → filter → penalize → merge → fallback → resolve |

## Defaults

| Parameter | Default | Description |
|---|---|---|
| `top_k` | 10 | Initial result count |
| `max_passages` | 5 | Final passage count |
| `max_chars` | 6000 | Max chars per passage |
| `merge_strategy` | `adjacent` | Passage merging: none, adjacent, all |
| `rerank` | `false` | Enable cross-encoder reranking |
| `min_passage_score` | 0.3 | Minimum score threshold |
