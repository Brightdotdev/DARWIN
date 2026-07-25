# Darwin Core

Core engine behind Darwin RAG. Four main subsystems:

## Module map

```
src/core/
  Storage/       DarwinStore — vector store, BM25 indices, pipeline registry
  Retrival/      Search orchestration: keyword (BM25), semantic (embeddings), hybrid (RRF fusion)
  Pipeline/      Ingestion (parsing → chunking) + Indexing (BM25 + embeddings)
  Orchestrators/ High-level orchestrators that wire Storage + Retrival + Pipeline together
  Logger/        Structured session logging
  Models/        Model registry (40+), inference factories, downloaders
```

## Orchestrators

The orchestrators are the main API surface. Each one exposes a single entry method with typed parameters from `lib.py`.

### `Orchestrators/lib.py`

Central parameter library. All defaults live here. Each parameter class has a `.build()` method that produces the internal config object.

```python
from src.core.Orchestrators.lib import RetrievalParameters, PipelineParameters

# Retrieval — Literal types give IDE autocomplete for options
params = RetrievalParameters(
    query="what is Darwin?",
    method="hybrid",          # Literal["hybrid", "semantic", "keyword"]
    merge_strategy="adjacent", # Literal["none", "adjacent", "aggressive"]
    fallback_strategy="both",  # Literal["expand", "relax", "both"]
)
cfg = params.build()
ctx = orchestrator.retrieve_and_resolve(cfg)

# Pipeline
params = PipelineParameters(
    target="./docs",
    ocr=True,
    recursive=True,
)
internal, stages = params.build()
result = orchestrator.build_pipeline(internal, stages)
```

## Data flow

```
User query → Orchestrators/lib.py (typed params)
                 ↓
           RetrievalOrchestrator
                 ↓
      ┌──────────┼──────────┐
      ↓          ↓          ↓
  Keyword     Semantic   Hybrid (RRF)
   BM25       Embedding   fusion
      ↓          ↓          ↓
      └──────────┼──────────┘
                 ↓
         Post-processing
     (filter, penalize, merge)
                 ↓
          ResolvedPassages
```
