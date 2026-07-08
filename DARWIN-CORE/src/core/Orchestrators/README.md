# Orchestrators — High-Level Business Logic

**What it is:** The orchestration layer sits between the CLI/MCP and the core engine. Each orchestrator encapsulates a domain: retrieval, pipeline, models, store, and logs.

**Why it exists:** The core engine (DarwinStore, Retrival, Pipeline stages, Models) is low-level. Orchestrators provide the high-level operations that CLI commands and MCP tools need, with proper error handling, session logging, and coordination between subsystems.

## Files

| File | Lines | Purpose |
|---|---|---|
| `RetrievalOrchestrator.py` | 582 | Full retrieval pipeline: fetch, filter, penalize, merge, fallback, resolve |
| `PipelineOrchestrator.py` | 284 | End-to-end pipeline: validate params, run stages, purge artifacts |
| `ModelOrchestrator.py` | 828 | Model lifecycle: install, switch, uninstall, upgrade, onboard, detect hardware |
| `StoreOrchestrator.py` | — | Store lifecycle: status, files, audit, repair, registry operations |
| `LoggerOrchestrator.py` | 182 | Log management: query, purge, export, retention |

## Usage

### RetrievalOrchestrator

```python
from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
from src.core.Retrival.OrchestratorTypes import OrchestratorConfig, MergeStrategy

store = DarwinStore()
orch = RetrievalOrchestrator(store)

config = OrchestratorConfig(
    query="machine learning",
    method="hybrid",
    top_k=10,
    max_passages=5,
    merge_strategy=MergeStrategy.ADJACENT,
    rerank=True,
)

context = orch.retrieve_and_resolve(config)
for p in context.passages:
    print(p.text)
```

### PipelineOrchestrator

```python
from src.core.Orchestrators.PipelineOrchestrator import (
    PipelineOrchestrator, PipelineParameters, PipelineStage
)

orch = PipelineOrchestrator(store)
params = PipelineParameters(
    parsing={"strategy": "auto", "ocr": True},
    chunking={"chunk_size": 512, "overlap": 64},
    indexing={"method": "hybrid"},
)
stages = [PipelineStage.INGESTION, PipelineStage.INDEXING]
result = orch.build_pipeline(params, stages, strict=True, log_mode="normal")

# Purge artifacts
orch.purge([PipelineStage.INGESTION], ["doc1.pdf", "doc2.pdf"])
```

### ModelOrchestrator

```python
from src.core.Orchestrators.ModelOrchestrator import modelsOrchestrator

# Detect hardware
info = modelsOrchestrator.detect_system()
print(f"RAM: {info.ram_gb}GB, Tier: {info.recommended_tier}")

# Install model
modelsOrchestrator.install("intfloat/multilingual-e5-small-onnx")

# Switch active model
modelsOrchestrator.switch("embedding", "intfloat/multilingual-e5-small-onnx")

# List registered
models = modelsOrchestrator.list_registered()
```

### StoreOrchestrator

```python
from src.core.Orchestrators.StoreOrchestrator import StoreOrchestrator

stores = StoreOrchestrator.list_stores()
info = StoreOrchestrator("store-id").info()
status = StoreOrchestrator("store-id").store_status()
file_s = StoreOrchestrator("store-id").file_status("doc.pdf")
```

### LoggerOrchestrator

```python
from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

logs = LoggerOrchestrator.get_logs(levels="error", limit=20)
LoggerOrchestrator.purge_logs(before_days=30)
LoggerOrchestrator.export_logs("export.json")
```
