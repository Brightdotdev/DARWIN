# Models — Model Registry & Inference

**What it is:** Model management for 40+ AI models across 5 categories (embedding, NLP, reranker, vision, OCR). Handles registration, download, lifecycle, and inference via ONNX, spaCy, and API providers.

**Why it exists:** Darwin needs local models for offline operation. This module provides a unified interface for model discovery ("what models can I run on my hardware?"), installation (download from HuggingFace), lifecycle (switch/uninstall/upgrade), and inference (ONNX runtime, spaCy, API fallbacks).

## Files

| File | Lines | Purpose |
|---|---|---|
| `ModelManager.py` | 862 | Unified high-level API: init, ensure, use, unload |
| `ModelOrchestrator.py` | 828 | Orchestrator: install, switch, uninstall, upgrade, detect hardware |
| `ModelConfig.py` | — | Active model config persistence (JSON) |
| `ModelStore.py` | — | Model storage management |
| `ModelUtils.py` | — | Utilities |
| `ModelBatchSizing.py` | — | Smart batch size calculation |
| `EmbeddingConfig.py` | — | Embedding configuration |
| `ModeInterfaces.py` | — | Abstract base classes for inference |
| `Lib/ModelRegistry.py` | 566 | Master MODEL_SPECS list (40+ models) |
| `Lib/ModelSpec.py` | — | ModelSpec, ModelCategory, ModelTier, LOADER_TYPES |
| `Lib/ProviderRegistry.py` | — | Provider configuration (9 providers) |
| `Infrence/EmbeddingInferenceFactory.py` | — | ONNX + API embedding inference |
| `Infrence/RerankerInferenceFactory.py` | — | ONNX cross-encoder reranking |
| `Infrence/VisionInfrenceFactory.py` | — | Vision model inference |
| `Infrence/NLPInferenceFactory.py` | — | spaCy NLP inference |
| `Infrence/OCRInferenceFactory.py` | — | RapidOCR inference |
| `ModelDownloading/OnnxEmbeddingDownloader.py` | — | Download ONNX embedding from HuggingFace |
| `ModelDownloading/OnnxRerankerDownloader.py` | — | Download ONNX reranker from HuggingFace |
| `ModelDownloading/SpacyNLPModelDownloader.py` | — | Download spaCy models |

## Model Categories

| Category | Tier | Example Model | Loader |
|---|---|---|---|
| `embedding` | DOLPHIN | `intfloat/multilingual-e5-small-onnx` | ONNX |
| `embedding` | ORCA | `Alibaba-NLP/gte-Qwen2-7B-instruct` | API |
| `nlp` | DOLPHIN | `en_core_web_md` | spaCy |
| `reranker` | DOLPHIN | `cross-encoder/ms-marco-MiniLM-L6-v2-onnx` | ONNX |
| `vision` | DOLPHIN | `HuggingFaceTB/SmolVLM-256M-Instruct-onnx` | ONNX |
| `ocr` | DOLPHIN | `SWHL/RapidOCR` | ONNX |

## Usage

### Via Orchestrator

```python
from src.core.Orchestrators.ModelOrchestrator import modelsOrchestrator

# Detect hardware
sysinfo = modelsOrchestrator.detect_system()
print(f"RAM: {sysinfo.ram_gb}GB, GPU: {sysinfo.gpu_name}")

# Get recommendations
recs = modelsOrchestrator.get_recommendations(sysinfo)
for cat, model in recs.recommendations.items():
    print(f"{cat}: {model.model_id}")

# Install
result = modelsOrchestrator.install("intfloat/multilingual-e5-small-onnx")

# Switch
result = modelsOrchestrator.switch("embedding", "intfloat/multilingual-e5-small-onnx")

# List
models = modelsOrchestrator.list_registered()
models = modelsOrchestrator.list_registered(ModelCategory.EMBEDDING)

# Check before switching
can, reason = modelsOrchestrator.can_switch(category, model_id)
```

### Via ModelManager

```python
from src.core.Models.ModelManager import ModelManager

manager = ModelManager()
manager.initialize()

# Ensure a model is ready
status = manager.ensure_model("embedding", "intfloat/multilingual-e5-small-onnx")

# Get inference engine
engine = manager.use_model("embedding")
embeddings = engine.encode(["hello world"])
```

### Inference Factories (Direct)

```python
from src.core.Models.Infrence.EmbeddingInferenceFactory import create_embedding_inference

engine = create_embedding_inference(
    model_id="intfloat/multilingual-e5-small-onnx",
    model_path="/path/to/model",
)
embeddings = engine.encode(["text to embed"])
```

### Via CLI

```bash
darwin-admin models status
darwin-admin models list
darwin-admin models install intfloat/multilingual-e5-small-onnx
darwin-admin models switch embedding intfloat/multilingual-e5-small-onnx
```

## Provider Registry

9 API providers supported (from `ProviderRegistry.py`):

| Provider | Key Env Var |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Cohere | `COHERE_API_KEY` |
| Together | `TOGETHER_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
