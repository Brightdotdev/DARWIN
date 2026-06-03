# Darwin Model Management System

A modular, production‑grade system for downloading, caching, verifying, and loading AI models. Supports local PyTorch, ONNX, spaCy, and API‑based models with automatic hardware validation, plugin management, and a CLI for full control.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                    ModelManager                  │
│  (orchestrator, user-facing API & CLI backend)   │
└──────────┬──────────┬──────────────┬────────────┘
           │          │              │
    ┌──────▼──────┐ ┌─▼──────────┐ ┌▼─────────────────┐
    │ ModelLoader │ │ ModelStore │ │ PluginManager     │
    │(caching,     │ │(download,  │ │(torch plugin, etc)│
    │ loading,     │ │ verify,    │ └───────────────────┘
    │ fallback)    │ │ delete)    │
    └──────┬───────┘ └─┬──────────┘
           │           │
    ┌──────▼───────┐ ┌─▼──────────────────────┐
    │  Wrappers    │ │  Downloaders            │
    │  (ONNX, HF,  │ │  (PyTorch, ONNX, spaCy) │
    │   API)       │ └─────────────────────────┘
    └──────────────┘
```

- **ModelRegistry** – static definitions of all supported models (`ModelSpec`).
- **ModelPreset** – chunking & prompting parameters per model (in `EmbeddingConfig`).
- **ModelStore** – manages disk storage, downloads, and verification; delegates to specialised downloaders.
- **ModelLoader** – loads models on‑demand with LRU cache, plugin checks, and ONNX fallback.
- **ModelManager** – high‑level API for setup, switching, migrating, and status.
- **PluginManager / TorchPlugin** – optional heavy dependency installation (PyTorch, transformers).
- **CLI** – command‑line interface to all functionality.

---

## 1. Model Registry (`ModelSpec`)

Every model is defined in `REGISTRY` (in `ModelRegistry.py`). Key fields:

```python
@dataclass
class ModelSpec:
    model_id: str          # HuggingFace ID, e.g. "BAAI/bge-m3"
    category: str          # "embedding", "reranker", "vision", "tokenizer"
    tier: str              # "light", "medium", "large", "overkill"
    ram_gb: int            # required system RAM
    disk_mb: int           # download size (0 for API models)
    loader: str            # e.g. "onnx_embedding", "embedding_hf", "api_embedding"
    required: bool         # always installed?
    experimental: bool
    notes: str
    requires_plugin: str   # e.g. "torch"
    onnx_available: bool
    min_length: int        # tokeniser minimum
    max_length: int        # tokeniser maximum
    normalize: bool        # L2‑normalise embeddings
```



## 2. Model Presets (`EmbeddingConfig`)

Defines chunking strategy, prompting prefixes, and instruction templates for each embedding model.

```python
from src.core.Models.EmbeddingConfig import get_preset

preset = get_preset("BAAI/bge-m3-onnx")
print(preset.max_tokens, preset.overlap, preset.category)
# 8000, 5, "semantic_markdown"
```

Used by the chunking pipeline to pre‑split documents before embedding.

---

## 3. ModelStore (download, cache, verify)

Singleton: `get_model_store()`.  
Handles all storage operations – never need to instantiate manually.

### Public API

| Method | Description |
|--------|-------------|
| `store.model_path(key, format=None)` | Get canonical disk path for a model. |
| `store.is_downloaded(key, format=None)` | `True` if model is fully cached and valid. |
| `store.ensure(key, force=False)` | Download if missing; returns the local `Path`. |
| `store.download(key, force=False, minimal=True)` | Trigger download for a model. |
| `store.delete(key, format=None)` | Remove a model from disk. |
| `store.list_installed()` | List keys of all cached models. |
| `store.installed_size(key)` | Human‑readable size of installed model. |
| `store.get_format(key)` | Return storage format (`"onnx"`, `"pytorch"`, `"spacy"`). |

### Usage

```python
from src.core.Models.ModelStore import get_model_store

store = get_model_store()

# Ensure a model is ready
path = store.ensure("BAAI/bge-m3-onnx")  # downloads if not cached

# Check status
if store.is_downloaded("intfloat/e5-small-v2"):
    print("Cached")

# Delete a model
store.delete("intfloat/e5-small-v2", format="onnx")

# List all installed
print(store.list_installed())
```

---

## 4. ModelLoader (load models into memory)

Uses the store to get local paths, wraps models into a uniform interface, caches loaded objects (LRU).

### Public API

| Method | Description |
|--------|-------------|
| `loader.get(category: ModelCategory)` | Load the active model for a category. |
| `loader.preferences` | Dict of category → model key currently set. |
| `loader.set_preference(category, key)` | Switch the active model for a category. |
| `loader.resolve_key(key, category)` | Resolve a user‑facing key to a registry key. |
| `loader.unload(category)` | Remove a model from memory (and clear GPU cache if applicable). |
| `loader.ensure_ready(category)` | Return the loaded model or `None` if not ready. |

### Usage

```python
from src.core.Models.ModelLoader import ModelLoader
from src.core.Models.ModelRegistry import ModelCategory

loader = ModelLoader()  # auto-loads preferences from config

# Get the active embedding model
embedder = loader.get(ModelCategory.EMBEDDING)
# embedder is an ONNXEmbeddingWrapper, SentenceTransformer, or API wrapper

# Use it directly
vectors = embedder.encode(["Hello world"])  # numpy array

# Get reranker
reranker = loader.get(ModelCategory.RERANKER)
scores = reranker.predict([("query", "document")])

# Switch embedding model
loader.set_preference("embedding", "BAAI/bge-m3-onnx")
# Next call to loader.get(EMBEDDING) will use the new model.

# Unload to free memory
loader.unload(ModelCategory.EMBEDDING)
```

**Important:** `ModelLoader` does **not** download plugins automatically. If a model requires `torch` and it’s missing, a `PluginRequiredError` is raised. The user must install the plugin manually (see Section 7).

---

## 5. ModelManager (high‑level orchestration)

Wraps `ModelLoader` and `ModelStore`, adds setup wizards, migration, and model listing.

### Public API

| Method | Description |
|--------|-------------|
| `manager.status()` | Print installed model status. |
| `manager.list_models(category=None)` | Print available models. |
| `manager.list_available_models(category=None)` | Return detailed list for UI. |
| `manager.install(key, force=False)` | Download and cache a model. |
| `manager.uninstall(key)` | Remove a model (with safety checks). |
| `manager.switch(category, model, migrate=False)` | Switch the active model in a category. |
| `manager.switch_with_format(category, base_model, prefer_format=None, migrate=False)` | Switch and auto‑select ONNX or PyTorch variant. |
| `manager.setup(preferences, skip_download=False)` | Complete first‑time setup (downloads). |
| `manager.migrate_embedding(file=None, force=False)` | Re‑embed all files with the current embedding model. |
| `manager.get_loader()` | Return the global `ModelLoader` singleton. |

### Usage

```python
from src.core.Models.ModelManager import ModelManager

manager = ModelManager()

# Setup from defaults (interactive onboarding via CLI)
manager.setup_recommended()

# Switch to ONNX variant of BGE-M3 with auto‑migration
manager.switch_with_format("embedding", "BAAI/bge-m3", prefer_format="onnx", migrate=True)

# List available embedding models for UI
models = manager.list_available_models("embedding")
for m in models:
    print(m["key"], m["status"], m["tags"])

# Get active model keys
print(manager.get_active_models())
```

**Note:** `manager.setup()` internally calls `ModelManager.setup()` static method which downloads all required local models based on hardware and preferences.

---

## 6. Plugin Management (TorchPlugin)

Heavy PyTorch‑dependent models require the `torch` plugin. The plugin is **never auto‑installed** by the loader/manager; the user must explicitly trigger installation.

### Check if torch is available

```python
from src.core.Plugins.TorchPlugin import TorchPlugin

tp = TorchPlugin()
if tp.is_installed():
    print("Torch ready")
else:
    print("Torch plugin missing – install from Settings or CLI")
```

### Install the plugin manually

```python
tp.install_if_needed()  # downloads and installs from GitHub Releases
```

### CLI equivalent

```bash
python -m src.core.Plugins.cli install torch   # if a plugin CLI exists
```

When a model that needs torch is requested and the plugin is missing, a `PluginRequiredError` is raised:

```
PluginRequiredError: Plugin 'torch' is required for model BAAI/bge-m3. Install it from Settings.
```

This error can be caught by the frontend to show an “Install Plugin” button.

---

## 7. CLI – Full Command Reference

All commands are run from the project root:

```bash
python -m src.core.Models.cli <command> [options]
```

### 7.1 Status & Listing

```bash
# Show all installed models and system info
python -m src.core.Models.cli status

# List all models (or filter by category)
python -m src.core.Models.cli list
python -m src.core.Models.cli list --category embedding
```

### 7.2 Installation

```bash
# Install specific models
python -m src.core.Models.cli install --model BAAI/bge-m3-onnx
python -m src.core.Models.cli install --model intfloat/e5-small-v2-onnx --force

# Install all models of a category
python -m src.core.Models.cli install --category reranker

# Install only 'light' tier models
python -m src.core.Models.cli install --category embedding --tier light

# Install the recommended defaults for your hardware
python -m src.core.Models.cli setup --profile recommended

# Or install only required models
python -m src.core.Models.cli setup --profile required
```

### 7.3 Switching Models

```bash
# Interactive switch (shows available models with format tags)
python -m src.core.Models.cli switch --category embedding --interactive

# Direct switch by registry key
python -m src.core.Models.cli switch --category embedding --model BAAI/bge-m3-onnx

# Switch with format preference (auto‑select ONNX/PyTorch variant)
python -m src.core.Models.cli switch-format --category embedding --base-model BAAI/bge-m3 --format onnx

# Switch and migrate embeddings (re‑index all files)
python -m src.core.Models.cli switch --category embedding --model BAAI/bge-m3 --migrate
```

### 7.4 Uninstalling

```bash
python -m src.core.Models.cli uninstall --model intfloat/e5-small-v2-onnx
```

### 7.5 Upgrading

```bash
# Upgrade a specific model
python -m src.core.Models.cli upgrade --model BAAI/bge-m3-onnx

# Upgrade all installed models in a category
python -m src.core.Models.cli upgrade --category embedding
```

### 7.6 Validation

Check if a model can run on your hardware:

```bash
python -m src.core.Models.cli validate --model GritLM/GritLM-7B
python -m src.core.Models.cli validate --category vision
```

### 7.7 API Key Management

```bash
# Set an API key (saved to OS keychain)
python -m src.core.Models.cli key set openai sk-...

# Remove a key
python -m src.core.Models.cli key remove openai

# List configured providers and key presence
python -m src.core.Models.cli key list
```

### 7.8 Custom Onboarding (interactive, cross‑platform)

```bash
python -m src.core.Models.cli setup-custom
```

This launches the async onboarding wizard (works in terminal and browser via Darwin WebSocket).

---

## 8. Full Workflow Examples

### First‑time user (automatic setup)

```bash
# 1. Run interactive setup (selects models, prompts for API keys)
python -m src.core.Models.cli setup-custom

# 2. Verify
python -m src.core.Models.cli status

# 3. Index some documents (example)
python -m src.features.Ingestion.main --target ./docs
```

### Switching to ONNX for faster CPU inference

```bash
# Switch embedding to ONNX variant
python -m src.core.Models.cli switch --category embedding --model BAAI/bge-m3-onnx

# Re‑embed existing files with the new model
python -m src.core.Models.cli switch --category embedding --model BAAI/bge-m3-onnx --migrate
```

### Adding a new model programmatically

```python
from src.core.Models.ModelStore import get_model_store
from src.core.Models.ModelLoader import ModelLoader
from src.core.Models.ModelRegistry import ModelCategory

store = get_model_store()
loader = ModelLoader(store)

# Download if not cached
store.ensure("HuggingFaceTB/SmolVLM-256M-Instruct")

# Activate as vision model
loader.set_preference("vision", "HuggingFaceTB/SmolVLM-256M-Instruct")

# Use
vision = loader.get(ModelCategory.VISION)
caption = vision.caption("image.jpg")
```

### Listing models for a custom UI

```python
manager = ModelManager()
models = manager.list_available_models("embedding")
# returns list of dicts with 'key', 'status', 'tags', etc.
```

---

## 9. Error Handling & Recovery

| Error | Cause | Action |
|-------|-------|--------|
| `PluginRequiredError` | A required plugin (torch) is not installed. | Install plugin manually or via Settings. |
| `DownloadError` | Network issue or corrupt download. | The CLI automatically retries; you can also resume later. |
| `IntegrityError` | Cached model is incomplete or corrupted. | Delete the model (`uninstall`) and re‑download. |
| `ValueError` with "Unknown model" | The model key is not in the registry. | Check spelling or use `list` to see available keys. |
| `KeyError` in `get_spec` | Registry key mismatch (e.g., using model_id instead of key). | Use `resolve_key` or the full registry key (e.g., `"BAAI/bge-m3-onnx"`). |

### Recovery helpers

```python
from src.core.Models.ModelSetup import handle_download_error

try:
    store.download("some-model")
except Exception as e:
    handle_download_error(e, "some-model")  # prints actionable steps
```

---

## 10. Extending the System

### Adding a new model

1. Add an entry to `REGISTRY` in `ModelRegistry.py` with appropriate `loader` and `requires_plugin`.
2. If it’s an embedding model, add a `ModelPreset` in `EmbeddingConfig.py`.
3. If it requires a new type of loader, implement a wrapper in `ModelLoading/Wrappers.py` and register the loader type in `ModelLoader`.

### Adding a new plugin

1. Create a ZIP containing a virtualenv’s `site-packages` (see `scripts/build_torch_plugin.py` as example).
2. Add detection logic in a plugin class similar to `TorchPlugin`.
3. Add the download URL to GitHub Releases.
4. The `ModelManager` and `ModelLoader` will automatically check `spec.requires_plugin` and raise `PluginRequiredError`.

---

## 11. Development & Testing

### Running the test suite (if available)

```bash
pytest tests/core/Models/
```

### Simulating missing torch

```python
# Before importing, mock the plugin check
import src.core.Plugins.TorchPlugin as tp
tp.TorchPlugin.is_installed = lambda self: False
```

### Inspecting cache

```python
store = get_model_store()
print(store.resolver.root)  # ~/.darwin/models/
```

All model files are stored under `~/.darwin/models/{category}/{org}/{name}/{format}/`.

---

## Summary

This system gives you:
- **Zero‑install** for ONNX and API models.
- **One‑click plugin** for heavy PyTorch models.
- **Hardware‑aware** model recommendations.
- **Full CLI** for scripting and automation.
- **Clean separation** of concerns for maintainability.

Use `ModelManager` for everyday tasks, `ModelLoader` for programmatic loading, and the CLI for administration. All components are thread‑safe and emit progress via the Darwin event system.