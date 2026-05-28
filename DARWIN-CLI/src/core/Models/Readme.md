```markdown
# Darwin Model Manager

A complete CLI and Python API for managing AI models in the Darwin ecosystem. Download, switch, and uninstall local models; configure API providers; and maintain a persistent, config‑driven active‑model state per category.

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [CLI Commands](#cli-commands)
- [Python API](#python-api)
- [Config File](#config-file)
- [Workflows](#workflows)
- [FAQ](#faq)

---

## Architecture

```
┌──────────────┐       ┌──────────────────┐       ┌─────────────────┐
│   cli.py     │──────▶│  ModelManager    │──────▶│  ModelConfig    │
│  (argparse)  │       │  (orchestrator)  │       │  (JSON on disk) │
└──────────────┘       └────────┬─────────┘       └─────────────────┘
                                │
                   ┌────────────┼────────────┐
                   │            │            │
              ModelStore   ModelLoader   DarwinStore
              (downloads)  (inference)   (migration)
```

- **`ModelStore`** – Singleton that downloads/verifies/deletes model files on disk.  
- **`ModelLoader`** – Lazily loads models into memory, wrapping them for unified inference.  
- **`ModelConfig`** – Persists `{category: active_model_id}` to `~/.darwin/active_models.json`.  
- **`ModelManager`** – User‑facing orchestrator; handles switching, installing, uninstalling, and migration.  
- **`DarwinStore`** – Tracks which files were embedded with which model (only needed for migration).

---

## Quick Start

### 1. First‑time Setup (Interactive)

```bash
python -m src.core.Models.cli setup-custom
```

You'll be guided through choosing models for each category. Your selections are saved to the config file and any required local models are downloaded automatically.

### 2. Add API Keys (Optional)

```bash
python -m src.core.Models.cli key set openai sk-your-key-here
python -m src.core.Models.cli key set cohere your-cohere-key
python -m src.core.Models.cli key list
```

Keys are stored in your system keychain and loaded into environment variables when the manager starts.

### 3. See What's Installed

```bash
python -m src.core.Models.cli status
```

---

## CLI Commands

All commands are run from the project root.

```
python -m src.core.Models.cli <command> [options]
```

### `key` – Manage API Keys

| Subcommand | Options | Description |
|------------|---------|-------------|
| `set <provider> <key>` | `--no-validate` | Store an API key |
| `remove <provider>` | | Remove a stored key |
| `list` | | Show all providers and key status |

**Providers:** `openai`, `cohere`, `voyage`, `gemini`, `openrouter`

### `status`

Show every model in the registry with installation status and size.

```
python -m src.core.Models.cli status
```

### `list`

List available models, optionally filtered by category.

```
python -m src.core.Models.cli list --category embedding
```

### `install`

Download one or more models. Does **not** activate them.

| Option | Description |
|--------|-------------|
| `--model`, `-m` | Model key(s) or model_id(s) |
| `--category`, `-c` | Install all models in a category |
| `--tier`, `-t` | Filter category install by tier (`light`, `medium`, `large`, `overkill`) |
| `--force`, `-f` | Re‑download even if already installed |

```bash
python -m src.core.Models.cli install --model BAAI/bge-m3
python -m src.core.Models.cli install --category vision --tier light
```

### `switch`

Change the active model for a category.

| Option | Description |
|--------|-------------|
| `--category`, `-c` | **Required.** Category to switch |
| `--model`, `-m` | Model key or model_id (omit for interactive selection) |
| `--interactive`, `-i` | Force interactive mode |
| `--migrate` | Re‑embed files after switching (embedding only) |

```bash
# Interactive (recommended)
python -m src.core.Models.cli switch --category embedding

# Direct switch
python -m src.core.Models.cli switch --category embedding --model BAAI/bge-m3

# Direct switch with embedding migration
python -m src.core.Models.cli switch --category embedding --model BAAI/bge-m3 --migrate
```

**Interactive mode** shows only:

- Local models (always)
- API models (only if their key is set)

Each entry is numbered – type the number to select it.

### `uninstall`

Remove a model from disk. **Cannot uninstall the currently active model.**

```bash
python -m src.core.Models.cli uninstall --model intfloat/e5-small-v2
```

For API models, you'll be asked if you want to deactivate it.

### `upgrade`

Force re‑download a model (or all installed in a category).

```bash
python -m src.core.Models.cli upgrade --model BAAI/bge-m3
python -m src.core.Models.cli upgrade --category embedding
```

### `validate`

Check hardware compatibility.

```bash
python -m src.core.Models.cli validate --model GritLM/GritLM-7B
python -m src.core.Models.cli validate --category vision
```

### `setup`

Install required or recommended models without interaction.

```bash
python -m src.core.Models.cli setup --profile recommended
```

---

## Python API

```python
from src.core.Models.ModelManager import ModelManager
from src.core.Models.ModelRegistry import ModelCategory
from src.core.Models.ModelStore import get_model_store

# ── Manager ────────────────────────────────────────────────────────
manager = ModelManager()

# See active models
active = manager.get_active_models()
# {'embedding': 'BAAI/bge-m3', 'reranker': '...', ...}

# Switch embedding (no migration)
manager.switch("embedding", "intfloat/e5-small-v2")

# Switch embedding with migration
manager.switch_embedding("BAAI/bge-m3", migrate=True)

# Install a model without activating
manager.install("vikhyatk/moondream2")

# Uninstall (won't work if it's the active model)
manager.uninstall("intfloat/e5-small-v2")

# ── Store (singleton) ──────────────────────────────────────────────
store = get_model_store()
installed = store.list_installed()
size_info = store.get_size_info()

# ── Loader ─────────────────────────────────────────────────────────
loader = ModelManager.get_loader()

# Load the active embedding model
embed_model = loader.get(ModelCategory.EMBEDDING)

# Load the active reranker
reranker = loader.get(ModelCategory.RERANKER)

# Check if a category is ready
loader.ensure_ready(ModelCategory.EMBEDDING)

# Unload to free memory
loader.unload(ModelCategory.EMBEDDING)
```

All switches are immediately persisted to the config file and survive restarts.

---

## Config File

**Location:** `~/.darwin/active_models.json`

**Format:**

```json
{
  "embedding": "BAAI/bge-m3",
  "reranker": "cross-encoder/ms-marco-MiniLM-L-12-v2",
  "vision": null,
  "tokenizer": "en_core_web_md"
}
```

- Written automatically by `switch`, `setup-custom`, and any `set_preference` call.  
- Read by `ModelLoader` on instantiation if no explicit preferences are passed.  
- `null` means no model is active for that category.  
- You can manually edit it, but the recommended way is through the CLI or Python API.

---

## Workflows

### Workflow 1: First Run

1. `setup-custom` – pick models, download them.  
2. `key set openai ...` – add any API keys you plan to use.  
3. `status` – verify everything is installed.

### Workflow 2: Switching Between Local Models

You have `intfloat/e5-small-v2` and `BAAI/bge-m3` installed.

- Work on general documents → `switch -c embedding` → choose `BAAI/bge-m3`.  
- Work on NDA documents where you want a specific model → `switch -c embedding` → choose `intfloat/e5-small-v2`.  
- No migration needed because each file already knows which embedding model was used.

### Workflow 3: Trying an API Model

1. `key set openai sk-...`  
2. `switch -c embedding` → the OpenAI model now appears in the list.  
3. Select it. No download occurs; the config is updated.  
4. Search/embed calls will now use the API wrapper.

### Workflow 4: Migrating After Embedding Switch

1. `switch -c embedding --model BAAI/bge-m3 --migrate`  
2. All files embedded with the previous model are re‑ingested and re‑indexed.  
3. Progress is logged; failed files are reported.

### Workflow 5: Cleaning Up

1. `uninstall --model intfloat/e5-small-v2`  
   - If it's the active model → command aborts.  
   - Switch to another model first, then uninstall.

---

## FAQ

**Q: I already have models downloaded. How do I tell the system to use them?**  
A: Just run `switch --category <category>` and pick the model from the list. If it's already on disk, switching is instant.

**Q: How do I know which model is currently active?**  
A: Check `~/.darwin/active_models.json` or call `manager.get_active_models()` in Python.

**Q: Can I use an API model for embedding and a local model for vision?**  
A: Yes. Each category is independent. The config file stores one model per category.

**Q: Why can't I uninstall my active model?**  
A: To prevent accidentally breaking search/indexing. Switch to another model first.

**Q: What happens if I switch to an API model but haven't set the key?**  
A: The model won't appear in the interactive list. If you force it via direct switch, you'll get a warning and the loader will fail at inference time.

**Q: Does switching automatically re‑embed my files?**  
A: No. Migration is opt‑in via `--migrate`. Files retain metadata about which embedding model was used.

**Q: Where are models stored on disk?**  
A: Under the directory returned by `default_models_dir()` (typically `~/.darwin/models/`).
```