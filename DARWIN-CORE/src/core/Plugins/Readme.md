# Torch Plugin for Darwin

The **Torch Plugin** provides optional heavy‑dependencies (PyTorch, Transformers, Sentence‑Transformers, etc.) for running local PyTorch models like BGE‑M3, GritLM, SmolVLM, and moondream.  
It is **never installed automatically** – users must explicitly install it when they need those models.  
The plugin is delivered as a pre‑built ZIP from GitHub Releases and integrates seamlessly with Darwin’s model manager.

---

## Architecture

```
User requests a PyTorch model
        │
        ▼
ModelLoader.get() / ModelManager.switch()
        │
        ├── spec.requires_plugin == "torch" ?
        │         │
        │         ▼
        │   TorchPlugin.is_installed() ?
        │         ├── Yes → import torch & load model
        │         └── No  → raise PluginRequiredError
        │                      │
        │                      ▼
        │              Frontend / CLI catches error
        │              and shows "Install Torch Plugin" button / command
        │                      │
        │          User approves → TorchPlugin.install_if_needed()
        │                      │
        │                      ▼
        │          PluginManager.install_from_zip()
        │          (downloads ZIP, shows Darwin progress)
        │                      │
        │                      ▼
        │          ZIP extracted to ~/.darwin/plugins/torch/
        │          site-packages added to sys.path
        │                      │
        └──────────────────────┴──> model loads successfully
```

---

## Plugin Contents

The built ZIP contains a complete **Python virtual environment’s `site-packages`** with:

- `torch` (CPU or CUDA variant depending on platform)
- `transformers`
- `sentence-transformers`
- `accelerate`
- `Pillow`
- `numpy`
- All their sub‑dependencies

No system‑wide installation required. The plugin is isolated in `~/.darwin/plugins/torch/`.

---

## Building the Plugin

### Prerequisites

- Python 3.11
- Git
- Internet access (to download PyPI packages)

### Build locally

```bash
# CPU version
python scripts/build_torch_plugin.py

# CUDA version (run on a machine with NVIDIA GPU)
python scripts/build_torch_plugin.py --cuda

# For Apple Silicon
python scripts/build_torch_plugin.py --arch arm64
```

The script will:
1. Create a temporary virtual environment.
2. Install all required packages into it.
3. Copy the `site-packages` directory to `build/torch-plugin-.../python/`.
4. Create a ZIP file named `darwin-torch-{cpu|cuda}-{os}-{arch}-v2.1.0.zip`.

### Output example

```
Built: darwin-torch-cpu-linux-amd64-v2.1.0.zip
Size: 487.3 MB
```

---

## Releasing via GitHub Actions

The workflow `.github/workflows/torch-plugin.yml` automatically builds the plugin for all supported platforms when a tag like `torch-plugin-v*` is pushed.

### How to release

1. Tag the commit:
   ```bash
   git tag torch-plugin-v2.1.0
   git push origin torch-plugin-v2.1.0
   ```

2. GitHub Actions will:
   - Build CPU versions for `ubuntu-22.04`, `windows-2022`, `macos-13` (Intel), `macos-14` (Apple Silicon).
   - Build CUDA versions for `ubuntu-22.04` and `windows-2022` (optional).
   - Upload all ZIPs as artifacts.
   - Create a **Release** with all ZIPs attached.

3. The download URL pattern used by `TorchPlugin._get_download_url()` is:
   ```
   https://github.com/Brightdotdev/darwin/releases/download/torch-plugin-v{version}/darwin-torch-{suffix}-{os}-{arch}-v{version}.zip
   ```

---

## Installation (User Side)

### Check if the plugin is installed

```python
from src.core.Plugins.TorchPlugin import TorchPlugin
tp = TorchPlugin()
print(tp.is_installed())   # True / False
```

### Install manually via CLI

```bash
# If a plugin CLI is provided:
python -m src.core.Plugins.cli install torch
```

### Install programmatically

```python
tp = TorchPlugin()
tp.install_if_needed()   # Downloads + extracts, shows Darwin spinner/progress
```

After installation, the plugin’s `site-packages` are automatically added to `sys.path` by `PluginManager.add_to_path("torch")`.

---

## Usage in the Application

### Automatic detection

When a model with `requires_plugin = "torch"` is requested, the system checks `TorchPlugin.is_installed()`. If missing, a `PluginRequiredError` is raised **without** attempting any download.

### Example: loading a PyTorch vision model

```python
from src.core.Models.ModelLoader import ModelLoader
from src.core.Models.ModelRegistry import ModelCategory
from src.core.Plugins.TorchPlugin import PluginRequiredError

loader = ModelLoader()

try:
    vision = loader.get(ModelCategory.VISION)
    # If HuggingFaceTB/SmolVLM-256M-Instruct is active and plugin missing,
    # PluginRequiredError is raised.
except PluginRequiredError as e:
    print(f"Torch plugin needed: {e}")
    # UI can offer to install the plugin
```

### UI integration

Frontend catches `PluginRequiredError` and displays a message:

> “This model requires the Torch plugin. It's about 500 MB. [Install]”

On user click, call:

```python
from src.core.Plugins.TorchPlugin import TorchPlugin
TorchPlugin().install_if_needed()
```

After installation, the model can be loaded immediately without restarting the application (plugin path is added dynamically).

---

## Switching Between ONNX and PyTorch

Many models (e.g., `BAAI/bge-m3`) have both an ONNX and a PyTorch variant.  
You can switch using `ModelManager.switch_with_format()`:

```python
manager = ModelManager()

# Switch to ONNX variant (runs without torch plugin)
manager.switch_with_format("embedding", "BAAI/bge-m3", prefer_format="onnx")

# Switch to PyTorch variant (requires torch plugin)
manager.switch_with_format("embedding", "BAAI/bge-m3", prefer_format="pytorch")

# Auto‑select (prefers ONNX if available, else PyTorch if plugin installed)
manager.switch_with_format("embedding", "BAAI/bge-m3")
```

**CLI equivalent:**

```bash
# Prefer ONNX
python -m src.core.Models.cli switch-format --category embedding --base-model BAAI/bge-m3 --format onnx

# Auto‑detect
python -m src.core.Models.cli switch-format --category embedding --base-model BAAI/bge-m3
```

If the required plugin is missing, the command prints an error and instructs the user to install the plugin.

---

## Uninstalling the Plugin

```python
from src.core.Plugins.PluginManager import PluginManager
PluginManager().uninstall("torch")
```

Models that depend on torch will then raise `PluginRequiredError` again until the plugin is reinstalled.

---

## Security & Isolation

- The plugin lives in `~/.darwin/plugins/torch/` and does **not** modify the system Python.
- It uses a pre‑built `site-packages` directory; no arbitrary code execution during installation.
- Adding to `sys.path` happens only after user consent and is reversible (by removing the plugin directory).

---

## Troubleshooting

| Symptom | Solution |
|---------|----------|
| `PluginRequiredError` but you installed the plugin | Ensure the plugin’s `site-packages` path is in `sys.path` – call `PluginManager().add_to_path("torch")`. |
| `import torch` fails after plugin installation | The plugin may be for a different Python version or OS. Re‑download the correct ZIP. |
| “No module named 'torch'” after switching to PyTorch model | The plugin was installed but path wasn’t added. Call `PluginManager().add_to_path("torch")`. |
| Download fails | The GitHub Release URL may be wrong. Check the tag version and platform/arch. |
| Out of disk space | The plugin is ~500 MB. Free up space or use ONNX models instead. |

---

## Summary

- **Build once, run anywhere** – pre‑compiled plugin for each OS/arch.
- **Explicit install** – no background downloads; user stays in control.
- **Graceful fallback** – missing plugin gives clear error, not a crash.
- **Easy format switching** – ONNX for lean core, PyTorch for full power.

The Torch Plugin brings heavy‑lifting AI models to Darwin users without burdening the base installation.