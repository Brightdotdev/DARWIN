# Lib — Shared Utilities

**What it is:** Shared utility modules used across all Darwin components — platform path resolution, API key management, terminal helpers, internet checking, and general utilities.

**Why it exists:** These are cross-cutting concerns that don't belong to any specific module. Centralizing them avoids duplication and ensures consistent behavior (e.g., all modules resolve the Darwin directory through `AppPaths`).

## Files

| File | Lines | Purpose |
|---|---|---|
| `AppPaths.py` | — | Darwin base directory resolution (platform-aware) |
| `KeyManager.py` | 228 | Secure API key management via keyring + env vars |
| `Utils.py` | — | General utilities (open_in_explorer, scan_folder, format_size) |
| `Terminal.py` | 591 | Terminal UI helpers (Rich-based, partially superseded by Logger) |
| `InternetChecker.py` | — | Internet connectivity check |
| `Prompts/IntentAnalyzer.md` | — | Intent analysis prompt template for LLM routing |

## Usage

### AppPaths

```python
from src.lib.AppPaths import (
    set_darwin_base_dir,
    default_models_dir,
    default_config_dir,
    default_logs_dir,
    ensure_directories,
)

set_darwin_base_dir("/custom/path")          # Override default
models_dir = default_models_dir()            # Platform-specific model path
config_dir = default_config_dir()            # Config directory
ensure_directories()                         # Create all required dirs
```

Platform defaults:

| OS | Default Base Dir |
|---|---|
| Windows | `%LOCALAPPDATA%/Darwin` |
| macOS | `~/Library/Application Support/Darwin` |
| Linux | `$XDG_DATA_HOME/darwin` (or `~/.local/share/darwin`) |

The base directory can be overridden with:
1. `set_darwin_base_dir()` function call
2. `DARWIN_BASE_DIR` environment variable

### KeyManager

```python
from src.lib.KeyManager import set_api_key, clear_api_key, is_key_set

set_api_key("openai", "sk-...")              # Store in OS keychain
is_key_set("openai")                         # Check if set
key = is_key_set("openai", return_key=True)  # Retrieve
clear_api_key("openai")                      # Remove from keychain
```

Supported providers: OpenAI, Anthropic, Google, Mistral, Groq, Cohere, Together, OpenRouter, DeepSeek.

### Utils

```python
from src.lib.Utils import open_in_explorer, scan_folder, format_size

open_in_explorer("/path/to/folder")          # Open in file manager
files = scan_folder("/path/to/docs", [".pdf", ".md"])  # Recursive scan
human_size = format_size(1048576)            # "1.0 MB"
```
