"""
src/core/LLM/ModelPicker.py
-----------------------------------
Selects the fastest useful model per provider for LLM calls.
Returns bare model IDs (no provider prefix) — LLMClient handles prefixing.

Tier philosophy:
  fast   → latency & cost optimized (small/flash models, sub-2s target)
  normal → balanced quality/speed (mid-size or standard flagship)
  think  → reasoning optimized (largest/capable, slower acceptable)
"""

import json
import os
import subprocess
from typing import Optional, Literal

from src.core.Logger.DarwinLogger import darwin
from src.agent.LLM.LLMConfig import PROVIDER_ENV_MAP

ModelGrade = Literal["fast", "think", "normal"]


# ═══════════════════════════════════════════════════════════════════════════════
# Provider model maps — bare model IDs (LLMClient will prepend provider/)
# ═══════════════════════════════════════════════════════════════════════════════

_PROVIDER_MAP: dict[str, dict[str, str]] = {
    "openai": {
        "fast": "gpt-4o-mini",
        "think": "gpt-4o",
        "normal": "gpt-4o",
    },
    "openrouter": {
    "fast":  "google/gemini-2.5-flash",
    "think": "deepseek/deepseek-chat",       # DeepSeek-V3 (latest)
    "normal":"deepseek/deepseek-chat",
},
    "anthropic": {
        "fast": "claude-3-haiku-20240307",
        "think": "claude-3-7-sonnet-20250219",
        "normal": "claude-3-7-sonnet-20250219",
    },
    "gemini": {
        "fast": "gemini-2.5-flash",
        "think": "gemini-2.5-pro",
        "normal": "gemini-2.5-pro",
    },
    "groq": {
        "fast": "llama-3.1-8b-instant",
        "think": "llama-3.3-70b-versatile",
        "normal": "llama-3.3-70b-versatile",
    },
    "together": {
        "fast": "meta-llama/Llama-3.2-3B-Instruct",
        "think": "meta-llama/Llama-3.3-70B-Instruct",
        "normal": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "cohere": {
        "fast": "command-r7b-12-2024",
        "think": "command-r-plus-08-2024",
        "normal": "command-r-plus-08-2024",
    },
    "deepseek": {
        "fast": "deepseek-v4-flash",
        "normal": "deepseek-v4-pro",
        "think": "deepseek-v4-pro",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Ollama discovery
# ═══════════════════════════════════════════════════════════════════════════════

_OLLAMA_CACHE: Optional[list[tuple[str, str, int]]] = None

_NON_LLM_PATTERNS = {
    "nomic-embed-text",
    "all-minilm",
    "bge-large",
    "mxbai-embed-large",
    "snowflake-arctic-embed",
    "embed-",
    "reranker-",
    "stella",
    "paraphrase-",
    "minilm",
    "bert-",
}


def is_likely_llm_model(name: str) -> bool:
    """Heuristic to exclude embedding / reranker models."""
    lower = name.lower()
    for pattern in _NON_LLM_PATTERNS:
        if pattern in lower:
            return False
    return True


def get_default_models(provider: str) -> dict[str, str]:
    """Return default bare model IDs for a cloud provider's tiers."""
    provider = provider.lower().strip()
    if provider not in _PROVIDER_MAP:
        return {"fast": "", "smart": "", "normal": ""}
    return {
        "fast": _PROVIDER_MAP[provider]["fast"],
        "smart": _PROVIDER_MAP[provider]["normal"],
        "normal": _PROVIDER_MAP[provider]["normal"],
    }


def _discover_ollama_models() -> list[tuple[str, str, int]]:
    """List installed Ollama LLMs sorted by size (smallest first)."""
    global _OLLAMA_CACHE
    if _OLLAMA_CACHE is not None:
        return _OLLAMA_CACHE

    # ------------------------------------------------------------------
    # Attempt 1: JSON output (Ollama >= 0.1.32)
    # ------------------------------------------------------------------
    try:
        result = subprocess.run(
            ["ollama", "list", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            models = []
            for entry in data:
                name = entry.get("name", "").split(":")[0]
                if not is_likely_llm_model(name):
                    continue

                # Ollama JSON may return size as int (bytes) or str ("1.9 GB")
                size_raw = entry.get("size", "0 B")
                if isinstance(size_raw, (int, float)):
                    mb = int(size_raw / (1024 * 1024))
                    size_str = f"{mb} MB"
                else:
                    size_str = str(size_raw)
                    parts = size_str.split()
                    if len(parts) >= 2:
                        try:
                            val = float(parts[-2])
                            unit = parts[-1].upper()
                            mb = int(val * {"B": 1/1e6, "KB": 1/1e3, "MB": 1, "GB": 1e3, "TB": 1e6}.get(unit, 1))
                        except (ValueError, IndexError):
                            mb = 999999
                    else:
                        mb = 999999

                models.append((name, size_str, mb))

            models.sort(key=lambda x: x[2])
            _OLLAMA_CACHE = models
            darwin.info(f"Discovered Ollama models via JSON: {[m[0] for m in models]}")
            return _OLLAMA_CACHE
    except FileNotFoundError:
        darwin.warning("Ollama not found — not installed or missing from PATH")
        _OLLAMA_CACHE = []
        return _OLLAMA_CACHE
    except Exception as e:
        darwin.info(f"JSON discovery failed, falling back to text parsing: {e}")

    # ------------------------------------------------------------------
    # Attempt 2: Fallback – text parsing of `ollama list`
    # ------------------------------------------------------------------
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            darwin.warning(f"ollama list command failed: {result.stderr}")
            _OLLAMA_CACHE = []
            return _OLLAMA_CACHE

        models = []
        lines = result.stdout.strip().split("\n")[1:]

        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            name = parts[0]
            if not is_likely_llm_model(name):
                continue
            try:
                val = float(parts[-2])
                unit = parts[-1].upper()
                mb = int(val * {"B": 1/1e6, "KB": 1/1e3, "MB": 1, "GB": 1e3, "TB": 1e6}.get(unit, 1))
            except (ValueError, IndexError):
                mb = 999999
            models.append((name, f"{parts[-2]} {parts[-1]}", mb))

        models.sort(key=lambda x: x[2])
        _OLLAMA_CACHE = models
        darwin.info(f"Discovered Ollama models via text parsing: {[m[0] for m in models]}")
        return _OLLAMA_CACHE

    except FileNotFoundError:
        darwin.warning("Ollama not found — not installed or missing from PATH")
    except Exception as e:
        darwin.warning(f"Ollama discovery error: {e}")

    _OLLAMA_CACHE = []
    return _OLLAMA_CACHE


def _pick_ollama(grade: ModelGrade) -> str:
    """Pick best Ollama model by size and grade. Returns bare model name."""
    models = _discover_ollama_models()
    if not models:
        darwin.warning("No Ollama models found — falling back to llama3.2")
        return "llama3.2"

    names = [m[0] for m in models]

    if grade == "fast":
        return names[0]
    elif grade == "think":
        idx = min(len(names) // 2, len(names) - 1)
        return names[idx]
    else:
        return names[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def pick_model(provider: Optional[str] = None, grade: ModelGrade = "normal") -> str:
    if provider is None:
        provider = _detect_provider()

    provider = provider.lower().strip()

    if provider == "ollama":
        return _pick_ollama(grade)

    if provider in _PROVIDER_MAP:
        model = _PROVIDER_MAP[provider].get(grade, _PROVIDER_MAP[provider]["normal"])
        darwin.info(f"Picked model: {provider}/{model} (grade={grade})")
        return model

    darwin.warning(f"Unknown provider '{provider}' — using raw value as model ID")
    return provider


def _detect_provider() -> str:
    for name, env in PROVIDER_ENV_MAP.items():
        if os.getenv(env):
            return name

    try:
        subprocess.run(["ollama", "--version"], capture_output=True, timeout=5)
        return "ollama"
    except Exception:
        pass

    return "openai"


def list_available() -> dict[str, list[str]]:
    result = {}
    for provider, grades in _PROVIDER_MAP.items():
        result[provider] = [grades["fast"], grades["think"], grades["normal"]]

    ollama_models = _discover_ollama_models()
    if ollama_models:
        names = [m[0] for m in ollama_models]
        result["ollama"] = [
            names[0] if names else "llama3.2",
            names[len(names)//2] if len(names) > 1 else names[0],
            names[-1] if names else "llama3.2",
        ]
    return result


if __name__ == "__main__":
    print("Available:", list_available())
    print("Auto-detect:", _detect_provider())
    print("Fast:", pick_model(grade="fast"))
    print("Think:", pick_model(grade="think"))
    print("Normal:", pick_model(grade="normal"))