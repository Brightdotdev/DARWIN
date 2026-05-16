"""
src/core/LLM/ModelPicker.py
-----------------------------------
Selects the fastest useful model per provider for LLM calls.
Grades: fast (cheap decisions), think (reasoning), normal (full quality).

Usage:
    from src.core.LLM.ModelPicker import pick_model, ModelGrade

    model = pick_model("openai", grade="fast")   # "openai/gpt-4o-mini"
    model = pick_model("ollama", grade="think")  # "ollama/phi4" or similar
"""

import os
import subprocess
from typing import Optional, Literal

from src.core.Logger.DarwinLogger import darwin, LogType


ModelGrade = Literal["fast", "think", "normal"]


# ═══════════════════════════════════════════════════════════════════════════════
# Provider model maps — fastest useful per grade
# ═══════════════════════════════════════════════════════════════════════════════

_PROVIDER_MAP: dict[str, dict[str, str]] = {
    "openai": {
        "fast": "openai/gpt-4o-mini",
        "think": "openai/gpt-4o",
        "normal": "openai/gpt-4o",
    },
    "openrouter": {
        "fast": "openrouter/google/gemini-2.0-flash-001",
        "think": "openrouter/anthropic/claude-3.5-sonnet",
        "normal": "openrouter/anthropic/claude-3.5-sonnet",
    },
    "anthropic": {
        "fast": "anthropic/claude-3-haiku-20240307",
        "think": "anthropic/claude-3-5-sonnet-20241022",
        "normal": "anthropic/claude-3-5-sonnet-20241022",
    },
    "gemini": {
        "fast": "gemini/gemini-2.0-flash",
        "think": "gemini/gemini-1.5-pro",
        "normal": "gemini/gemini-1.5-pro",
    },
    "groq": {
        "fast": "groq/llama-3.1-8b-instant",
        "think": "groq/llama-3.3-70b-versatile",
        "normal": "groq/llama-3.3-70b-versatile",
    },
    "together": {
        "fast": "together/meta-llama/Llama-3.2-3B-Instruct",
        "think": "together/meta-llama/Llama-3.3-70B-Instruct",
        "normal": "together/meta-llama/Llama-3.3-70B-Instruct",
    },
    "cohere": {
        "fast": "cohere/command-r7b-12-2024",
        "think": "cohere/command-r-plus-08-2024",
        "normal": "cohere/command-r-plus-08-2024",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Ollama discovery
# ═══════════════════════════════════════════════════════════════════════════════

_OLLAMA_CACHE: Optional[list[tuple[str, str, int]]] = None  # [(name, size_str, param_mb)]

# Models that are definitely not LLMs (embeddings, rerankers, etc.)
_NON_LLM_PATTERNS = {
    "nomic-embed-text",
    "all-minilm",
    "bge-large",
    "mxbai-embed-large",
    "snowflake-arctic-embed",
    "embed-",          # any model name containing "embed-"
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
    """Return default model IDs for a cloud provider’s 'fast' and 'smart' tiers."""
    provider = provider.lower().strip()
    if provider not in _PROVIDER_MAP:
        return {"fast": "", "smart": ""}
    # In LLMManager we use "fast" and "smart" tiers; map to "fast" and "normal" grades
    return {
        "fast": _PROVIDER_MAP[provider]["fast"],
        "smart": _PROVIDER_MAP[provider]["normal"],
    }


import json  # needed for JSON parsing

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
            data = json.loads(result.stdout)  # expects a list of objects
            models = []
            for entry in data:
                name = entry.get("name", "").split(":")[0]  # strip tag
                if not is_likely_llm_model(name):
                    continue
                # Parse size field like "4.7 GB"
                size_raw = entry.get("size", "0 B")
                parts = size_raw.split()
                if len(parts) >= 2:
                    try:
                        val = float(parts[-2])
                        unit = parts[-1].upper()
                        mb = int(val * {"B": 1/1e6, "KB": 1/1e3, "MB": 1, "GB": 1e3, "TB": 1e6}.get(unit, 1))
                    except (ValueError, IndexError):
                        mb = 999999
                else:
                    mb = 999999
                models.append((name, size_raw, mb))
            models.sort(key=lambda x: x[2])
            _OLLAMA_CACHE = models
            darwin.log(f"Ollama LLM models discovered (JSON): {[m[0] for m in models]}")
            return _OLLAMA_CACHE
    except FileNotFoundError:
        darwin.log("Ollama not installed or not in PATH", LogType.WARNING)
        _OLLAMA_CACHE = []
        return _OLLAMA_CACHE
    except Exception as e:
        darwin.log(f"Ollama JSON discovery failed, falling back to text parsing: {e}", LogType.DEBUG)

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
            darwin.log(f"Ollama list failed: {result.stderr}", LogType.WARNING)
            _OLLAMA_CACHE = []
            return _OLLAMA_CACHE

        models = []
        lines = result.stdout.strip().split("\n")[1:]  # skip header

        for line in lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            name = parts[0]
            if not is_likely_llm_model(name):
                continue
            size_str = parts[-2] + " " + parts[-1]  # e.g. "4.7 GB"
            try:
                val = float(parts[-2])
                unit = parts[-1].upper()
                mb = int(val * {"B": 1/1e6, "KB": 1/1e3, "MB": 1, "GB": 1e3, "TB": 1e6}.get(unit, 1))
            except (ValueError, IndexError):
                mb = 999999
            models.append((name, size_str, mb))

        models.sort(key=lambda x: x[2])
        _OLLAMA_CACHE = models
        darwin.log(f"Ollama LLM models discovered (text): {[m[0] for m in models]}")
        return _OLLAMA_CACHE

    except FileNotFoundError:
        darwin.log("Ollama not installed or not in PATH", LogType.WARNING)
    except Exception as e:
        darwin.log(f"Ollama discovery error: {e}", LogType.WARNING)

    _OLLAMA_CACHE = []
    return _OLLAMA_CACHE


def _pick_ollama(grade: ModelGrade) -> str:
    """Pick best Ollama model by size and grade."""
    models = _discover_ollama_models()
    if not models:
        return "ollama/llama3.2"  # safe fallback

    names = [m[0] for m in models]

    if grade == "fast":
        return f"ollama/{names[0]}"
    elif grade == "think":
        idx = min(len(names) // 2, len(names) - 1)
        return f"ollama/{names[idx]}"
    else:
        return f"ollama/{names[-1]}"


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
        darwin.log(f"Model picked: {model} (grade={grade})", LogType.DEBUG)
        return model

    darwin.log(f"Unknown provider '{provider}', using as-is", LogType.WARNING)
    return provider


def _detect_provider() -> str:
    env_map = {
        "OPENAI_API_KEY": "openai",
        "OPENROUTER_API_KEY": "openrouter",
        "ANTHROPIC_API_KEY": "anthropic",
        "GEMINI_API_KEY": "gemini",
        "GROQ_API_KEY": "groq",
        "TOGETHER_API_KEY": "together",
        "COHERE_API_KEY": "cohere",
    }
    for env, name in env_map.items():
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
            f"ollama/{names[0] if names else 'llama3.2'}",
            f"ollama/{names[len(names)//2] if len(names) > 1 else names[0]}",
            f"ollama/{names[-1] if names else 'llama3.2'}",
        ]
    return result


if __name__ == "__main__":
    print("Available:", list_available())
    print("Auto-detect:", _detect_provider())
    print("Fast:", pick_model(grade="fast"))
    print("Think:", pick_model(grade="think"))
    print("Normal:", pick_model(grade="normal"))