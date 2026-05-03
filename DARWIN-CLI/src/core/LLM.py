"""
src/core/LLM.py
---------------
LiteLLM wrapper for Darwin.

Modes:
    none  → no LLM, returns None immediately
    local → Ollama, auto-detects installed models
    api   → external provider, exponential backoff on rate limits

API keys injected into env at runtime from keychain.
Never written to disk as plaintext.

Usage:
    from src.core.Onboarding import get_llm_config
    llm = LLM.from_config(get_llm_config(store))
    response = llm.complete("your prompt")
"""

import os
import time
import random
from dataclasses import dataclass

from src.lib.Terminal import TerminalUI


# ── Provider env var mapping ───────────────────────────────────────────────────

PROVIDER_ENV_VARS = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "deepseek":   "DEEPSEEK_API_KEY",
}


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    text:              str
    mode:              str
    model:             str
    prompt_tokens:     int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class OllamaModel:
    id:          str    # "ollama/llama3.2"
    name:        str    # "llama3.2"
    size_gb:     float  # disk size
    params:      str    # "3B", "8B" etc


# ── Ollama detection ───────────────────────────────────────────────────────────

def detect_ollama_models(host: str = "http://localhost:11434") -> list[OllamaModel]:
    """
    Detect installed Ollama models by hitting the local API.
    Returns empty list if Ollama isn't running — never raises.
    """
    try:
        import requests
        response = requests.get(f"{host}/api/tags", timeout=5)
        if response.status_code != 200:
            return []

        models = []
        for m in response.json().get("models", []):
            name     = m.get("name", "").split(":")[0]  # strip tag
            size_gb  = m.get("size", 0) / (1024 ** 3)

            # guess param count from name or file size
            params = "unknown"
            n = name.lower()
            if   "3b"  in n or "3.2" in n: params = "3B"
            elif "7b"  in n or "8b"  in n: params = "8B"
            elif "13b" in n:               params = "13B"
            elif "70b" in n:               params = "70B"
            elif size_gb > 40:             params = "70B"
            elif size_gb > 20:             params = "13B"
            elif size_gb > 8:              params = "8B"
            elif size_gb > 2:              params = "3B"

            models.append(OllamaModel(
                id      = f"ollama/{name}",
                name    = name,
                size_gb = round(size_gb, 1),
                params  = params,
            ))

        return models

    except Exception:
        return []


def ollama_is_running() -> bool:
    return len(detect_ollama_models()) > 0


# ── Main LLM class ─────────────────────────────────────────────────────────────

class LLM:
    """
    Unified LLM interface for Darwin.

    Args:
        mode:        "none" | "local" | "api"
        model:       LiteLLM model string e.g. "ollama/llama3.2"
        api_key:     raw key from keychain (injected into env)
        provider:    provider id for env var mapping
        max_retries: retries on rate limit (api only)
        base_delay:  initial backoff seconds
    """

    def __init__(
        self,
        mode:        str        = "none",
        model:       str        = "",
        api_key:     str | None = None,
        provider:    str        = "",
        max_retries: int        = 4,
        base_delay:  float      = 1.0,
    ):
        self.mode        = mode
        self.model       = model
        self.api_key     = api_key
        self.provider    = provider
        self.max_retries = max_retries
        self.base_delay  = base_delay

        # inject key into env so LiteLLM picks it up
        if api_key and provider in PROVIDER_ENV_VARS:
            os.environ[PROVIDER_ENV_VARS[provider]] = api_key

        # OpenRouter needs base URL + optional headers
        if provider == "openrouter":
            os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"

    @classmethod
    def from_config(cls, config: dict) -> "LLM":
        """Build from config dict returned by get_llm_config()."""
        return cls(
            mode     = config.get("llm_mode",     "none"),
            model    = config.get("llm_model",    ""),
            api_key  = config.get("api_key",      None),
            provider = config.get("llm_provider", ""),
        )

    @property
    def is_available(self) -> bool:
        """True if LLM will actually do something when called."""
        if self.mode == "none" or not self.model:
            return False
        if self.mode == "api" and not self.api_key:
            return False
        return True

    def validate(self) -> tuple[bool, str]:
        """
        Check that the configured model is actually reachable.
        Returns (ok, message).
        """
        if self.mode == "none":
            return True, "no LLM configured"

        if self.mode == "local":
            models = detect_ollama_models()
            if not models:
                return False, "Ollama not running — start with: ollama serve"
            ids = [m.id for m in models]
            if self.model not in ids:
                names = ", ".join(m.name for m in models)
                return False, f"model '{self.model}' not installed. available: {names}"
            return True, f"ollama model ready"

        if self.mode == "api":
            # quick ping
            r = self.complete("Reply with one word: OK", max_tokens=10)
            if r and "ok" in r.text.lower():
                return True, "api connection ok"
            return False, "api test failed — check your key"

        return False, f"unknown mode: {self.mode}"

    def complete(
        self,
        prompt:     str,
        system:     str = "",
        max_tokens: int = 1024,
    ) -> LLMResponse | None:
        """
        Send a prompt and return a response.
        Returns None if unavailable or all retries fail.
        Never raises.
        """
        if not self.is_available:
            return None

        if self.mode == "local":
            return self._call_local(prompt, system, max_tokens)

        if self.mode == "api":
            return self._call_api_with_backoff(prompt, system, max_tokens)

        return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _messages(self, prompt: str, system: str) -> list[dict]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def _parse_response(self, response, mode: str) -> LLMResponse:
        text  = response.choices[0].message.content or ""
        usage = response.usage or {}
        return LLMResponse(
            text              = text.strip(),
            mode              = mode,
            model             = self.model,
            prompt_tokens     = getattr(usage, "prompt_tokens",     0),
            completion_tokens = getattr(usage, "completion_tokens", 0),
        )

    def _call_local(self, prompt, system, max_tokens) -> LLMResponse | None:
        try:
            from litellm import completion
            r = completion(
                model      = self.model,
                messages   = self._messages(prompt, system),
                max_tokens = max_tokens,
            )
            return self._parse_response(r, "local")
        except Exception as e:
            print(f"  [llm] local call failed: {type(e).__name__}: {e}")
            return None

    def _call_api_with_backoff(self, prompt, system, max_tokens) -> LLMResponse | None:
        try:
            from litellm import completion, RateLimitError
        except ImportError:
            print("  [llm] litellm not installed: pip install litellm")
            return None

        msgs = self._messages(prompt, system)

        for attempt in range(self.max_retries):
            try:
                r = completion(
                    model      = self.model,
                    messages   = msgs,
                    max_tokens = max_tokens,
                )
                return self._parse_response(r, "api")

            except RateLimitError:
                if attempt == self.max_retries - 1:
                    TerminalUI.error(f" [llm] rate limited — giving up after  {self.max_retries} attempts")
                    return None
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                TerminalUI.dim(f"  (attempt {attempt + 1} of {self.max_retries})")
                time.sleep(delay)

            except Exception as e:
                TerminalUI.error(f" [llm] api call failed: {type(e).__name__}: {e}")
                return None

        return None