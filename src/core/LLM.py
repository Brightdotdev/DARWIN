"""
src/core/LLM.py
---------------
LiteLLM wrapper for Darwin.

Handles three modes:
    none  → no LLM, returns None for all calls
    local → Ollama, no rate limiting
    api   → external provider with exponential backoff

API keys are read from the OS keychain at runtime.
Never stored in config files or environment variables.

Usage:
    from src.core.Onboarding import get_llm_config
    from src.core.LLM import LLM

    llm_config = get_llm_config(store)
    llm        = LLM.from_config(llm_config)

    response = llm.complete(prompt)
    if response is None:
        # mode is none or call failed
        ...
"""

import os
import time
import random
from dataclasses import dataclass


# ── Data model ─────────────────────────────────────────────────────────────────

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


# ── Provider → LiteLLM env var mapping ────────────────────────────────────────

PROVIDER_ENV_VARS = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "deepseek":   "DEEPSEEK_API_KEY",
}


# ── Main LLM class ─────────────────────────────────────────────────────────────

class LLM:
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

        if api_key and provider in PROVIDER_ENV_VARS:
            os.environ[PROVIDER_ENV_VARS[provider]] = api_key

        if provider == "openrouter":
            os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1/chat/completions"

    @classmethod
    def from_config(cls, config: dict) -> "LLM":
        return cls(
            mode     = config.get("llm_mode",     "none"),
            model    = config.get("llm_model",    ""),
            api_key  = config.get("api_key",      None),
            provider = config.get("llm_provider", ""),
        )

    @property
    def is_available(self) -> bool:
        if self.mode == "none" or not self.model:
            return False
        if self.mode == "api" and not self.api_key:
            return False
        return True

    def complete(
        self,
        prompt:     str,
        system:     str = "",
        max_tokens: int = 1024,
    ) -> LLMResponse | None:
        if not self.is_available:
            return None
        if self.mode == "local":
            return self._call_local(prompt, system, max_tokens)
        if self.mode == "api":
            return self._call_api_with_backoff(prompt, system, max_tokens)
        return None

    def _build_messages(self, prompt: str, system: str) -> list[dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _call_local(self, prompt, system, max_tokens) -> LLMResponse | None:
        try:
            from litellm import completion
            response = completion(
                model      = self.model,
                messages   = self._build_messages(prompt, system),
                max_tokens = max_tokens,
            )
            text  = response.choices[0].message.content or ""
            usage = response.usage or {}
            return LLMResponse(
                text              = text.strip(),
                mode              = "local",
                model             = self.model,
                prompt_tokens     = getattr(usage, "prompt_tokens",     0),
                completion_tokens = getattr(usage, "completion_tokens", 0),
            )
        except Exception as e:
            print(f"  [llm] local call failed: {type(e).__name__}: {e}")
            return None

    def _call_api_with_backoff(self, prompt, system, max_tokens) -> LLMResponse | None:
        try:
            from litellm import completion, RateLimitError
        except ImportError:
            print("  [llm] litellm not installed: pip install litellm")
            return None

        messages = self._build_messages(prompt, system)

        for attempt in range(self.max_retries):
            try:
                response = completion(
                    model      = self.model,
                    messages   = messages,
                    max_tokens = max_tokens,
                )
                text  = response.choices[0].message.content or ""
                usage = response.usage or {}
                return LLMResponse(
                    text              = text.strip(),
                    mode              = "api",
                    model             = self.model,
                    prompt_tokens     = getattr(usage, "prompt_tokens",     0),
                    completion_tokens = getattr(usage, "completion_tokens", 0),
                )
            except RateLimitError:
                if attempt == self.max_retries - 1:
                    print(f"  [llm] rate limited — giving up")
                    return None
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"  [llm] rate limited — waiting {delay:.1f}s")
                time.sleep(delay)
            except Exception as e:
                print(f"  [llm] api call failed: {type(e).__name__}: {e}")
                return None
        return None