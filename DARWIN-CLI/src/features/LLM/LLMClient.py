import os
import time
import asyncio
import random
import re
from typing import Optional, List, Dict, Any, Union, AsyncGenerator, Generator, Callable, Awaitable

import litellm
from litellm import completion, acompletion
from litellm.caching import Cache

from src.features.LLM.LLMConfig import ProviderConfig, TierConfig
from src.features.LLM.LLMTypes import LLMResponse
from src.core.Logger.DarwinLogger import darwin, LogType

PROVIDER_ENV_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "cohere": "COHERE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

RETRYABLE_EXCEPTIONS = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.APIConnectionError,
)

# Providers that support the "thinking" extra_body parameter
THINKING_SUPPORTED_PROVIDERS = {"openai", "anthropic"}

# Default timeout for LLM calls (seconds)
DEFAULT_TIMEOUT = 30


def _extract_json(text: str) -> str:
    """Strip markdown fences and extra whitespace before JSON parsing."""
    text = text.strip()
    # Remove ```json ... ``` or just ``` ... ```
    if text.startswith("```"):
        # find the first newline after opening ```
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # remove trailing ```
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


class LLMClient:
    def __init__(
        self,
        providers: Dict[str, ProviderConfig],
        default_provider: str,
        enable_cache: bool = False,
        cache_dir: Optional[str] = None,
        max_retries: int = 2,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.providers = providers
        self.default_provider = default_provider
        self.max_retries = max_retries
        self.timeout = timeout

        litellm.enable_json_schema_validation = True
        litellm.suppress_debug_info = True

        if enable_cache:
            litellm.cache = Cache(disk_cache_dir=cache_dir)

        for prov in self.providers:
            env_var = PROVIDER_ENV_MAP.get(prov)
            if env_var and not os.environ.get(env_var):
                darwin.log(
                    f"Environment variable {env_var} not set. Provider '{prov}' will fail unless key is provided later.",
                    LogType.WARNING,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_provider_config(
        self, provider: Optional[str]
    ) -> tuple[str, ProviderConfig]:
        prov = provider or self.default_provider
        if prov not in self.providers:
            raise ValueError(f"Provider '{prov}' not configured")
        return prov, self.providers[prov]

    def _build_litellm_model(self, provider: str, model: str) -> str:
        return f"{provider}/{model}"

    def _prepare_kwargs(
        self,
        messages: List[Dict[str, str]],
        tier: str,
        provider: str,
        json_schema: Optional[Dict[str, Any]],
        override_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        prov, provider_config = self._resolve_provider_config(provider)
        tier_config: TierConfig = getattr(provider_config, tier)

        kwargs = {
            "model": self._build_litellm_model(prov, tier_config.model),
            "messages": messages,
            "temperature": tier_config.temperature,
            "max_tokens": tier_config.max_tokens,
            "timeout": self.timeout,          # ★ new
        }

        extra_body = dict(tier_config.extra_body) if tier_config.extra_body else {}

        # Fast tier: disable thinking, but only for providers that support it
        if tier == "fast" and prov in THINKING_SUPPORTED_PROVIDERS:
            extra_body["thinking"] = {"type": "disabled"}

        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        kwargs["extra_body"] = extra_body

        if override_kwargs:
            user_extra = override_kwargs.pop("extra_body", None)
            kwargs.update(override_kwargs)
            if user_extra:
                kwargs["extra_body"].update(user_extra)

        return kwargs

    # ------------------------------------------------------------------
    # Core retry loop (deduplicated for sync/async)
    # ------------------------------------------------------------------
    def _run_retry_loop(self, call_fn, prov: str, is_async: bool = False):
        """Execute call_fn with retry and exponential backoff + jitter."""
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                response = call_fn()
                raw_text = response.choices[0].message.content
                return raw_text
            except RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt + random.uniform(0, 0.5 * 2 ** attempt)  # ★ jitter
                    darwin.log(
                        f"Transient error on {prov} (attempt {attempt+1}/{self.max_retries+1}): {e}. "
                        f"Retrying in {wait:.1f}s…",
                        LogType.WARNING,
                    )
                    if is_async:
                        import asyncio
                        asyncio.sleep(wait)  # will be awaited by caller
                    else:
                        time.sleep(wait)
                else:
                    darwin.log(f"All retries exhausted for {prov}: {e}", LogType.ERROR)
            except Exception as e:
                darwin.log(f"Non-retryable error for {prov}: {e}", LogType.ERROR)
                raise  # re-raise to be caught by outer caller

        raise last_exception  # all retries failed

    # ------------------------------------------------------------------
    # Sync call
    # ------------------------------------------------------------------
    def _call(
        self,
        messages: List[Dict[str, str]],
        tier: str,
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **override_kwargs,
    ) -> Union[LLMResponse, Generator]:
        if stream:
            raise NotImplementedError("Streaming is not yet implemented.")

        prov, _ = self._resolve_provider_config(provider)
        kwargs = self._prepare_kwargs(messages, tier, prov, json_schema, override_kwargs)

        def sync_call():
            return completion(**kwargs)

        try:
            raw_text = self._run_retry_loop(sync_call, prov, is_async=False)
            parsed = None
            if json_schema is not None:
                cleaned = _extract_json(raw_text)            # ★ strip markdown
                import json as json_mod
                try:
                    parsed = json_mod.loads(cleaned)
                except Exception as e:
                    return LLMResponse(
                        text=raw_text,
                        error=f"Failed to parse JSON response: {e}",
                    )
            return LLMResponse(text=raw_text, json=parsed)
        except Exception as e:
            return LLMResponse(error=str(e))

    # ------------------------------------------------------------------
    # Async call
    # ------------------------------------------------------------------
    async def _acall(
        self,
        messages: List[Dict[str, str]],
        tier: str,
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **override_kwargs,
    ) -> Union[LLMResponse, AsyncGenerator]:
        if stream:
            raise NotImplementedError("Streaming is not yet implemented.")

        prov, _ = self._resolve_provider_config(provider)
        kwargs = self._prepare_kwargs(messages, tier, prov, json_schema, override_kwargs)

        async def async_call():
            return await acompletion(**kwargs)

        # Async retry loop with await for sleep
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await async_call()
                raw_text = response.choices[0].message.content
                # success: parse JSON if needed
                parsed = None
                if json_schema is not None:
                    cleaned = _extract_json(raw_text)      # ★ strip markdown
                    import json as json_mod
                    try:
                        parsed = json_mod.loads(cleaned)
                    except Exception as e:
                        return LLMResponse(
                            text=raw_text,
                            error=f"Failed to parse JSON response: {e}",
                        )
                return LLMResponse(text=raw_text, json=parsed)
            except RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt + random.uniform(0, 0.5 * 2 ** attempt)  # ★ jitter
                    darwin.log(
                        f"Transient error on {prov} (async attempt {attempt+1}/{self.max_retries+1}): {e}. "
                        f"Retrying in {wait:.1f}s…",
                        LogType.WARNING,
                    )
                    await asyncio.sleep(wait)
                else:
                    darwin.log(f"All async retries exhausted for {prov}: {e}", LogType.ERROR)
            except Exception as e:
                darwin.log(f"Non-retryable async error for {prov}: {e}", LogType.ERROR)
                return LLMResponse(error=str(e))

        return LLMResponse(error=str(last_exception))

    # Public sync interface unchanged (call_small, call_smart)
    def call_small(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **override_kwargs,
    ) -> Union[LLMResponse, Generator]:
        return self._call(messages, "fast", provider=provider, json_schema=json_schema, stream=stream, **override_kwargs)

    def call_smart(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **override_kwargs,
    ) -> Union[LLMResponse, Generator]:
        return self._call(messages, "smart", provider=provider, json_schema=json_schema, stream=stream, **override_kwargs)

    async def acall_small(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **override_kwargs,
    ) -> Union[LLMResponse, AsyncGenerator]:
        return await self._acall(messages, "fast", provider=provider, json_schema=json_schema, stream=stream, **override_kwargs)

    async def acall_smart(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **override_kwargs,
    ) -> Union[LLMResponse, AsyncGenerator]:
        return await self._acall(messages, "smart", provider=provider, json_schema=json_schema, stream=stream, **override_kwargs)