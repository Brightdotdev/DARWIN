import os
import time
import asyncio
from typing import Optional, List, Dict, Any, Union, AsyncGenerator, Generator

import litellm
from litellm import completion, acompletion
from litellm.caching import Cache

from src.features.LLM.LLMConfig import ProviderConfig, TierConfig
from src.features.LLM.LLMTypes import LLMResponse   # new
from src.core.Logger.DarwinLogger import darwin, LogType

# Map provider name to env var (needed for early warning, but also used nowhere else)
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

# Transient errors worth retrying
RETRYABLE_EXCEPTIONS = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.APIConnectionError,
)

class LLMClient:
    def __init__(
        self,
        providers: Dict[str, ProviderConfig],
        default_provider: str,
        enable_cache: bool = False,
        cache_dir: Optional[str] = None,
        max_retries: int = 2,              
    ):
        """
        :param providers: dict of provider name -> ProviderConfig
        :param default_provider: fallback provider
        :param enable_cache: enable LiteLLM disk cache
        :param cache_dir: directory for cache storage
        :param max_retries: additional retry attempts for transient errors (total tries = 1 + max_retries)
        """
        self.providers = providers
        self.default_provider = default_provider
        self.max_retries = max_retries

        # Global litellm settings -----------------------------------------------
        litellm.enable_json_schema_validation = True
        litellm.suppress_debug_info = True

        if enable_cache:
            litellm.cache = Cache(disk_cache_dir=cache_dir)

        # Warn if any provider lacks its environment variable
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
        """Build the kwargs dict for litellm call (without env handling)."""
        prov, provider_config = self._resolve_provider_config(provider)
        tier_config: TierConfig = getattr(provider_config, tier)

        kwargs = {
            "model": self._build_litellm_model(prov, tier_config.model),
            "messages": messages,
            "temperature": tier_config.temperature,
            "max_tokens": tier_config.max_tokens,
        }

        # Prepare extra_body
        extra_body = dict(tier_config.extra_body) if tier_config.extra_body else {}

        # Fast tier: force disable thinking (unless user explicitly overrides later)
        if tier == "fast":
            extra_body["thinking"] = {"type": "disabled"}

        # Structured output – response_format is top-level, not inside extra_body
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

        # Merge user overrides safely
        if override_kwargs:
            user_extra = override_kwargs.pop("extra_body", None)
            # User can override temperature, max_tokens, etc.
            kwargs.update(override_kwargs)
            if user_extra:
                # Deep-merge user extra_body (so thinking can be re-enabled)
                kwargs["extra_body"].update(user_extra)

        return kwargs

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
        """Synchronous LLM call with retry and graceful error handling."""
        if stream:
            # We'll implement streaming later; raise for now (or return a generator)
            raise NotImplementedError("Streaming is not yet implemented.")

        prov, _ = self._resolve_provider_config(provider)
        kwargs = self._prepare_kwargs(messages, tier, prov, json_schema, override_kwargs)

        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                response = completion(**kwargs)
                raw_text = response.choices[0].message.content
                # If a schema was requested, parse JSON
                parsed = None
                if json_schema is not None:
                    try:
                        import json as json_mod
                        parsed = json_mod.loads(raw_text)
                    except Exception as e:
                        # Parsing failed – return with error
                        return LLMResponse(
                            text=raw_text,
                            error=f"Failed to parse JSON response: {e}",
                        )
                return LLMResponse(text=raw_text, json=parsed)
            except RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    darwin.log(
                        f"Transient error on {prov} (attempt {attempt+1}/{self.max_retries+1}): {e}. Retrying in {wait}s…",
                        LogType.WARNING,
                    )
                    time.sleep(wait)
                else:
                    darwin.log(
                        f"All retries exhausted for {prov}: {e}",
                        LogType.ERROR,
                    )
            except Exception as e:
                darwin.log(
                    f"Non-retryable error for {prov}: {e}",
                    LogType.ERROR,
                )
                return LLMResponse(error=str(e))

        # If we exit loop without returning, it means all retries failed
        return LLMResponse(error=str(last_exception))

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
        """Asynchronous LLM call with retry and graceful error handling."""
        if stream:
            raise NotImplementedError("Streaming is not yet implemented.")

        prov, _ = self._resolve_provider_config(provider)
        kwargs = self._prepare_kwargs(messages, tier, prov, json_schema, override_kwargs)

        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await acompletion(**kwargs)
                raw_text = response.choices[0].message.content
                parsed = None
                if json_schema is not None:
                    import json as json_mod
                    try:
                        parsed = json_mod.loads(raw_text)
                    except Exception as e:
                        return LLMResponse(
                            text=raw_text,
                            error=f"Failed to parse JSON response: {e}",
                        )
                return LLMResponse(text=raw_text, json=parsed)
            except RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    darwin.log(
                        f"Transient error on {prov} (async attempt {attempt+1}/{self.max_retries+1}): {e}. Retrying in {wait}s…",
                        LogType.WARNING,
                    )
                    await asyncio.sleep(wait)
                else:
                    darwin.log(
                        f"All async retries exhausted for {prov}: {e}",
                        LogType.ERROR,
                    )
            except Exception as e:
                darwin.log(
                    f"Non-retryable async error for {prov}: {e}",
                    LogType.ERROR,
                )
                return LLMResponse(error=str(e))

        return LLMResponse(error=str(last_exception))

    # ------------------------------------------------------------------
    # Public sync interface
    # ------------------------------------------------------------------
    def call_small(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **override_kwargs,
    ) -> Union[LLMResponse, Generator]:
        """Fast, cheap LLM call (sync)."""
        return self._call(
            messages,
            "fast",
            provider=provider,
            json_schema=json_schema,
            stream=stream,
            **override_kwargs,
        )

    def call_smart(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **override_kwargs,
    ) -> Union[LLMResponse, Generator]:
        """Full‑power LLM call (sync)."""
        return self._call(
            messages,
            "smart",
            provider=provider,
            json_schema=json_schema,
            stream=stream,
            **override_kwargs,
        )

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------
    async def acall_small(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **override_kwargs,
    ) -> Union[LLMResponse, AsyncGenerator]:
        """Fast, cheap LLM call (async)."""
        return await self._acall(
            messages,
            "fast",
            provider=provider,
            json_schema=json_schema,
            stream=stream,
            **override_kwargs,
        )

    async def acall_smart(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **override_kwargs,
    ) -> Union[LLMResponse, AsyncGenerator]:
        """Full‑power LLM call (async)."""
        return await self._acall(
            messages,
            "smart",
            provider=provider,
            json_schema=json_schema,
            stream=stream,
            **override_kwargs,
        )