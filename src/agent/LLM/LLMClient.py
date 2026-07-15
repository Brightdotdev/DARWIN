import os
import time
import asyncio
import random
import re
import json as json_mod
from typing import Optional, List, Dict, Any, Union, AsyncGenerator, Generator
import logging

# ── Suppress litellm BEFORE importing it ──────────────────────────────
# Set environment variables first — litellm reads these on import
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_SUPPRESS_LOGGING"] = "1"

# Also nuke the root logger level early
logging.getLogger("litellm").setLevel(logging.ERROR)

try:
    import litellm
except ImportError:
    litellm = None

if litellm is not None:
    # Now configure litellm silently
    litellm.suppress_debug_info = True
    litellm.telemetry = False
    litellm.turn_off_message_logging = True
else:
    raise ImportError("litellm is required for LLMClient")

from litellm import completion, acompletion
from litellm.caching import Cache

from src.agent.LLM.LLMConfig import ProviderConfig, TierConfig, PROVIDER_ENV_MAP
from src.agent.LLM.LLMTypes import LLMResponse
from src.core.Logger.DarwinLogger import darwin




# LiteLLM exceptions we consider transient and worth retrying
RETRYABLE_EXCEPTIONS = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.Timeout,
)

# Providers that support the "thinking" extra_body parameter
THINKING_SUPPORTED_PROVIDERS = {"openai", "anthropic"}

DEFAULT_TIMEOUT = 30

# Regex to strip outer markdown JSON fences robustly
_JSON_FENCE_RE = re.compile(r'^```(?:json)?\s*\n?(.*?)\n?```$', re.DOTALL)


def _extract_json(text: str) -> str:
    """Strip outer markdown fences if present; preserve inner content."""
    text = text.strip()
    match = _JSON_FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
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

        # Warn early about missing keys (informational only; we pass keys explicitly)
        for prov in self.providers:
            env_var = PROVIDER_ENV_MAP.get(prov)
            if env_var and not os.environ.get(env_var):
                if not self.providers[prov].api_key:
                    darwin.warning(
                        f"Environment variable {env_var} not set and no explicit api_key "
                        f"for provider '{prov}'. Calls may fail unless key is provided."
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
        """
        Build a LiteLLM model identifier.
        Idempotent: handles both bare model IDs and already-prefixed IDs
        for backward compatibility with existing config files.
        """
        prefix = f"{provider}/"
        if model.startswith(prefix):
            return model
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

        kwargs: Dict[str, Any] = {
            "model": self._build_litellm_model(prov, tier_config.model),
            "messages": messages,
            "temperature": tier_config.temperature,
            "max_tokens": tier_config.max_tokens,
            "timeout": self.timeout,
        }

        # Pass explicit API key if available in config; otherwise LiteLLM falls back to env
        if provider_config.api_key:
            kwargs["api_key"] = provider_config.api_key

        # Pass base_url / api_base for custom endpoints (Ollama, proxies, etc.)
        if provider_config.base_url:
            kwargs["api_base"] = provider_config.base_url

        extra_body = dict(tier_config.extra_body) if tier_config.extra_body else {}

        # Fast tier: disable thinking where supported
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
            if user_extra and isinstance(user_extra, dict):
                kwargs["extra_body"].update(user_extra)

        return kwargs

    # ------------------------------------------------------------------
    # Sync retry loop
    # ------------------------------------------------------------------
    def _run_retry_loop(self, call_fn, prov: str) -> str:
        """Execute call_fn with retry, exponential backoff, and jitter."""
        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = call_fn()
                return response.choices[0].message.content
            except RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait = (2 ** attempt) + random.uniform(0, 0.5 * (2 ** attempt))
                    darwin.warning(
                        f"Transient error on {prov} (attempt {attempt+1}/{self.max_retries+1}): {e}. "
                        f"Retrying in {wait:.1f}s…"
                    )
                    time.sleep(wait)
                else:
                    darwin.error(f"All retries exhausted for {prov}: {e}")
            except Exception:
                # Non-retryable errors (auth, bad request, coding bugs) propagate immediately
                raise

        # All retries failed on transient errors
        if last_exception is not None:
            darwin.error(str(last_exception))
            return f"Error: {last_exception}"
        return "Error: Unknown error after retries"

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

        # Guard: resolve provider & build kwargs so ValueError becomes a clean LLMResponse
        try:
            prov, provider_config = self._resolve_provider_config(provider)
            kwargs = self._prepare_kwargs(messages, tier, prov, json_schema, override_kwargs)
        except ValueError as e:
            darwin.error(str(e))
            return LLMResponse(error=str(e))
        except Exception as e:
            import traceback
            darwin.error(f"Unexpected setup error in _call: {traceback.format_exc()}")
            return LLMResponse(error=f"Setup error: {e}")

        def sync_call():
            return completion(**kwargs)

        try:
            raw_text = self._run_retry_loop(sync_call, prov)
            parsed = None
            if json_schema is not None:
                cleaned = _extract_json(raw_text)
                try:
                    parsed = json_mod.loads(cleaned)
                except Exception as e:
                    return LLMResponse(
                        text=raw_text,
                        error=f"Failed to parse JSON response: {e}",
                    )
            return LLMResponse(text=raw_text, json=parsed)

        except litellm.exceptions.AuthenticationError as e:
            darwin.error(f"Authentication failed for provider '{prov}': {e}")
            return LLMResponse(error=f"Authentication failed for {prov}: {e}")
        except litellm.exceptions.BadRequestError as e:
            darwin.error(f"Bad request to provider '{prov}': {e}")
            return LLMResponse(error=f"Bad request to {prov}: {e}")
        except RETRYABLE_EXCEPTIONS as e:
            # Transient error that survived all retries
            return LLMResponse(error=f"Persistent transient error for {prov}: {e}")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            import traceback
            darwin.error(
                f"Unexpected error in LLM call to {prov}:\n{traceback.format_exc()}",
            )
            return LLMResponse(error=f"Unexpected error: {e}")

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

        # Guard: resolve provider & build kwargs so ValueError becomes a clean LLMResponse
        try:
            prov, provider_config = self._resolve_provider_config(provider)
            kwargs = self._prepare_kwargs(messages, tier, prov, json_schema, override_kwargs)
        except ValueError as e:
            darwin.error(str(e))
            return LLMResponse(error=str(e))
        except Exception as e:
            import traceback
            darwin.error(f"Unexpected setup error in _acall: {traceback.format_exc()}")
            return LLMResponse(error=f"Setup error: {e}")

        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await acompletion(**kwargs)
                raw_text = response.choices[0].message.content

                parsed = None
                if json_schema is not None:
                    cleaned = _extract_json(raw_text)
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
                    wait = (2 ** attempt) + random.uniform(0, 0.5 * (2 ** attempt))
                    darwin.warning(
                        f"Transient error on {prov} (async attempt {attempt+1}/{self.max_retries+1}): {e}. "
                        f"Retrying in {wait:.1f}s…",
                    )
                    await asyncio.sleep(wait)
                else:
                    darwin.error(f"All async retries exhausted for {prov}: {e}")
            except litellm.exceptions.AuthenticationError as e:
                darwin.error(f"Authentication failed for provider '{prov}': {e}")
                return LLMResponse(error=f"Authentication failed for {prov}: {e}")
            except litellm.exceptions.BadRequestError as e:
                darwin.error(f"Bad request to provider '{prov}': {e}")
                return LLMResponse(error=f"Bad request to {prov}: {e}")
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                import traceback
                darwin.error(
                    f"Unexpected async error in LLM call to {prov}:\n{traceback.format_exc()}",
                )
                return LLMResponse(error=f"Unexpected error: {e}")

        # All retries exhausted
        assert last_exception is not None
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
        return self._call(messages, "fast", provider=provider, json_schema=json_schema, stream=stream, **override_kwargs)

    def call_normal(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **override_kwargs,
    ) -> Union[LLMResponse, Generator]:
        return self._call(messages, "normal", provider=provider, json_schema=json_schema, stream=stream, **override_kwargs)

    def call_smart(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **override_kwargs,
    ) -> Union[LLMResponse, Generator]:
        return self._call(messages, "smart", provider=provider, json_schema=json_schema, stream=stream, **override_kwargs)

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
        return await self._acall(messages, "fast", provider=provider, json_schema=json_schema, stream=stream, **override_kwargs)

    async def acall_normal(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **override_kwargs,
    ) -> Union[LLMResponse, AsyncGenerator]:
        return await self._acall(messages, "normal", provider=provider, json_schema=json_schema, stream=stream, **override_kwargs)

    async def acall_smart(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **override_kwargs,
    ) -> Union[LLMResponse, AsyncGenerator]:
        return await self._acall(messages, "smart", provider=provider, json_schema=json_schema, stream=stream, **override_kwargs)