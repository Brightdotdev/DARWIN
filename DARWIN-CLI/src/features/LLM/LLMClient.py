import os
from typing import Optional, List, Dict, Any

import litellm
from litellm import completion
from litellm.caching import Cache

from src.features.LLM.LLMConfig import ProviderConfig, TierConfig
from src.features.LLM.KeyStore import KeyStore
from src.core.Logger.DarwinLogger import darwin, LogType


class LLMClient:
    def __init__(
        self,
        providers: Dict[str, ProviderConfig],
        default_provider: str,
        keystore: KeyStore,
        enable_cache: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        :param providers: dict mapping provider names to ProviderConfig
        :param default_provider: fallback provider when none is given
        :param keystore: KeyStore instance for secure API key retrieval
        :param enable_cache: if True, enable LiteLLM disk caching (speeds repeated prompts)
        :param cache_dir: directory for cache storage
        """
        self.providers = providers
        self.default_provider = default_provider
        self.keystore = keystore

        # Global LiteLLM optimisations ------------------------------------------------
        litellm.enable_json_schema_validation = True   # auto‑validate structured outputs
        litellm.suppress_debug_info = True             # cleaner logs, better performance

        # Optional caching – can dramatically speed up repeated identical calls
        if enable_cache:
            litellm.cache = Cache(disk_cache_dir=cache_dir)

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def _resolve_provider_config(
        self, provider: Optional[str]
    ) -> tuple[str, ProviderConfig]:
        prov = provider or self.default_provider
        if prov not in self.providers:
            raise ValueError(f"Provider '{prov}' not configured")
        return prov, self.providers[prov]

    def _set_env_for_provider(self, provider: str):
        """Set necessary environment variables (API keys, base URLs)."""
        if provider == "ollama":
            # Ollama needs only OLLAMA_API_BASE; no API key
            base_url = self.providers[provider].base_url or "http://localhost:11434"
            os.environ["OLLAMA_API_BASE"] = base_url
        else:
            # Let the KeyStore rotate to the correct key and set the env var
            self.keystore.set_env_for_provider(provider)

    def _build_litellm_model(self, provider: str, model: str) -> str:
        """Construct the model string expected by LiteLLM (e.g. 'openai/gpt-4o-mini')."""
        return f"{provider}/{model}"

    # ----------------------------------------------------------------------
    # Core call logic
    # ----------------------------------------------------------------------
    def _call(
        self,
        messages: List[Dict[str, str]],
        tier: str,                    # "fast" or "smart" from TierConfig
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,   # new structured output support
        **override_kwargs,
    ) -> str:
        prov, provider_config = self._resolve_provider_config(provider)
        tier_config: TierConfig = getattr(provider_config, tier)  # fast or smart

        # Base parameters from configuration ------------------------------------------
        kwargs = {
            "model": self._build_litellm_model(prov, tier_config.model),
            "messages": messages,
            "temperature": tier_config.temperature,
            "max_tokens": tier_config.max_tokens,
        }

        # **Speed optimisation for "fast" tier**:
        #   - lock temperature to 0.0 (deterministic, cheapest)
        #   - clamp max_tokens to 256 (or smaller) unless explicitly overridden
        #   - disable "thinking" for reasoning models (Anthropic, etc.)
        if tier == "fast":
            if "temperature" not in override_kwargs:
                kwargs["temperature"] = 0.0
            if "max_tokens" not in override_kwargs:
                kwargs["max_tokens"] = min(tier_config.max_tokens, 256)
            # Disable extended thinking (provider-specific, stored in extra_body)
            thinking_disabled = {"type": "disabled"}
            # We'll merge it later with the rest of extra_body

        # Merge provider‑specific extra_body from config ------------------------------
        extra_body = dict(tier_config.extra_body) if tier_config.extra_body else {}

        # If a JSON schema was requested, build the response_format for LiteLLM
        if json_schema is not None:
            extra_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        # Apply the "thinking disabled" for fast tier **after** everything else
        if tier == "fast":
            extra_body.setdefault("thinking", thinking_disabled)

        kwargs["extra_body"] = extra_body

        # Override with any extra user‑provided arguments (e.g. temperature)
        kwargs.update(override_kwargs)

        # Set environment for the chosen provider
        self._set_env_for_provider(prov)

        # Retry loop with automatic key rotation on rate limits ------------------------
        max_retries = (
            len(self.keystore.get_keys(prov)) if prov != "ollama" and self.keystore.get_keys(prov) else 1
        )
        for attempt in range(max_retries):
            try:
                response = completion(**kwargs)
                return response.choices[0].message.content
            except litellm.exceptions.RateLimitError as e:
                if prov == "ollama":
                    raise   # no fallback keys for local Ollama
                darwin.log(
                    f"Rate limit hit for {prov}, rotating key (attempt {attempt+1}/{max_retries})",
                    LogType.WARNING,
                )
                # Rotate to the next API key and update environment
                self.keystore.rotate_active(prov)
                self.keystore.set_env_for_provider(prov)
            except Exception as e:
                darwin.log(
                    f"LLM call failed ({prov}, {tier}): {e}",
                    LogType.ERROR,
                )
                raise

        raise RuntimeError(f"All keys exhausted for provider {prov}")

    # ----------------------------------------------------------------------
    # Public interface – call_small / call_smart
    # ----------------------------------------------------------------------
    def call_small(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **override_kwargs,
    ) -> str:
        """
        Fast, cheap LLM call.
        - Temperature forced to 0.0 (unless overridden)
        - max_tokens capped at 256 (config can be lower)
        - Extended thinking disabled
        - Accepts an optional JSON schema for structured output
        """
        return self._call(
            messages,
            "fast",
            provider=provider,
            json_schema=json_schema,
            **override_kwargs,
        )

    def call_smart(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **override_kwargs,
    ) -> str:
        """
        Full‑power LLM call (higher quality, slower).
        - Uses the configured "smart" tier settings
        - Accepts an optional JSON schema for structured output
        """
        return self._call(
            messages,
            "smart",
            provider=provider,
            json_schema=json_schema,
            **override_kwargs,
        )