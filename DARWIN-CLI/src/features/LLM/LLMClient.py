import json
from typing import Optional, List, Dict, Any
import litellm
from litellm import completion
import os

from src.features.LLM.LLMConfig import ProviderConfig, TierConfig
from src.features.LLM.KeyStore import KeyStore
from src.core.Logger.DarwinLogger import darwin, LogType

class LLMClient:
    def __init__(self, providers: Dict[str, ProviderConfig], default_provider: str,
                 keystore: KeyStore):
        self.providers = providers
        self.default_provider = default_provider
        self.keystore = keystore

    def _resolve_provider_config(self, provider: Optional[str]) -> tuple[str, ProviderConfig]:
        prov = provider or self.default_provider
        if prov not in self.providers:
            raise ValueError(f"Provider '{prov}' not configured")
        return prov, self.providers[prov]

    def _set_provider_env(self, provider: str):
        # Only for external providers; Ollama handled via base_url
        if provider != "ollama":
            self.keystore.set_env_for_provider(provider)
            # For Ollama, set OLLAMA_API_BASE from config if present
        if provider == "ollama":
            base_url = self.providers[provider].base_url or "http://localhost:11434"
            os.environ["OLLAMA_API_BASE"] = base_url

    def _build_litellm_model(self, provider: str, model: str) -> str:
        # Ollama uses provider/model format with base_url via env
        # Others: "openai/gpt-4o-mini"
        return f"{provider}/{model}"

    def _call(self, messages: List[Dict[str, str]], tier: str,
              provider: Optional[str] = None, **override_kwargs) -> str:
        prov, provider_config = self._resolve_provider_config(provider)
        tier_config: TierConfig = getattr(provider_config, tier)  # fast or smart

        # Merge kwargs
        kwargs = {
            "model": self._build_litellm_model(prov, tier_config.model),
            "messages": messages,
            "temperature": tier_config.temperature,
            "max_tokens": tier_config.max_tokens,
        }
        kwargs.update(tier_config.extra_body)
        kwargs.update(override_kwargs)

        self._set_provider_env(prov)

        max_retries = len(self.keystore.get_keys(prov)) if prov != "ollama" else 1
        for attempt in range(max_retries):
            try:
                response = completion(**kwargs)
                return response.choices[0].message.content
            except litellm.exceptions.RateLimitError as e:
                if prov == "ollama":
                    raise  # no key rotation for Ollama
                darwin.log(f"Rate limit hit for {prov}, rotating key...", LogType.WARNING)
                # rotate key
                # For simplicity, we track key index? We'll just try the next one via retry.
                # Use a simple retry mechanic: set next key from keystore
                new_index = self.keystore.rotate_active(prov, attempt % len(self.keystore.get_keys(prov)))
                self.keystore.set_env_for_provider(prov, new_index)
                continue
            except Exception as e:
                darwin.log(f"LLM call failed ({prov}, {tier}): {e}", LogType.ERROR)
                raise

        raise RuntimeError(f"All keys exhausted for provider {prov}")

    def call_small(self, messages: List[Dict[str, str]], provider: Optional[str] = None,
                   **override_kwargs) -> str:
        return self._call(messages, "fast", provider=provider, **override_kwargs)

    def call_smart(self, messages: List[Dict[str, str]], provider: Optional[str] = None,
                   **override_kwargs) -> str:
        return self._call(messages, "smart", provider=provider, **override_kwargs)