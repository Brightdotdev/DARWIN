from dataclasses import dataclass, field
from typing import Optional, Dict

# ═══════════════════════════════════════════════════════════════════════════════
# Single source of truth: provider → environment variable mapping
# Consumed by both LLMManager/LLMClient and src.lib.KeyManager
# ═══════════════════════════════════════════════════════════════════════════════
PROVIDER_ENV_MAP: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "cohere": "COHERE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


@dataclass
class TierConfig:
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    extra_body: dict = field(default_factory=dict)


@dataclass
class ProviderConfig:
    fast: TierConfig
    smart: TierConfig
    normal: TierConfig
    base_url: Optional[str] = None
    # Runtime-only: hydrated from keychain/env during initialization.
    # NEVER written to disk by LLMManager._serialize_config().
    api_key: Optional[str] = None


@dataclass
class LLMConfig:
    default_provider: str = "openai"
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)