"""Provider configuration: API keys, env vars, validation.
Completely decoupled from ModelRegistry.
"""

from dataclasses import dataclass
from typing import Dict



@dataclass(frozen=True)
class ProviderSpec:
    name: str
    env_var: str
    

PROVIDER_REGISTRY: Dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        env_var="OPENAI_API_KEY",
    ),

    "anthropic": ProviderSpec(
        name="anthropic",
        env_var="ANTHROPIC_API_KEY",
    ),

    "google": ProviderSpec(
        name="google",
        env_var="GEMINI_API_KEY",
    ),

    "mistral": ProviderSpec(
        name="mistral",
        env_var="MISTRAL_API_KEY",
    ),

    "groq": ProviderSpec(
        name="groq",
        env_var="GROQ_API_KEY",
    ),

    "cohere": ProviderSpec(
        name="cohere",
        env_var="COHERE_API_KEY",
    ),

    "together": ProviderSpec(
        name="together",
        env_var="TOGETHER_API_KEY",
    ),

    "openrouter": ProviderSpec(
        name="openrouter",
        env_var="OPENROUTER_API_KEY",
    ),

    "deepseek": ProviderSpec(
        name="deepseek",
        env_var="DEEPSEEK_API_KEY",
    ),
}