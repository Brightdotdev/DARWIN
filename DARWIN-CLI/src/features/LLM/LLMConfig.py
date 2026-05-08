from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class TierConfig:
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    extra_body: dict = field(default_factory=dict)  # e.g., {"thinking": False}

@dataclass
class ProviderConfig:
    fast: TierConfig
    smart: TierConfig
    base_url: Optional[str] = None  # for Ollama

@dataclass
class LLMConfig:
    default_provider: str = "openai"
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)