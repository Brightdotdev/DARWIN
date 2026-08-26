# src/core/Models/Lib/ModelSpec.py
"""
Single source of truth for all model specifications.
Defines ModelSpec, ModelCategory, ModelTier, LOADER_TYPES, ProviderSpec, PROVIDER_REGISTRY.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Literal
from enum import Enum, auto

from core.Logger.DarwinLogger import darwin


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class ModelCategory(Enum):
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    VISION = "vision"
    OCR = "ocr"
    NLP = "nlp"


class ModelTier(Enum):
    DOLPHIN = auto()
    ORCA = auto()
    WHALE = auto()


# ═══════════════════════════════════════════════════════════════════════════
# Loader type — single source of truth
# ═══════════════════════════════════════════════════════════════════════════

# Allowed loader strings (used for validation and filtering)
ALLOWED_LOADERS = (
    "onnx_embedding",
    "onnx_reranker",
    "onnx_vision",
    "onnx_ocr",
    "api_embedding",
    "api_vision",
    "rapidocr",
    "spacy_nlp",
)

# Type alias for type annotations
LOADER_TYPES = Literal[
    "onnx_embedding",
    "onnx_reranker",
    "onnx_vision",
    "onnx_ocr",
    "api_embedding",
    "api_vision",
    "rapidocr",
    "spacy_nlp",
]


# ═══════════════════════════════════════════════════════════════════════════
# ModelSpec — canonical model definition
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    category: ModelCategory
    tier: ModelTier
    loader: str                            # must be in ALLOWED_LOADERS
    ram_gb: int = 0
    disk_mb: int = 0
    required: bool = False
    experimental: bool = False
    notes: str = ""
    requires_provider: Optional[str] = None
    onnx_available: bool = False
    repo_id: Optional[str] = None

    def __post_init__(self):
        if self.loader not in ALLOWED_LOADERS:
            darwin.error(
                f"ModelSpec '{self.model_id}': loader '{self.loader}' not in ALLOWED_LOADERS. "
                f"Valid: {sorted(ALLOWED_LOADERS)}"
            )
            return None

        # Validate provider if specified
        if self.requires_provider and self.requires_provider not in PROVIDER_REGISTRY:
            darwin.error(
                f"ModelSpec '{self.model_id}': provider '{self.requires_provider}' "
                f"not in PROVIDER_REGISTRY"
            )
            return None

        # ONNX flag consistency
        is_onnx = self.loader.startswith("onnx_") or self.loader == "rapidocr"
        if is_onnx and not self.onnx_available:
            darwin.error(
                f"ModelSpec '{self.model_id}': ONNX loader '{self.loader}' "
                f"requires onnx_available=True"
            )
            return None

    @property
    def is_api(self) -> bool:
        return self.loader.startswith("api_")

    @property
    def is_local(self) -> bool:
        return not self.is_api


# ═══════════════════════════════════════════════════════════════════════════
# Provider Registry
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProviderSpec:
    name: str
    env_var: str
    base_url: Optional[str] = None
    default_model: Optional[str] = None


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
        base_url="https://openrouter.ai/api/v1",
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        env_var="DEEPSEEK_API_KEY",
    ),
}