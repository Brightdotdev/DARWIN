# src/core/Models/ModelRegistry.py

from dataclasses import dataclass
from typing import Literal
from enum import Enum


class ModelCategory(Enum):
    EMBEDDING = "embedding"
    RERANKER  = "reranker"
    VISION    = "vision"
    TOKENIZER = "tokenizer"  # spaCy — optional NLP


@dataclass
class ModelSpec:
    model_id:     str
    category:     str
    tier:         Literal["light", "medium", "large", "overkill"]
    ram_gb:       int              # minimum system RAM to run comfortably
    disk_mb:      int              # actual download size in MB
    loader:       Literal["sentence_transformers", "cross_encoder", "transformers", "spacy", "none", "embedding_hf", "embedding_st"]
    required:     bool = True      # core pipeline needs this? False = nice-to-have
    experimental: bool = False
    notes:        str  = ""


REGISTRY: dict[str, ModelSpec] = {
    # ═══════════════════════════════════════════════════════════════════════
    # Embedding (required — search doesn't work without one)
    # ═══════════════════════════════════════════════════════════════════════
    "intfloat/e5-small-v2": ModelSpec(
        model_id  = "intfloat/e5-small-v2",
        category  = "embedding",
        tier      = "light",
        ram_gb    = 8,
        disk_mb   = 130,
        loader    = "embedding_hf",
        required  = True,
        notes     = "384-dim, English-only. Fast, low memory.",
    ),
    "BAAI/bge-m3": ModelSpec(
        model_id  = "BAAI/bge-m3",
        category  = "embedding",
        tier      = "medium",
        ram_gb    = 16,
        disk_mb   = 2200,
        loader    = "embedding_st",
        required  = True,
        notes     = "1024-dim, multilingual, understands markdown structure. Recommended.",
    ),
    "GritLM/GritLM-7B": ModelSpec(
        model_id  = "GritLM/GritLM-7B",
        category  = "embedding",
        tier      = "overkill",
        ram_gb    = 32,
        disk_mb   = 14000,
        loader    = "sentence_transformers",
        required  = False,
        experimental = True,
        notes     = "7B LLM repurposed as embedder. 4096-dim. Needs 32GB+ RAM or GPU.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # Reranker (required — one is always installed, tiny)
    # ═══════════════════════════════════════════════════════════════════════
    "cross-encoder/ms-marco-MiniLM-L-12-v2": ModelSpec(
        model_id  = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        category  = "reranker",
        tier      = "medium",
        ram_gb    = 2,
        disk_mb   = 33,
        loader    = "cross_encoder",
        required  = True,
        notes     = "33MB, ~50ms CPU. Perfect for everyone. Always installed.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # Vision (optional — only needed with --images --caption)
    # ═══════════════════════════════════════════════════════════════════════
    "HuggingFaceTB/SmolVLM-256M-Instruct": ModelSpec(
        model_id  = "HuggingFaceTB/SmolVLM-256M-Instruct",
        category  = "vision",
        tier      = "light",
        ram_gb    = 4,
        disk_mb   = 500,
        loader    = "transformers",
        required  = True,
        notes     = "256M params. Fast captions, low memory. Good enough for most docs.",
    ),
    "vikhyatk/moondream2": ModelSpec(
        model_id  = "vikhyatk/moondream2",
        category  = "vision",
        tier      = "medium",
        ram_gb    = 16,
        disk_mb   = 1400,
        loader    = "transformers",
        required  = True,
        notes     = "1.8B params. Better caption quality, understands charts/diagrams.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # Tokenizer / NLP (optional — nice-to-have for better sentence splitting)
    # ═══════════════════════════════════════════════════════════════════════
    "en_core_web_md": ModelSpec(
        model_id  = "en_core_web_md",
        category  = "tokenizer",
        tier      = "light",
        ram_gb    = 1,
        disk_mb   = 45,
        loader    = "spacy",
        required  = True,
        notes     = "Medium English model. Optimized for concept extraction (NER, parsing, lemmatization). 45MB.",
    ),
    "en_core_web_lg": ModelSpec(
        model_id  = "en_core_web_lg",
        category  = "tokenizer",
        tier      = "medium",
        ram_gb    = 2,
        disk_mb   = 560,
        loader    = "spacy",
        required  = False,
        notes     = "Large English model. Better accuracy, word vectors. 560MB.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Tier labels for the UI
# ═══════════════════════════════════════════════════════════════════════════

TIER_LABELS = {
    "light":    "⚡ Fast",
    "medium":   "⭐ Recommended",
    "large":    "💪 Powerful",
    "overkill": "🔥 Maximum",
}


# ═══════════════════════════════════════════════════════════════════════════
# Lookup helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_spec(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key '{key}'. Available: {list(REGISTRY)}")
    return REGISTRY[key]


def required_models() -> dict[str, ModelSpec]:
    """Models that MUST be installed for the app to function."""
    return {k: v for k, v in REGISTRY.items() if v.required}


def optional_models() -> dict[str, ModelSpec]:
    """Nice-to-have models. Only downloaded when the user opts in or needs them."""
    return {k: v for k, v in REGISTRY.items() if not v.required}


def by_category(category: str) -> dict[str, ModelSpec]:
    return {k: v for k, v in REGISTRY.items() if v.category == category}


# ═══════════════════════════════════════════════════════════════════════════
# Validation — catch registry/loader mismatches at import time
# ═══════════════════════════════════════════════════════════════════════════

_VALID_LOADERS = {
    "sentence_transformers", "cross_encoder", "transformers",
    "spacy", "none", "embedding_hf", "embedding_st",
}

def _validate_registry() -> None:
    for key, spec in REGISTRY.items():
        if spec.loader not in _VALID_LOADERS:
            raise ValueError(
                f"Registry entry '{key}' has unknown loader '{spec.loader}'. "
                f"Valid loaders: {sorted(_VALID_LOADERS)}"
            )

_validate_registry()
