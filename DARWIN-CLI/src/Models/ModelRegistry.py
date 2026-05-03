# src/core/Models/ModelRegistry.py

from dataclasses import dataclass
from typing import Literal
from enum import Enum


class ModelCategory(Enum):
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    VISION = "vision"
    TOKENIZER = "tokenizer"


@dataclass
class ModelSpec:
    model_id: str
    category: str
    tier: Literal["light", "medium", "large", "overkill"]
    ram_gb: int
    disk_mb: int
    loader: Literal["sentence_transformers", "cross_encoder", "transformers", "spacy", "none", "embedding_hf", "embedding_st"]
    required: bool = True
    experimental: bool = False
    notes: str = ""

REGISTRY: dict[str, ModelSpec] = {
    # ═══════════════════════════════════════════════════════════════════════
    # Embedding (required)
    # ═══════════════════════════════════════════════════════════════════════
    "intfloat/e5-small-v2": ModelSpec(
        model_id="intfloat/e5-small-v2",
        category="embedding",
        tier="light",
        ram_gb=8,
        disk_mb=130,
        loader="embedding_hf",
        required=True,
        notes="384-dim, English-only. Fast, low memory.",
    ),
    "BAAI/bge-m3": ModelSpec(
        model_id="BAAI/bge-m3",
        category="embedding",
        tier="medium",
        ram_gb=16,
        disk_mb=2200,
        loader="embedding_st",
        required=True,
        notes="1024-dim, multilingual, understands markdown structure. Recommended.",
    ),
    "GritLM/GritLM-7B": ModelSpec(
        model_id="GritLM/GritLM-7B",
        category="embedding",
        tier="overkill",
        ram_gb=32,
        disk_mb=14000,
        loader="sentence_transformers",
        required=False,
        experimental=True,
        notes="7B LLM repurposed as embedder. 4096-dim. GPU strongly recommended; CPU inference will be prohibitively slow (minutes per document)."
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # Reranker (required)
    # ═══════════════════════════════════════════════════════════════════════
    "cross-encoder/ms-marco-MiniLM-L-12-v2": ModelSpec(
        model_id="cross-encoder/ms-marco-MiniLM-L-12-v2",
        category="reranker",
        tier="medium",
        ram_gb=2,
        disk_mb=33,
        loader="cross_encoder",
        required=True,
        notes="33MB, ~50ms CPU. Perfect for everyone. Always installed.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # Vision (optional)
    # ═══════════════════════════════════════════════════════════════════════
    "HuggingFaceTB/SmolVLM-256M-Instruct": ModelSpec(
        model_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        category="vision",
        tier="light",
        ram_gb=4,
        disk_mb=500,
        loader="transformers",
        required=False,
        notes="256M params. Fast captions, low memory. Good enough for most docs.",
    ),
    "vikhyatk/moondream2": ModelSpec(
        model_id="vikhyatk/moondream2",
        category="vision",
        tier="medium",
        ram_gb=16,
        disk_mb=1400,
        loader="transformers",
        required=False,
        notes="1.8B params. Better caption quality, understands charts/diagrams.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # Tokenizer / NLP (optional)
    # ═══════════════════════════════════════════════════════════════════════
    "en_core_web_md": ModelSpec(
        model_id="en_core_web_md",
        category="tokenizer",
        tier="light",
        ram_gb=1,
        disk_mb=45,
        loader="spacy",
        required=False,
        notes="Medium English model. Optimized for concept extraction (NER, parsing, lemmatization). 45MB.",
    ),
    "en_core_web_lg": ModelSpec(
        model_id="en_core_web_lg",
        category="tokenizer",
        tier="medium",
        ram_gb=2,
        disk_mb=560,
        loader="spacy",
        required=False,
        notes="Large English model. Better accuracy, word vectors. 560MB.",
    ),





# api based models


    "openai/text-embedding-3-small": ModelSpec(
        model_id="openai/text-embedding-3-small",
        category="embedding",
        tier="medium",
        ram_gb=0,
        disk_mb=0,
        loader="api_embedding",
        required=False,
        notes="OpenAI embedding API. 1536-dim. $0.02/1M tokens. Requires OPENAI_API_KEY.",
    ),
    "cohere/embed-english-v3.0": ModelSpec(
        model_id="cohere/embed-english-v3.0",
        category="embedding",
        tier="medium",
        ram_gb=0,
        disk_mb=0,
        loader="api_embedding",
        required=False,
        notes="Cohere embedding API. 1024-dim. Requires COHERE_API_KEY.",
    ),
    "voyage/voyage-3-lite": ModelSpec(
        model_id="voyage/voyage-3-lite",
        category="embedding",
        tier="medium",
        ram_gb=0,
        disk_mb=0,
        loader="api_embedding",
        required=False,
        notes="Voyage AI embeddings. 1024-dim. Requires VOYAGE_API_KEY.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # API-based Reranker (zero local compute)
    # ═══════════════════════════════════════════════════════════════════════
    "cohere/rerank-english-v3.0": ModelSpec(
        model_id="cohere/rerank-english-v3.0",
        category="reranker",
        tier="medium",
        ram_gb=0,
        disk_mb=0,
        loader="api_reranker",
        required=False,
        notes="Cohere rerank API. Requires COHERE_API_KEY.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # API-based Vision (zero local compute)
    # ═══════════════════════════════════════════════════════════════════════
    "openai/gpt-4o": ModelSpec(
        model_id="openai/gpt-4o",
        category="vision",
        tier="large",
        ram_gb=0,
        disk_mb=0,
        loader="api_vision",
        required=False,
        notes="OpenAI GPT-4o vision. Image captioning via API. Requires OPENAI_API_KEY.",
    ),
    "gemini/gemini-2.0-flash": ModelSpec(
        model_id="gemini/gemini-2.0-flash",
        category="vision",
        tier="medium",
        ram_gb=0,
        disk_mb=0,
        loader="api_vision",
        required=False,
        notes="Google Gemini Flash vision. Requires GEMINI_API_KEY.",
    ),

     "openrouter/openai/gpt-4o": ModelSpec(
        model_id="openrouter/openai/gpt-4o",
        category="vision",
        tier="large",
        ram_gb=0,
        disk_mb=0,
        loader="api_vision",
        required=False,
        notes="OpenRouter GPT-4o vision. Requires OPENROUTER_API_KEY.",
    ),
    "openrouter/meta-llama/llama-4-maverick": ModelSpec(
        model_id="openrouter/meta-llama/llama-4-maverick",
        category="vision",
        tier="large",
        ram_gb=0,
        disk_mb=0,
        loader="api_vision",
        required=False,
        notes="OpenRouter Llama 4 Maverick vision. Requires OPENROUTER_API_KEY.",
    ),
    "openrouter/google/gemini-2.0-flash-001": ModelSpec(
        model_id="openrouter/google/gemini-2.0-flash-001",
        category="vision",
        tier="medium",
        ram_gb=0,
        disk_mb=0,
        loader="api_vision",
        required=False,
        notes="OpenRouter Gemini Flash vision. Requires OPENROUTER_API_KEY.",
    ),
}


TIER_LABELS = {
    "light": "⚡ Fast",
    "medium": "⭐ Recommended",
    "large": "💪 Powerful",
    "overkill": "🔥 Maximum",
}


def get_spec(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key '{key}'. Available: {list(REGISTRY)}")
    return REGISTRY[key]


def required_models() -> dict[str, ModelSpec]:
    return {k: v for k, v in REGISTRY.items() if v.required}


def optional_models() -> dict[str, ModelSpec]:
    return {k: v for k, v in REGISTRY.items() if not v.required}


def by_category(category: str) -> dict[str, ModelSpec]:
    return {k: v for k, v in REGISTRY.items() if v.category == category}


_VALID_LOADERS = {
    "sentence_transformers", "cross_encoder", "transformers",
    "spacy", "none", "embedding_hf", "embedding_st",
    "api_embedding", "api_reranker", "api_vision",  # ← new
}


def _validate_registry() -> None:
    for key, spec in REGISTRY.items():
        if spec.loader not in _VALID_LOADERS:
            raise ValueError(
                f"Registry entry '{key}' has unknown loader '{spec.loader}'. "
                f"Valid loaders: {sorted(_VALID_LOADERS)}"
            )


_validate_registry()