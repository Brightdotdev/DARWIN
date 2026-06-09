# src/core/Models/ModelRegistry.py

from dataclasses import dataclass, field
from typing import Literal, Optional, Set
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
    tier: Literal["minnow", "dolphin", "orca", "whale"]
    ram_gb: int
    disk_mb: int
    loader: Literal[
        "sentence_transformers", "cross_encoder", "transformers", "spacy",
        "embedding_hf", "embedding_st", "onnx_embedding", "onnx_reranker",
        "api_embedding", "api_reranker", "api_vision"
    ]
    required: bool = False          # CHANGED: per-model required is now opt-in fallback only
    experimental: bool = False
    notes: str = ""
    requires_plugin: Optional[str] = None
    onnx_available: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# REQUIRED CATEGORIES — the system must have *a* model for each of these
# ═══════════════════════════════════════════════════════════════════════════
REQUIRED_CATEGORIES: Set[str] = {"embedding", "reranker", "tokenizer"}

# Default preferred model per category (used when no user preference exists)
DEFAULT_PREFERRED: dict[str, str] = {
    "embedding": "BAAI/bge-m3",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "tokenizer": "en_core_web_md",
}

REGISTRY: dict[str, ModelSpec] = {
    # ═══════════════════════════════════════════════════════════════════════
    # Embedding
    # ═══════════════════════════════════════════════════════════════════════
    "intfloat/e5-small-v2": ModelSpec(
        model_id="intfloat/e5-small-v2",
        category="embedding",
        tier="minnow",
        ram_gb=8,
        disk_mb=130,
        loader="embedding_hf",
        notes="384-dim, English-only. Fast, low memory.",
    ),
    "BAAI/bge-m3": ModelSpec(
        model_id="BAAI/bge-m3",
        category="embedding",
        tier="dolphin",
        ram_gb=16,
        disk_mb=2200,
        loader="embedding_st",
        notes="1024-dim, multilingual, understands markdown structure. 16GB RAM recommended.",
    ),
    "GritLM/GritLM-7B": ModelSpec(
        model_id="GritLM/GritLM-7B",
        category="embedding",
        tier="whale",
        ram_gb=32,
        disk_mb=14000,
        loader="sentence_transformers",
        experimental=True,
        requires_plugin="torch",
        notes="7B LLM repurposed as embedder. 4096-dim. GPU strongly recommended.",
    ),

    # ── ONNX Embedding Models ──
    "BAAI/bge-m3-onnx": ModelSpec(
        model_id="BAAI/bge-m3-onnx",
        category="embedding",
        tier="dolphin",
        ram_gb=4,
        disk_mb=400,
        loader="onnx_embedding",
        onnx_available=True,
        notes="BGE-M3 via ONNX Runtime. 1024-dim, multilingual. ~4x faster than PyTorch on CPU.",
    ),
    "intfloat/e5-small-v2-onnx": ModelSpec(
        model_id="intfloat/e5-small-v2-onnx",
        category="embedding",
        tier="minnow",
        ram_gb=2,
        disk_mb=100,
        loader="onnx_embedding",
        onnx_available=True,
        notes="E5-small via ONNX Runtime. 384-dim, English-only. Extremely fast.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # Reranker
    # ═══════════════════════════════════════════════════════════════════════
    "cross-encoder/ms-marco-MiniLM-L-12-v2": ModelSpec(
        model_id="cross-encoder/ms-marco-MiniLM-L-12-v2",
        category="reranker",
        tier="dolphin",
        ram_gb=2,
        disk_mb=33,
        loader="cross_encoder",
        notes="33MB, ~50ms CPU. Perfect for everyone. Always installed.",
    ),
    "cross-encoder/ms-marco-MiniLM-L-6-v2-onnx": ModelSpec(
        model_id="cross-encoder/ms-marco-MiniLM-L-6-v2-onnx",
        category="reranker",
        tier="minnow",
        ram_gb=1,
        disk_mb=25,
        loader="onnx_reranker",
        onnx_available=True,
        notes="MiniLM cross-encoder via ONNX. Faster than PyTorch on CPU.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # Vision
    # ═══════════════════════════════════════════════════════════════════════
    "HuggingFaceTB/SmolVLM-256M-Instruct": ModelSpec(
        model_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        category="vision",
        tier="minnow",
        ram_gb=4,
        disk_mb=500,
        loader="transformers",
        requires_plugin="torch",
        notes="256M params. Fast captions, low memory.",
    ),
    "vikhyatk/moondream2": ModelSpec(
        model_id="vikhyatk/moondream2",
        category="vision",
        tier="dolphin",
        ram_gb=16,
        disk_mb=1400,
        loader="transformers",
        requires_plugin="torch",
        notes="1.8B params. Better caption quality, understands charts/diagrams.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # Tokenizer / NLP
    # ═══════════════════════════════════════════════════════════════════════
    "en_core_web_md": ModelSpec(
        model_id="en_core_web_md",
        category="tokenizer",
        tier="minnow",
        ram_gb=1,
        disk_mb=45,
        loader="spacy",
        notes="Medium English model. Optimized for concept extraction. 45MB.",
    ),
    "en_core_web_lg": ModelSpec(
        model_id="en_core_web_lg",
        category="tokenizer",
        tier="dolphin",
        ram_gb=2,
        disk_mb=560,
        loader="spacy",
        notes="Large English model. Better accuracy, word vectors. 560MB.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # API-based Embedding
    # ═══════════════════════════════════════════════════════════════════════
    "openai/text-embedding-3-small": ModelSpec(
        model_id="openai/text-embedding-3-small",
        category="embedding",
        tier="dolphin",
        ram_gb=0,
        disk_mb=0,
        loader="api_embedding",
        notes="OpenAI embedding API. 1536-dim. Requires OPENAI_API_KEY.",
    ),
    "cohere/embed-english-v3.0": ModelSpec(
        model_id="cohere/embed-english-v3.0",
        category="embedding",
        tier="dolphin",
        ram_gb=0,
        disk_mb=0,
        loader="api_embedding",
        notes="Cohere embedding API. 1024-dim. Requires COHERE_API_KEY.",
    ),
    "voyage/voyage-3-lite": ModelSpec(
        model_id="voyage/voyage-3-lite",
        category="embedding",
        tier="dolphin",
        ram_gb=0,
        disk_mb=0,
        loader="api_embedding",
        notes="Voyage AI embeddings. 512-dim. Requires VOYAGE_API_KEY.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # API-based Reranker
    # ═══════════════════════════════════════════════════════════════════════
    "cohere/rerank-english-v3.0": ModelSpec(
        model_id="cohere/rerank-english-v3.0",
        category="reranker",
        tier="dolphin",
        ram_gb=0,
        disk_mb=0,
        loader="api_reranker",
        notes="Cohere rerank API. Requires COHERE_API_KEY.",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # API-based Vision
    # ═══════════════════════════════════════════════════════════════════════
    "openai/gpt-4o": ModelSpec(
        model_id="openai/gpt-4o",
        category="vision",
        tier="orca",
        ram_gb=0,
        disk_mb=0,
        loader="api_vision",
        notes="OpenAI GPT-4o vision. Requires OPENAI_API_KEY.",
    ),
    "gemini/gemini-2.0-flash": ModelSpec(
        model_id="gemini/gemini-2.0-flash",
        category="vision",
        tier="dolphin",
        ram_gb=0,
        disk_mb=0,
        loader="api_vision",
        notes="Google Gemini Flash vision. Requires GEMINI_API_KEY.",
    ),
    "openrouter/openai/gpt-4o": ModelSpec(
        model_id="openrouter/openai/gpt-4o",
        category="vision",
        tier="orca",
        ram_gb=0,
        disk_mb=0,
        loader="api_vision",
        notes="OpenRouter GPT-4o vision. Requires OPENROUTER_API_KEY.",
    ),
    "openrouter/meta-llama/llama-4-maverick": ModelSpec(
        model_id="openrouter/meta-llama/llama-4-maverick",
        category="vision",
        tier="orca",
        ram_gb=0,
        disk_mb=0,
        loader="api_vision",
        notes="OpenRouter Llama 4 Maverick vision. Requires OPENROUTER_API_KEY.",
    ),
    "openrouter/google/gemini-2.0-flash-001": ModelSpec(
        model_id="openrouter/google/gemini-2.0-flash-001",
        category="vision",
        tier="dolphin",
        ram_gb=0,
        disk_mb=0,
        loader="api_vision",
        notes="OpenRouter Gemini Flash vision. Requires OPENROUTER_API_KEY.",
    ),
}


TIER_LABELS = {
    "minnow": "⚡ Fast",
    "dolphin": "⭐ Recommended",
    "orca": "💪 Powerful",
    "whale": "🔥 Maximum",
}


def get_spec(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model key '{key}'. Available: {list(REGISTRY)}")
    return REGISTRY[key]


def required_models() -> dict[str, ModelSpec]:
    """Return models that are individually marked required (legacy fallback)."""
    return {k: v for k, v in REGISTRY.items() if v.required}


def optional_models() -> dict[str, ModelSpec]:
    return {k: v for k, v in REGISTRY.items() if not v.required}


def by_category(category: str) -> dict[str, ModelSpec]:
    return {k: v for k, v in REGISTRY.items() if v.category == category}


_VALID_LOADERS = {
    "sentence_transformers", "cross_encoder", "transformers", "spacy",
    "embedding_hf", "embedding_st", "onnx_embedding", "onnx_reranker",
    "api_embedding", "api_reranker", "api_vision",
}


def _validate_registry() -> None:
    for key, spec in REGISTRY.items():
        if spec.loader not in _VALID_LOADERS:
            raise ValueError(
                f"Registry entry '{key}' has unknown loader '{spec.loader}'. "
                f"Valid loaders: {sorted(_VALID_LOADERS)}"
            )


_validate_registry()