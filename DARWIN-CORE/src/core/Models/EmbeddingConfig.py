# src/core/Models/EmbeddingConfig.py
"""
Model presets for chunking, prompting, and batch sizing.
PURE DATA MODULE — no runtime imports, no hardware detection.
All batch sizing logic lives in ModelBatchSizing.py.

FIXED from original:
  - Fixed reranker model_name (removed dash)
  - Added validate_presets() for self-checking
  - Added get_presets_by_loader() / get_presets_by_category()
  - Added is_local_preset() / is_api_preset() helpers
  - Added get_preset_with_spec() bridge to ModelRegistry
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Dict, List, Tuple, Any


# ═══════════════════════════════════════════════════════════════════════════
# Model preset — complete config for one model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelPreset:
    """Complete configuration for a specific model."""
    # ── Chunking ──
    max_tokens: int
    min_tokens: int
    overlap: int

    # ── Model type ──
    category: Literal["plain_text", "semantic_markdown", "reranker", "vision", "ocr", "nlp"]
    loader: str  # must match a LOADER_TYPES entry

    # ── Embedding-specific ──
    dimensions: int = 0          # 0 for non-embedding models
    needs_prefix: bool = False
    passage_prefix: str = ""
    query_prefix: str = ""
    is_instruction_tuned: bool = False
    instruction_template: Optional[str] = None

    # ── API batching ──
    api_batch_limit: Optional[int] = None

    # ── Metadata ──
    model_name: str = ""  # display name or HF ID


# ═══════════════════════════════════════════════════════════════════════════
# Registry — keyed by canonical model_id (must match ModelRegistry MODEL_SPECS)
# ═══════════════════════════════════════════════════════════════════════════

PRESETS: Dict[str, ModelPreset] = {

    # ── Embedding — ONNX (local) ──
    "intfloat/multilingual-e5-small-onnx": ModelPreset(
        max_tokens=512,
        min_tokens=128,
        dimensions=384,
        overlap=3,
        category="plain_text",
        loader="onnx_embedding",
        needs_prefix=True,
        passage_prefix="passage: ",
        query_prefix="query: ",
        model_name="intfloat/multilingual-e5-small",
    ),

    # ── Embedding — API: OpenAI ──
    "openai/text-embedding-3-small": ModelPreset(
        max_tokens=8191,
        min_tokens=100,
        dimensions=1536,
        overlap=5,
        category="semantic_markdown",
        loader="api_embedding",
        api_batch_limit=2048,
        model_name="openai/text-embedding-3-small",
    ),
    "openai/text-embedding-3-large": ModelPreset(
        max_tokens=8191,
        min_tokens=100,
        dimensions=3072,
        overlap=5,
        category="semantic_markdown",
        loader="api_embedding",
        api_batch_limit=2048,
        model_name="openai/text-embedding-3-large",
    ),
    "openai/text-embedding-ada-002": ModelPreset(
        max_tokens=8191,
        min_tokens=100,
        dimensions=1536,
        overlap=5,
        category="semantic_markdown",
        loader="api_embedding",
        api_batch_limit=2048,
        model_name="openai/text-embedding-ada-002",
    ),

    # ── Embedding — API: OpenRouter ──
    "openrouter/intfloat/multilingual-e5-large": ModelPreset(
        max_tokens=512,
        min_tokens=100,
        dimensions=1024,
        overlap=3,
        category="plain_text",
        loader="api_embedding",
        needs_prefix=True,
        passage_prefix="passage: ",
        query_prefix="query: ",
        api_batch_limit=100,
        model_name="intfloat/multilingual-e5-large",
    ),

    # ── Reranker — ONNX (local) ──
    "cross-encoder/ms-marco-MiniLM-L6-v2-onnx": ModelPreset(
        max_tokens=512,
        min_tokens=10,
        dimensions=0,           # rerankers output scores, not vectors
        overlap=0,
        category="reranker",
        loader="onnx_reranker",
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",  # FIXED: no dash
    ),

 
 
    # ── Vision — ONNX (local) ──
    "HuggingFaceTB/SmolVLM-256M-Instruct-onnx": ModelPreset(
        max_tokens=2048,
        min_tokens=1,
        dimensions=0,
        overlap=0,
        category="vision",
        loader="onnx_vision",
        model_name="SmolVLM-256M-Instruct",
    ),

    # ── Vision — API: OpenAI ──
    "openai/gpt-4o": ModelPreset(
        max_tokens=4096,
        min_tokens=1,
        dimensions=0,
        overlap=0,
        category="vision",
        loader="api_vision",
        api_batch_limit=100,
        model_name="gpt-4o",
    ),
    "openai/gpt-4o-mini": ModelPreset(
        max_tokens=4096,
        min_tokens=1,
        dimensions=0,
        overlap=0,
        category="vision",
        loader="api_vision",
        api_batch_limit=100,
        model_name="gpt-4o-mini",
    ),
    "openai/gpt-4-turbo": ModelPreset(
        max_tokens=4096,
        min_tokens=1,
        dimensions=0,
        overlap=0,
        category="vision",
        loader="api_vision",
        api_batch_limit=100,
        model_name="gpt-4-turbo",
    ),

    # ── Vision — API: Google ──
    "google/gemini-2.0-flash": ModelPreset(
        max_tokens=8192,
        min_tokens=1,
        dimensions=0,
        overlap=0,
        category="vision",
        loader="api_vision",
        api_batch_limit=100,
        model_name="gemini-2.0-flash",
    ),
    "google/gemini-1.5-pro": ModelPreset(
        max_tokens=8192,
        min_tokens=1,
        dimensions=0,
        overlap=0,
        category="vision",
        loader="api_vision",
        api_batch_limit=100,
        model_name="gemini-1.5-pro",
    ),
    "google/gemini-1.5-flash": ModelPreset(
        max_tokens=8192,
        min_tokens=1,
        dimensions=0,
        overlap=0,
        category="vision",
        loader="api_vision",
        api_batch_limit=100,
        model_name="gemini-1.5-flash",
    ),

    # ── Vision — API: Groq ──
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": ModelPreset(
        max_tokens=4096,
        min_tokens=1,
        dimensions=0,
        overlap=0,
        category="vision",
        loader="api_vision",
        api_batch_limit=100,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    ),

    # ── Vision — API: OpenRouter ──
    "openrouter/google/gemma-3-4b-it:free": ModelPreset(
        max_tokens=4096,
        min_tokens=1,
        dimensions=0,
        overlap=0,
        category="vision",
        loader="api_vision",
        api_batch_limit=100,
        model_name="google/gemma-3-4b-it:free",
    ),
    "openrouter/meta-llama/llama-3.2-11b-vision-instruct:free": ModelPreset(
        max_tokens=4096,
        min_tokens=1,
        dimensions=0,
        overlap=0,
        category="vision",
        loader="api_vision",
        api_batch_limit=100,
        model_name="meta-llama/llama-3.2-11b-vision-instruct:free",
    ),
    "openrouter/qwen/qwen2.5-vl-72b-instruct:free": ModelPreset(
        max_tokens=4096,
        min_tokens=1,
        dimensions=0,
        overlap=0,
        category="vision",
        loader="api_vision",
        api_batch_limit=100,
        model_name="qwen/qwen2.5-vl-72b-instruct:free",
    ),

    # ── OCR ──
    "SWHL/RapidOCR": ModelPreset(
        max_tokens=0,
        min_tokens=0,
        dimensions=0,
        overlap=0,
        category="ocr",
        loader="rapidocr",
        model_name="SWHL/RapidOCR",
    ),

    # ── NLP ──
    "en_core_web_md": ModelPreset(
        max_tokens=1000000,
        min_tokens=1,
        dimensions=0,
        overlap=0,
        category="nlp",
        loader="spacy_nlp",
        model_name="en_core_web_md",
    ),
    "en_core_web_lg": ModelPreset(
        max_tokens=1000000,
        min_tokens=1,
        dimensions=0,
        overlap=0,
        category="nlp",
        loader="spacy_nlp",
        model_name="en_core_web_lg",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Fallback preset
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_PRESET = ModelPreset(
    max_tokens=480,
    min_tokens=120,
    dimensions=384,
    overlap=3,
    category="plain_text",
    loader="onnx_embedding",
    model_name="unknown",
)


# ═══════════════════════════════════════════════════════════════════════════
# Core lookup
# ═══════════════════════════════════════════════════════════════════════════

def get_preset(model_key: Optional[str] = None) -> ModelPreset:
    """Returns the ModelPreset for a given model_id, or DEFAULT_PRESET."""
    if model_key is None:
        return DEFAULT_PRESET
    return PRESETS.get(model_key, DEFAULT_PRESET)


# ═══════════════════════════════════════════════════════════════════════════
# NEW: Local vs API detection helpers
# ═══════════════════════════════════════════════════════════════════════════
from src.core.Models.Lib.ModelSpec import LOADER_TYPES

_LOCAL_LOADERS = {t for t in LOADER_TYPES.__args__ if not t.startswith("api_")}
_API_LOADERS = {t for t in LOADER_TYPES.__args__ if t.startswith("api_")}
def is_local_preset(preset: ModelPreset) -> bool:
    """Return True if this preset is for a local (no API key) model."""
    return preset.loader in _LOCAL_LOADERS


def is_api_preset(preset: ModelPreset) -> bool:
    """Return True if this preset is for an API (requires key) model."""
    return preset.loader in _API_LOADERS


def get_local_presets() -> Dict[str, ModelPreset]:
    """Return all local model presets."""
    return {k: v for k, v in PRESETS.items() if is_local_preset(v)}


def get_api_presets() -> Dict[str, ModelPreset]:
    """Return all API model presets."""
    return {k: v for k, v in PRESETS.items() if is_api_preset(v)}


# ═══════════════════════════════════════════════════════════════════════════
# NEW: Filtered lookups
# ═══════════════════════════════════════════════════════════════════════════

def get_presets_by_loader(loader: str) -> Dict[str, ModelPreset]:
    """Return all presets that use a specific loader type."""
    return {k: v for k, v in PRESETS.items() if v.loader == loader}


def get_presets_by_category(
    category: str,
    local_only: bool = False,
    api_only: bool = False,
) -> Dict[str, ModelPreset]:
    """
    Return all presets for a given category string.

    Parameters
    ----------
    category : str
        Category to filter by (e.g. "vision", "reranker", "plain_text").
    local_only : bool
        If True, only return local models.
    api_only : bool
        If True, only return API models.
    """
    results = {k: v for k, v in PRESETS.items() if v.category == category}
    if local_only:
        results = {k: v for k, v in results.items() if is_local_preset(v)}
    if api_only:
        results = {k: v for k, v in results.items() if is_api_preset(v)}
    return results


# ═══════════════════════════════════════════════════════════════════════════
# NEW: Bridge to ModelRegistry
# ═══════════════════════════════════════════════════════════════════════════

def get_preset_with_spec(model_id: str) -> Tuple[ModelPreset, Optional[Any]]:
    """
    Return (ModelPreset, ModelSpec) tuple for a given model_id.

    ModelSpec is imported lazily to avoid circular imports.
    Returns (preset, None) if ModelRegistry is not available.
    """
    preset = get_preset(model_id)
    try:
        from src.core.Models.Lib.ModelRegistry import find_spec
        spec = find_spec(model_id)
        return preset, spec
    except ImportError:
        return preset, None


def get_local_presets_with_specs() -> List[Tuple[ModelPreset, Any]]:
    """Return all local presets paired with their ModelSpecs."""
    try:
        from src.core.Models.Lib.ModelRegistry import find_spec
    except ImportError:
        return [(v, None) for v in get_local_presets().values()]

    results = []
    for model_id, preset in get_local_presets().items():
        spec = find_spec(model_id)
        results.append((preset, spec))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# NEW: Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_presets() -> Dict[str, Any]:
    """
    Self-validate all presets. Returns dict with valid, errors, warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []
    valid_loaders = _LOCAL_LOADERS | _API_LOADERS

    for model_id, preset in PRESETS.items():
        # Loader validation
        if preset.loader not in valid_loaders:
            errors.append(
                f"{model_id}: unknown loader '{preset.loader}'. "
                f"Valid: {sorted(valid_loaders)}"
            )

        # Token range validation
        if preset.max_tokens < preset.min_tokens:
            errors.append(
                f"{model_id}: max_tokens ({preset.max_tokens}) < "
                f"min_tokens ({preset.min_tokens})"
            )

        # Embedding dimension validation
        if preset.loader == "onnx_embedding" and preset.dimensions <= 0:
            errors.append(
                f"{model_id}: embedding model has dimensions={preset.dimensions}"
            )
        if preset.loader == "api_embedding" and preset.dimensions <= 0:
            errors.append(
                f"{model_id}: API embedding model has dimensions={preset.dimensions}"
            )

        # API batch limit validation
        if is_api_preset(preset) and preset.api_batch_limit is None:
            warnings.append(
                f"{model_id}: API model has no api_batch_limit set"
            )

        # Prefix validation for embedding models
        if preset.loader in ("onnx_embedding", "api_embedding") and preset.needs_prefix:
            if not preset.passage_prefix or not preset.query_prefix:
                warnings.append(
                    f"{model_id}: needs_prefix=True but missing "
                    f"passage_prefix/query_prefix"
                )

        # ModelRegistry cross-check (optional)
        try:
            from src.core.Models.Lib.ModelRegistry import find_spec
            spec = find_spec(model_id)
            if spec is None:
                warnings.append(
                    f"{model_id}: not found in ModelRegistry — preset may be orphaned"
                )
            else:
                # Check loader consistency
                if spec.loader != preset.loader:
                    errors.append(
                        f"{model_id}: loader mismatch — "
                        f"ModelRegistry='{spec.loader}', "
                        f"EmbeddingConfig='{preset.loader}'"
                    )
        except ImportError:
            pass  # ModelRegistry not available, skip cross-check

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "total_presets": len(PRESETS),
        "local_count": len(get_local_presets()),
        "api_count": len(get_api_presets()),
    }