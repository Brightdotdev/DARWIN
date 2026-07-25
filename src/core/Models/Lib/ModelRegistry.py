# src/core/Models/Lib/ModelRegistry.py
"""
Master registry of all model specs (local + API).
Uses ModelSpec from ModelSpec.py as the single source of truth.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import asdict
import os

from src.core.Models.Lib.ModelSpec import (
    ModelSpec, ModelCategory, ModelTier,
    PROVIDER_REGISTRY,
)
from src.core.Logger.DarwinLogger import darwin




# ═══════════════════════════════════════════════════════════════════════════
# Master list of all model specs
# ═══════════════════════════════════════════════════════════════════════════

MODEL_SPECS: List[ModelSpec] = [
    # ── EMBEDDING — local ONNX ──
    ModelSpec(
        model_id="intfloat/multilingual-e5-small-onnx",
        category=ModelCategory.EMBEDDING,
        tier=ModelTier.DOLPHIN,
        loader="onnx_embedding",
        ram_gb=2, disk_mb=300,
        onnx_available=True,
        repo_id="intfloat/multilingual-e5-small",
        notes="Base Embedding Model (ONNX)",
    ),

    # ── EMBEDDING — API: OpenAI ──
    ModelSpec(
        model_id="openai/text-embedding-3-small",
        category=ModelCategory.EMBEDDING,
        tier=ModelTier.DOLPHIN,
        loader="api_embedding",
        requires_provider="openai",
        notes="1536 dimensions",
    ),
    ModelSpec(
        model_id="openai/text-embedding-3-large",
        category=ModelCategory.EMBEDDING,
        tier=ModelTier.ORCA,
        loader="api_embedding",
        requires_provider="openai",
        notes="3072 dimensions",
    ),
    ModelSpec(
        model_id="openai/text-embedding-ada-002",
        category=ModelCategory.EMBEDDING,
        tier=ModelTier.DOLPHIN,
        loader="api_embedding",
        requires_provider="openai",
        notes="Legacy Ada, 1536 dims",
    ),

    # ── EMBEDDING — API: OpenRouter ──
    ModelSpec(
        model_id="openrouter/intfloat/multilingual-e5-large",
        category=ModelCategory.EMBEDDING,
        tier=ModelTier.DOLPHIN,
        loader="api_embedding",
        requires_provider="openrouter",
        notes="E5-large via OpenRouter",
    ),

    # ── RERANKER — local ONNX ──
    ModelSpec(
        model_id="cross-encoder/ms-marco-MiniLM-L6-v2-onnx",
        category=ModelCategory.RERANKER,
        tier=ModelTier.DOLPHIN,
        loader="onnx_reranker",
        ram_gb=1, disk_mb=25,
        onnx_available=True,
        repo_id="cross-encoder/ms-marco-MiniLM-L6-v2",
        notes="MiniLM cross-encoder via ONNX. Fast, tiny.",
    ),




    # ── VISION — API: OpenAI ──
    ModelSpec(
        model_id="openai/gpt-4o",
        category=ModelCategory.VISION,
        tier=ModelTier.ORCA,
        loader="api_vision",
        requires_provider="openai",
        notes="OpenAI GPT-4o vision",
    ),
    ModelSpec(
        model_id="openai/gpt-4o-mini",
        category=ModelCategory.VISION,
        tier=ModelTier.DOLPHIN,
        loader="api_vision",
        requires_provider="openai",
        notes="Cheaper, faster vision",
    ),
    ModelSpec(
        model_id="openai/gpt-4-turbo",
        category=ModelCategory.VISION,
        tier=ModelTier.ORCA,
        loader="api_vision",
        requires_provider="openai",
        notes="GPT-4 Turbo with vision",
    ),

    # ── VISION — API: Google ──
    ModelSpec(
        model_id="google/gemini-2.0-flash",
        category=ModelCategory.VISION,
        tier=ModelTier.DOLPHIN,
        loader="api_vision",
        requires_provider="google",
        notes="Gemini 2.0 Flash vision",
    ),
    ModelSpec(
        model_id="google/gemini-1.5-pro",
        category=ModelCategory.VISION,
        tier=ModelTier.ORCA,
        loader="api_vision",
        requires_provider="google",
        notes="Gemini 1.5 Pro vision",
    ),
    ModelSpec(
        model_id="google/gemini-1.5-flash",
        category=ModelCategory.VISION,
        tier=ModelTier.DOLPHIN,
        loader="api_vision",
        requires_provider="google",
        notes="Gemini 1.5 Flash vision",
    ),

    # ── VISION — API: Groq ──
    ModelSpec(
        model_id="groq/meta-llama/llama-4-scout-17b-16e-instruct",
        category=ModelCategory.VISION,
        tier=ModelTier.ORCA,
        loader="api_vision",
        requires_provider="groq",
        notes="Llama 4 Scout vision on Groq",
    ),

    # ── VISION — API: OpenRouter ──
    ModelSpec(
        model_id="openrouter/google/gemma-3-4b-it:free",
        category=ModelCategory.VISION,
        tier=ModelTier.DOLPHIN,
        loader="api_vision",
        requires_provider="openrouter",
        notes="Gemma 3 4B vision via OpenRouter",
    ),
    ModelSpec(
        model_id="openrouter/meta-llama/llama-3.2-11b-vision-instruct:free",
        category=ModelCategory.VISION,
        tier=ModelTier.ORCA,
        loader="api_vision",
        requires_provider="openrouter",
        notes="Llama 3.2 11B vision via OpenRouter",
    ),
    ModelSpec(
        model_id="openrouter/qwen/qwen2.5-vl-72b-instruct:free",
        category=ModelCategory.VISION,
        tier=ModelTier.WHALE,
        loader="api_vision",
        requires_provider="openrouter",
        notes="Qwen2.5-VL 72B vision via OpenRouter",
    ),

    # ── NLP — local (spaCy) ──


    ModelSpec(
        model_id="en_core_web_md",
        category=ModelCategory.NLP,
        tier=ModelTier.DOLPHIN,
        loader="spacy_nlp",
        notes="spaCy medium English model",
        ram_gb=1,
        disk_mb=50,
    ),
    ModelSpec(
        model_id="en_core_web_lg",
        category=ModelCategory.NLP,
        tier=ModelTier.ORCA,
        loader="spacy_nlp",
        notes="spaCy large English model",
        ram_gb=2,
        disk_mb=400,
    ),

    ModelSpec(
    model_id="rapidocr",
    category=ModelCategory.OCR,
    tier=ModelTier.DOLPHIN,
    loader="rapidocr",
    ram_gb=0,
    disk_mb=16,
    notes="Lightweight ONNX OCR. English recognition. Dual-pass with synthetic-image preprocessing.",
    onnx_available=True,
    repo_id=None,
)
]


# ═══════════════════════════════════════════════════════════════════════════
# Build indexes for O(1) lookups
# ═══════════════════════════════════════════════════════════════════════════

# (model_id, category) -> ModelSpec
_SPEC_INDEX: Dict[tuple, ModelSpec] = {}
# model_id -> [ModelSpec] (a model_id could theoretically span categories)
_ID_INDEX: Dict[str, List[ModelSpec]] = {}
# loader -> [ModelSpec]
_LOADER_INDEX: Dict[str, List[ModelSpec]] = {}
# provider -> [ModelSpec]
_PROVIDER_INDEX: Dict[str, List[ModelSpec]] = {}
# category -> [ModelSpec]
_CATEGORY_INDEX: Dict[ModelCategory, List[ModelSpec]] = {}
# tier -> [ModelSpec]
_TIER_INDEX: Dict[ModelTier, List[ModelSpec]] = {}

def _build_indexes():
    _SPEC_INDEX.clear()
    _ID_INDEX.clear()
    _LOADER_INDEX.clear()
    _PROVIDER_INDEX.clear()
    _CATEGORY_INDEX.clear()
    _TIER_INDEX.clear()
    for spec in MODEL_SPECS:
        _SPEC_INDEX[(spec.model_id, spec.category)] = spec
        _ID_INDEX.setdefault(spec.model_id, []).append(spec)
        _LOADER_INDEX.setdefault(spec.loader, []).append(spec)
        if spec.requires_provider:
            _PROVIDER_INDEX.setdefault(spec.requires_provider, []).append(spec)
        _CATEGORY_INDEX.setdefault(spec.category, []).append(spec)
        _TIER_INDEX.setdefault(spec.tier, []).append(spec)

_build_indexes()


# ═══════════════════════════════════════════════════════════════════════════
# Core Query Functions
# ═══════════════════════════════════════════════════════════════════════════

def find_spec(model_id: str, category: Optional[ModelCategory] = None) -> Optional[ModelSpec]:
    """O(1) lookup by model_id + optional category."""
    if category is not None:
        return _SPEC_INDEX.get((model_id, category))
    specs = _ID_INDEX.get(model_id)
    return specs[0] if specs else None


def get_specs_by_loader(loader: str) -> List[ModelSpec]:
    """Get all specs that use a given loader type."""
    return list(_LOADER_INDEX.get(loader, []))


def get_specs_by_provider(provider: str) -> List[ModelSpec]:
    """Get all specs that require a given provider."""
    return list(_PROVIDER_INDEX.get(provider, []))


def get_specs_by_category(
    category: ModelCategory,
    tier: Optional[ModelTier] = None
) -> List[ModelSpec]:
    """Get all specs in a category, optionally filtered by tier."""
    results = list(_CATEGORY_INDEX.get(category, []))
    if tier is not None:
        results = [s for s in results if s.tier == tier]
    return results


def get_specs_by_tier(tier: ModelTier) -> List[ModelSpec]:
    """Get all specs at a given tier."""
    return list(_TIER_INDEX.get(tier, []))


# ═══════════════════════════════════════════════════════════════════════════
# BATTLE-TESTED: get_models_per_provider
# ═══════════════════════════════════════════════════════════════════════════

def get_models_per_provider(
    provider: str,
    category: Optional[ModelCategory] = None,
    tier: Optional[ModelTier] = None,
    include_local: bool = False,
    sort_by: str = "model_id",
    reverse: bool = False,
    as_dicts: bool = False,
    predicate: Optional[Callable[[ModelSpec], bool]] = None,
) -> List[ModelSpec] | List[Dict[str, Any]]:
    provider = provider.lower().strip()

    # Special case: "local" provider
    if provider == "local":
        results = [s for s in MODEL_SPECS if s.is_local]
    else:
        results = list(_PROVIDER_INDEX.get(provider, []))
        if not results and not include_local:
            # Check if provider exists in PROVIDER_REGISTRY
            if provider not in PROVIDER_REGISTRY:
                darwin.error(
                    f"Unknown provider: {provider!r}. "
                    f"Known providers: {sorted(PROVIDER_REGISTRY.keys())}"
                )
                return None
            # Provider exists but has no models — return empty, not error

    # ── Apply filters ─────────────────────────────────────────────────────
    if category is not None:
        results = [s for s in results if s.category == category]

    if tier is not None:
        results = [s for s in results if s.tier == tier]

    if predicate is not None:
        results = [s for s in results if predicate(s)]

    # ── Sort ───────────────────────────────────────────────────────────────
    if sort_by == "model_id":
        key_fn = lambda s: s.model_id
    elif sort_by == "tier":
        key_fn = lambda s: s.tier.value if hasattr(s.tier, "value") else str(s.tier)
    elif sort_by == "ram_gb":
        key_fn = lambda s: s.ram_gb
    elif sort_by == "disk_mb":
        key_fn = lambda s: s.disk_mb
    elif sort_by == "category":
        key_fn = lambda s: s.category.value if hasattr(s.category, "value") else str(s.category)
    else:
        key_fn = lambda s: getattr(s, sort_by, s.model_id)

    results = sorted(results, key=key_fn, reverse=reverse)

    # ── Return format ────────────────────────────────────────────────────
    if as_dicts:
        return [asdict(s) for s in results]
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Convenience wrappers around get_models_per_provider
# ═══════════════════════════════════════════════════════════════════════════

def get_local_models(
    category: Optional[ModelCategory] = None,
    tier: Optional[ModelTier] = None,
    sort_by: str = "model_id",
    reverse: bool = False,
) -> List[ModelSpec]:
    """Shorthand for get_models_per_provider("local", ...)."""
    return get_models_per_provider(
        "local",
        category=category,
        tier=tier,
        sort_by=sort_by,
        reverse=reverse,
    )


def get_api_models(
    category: Optional[ModelCategory] = None,
    tier: Optional[ModelTier] = None,
    sort_by: str = "model_id",
    reverse: bool = False,
) -> List[ModelSpec]:
    """Return all API models (those with a provider)."""
    results = [s for s in MODEL_SPECS if s.is_api]
    if category is not None:
        results = [s for s in results if s.category == category]
    if tier is not None:
        results = [s for s in results if s.tier == tier]
    key_fn = lambda s: getattr(s, sort_by, s.model_id)
    return sorted(results, key=key_fn, reverse=reverse)


def get_onnx_models(
    category: Optional[ModelCategory] = None,
    sort_by: str = "model_id",
    reverse: bool = False,
) -> List[ModelSpec]:
    """Return all models with ONNX availability."""
    results = [s for s in MODEL_SPECS if s.onnx_available]
    if category is not None:
        results = [s for s in results if s.category == category]
    key_fn = lambda s: getattr(s, sort_by, s.model_id)
    return sorted(results, key=key_fn, reverse=reverse)


# ═══════════════════════════════════════════════════════════════════════════
# Summary / Reporting
# ═══════════════════════════════════════════════════════════════════════════

def get_provider_summary(
    provider: Optional[str] = None,
    as_flat_list: bool = False,
) -> Dict[str, Any]:
    """
    Summary of models per provider.

    Parameters
    ----------
    provider : str, optional
        If given, only summarize this provider.
    as_flat_list : bool
        If True, return flat list of model_ids instead of nested dict.

    Returns
    -------
    dict
        Nested: {provider: {category: count, ...}, ...}
        or if as_flat_list: {provider: [model_id, ...], ...}
    """
    if provider:
        provider = provider.lower().strip()
        specs = _PROVIDER_INDEX.get(provider, [])
        if as_flat_list:
            return {provider: [s.model_id for s in specs]}
        cats: Dict[str, int] = {}
        for s in specs:
            cat = s.category.value if hasattr(s.category, "value") else str(s.category)
            cats[cat] = cats.get(cat, 0) + 1
        return {provider: cats}

    # All providers
    summary: Dict[str, Any] = {}
    for prov, specs in _PROVIDER_INDEX.items():
        if as_flat_list:
            summary[prov] = [s.model_id for s in specs]
        else:
            cats: Dict[str, int] = {}
            for s in specs:
                cat = s.category.value if hasattr(s.category, "value") else str(s.category)
                cats[cat] = cats.get(cat, 0) + 1
            summary[prov] = cats
    return summary


def get_registry_stats() -> Dict[str, Any]:
    """High-level stats about the entire registry."""
    local_count = sum(1 for s in MODEL_SPECS if s.is_local)
    api_count = sum(1 for s in MODEL_SPECS if s.is_api)
    onnx_count = sum(1 for s in MODEL_SPECS if s.onnx_available)

    categories: Dict[str, int] = {}
    for s in MODEL_SPECS:
        cat = s.category.value if hasattr(s.category, "value") else str(s.category)
        categories[cat] = categories.get(cat, 0) + 1

    tiers: Dict[str, int] = {}
    for s in MODEL_SPECS:
        tier = s.tier.value if hasattr(s.tier, "value") else str(s.tier)
        tiers[tier] = tiers.get(tier, 0) + 1

    return {
        "total_models": len(MODEL_SPECS),
        "local_models": local_count,
        "api_models": api_count,
        "onnx_models": onnx_count,
        "providers": sorted(_PROVIDER_INDEX.keys()),
        "categories": categories,
        "tiers": tiers,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_registry() -> Dict[str, Any]:
    """Validate all model specs. Returns dict with valid, errors, warnings."""
    errors: List[str] = []
    warnings: List[str] = []

    seen_ids: set = set()
    for idx, spec in enumerate(MODEL_SPECS):
        if not spec.model_id:
            errors.append(f"Index {idx}: missing model_id")
        if spec.model_id in seen_ids:
            errors.append(f"Duplicate model_id: {spec.model_id}")
        seen_ids.add(spec.model_id)

        # API key warnings
        if spec.requires_provider:
            prov = PROVIDER_REGISTRY.get(spec.requires_provider)
            if prov:
                if not os.getenv(prov.env_var):
                    warnings.append(
                        f"{spec.model_id}: provider '{spec.requires_provider}' "
                        f"requires env var {prov.env_var} which is not set"
                    )
            else:
                errors.append(
                    f"{spec.model_id}: provider '{spec.requires_provider}' "
                    f"not in PROVIDER_REGISTRY"
                )

        # RAM/disk sanity checks
        if spec.is_local and spec.ram_gb <= 0 and spec.disk_mb <= 0:
            warnings.append(
                f"{spec.model_id}: local model has zero RAM/disk — "
                f"should have resource estimates"
            )
        if spec.is_api and (spec.ram_gb > 0 or spec.disk_mb > 0):
            warnings.append(
                f"{spec.model_id}: API model has ram_gb={spec.ram_gb}/disk_mb={spec.disk_mb} — "
                f"should be 0 for API models"
            )

        # Repo ID consistency for local models
        if spec.is_local and spec.onnx_available and not spec.repo_id:
            warnings.append(
                f"{spec.model_id}: local ONNX model missing repo_id — "
                f"downloader won't know where to fetch from"
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "total_models": len(MODEL_SPECS),
    }