"""
src/core/Models/startup.py
---------------------------
Model initialization and onboarding.
Supports both terminal (print/input) and frontend (darwin.prompt) modes.
"""

from typing import Optional
from src.core.Models.ModelStore import ModelStore
from src.core.Models.ModelLoader import ModelLoader
from src.core.Models.ModelRegistry import REGISTRY, TIER_LABELS, get_spec
from src.core.Hardware.SystemDetector import detect_system, recommend_defaults, SystemInfo
from src.core.Models.ModelSetup import (
    validate_model, calculate_download_size, validate_all_selections,
)
from src.core.Logger.DarwinLogger import darwin, LogType


def create_loader(preferences=None, skip_download=False):
    """Create a ModelLoader. Auto-detects hardware if no preferences given."""
    store = ModelStore()
    system = detect_system()
    if preferences is None:
        preferences = recommend_defaults(system)
    loader = ModelLoader(store, preferences)
    if not skip_download:
        download_required(store, preferences, system)
    return loader, system


def download_required(store, preferences, system):
    """Download required models with hardware validation."""
    VALID_CATEGORIES = {"embedding", "reranker", "vision", "tokenizer"}
    selected = {cat: val for cat, val in preferences.items()
                if cat in VALID_CATEGORIES and val and val not in {"none", "not set"}}
    if not selected:
        return

    required_keys, optional_keys, failed_validation = [], [], []
    for cat, key in selected.items():
        if key not in REGISTRY: continue
        validation = validate_model(key, system, store)
        if not validation.can_run:
            failed_validation.append((key, validation.warnings))
        elif REGISTRY[key].required:
            required_keys.append(key)
        else:
            optional_keys.append(key)

    all_keys = required_keys + optional_keys
    download_info = calculate_download_size({k: k for k in all_keys}, store) if all_keys else {"total_gb": 0}

    # Use darwin.log so frontend sees these messages
    darwin.log(f"Models directory: {store.root}", LogType.INFO)
    darwin.log(f"System: {system.ram_gb}GB RAM, {system.disk_free_gb:.1f}GB free", LogType.INFO)

    if failed_validation:
        darwin.log(f"{len(failed_validation)} model(s) rejected (insufficient hardware)", LogType.WARNING)
        for key, warnings in failed_validation:
            darwin.log(f"  {key}: {', '.join(warnings)}", LogType.WARNING)

    new_downloads = [k for k in all_keys if not store.is_downloaded(k)]
    if not new_downloads:
        darwin.log("All compatible models already cached", LogType.SUCCESS)
        return

    if download_info["total_gb"] > system.disk_free_gb * 0.8:
        darwin.log(f"Downloads need {download_info['total_gb']:.1f}GB but only {system.disk_free_gb:.1f}GB free", LogType.WARNING)

    darwin.log(f"Downloading {len(new_downloads)} model(s) — ~{download_info['total_gb']:.1f}GB", LogType.INFO)

    for key in all_keys:
        spec = get_spec(key)
        if store.is_downloaded(key):
            darwin.log(f"{key} — cached ({store.installed_size(key)})", LogType.SUCCESS)
        else:
            darwin.log(f"Downloading {key} ({spec.disk_mb}MB)...", LogType.INFO)
            store.download(key)


# ═══════════════════════════════════════════════════════════════════════════
# Async onboarding — works with both terminal and frontend
# ═══════════════════════════════════════════════════════════════════════════

async def run_onboarding_async() -> ModelLoader:
    """
    Interactive first-run setup with custom model selection.
    Uses darwin.prompt() so it works in terminal AND browser.
    """
    store = ModelStore()
    system = detect_system()
    recommended = recommend_defaults(system)

    darwin.log("")
    darwin.log("Welcome to Darwin — Let's set up your AI models", LogType.INFO)
    darwin.log(f"System: {system.ram_gb}GB RAM | {system.cpu_cores} cores | {system.disk_free_gb:.0f}GB free disk", LogType.INFO)
    if system.has_gpu:
        darwin.log(f"GPU: {system.gpu_name}", LogType.INFO)
    darwin.log(f"Best tier for your hardware: {TIER_LABELS[system.tier]}", LogType.INFO)
    darwin.log("")

    choices: dict[str, str | None] = {}

    # ── STEP 1: Embedding (required) ─────────────────────────────────────
    choices["embedding"] = await _onboard_category_async(
        category="embedding",
        title="Embedding model (REQUIRED) — turns documents into searchable vectors",
        recommended_key=recommended["embedding"],
        system=system,
        store=store,
        required=True,
    )

    # ── STEP 2: Vision (optional) ────────────────────────────────────────
    choices["vision"] = await _onboard_category_async(
        category="vision",
        title="Vision model (OPTIONAL) — describes images in PDFs",
        recommended_key=recommended.get("vision"),
        system=system,
        store=store,
        required=False,
    )

    # ── STEP 3: Tokenizer (optional) ─────────────────────────────────────
    choices["tokenizer"] = await _onboard_category_async(
        category="tokenizer",
        title="NLP tokenizer (OPTIONAL) — better sentence splitting",
        recommended_key=recommended.get("tokenizer"),
        system=system,
        store=store,
        required=False,
    )

    # ── STEP 4: Reranker (optional) ──────────────────────────────────────
    choices["reranker"] = await _onboard_category_async(
        category="reranker",
        title="Reranker model (OPTIONAL) — re-ranks search results",
        recommended_key=recommended.get("reranker"),
        system=system,
        store=store,
        required=False,
    )

    # ── Validate and summarize ───────────────────────────────────────────
    validation = validate_all_selections(
        {k: v for k, v in choices.items() if v},
        system, store
    )

    darwin.log("")
    darwin.log("── Summary ──", LogType.INFO)
    for cat, key in choices.items():
        if key and key in REGISTRY:
            spec = REGISTRY[key]
            status = "cached" if store.is_downloaded(key) else f"{spec.disk_mb}MB to download"
            can = "fits" if system.can_run(spec) else "may not fit"
            darwin.log(f"  {cat}: {key} ({spec.tier}, {status}, {can})", LogType.INFO)
        elif key:
            darwin.log(f"  {cat}: {key}", LogType.INFO)
        else:
            darwin.log(f"  {cat}: (none)", LogType.INFO)

    if validation["rejected"]:
        darwin.log(f"{len(validation['rejected'])} model(s) rejected (insufficient hardware)", LogType.WARNING)

    if validation["download_info"] and validation["download_info"]["total_mb"] > 0:
        info = validation["download_info"]
        darwin.log(f"Total to download: ~{info['total_gb']:.1f}GB (free: {system.disk_free_gb:.1f}GB)", LogType.INFO)

    # Download valid models
    for category, key in choices.items():
        if key and key in REGISTRY and validate_model(key, system, store).can_run and not store.is_downloaded(key):
            darwin.log(f"Downloading {key}...", LogType.INFO)
            store.download(key)

    darwin.log("All models ready!", LogType.SUCCESS)
    return ModelLoader(store, {k: v for k, v in choices.items() if v})


async def _onboard_category_async(
    category: str,
    title: str,
    recommended_key: Optional[str],
    system: SystemInfo,
    store: ModelStore,
    required: bool = False,
) -> Optional[str]:
    """Async model selection for one category. Works in terminal + browser."""
    darwin.log("")
    darwin.log(f"── {title} ──", LogType.INFO)

    # Get all models with validation
    opts = []
    for key in REGISTRY:
        if REGISTRY[key].category != category:
            continue
        validation = validate_model(key, system, store)
        spec = REGISTRY[key]
        opts.append((key, spec, validation))

    if not opts:
        darwin.log("No models available for this category.", LogType.WARNING)
        return None if not required else recommended_key

    darwin.log(f"Available {category} models:", LogType.INFO)
    options = []
    for i, (key, spec, validation) in enumerate(opts):
        star = " ⭐" if key == recommended_key else ""
        cached = " [cached]" if store.is_downloaded(key) else ""
        can = "✅" if validation.can_run else "⚠️"
        label = f"[{i+1}] {key} ({spec.tier}, {spec.disk_mb}MB, {spec.ram_gb}GB RAM) {can}{star}{cached}"
        options.append(label)
        darwin.log(f"  {label}", LogType.INFO)

    if not required:
        options.append(f"[s] Skip — no {category} model for now")
        if recommended_key:
            options.append(f"[r] Use recommended: {recommended_key}")

    # Use darwin.prompt() — works in terminal AND WebSocket browser
    choice = await darwin.prompt(
        id=f"onboard_{category}",
        message=f"Select {category} model" + (" (required)" if required else ""),
        options=options,
        action="onboard_select",
        params={"category": category, "opts": opts, "recommended": recommended_key, "required": required},
    )

    # Parse the choice
    if choice == "s" and not required:
        darwin.log(f"Skipped {category}. Install later from settings.", LogType.INFO)
        return None
    if choice == "r" and recommended_key:
        return recommended_key

    # Try to parse as option number
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(opts):
            key, spec, validation = opts[idx]
            if not validation.can_run:
                darwin.log(f"This model may not run well: {', '.join(validation.warnings)}", LogType.WARNING)
            return key
    except (ValueError, TypeError):
        pass

    # Fallback to recommended
    if recommended_key:
        darwin.log(f"Using recommended: {recommended_key}", LogType.INFO)
        return recommended_key
    return opts[0][0] if opts else None


# ═══════════════════════════════════════════════════════════════════════════
# Onboarding action handler — register this with darwin.register_action()
# ═══════════════════════════════════════════════════════════════════════════

def onboard_select_action(choice: str, **params) -> str:
    """
    Action handler for darwin.prompt() during onboarding.
    Parses the user's model selection and returns the resolved choice.
    """
    opts = params.get("opts", [])
    recommended = params.get("recommended")
    required = params.get("required", False)

    # Parse choice format: "[1] model_name (...)" → extract "1"
    import re
    match = re.match(r'\[(\d+)\]', choice)
    if match:
        return match.group(1)  # Return just the number
    if choice.startswith("[s]"):
        return "s"
    if choice.startswith("[r]"):
        return "r"
    return choice  # Return as-is for fallback