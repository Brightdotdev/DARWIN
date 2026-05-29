"""
src/core/Models/startup.py
---------------------------
Model initialization and onboarding.
Supports both terminal (print/input) and frontend (darwin.prompt) modes.
Now integrates OS keychain for API key storage.
"""

from typing import Optional
from src.core.Models.ModelStore import ModelStore
from src.core.Models.ModelLoader import ModelLoader
from src.core.Models.ModelRegistry import REGISTRY, TIER_LABELS, get_spec
from src.features.Hardware.SystemDetector import detect_system, recommend_defaults, SystemInfo
from src.core.Models.ModelSetup import (
    validate_model, calculate_download_size, validate_all_selections,
)
from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.KeyManager import (
    load_keys_from_keychain,
    set_api_key,
    PROVIDER_REGISTRY,
)


def _detect_api_junkies() -> dict[str, bool]:
    """Check which API keys are available (env + keychain)."""
    import os
    # Ensure keychain is loaded before checking
    load_keys_from_keychain()
    return {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
        "cohere": bool(os.getenv("COHERE_API_KEY")),
        "voyage": bool(os.getenv("VOYAGE_API_KEY")),
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "together": bool(os.getenv("TOGETHER_API_KEY")),
    }


def download_required(store, preferences, system):
    """Download required *local* models; API models are skipped with a check."""
    VALID_CATEGORIES = {"embedding", "reranker", "vision", "tokenizer"}
    selected = {cat: val for cat, val in preferences.items()
                if cat in VALID_CATEGORIES and val and val not in {"none", "not set"}}

    to_download = []
    for cat, key in selected.items():
        if key not in REGISTRY:
            continue
        spec = REGISTRY[key]
        if spec.disk_mb == 0:  # API model, nothing to download
            provider = key.split("/")[0]
            env_var = PROVIDER_REGISTRY.get(provider, (None,))[0]
            import os
            if env_var and not os.getenv(env_var):
                darwin.log(f" {key} requires API key for {provider}, which is not set.", LogType.WARNING)
            else:
                darwin.log(f"{key} is API‑only, key present.", LogType.SUCCESS)
            continue
        validation = validate_model(key, system, store)
        if validation.can_run:
            to_download.append(key)
        else:
            darwin.log(f"Skipping {key}: {', '.join(validation.warnings)}", LogType.WARNING)

    if not to_download:
        darwin.log("All compatible local models already cached or skipped.", LogType.SUCCESS)
        return

    download_info = calculate_download_size({k: k for k in to_download}, store)
    darwin.log(f"Models directory: {store.root}", LogType.INFO)
    darwin.log(f"System: {system.ram_gb}GB RAM, {system.disk_free_gb:.1f}GB free", LogType.INFO)

    if download_info["total_gb"] > system.disk_free_gb * 0.8:
        darwin.log(f"Downloads need {download_info['total_gb']:.1f}GB but only {system.disk_free_gb:.1f}GB free", LogType.WARNING)

    new_downloads = [k for k in to_download if not store.is_downloaded(k)]
    darwin.log(f"Downloading {len(new_downloads)} local model(s) — ~{download_info['total_gb']:.1f}GB", LogType.INFO)

    for key in to_download:
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
    Keys are loaded from keychain at startup and can be entered during selection.
    """
    store = ModelStore()
    system = detect_system()
    recommended = recommend_defaults(system)

    # Load any previously saved API keys into environment
    load_keys_from_keychain()

    darwin.log("")
    darwin.log(
        "Welcome to Darwin — Let's set up your AI models",
        LogType.INFO
    )

    darwin.log(
        f"System: {system.ram_gb}GB RAM | "
        f"{system.cpu_cores} cores | "
        f"{system.disk_free_gb:.0f}GB free disk",
        LogType.INFO,
    )

    if system.has_gpu:
        darwin.log(f"GPU: {system.gpu_name}", LogType.INFO)

    darwin.log(
        f"Best tier for your hardware: {TIER_LABELS[system.tier]}",
        LogType.INFO,
    )

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
        system,
        store,
    )

    darwin.log("")
    darwin.log("── Summary ──", LogType.INFO)

    for cat, key in choices.items():

        if key and key in REGISTRY:
            spec = REGISTRY[key]

            if spec.disk_mb == 0:
                # API model
                provider = key.split("/")[0]
                api_junkies = _detect_api_junkies()
                has_key = api_junkies.get(provider, False)

                status = (
                    "🔑 ready"
                    if has_key
                    else "❌ missing key"
                )

            else:
                status = (
                    "cached"
                    if store.is_downloaded(key)
                    else f"{spec.disk_mb}MB to download"
                )

            can = (
                "fits"
                if system.can_run(spec)
                else "may not fit"
            )

            darwin.log(
                f"  {cat}: {key} ({spec.tier}, {status}, {can})",
                LogType.INFO,
            )

        elif key:
            darwin.log(f"  {cat}: {key}", LogType.INFO)

        else:
            darwin.log(f"  {cat}: (none)", LogType.INFO)

    if validation["rejected"]:
        darwin.log(
            f"{len(validation['rejected'])} model(s) rejected "
            f"(insufficient hardware)",
            LogType.WARNING,
        )

    if (
        validation["download_info"]
        and validation["download_info"]["total_mb"] > 0
    ):
        info = validation["download_info"]

        darwin.log(
            f"Total to download: ~{info['total_gb']:.1f}GB "
            f"(free: {system.disk_free_gb:.1f}GB)",
            LogType.INFO,
        )

    # ── Download selected models ─────────────────────────────────────────

    for category, key in choices.items():

        if key and key in REGISTRY:

            spec = REGISTRY[key]

            should_download = (
                spec.disk_mb > 0
                and validate_model(key, system, store).can_run
                and not store.is_downloaded(key)
            )

            if should_download:
                darwin.log(f"Downloading {key}...", LogType.INFO)
                store.download(key)

    # ── Persist choices to config ────────────────────────────────────────

    from src.core.Models.ModelConfig import save_active_models

    active_config: dict[str, str | None] = {}

    for cat, model in choices.items():
        active_config[cat] = (
            model
            if model and model != "none"
            else None
        )

    save_active_models(active_config)

    darwin.log("All models ready!", LogType.SUCCESS)

    return ModelLoader(
        store,
        {k: v for k, v in choices.items() if v}
    )


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
    api_junkies = _detect_api_junkies()  # fresh check
    for i, (key, spec, validation) in enumerate(opts):
        star = " ⭐" if key == recommended_key else ""

        # API models show key status, local models show disk size + cache
        if spec.disk_mb == 0:
            provider = key.split("/")[0]
            has_key = api_junkies.get(provider, False)
            cached = f" [🔑 {'ready' if has_key else 'missing key'}]"
            size_str = "API"
        else:
            cached = " [cached]" if store.is_downloaded(key) else ""
            size_str = f"{spec.disk_mb}MB"

        can = "✅" if validation.can_run else "⚠️"
        label = f"[{i+1}] {key} ({spec.tier}, {size_str}, {spec.ram_gb}GB RAM) {can}{star}{cached}"
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

    # Parse the choice (returns option number, 's', 'r', or raw string)
    selected_key = None
    if choice == "s" and not required:
        darwin.log(f"Skipped {category}. Install later from settings.", LogType.INFO)
        return None
    if choice == "r" and recommended_key:
        selected_key = recommended_key
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(opts):
                key, spec, validation = opts[idx]
                selected_key = key
        except (ValueError, TypeError):
            pass

    if selected_key is None:
        # Fallback to recommended or first available
        selected_key = recommended_key if recommended_key else (opts[0][0] if opts else None)

    if selected_key:
        spec = get_spec(selected_key)
        # If API model and key missing, prompt for it now
        if spec and spec.disk_mb == 0:
            provider = selected_key.split("/")[0]
            if not api_junkies.get(provider, False):
                darwin.log(f"The model {selected_key} requires an API key for {provider}.", LogType.WARNING)
                # Prompt for the key (secure input if possible)
                try:
                    api_key = await darwin.prompt_password(
                        id=f"key_{provider}",
                        message=f"Enter your {provider.upper()} API key:",
                        action="save_api_key",
                        params={"provider": provider},
                    )
                except AttributeError:
                    # fallback if darwin.prompt_password doesn't exist
                    api_key = await darwin.prompt(
                        id=f"key_{provider}",
                        message=f"Enter your {provider.upper()} API key (input will be visible):",
                        action="save_api_key",
                        params={"provider": provider},
                    )
                if api_key and api_key.strip():
                    try:
                        set_api_key(provider, api_key, validate=True)
                        darwin.log(f"✅ API key for {provider} saved.", LogType.SUCCESS)
                    except Exception as e:
                        darwin.log(f"❌ Failed: {e}. Key not saved.", LogType.ERROR)
                else:
                    darwin.log("No key entered. The API model will not work until you set it.", LogType.WARNING)

    # Ensure we don't return an API model that can't work without a key? We still return it; setup will warn later.
    return selected_key


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


# Optionally, register an action to handle key save from prompt
def save_api_key_action(choice: str, **params) -> str:
    """Action handler for saving API key via darwin.prompt."""
    return choice  # just pass through the entered key