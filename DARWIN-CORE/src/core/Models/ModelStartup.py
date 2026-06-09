# src/core/Models/startup.py
import os
from typing import Optional
from src.core.Models.ModelStore import ModelStore
from src.core.Models.ModelLoader import ModelLoader
from src.core.Models.ModelRegistry import REGISTRY, TIER_LABELS, get_spec
from src.features.Hardware.SystemDetector import detect_system, recommend_defaults
from src.core.Models.ModelSetup import (
    validate_model, calculate_download_size, validate_all_selections,
)
from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.KeyManager import (
    load_keys_from_keychain,
    set_api_key,
    is_key_set,
)


def download_required(store, preferences, system):
    """
    Download required **local** models (skip API models).

    preferences : dict like {'embedding': 'BAAI/bge-m3-onnx', ...}
    system      : SystemInfo instance
    """
    VALID_CATEGORIES = {"embedding", "reranker", "vision", "tokenizer"}

    selected = {
        cat: val
        for cat, val in preferences.items()
        if cat in VALID_CATEGORIES and val and val not in {"none", "not set"}
    }

    # Separate models into API-only vs local
    api_models = []
    local_models = []

    for cat, key in selected.items():
        if key not in REGISTRY:
            darwin.log(f"Model '{key}' not found in registry – skipping.", LogType.WARNING)
            continue
        spec = REGISTRY[key]

        if spec.disk_mb == 0:          # API model
            provider = key.split("/")[0]
            if not is_key_set(provider):
                darwin.log(f"{key} requires API key for {provider}, which is not set.", LogType.WARNING)
            else:
                darwin.log(f"{key} is API‑only, key present.", LogType.SUCCESS)
            api_models.append(key)
        else:                          # Local model
            local_models.append(key)

    # ── API models nothing to download ──
    if not local_models:
        darwin.log("All selected models are API‑based – no local downloads needed.", LogType.SUCCESS)
        return

    # ── Check which local models still need downloading ──
    to_download = []
    for key in local_models:
        if store.is_downloaded(key):
            spec = get_spec(key)
            darwin.log(f"[{spec.category.capitalize()}]  {key} ({store.installed_size(key)})", LogType.SUCCESS)
            continue

        # Only validate system requirements for models NOT yet downloaded
        validation = validate_model(key, system, store)
        if validation.can_run:
            to_download.append(key)
        else:
            darwin.log(f"Skipping {key}: {', '.join(validation.warnings)}", LogType.WARNING)

    if not to_download:
        darwin.log("All required local models are already cached.  ", LogType.SUCCESS)
        return

    # ── Proceed with downloads ──
    download_info = calculate_download_size({k: k for k in to_download}, store)
    darwin.log(f"[MODELS DIRECTORY]: {store.resolver.root}", LogType.INFO)
    darwin.log(f"System: {system.ram_gb}GB RAM, {system.disk_free_gb:.1f}GB free", LogType.INFO)

    if download_info["total_gb"] > system.disk_free_gb * 0.8:
        darwin.log(
            f"Downloads need {download_info['total_gb']:.1f}GB but only {system.disk_free_gb:.1f}GB free",
            LogType.WARNING,
        )

    darwin.log(
        f"Downloading {len(to_download)} local model(s) — ~{download_info['total_gb']:.1f}GB",
        LogType.INFO,
    )

    for key in to_download:
        spec = get_spec(key)
        darwin.log(f"[{spec.category.capitalize()}] Downloading {key} ({spec.disk_mb}MB)...", LogType.INFO)
        store.download(key)


async def run_onboarding_async() -> ModelLoader:
    store = ModelStore()
    system = detect_system()
    recommended = recommend_defaults(system)
    load_keys_from_keychain()

    darwin.log("\nWelcome to Darwin — Let's set up your AI models\n", LogType.INFO)
    darwin.log(f"System: {system.ram_gb}GB RAM | {system.cpu_cores} cores | {system.disk_free_gb:.0f}GB free", LogType.INFO)
    if system.has_gpu:
        darwin.log(f"GPU: {system.gpu_name}", LogType.INFO)
    darwin.log(f"Best tier for your hardware: {TIER_LABELS[system.tier]}", LogType.INFO)

    choices: dict[str, Optional[str]] = {}
    choices["embedding"] = await _onboard_category("embedding", "Embedding model (REQUIRED)", recommended["embedding"], system, store, required=True)
    choices["vision"] = await _onboard_category("vision", "Vision model (OPTIONAL)(REQUIRES PYTOCH RUNTIME)", recommended.get("vision"), system, store, required=False)
    choices["tokenizer"] = await _onboard_category("tokenizer", "NLP tokenizer (OPTIONAL)(IF YOU NEED ENTITY GRAPH AND WIKI KNOWLEDGE OF YOUR DOCUMENT)", recommended.get("tokenizer"), system, store, required=False)
    choices["reranker"] = await _onboard_category("reranker", "Reranker model (OPTIONAL)(ONE MORE STEP BEFORE FINAL RESULTS RECOMENDED BUT OPTIONAL)", recommended.get("reranker"), system, store, required=False)

    validation = validate_all_selections({k: v for k, v in choices.items() if v}, system, store)

    darwin.log("\n── Summary ──", LogType.INFO)
    for cat, key in choices.items():
        if key and key in REGISTRY:
            spec = REGISTRY[key]
            if spec.disk_mb == 0:
                provider = key.split("/")[0]
                status = "🔑 ready" if is_key_set(provider) else "❌ missing key"
            else:
                status = "cached" if store.is_downloaded(key) else f"{spec.disk_mb}MB to download"
            can = "fits" if system.can_run(spec) else "may not fit"
            darwin.log(f"  {cat}: {key} ({spec.tier}, {status}, {can})", LogType.INFO)
        elif key:
            darwin.log(f"  {cat}: {key}", LogType.INFO)
        else:
            darwin.log(f"  {cat}: (none)", LogType.INFO)

    if validation["rejected"]:
        darwin.log(f"{len(validation['rejected'])} model(s) rejected (insufficient hardware)", LogType.WARNING)

    if validation.get("download_info") and validation["download_info"]["total_mb"] > 0:
        info = validation["download_info"]
        darwin.log(f"Total to download: ~{info['total_gb']:.1f}GB (free: {system.disk_free_gb:.1f}GB)", LogType.INFO)

    # Download selected local models
    for cat, key in choices.items():
        if key and key in REGISTRY:
            spec = REGISTRY[key]
            if spec.disk_mb > 0 and validate_model(key, system, store).can_run and not store.is_downloaded(key):
                darwin.log(f"Downloading {key}...", LogType.INFO)
                store.download(key)

    from src.core.Models.ModelConfig import save_active_models
    active_config = {cat: (key if key and key != "none" else None) for cat, key in choices.items()}
    save_active_models(active_config)

    darwin.log("All models ready!", LogType.SUCCESS)
    return ModelLoader(store, {k: v for k, v in choices.items() if v})


async def _onboard_category(category, title, recommended_key, system, store, required):
    darwin.log(f"\n── {title} ──", LogType.INFO)
    opts = []
    for key in REGISTRY:
        if REGISTRY[key].category != category:
            continue
        validation = validate_model(key, system, store)
        spec = REGISTRY[key]
        opts.append((key, spec, validation))

    if not opts:
        darwin.log("No models available.", LogType.WARNING)
        return None if not required else recommended_key

    options = []
    for i, (key, spec, validation) in enumerate(opts):
        star = " ⭐" if key == recommended_key else ""
        if spec.disk_mb == 0:
            provider = key.split("/")[0]
            has_key = is_key_set(provider)
            cached = f" [🔑 {'ready' if has_key else 'missing key'}]"
            size_str = "API"
        else:
            cached = " [cached]" if store.is_downloaded(key) else ""
            size_str = f"{spec.disk_mb}MB"
        can = " " if validation.can_run else "⚠️"
        options.append(f"[{i+1}] {key} ({spec.tier}, {size_str}, {spec.ram_gb}GB RAM) {can}{star}{cached}")
        darwin.log(f"  {options[-1]}", LogType.INFO)

    if not required:
        options.append("[s] Skip")
        if recommended_key:
            options.append(f"[r] Use recommended: {recommended_key}")

    choice = await darwin.prompt(
        id=f"onboard_{category}",
        message=f"Select {category} model" + (" (required)" if required else ""),
        options=options,
        action="onboard_select",
        params={"category": category, "opts": opts, "recommended": recommended_key, "required": required},
    )

    selected_key = None
    if choice == "s" and not required:
        return None
    elif choice == "r" and recommended_key:
        selected_key = recommended_key
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(opts):
                selected_key = opts[idx][0]
        except ValueError:
            pass

    if selected_key is None:
        selected_key = recommended_key if recommended_key else (opts[0][0] if opts else None)

    if selected_key:
        spec = get_spec(selected_key)
        if spec and spec.disk_mb == 0:
            provider = selected_key.split("/")[0]
            if not is_key_set(provider):
                darwin.log(f"The model {selected_key} requires an API key for {provider}.", LogType.WARNING)
                try:
                    api_key = await darwin.prompt_password(
                        id=f"key_{provider}",
                        message=f"Enter your {provider.upper()} API key:",
                        action="save_api_key",
                        params={"provider": provider},
                    )
                except AttributeError:
                    api_key = await darwin.prompt(
                        id=f"key_{provider}",
                        message=f"Enter your {provider.upper()} API key (input will be visible):",
                        action="save_api_key",
                        params={"provider": provider},
                    )
                if api_key and api_key.strip():
                    try:
                        set_api_key(provider, api_key, validate=True)
                        darwin.log(f"  API key for {provider} saved.", LogType.SUCCESS)
                    except Exception as e:
                        darwin.log(f"❌ Failed: {e}. Key not saved.", LogType.ERROR)
                else:
                    darwin.log("No key entered. The API model will not work until you set it.", LogType.WARNING)
    return selected_key