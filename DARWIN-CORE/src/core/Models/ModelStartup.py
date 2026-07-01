# src/core/Models/startup.py
"""
Interactive model onboarding with terminal UI.
Uses rich for arrow-key navigation (fallback to simple input).
Supports headless mode via CLI args.
"""

import sys
import asyncio
import argparse
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

# ── Rich UI (optional) ──────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None

from src.core.Models.Lib.ModelRegistry import ModelCategory, find_spec, MODEL_SPECS, TIER_LABELS
from src.core.Models.ModelManager import modelManager, ModelManagerError
from src.core.Logger.DarwinLogger import darwin, LogType
from src.lib.KeyManager import load_keys_from_keychain, set_api_key, is_key_set


# ── System detection ───────────────────────────────────────────────────

@dataclass
class SystemInfo:
    ram_gb: float
    disk_free_gb: float
    cpu_cores: int
    has_gpu: bool = False
    gpu_name: str = ""
    tier: str = "light"


def detect_system() -> SystemInfo:
    try:
        import psutil
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_count = psutil.cpu_count(logical=True) or 1
        ram_gb = ram.total / (1024**3)
        disk_free_gb = disk.free / (1024**3)
    except ImportError:
        ram_gb = 8.0
        disk_free_gb = 50.0
        cpu_count = 4

    has_gpu = False
    gpu_name = ""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            has_gpu = True
            gpu_name = result.stdout.strip().split('\n')[0]
    except Exception:
        pass

    if ram_gb >= 32:
        tier = "overkill"
    elif ram_gb >= 16:
        tier = "large"
    elif ram_gb >= 8:
        tier = "medium"
    else:
        tier = "light"

    return SystemInfo(
        ram_gb=round(ram_gb, 1),
        disk_free_gb=round(disk_free_gb, 1),
        cpu_cores=cpu_count,
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        tier=tier,
    )


def recommend_defaults(system: SystemInfo) -> Dict[str, Optional[str]]:
    recs = {
        "light": {
            "embedding": "intfloat/multilingual-e5-small-onnx",
            "nlp": "en_core_web_md",
            "ocr": "SWHL/RapidOCR",
            "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2-onnx",
            "vision": None,
        },
        "medium": {
            "embedding": "intfloat/multilingual-e5-small-onnx",
            "nlp": "en_core_web_md",
            "ocr": "SWHL/RapidOCR",
            "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2-onnx",
            "vision": "HuggingFaceTB/SmolVLM-256M-Instruct-onnx",
        },
        "large": {
            "embedding": "intfloat/multilingual-e5-small-onnx",
            "nlp": "en_core_web_lg",
            "ocr": "SWHL/RapidOCR",
            "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2-onnx",
            "vision": "HuggingFaceTB/SmolVLM-256M-Instruct-onnx",
        },
        "overkill": {
            "embedding": "intfloat/multilingual-e5-small-onnx",
            "nlp": "en_core_web_lg",
            "ocr": "SWHL/RapidOCR",
            "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2-onnx",
            "vision": "HuggingFaceTB/SmolVLM-256M-Instruct-onnx",
        },
    }
    result = {}
    for cat, mid in recs.get(system.tier, recs["light"]).items():
        if mid and any(s.model_id == mid for s in MODEL_SPECS):
            result[cat] = mid
        else:
            result[cat] = None
    return result


# ── Logging helper ─────────────────────────────────────────────────────

def _log_verbose(msg: str, level: LogType = LogType.INFO):
    darwin.log(f"[STARTUP] {msg}", level)


# ── Interactive prompt (rich or fallback) ─────────────────────────────

async def _prompt_choice_interactive(
    category: ModelCategory,
    options: List[Tuple[str, Any, str]],  # (model_id, spec, status_str)
    title: str,
    required: bool,
    recommended_key: Optional[str],
) -> str:
    """
    Display an interactive menu with arrow-key navigation if rich is available.
    Returns 's' (skip), 'r' (recommended), or a number string (1-based index).
    """
    if RICH_AVAILABLE:
        table = Table(title=f"Available {title} Models", show_lines=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Model ID", style="white", no_wrap=False)
        table.add_column("Size", style="green", width=10)
        table.add_column("Status", style="yellow")
        table.add_column("RAM", style="dim", width=10)
        table.add_column("Notes", style="dim")

        for idx, (model_id, spec, status) in enumerate(options, 1):
            size_str = f"{spec.disk_mb}MB" if spec.disk_mb > 0 else "API"
            ram_str = f"{spec.ram_gb}GB" if spec.ram_gb > 0 else "N/A"
            star = " ⭐" if model_id == recommended_key else ""
            table.add_row(str(idx), model_id + star, size_str, status, ram_str, spec.notes or "")
        
        console.print(table)

        choices = [str(i) for i in range(1, len(options) + 1)]
        if not required:
            choices.append("s")
        if recommended_key:
            choices.append("r")

        prompt_text = f"Select {category.name.lower()} model" + (" (required)" if required else " ('s' to skip, 'r' for recommended)")
        default = "1"
        if recommended_key:
            for idx, (mid, _, _) in enumerate(options, 1):
                if mid == recommended_key:
                    default = str(idx)
                    break

        choice = Prompt.ask(prompt_text, choices=choices, default=default)
        return choice

    else:
        # Fallback: simple numbered list with input()
        print(f"\n{title}")
        print("-" * 60)
        for idx, (model_id, spec, status) in enumerate(options, 1):
            size_str = f"{spec.disk_mb}MB" if spec.disk_mb > 0 else "API"
            star = " ⭐" if model_id == recommended_key else ""
            print(f"  [{idx}] {model_id}{star} ({size_str}) {status}")
        if not required:
            print("  [s] Skip")
        if recommended_key:
            print("  [r] Use recommended: " + recommended_key)
        
        while True:
            choice = input("Select: ").strip().lower()
            if not required and choice == "s":
                return "s"
            if recommended_key and choice == "r":
                return "r"
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return choice
            print("Invalid choice. Try again.")


async def _prompt_api_key(provider: str) -> str:
    if RICH_AVAILABLE:
        return Prompt.ask(f"Enter your {provider.upper()} API key")
    else:
        print(f"Please enter your {provider.upper()} API key:")
        return input("> ").strip()


# ── Onboarding core ─────────────────────────────────────────────────────

async def run_onboarding_async(args: argparse.Namespace) -> None:
    load_keys_from_keychain()
    system = detect_system()
    recommended = recommend_defaults(system)

    _log_verbose("=" * 60, LogType.INFO)
    _log_verbose("🚀 Darwin Model Setup", LogType.INFO)
    _log_verbose("=" * 60, LogType.INFO)
    _log_verbose(f"System: {system.ram_gb}GB RAM | {system.cpu_cores} cores | {system.disk_free_gb:.0f}GB free", LogType.INFO)
    if system.has_gpu:
        _log_verbose(f"GPU: {system.gpu_name}", LogType.INFO)
    _log_verbose(f"Best tier for your hardware: {TIER_LABELS.get(system.tier, 'unknown')}", LogType.INFO)

    _log_verbose("Scanning existing models...", LogType.INFO)
    status = modelManager.initialize(auto_setup=False)
    _log_verbose(status.summary, LogType.INFO)

    choices: Dict[str, Optional[str]] = {}

    categories = [
        ("embedding", "Embedding model", True),
        ("vision", "Vision model", False),
        ("nlp", "NLP model", False),
        ("reranker", "Reranker model", False),
        ("ocr", "OCR model", False),
    ]

    for cat_name, title, required in categories:
        if cat_name not in ModelCategory.__members__:
            _log_verbose(f"Unknown category {cat_name} – skipping", LogType.WARNING)
            continue
        
        cat_enum = ModelCategory[cat_name.upper()]
        cli_model = getattr(args, cat_name, None)
        if cli_model:
            spec = find_spec(cli_model)
            if spec and spec.category == cat_enum:
                choices[cat_name] = cli_model
                _log_verbose(f"Using CLI‑provided model for {cat_name}: {cli_model}", LogType.INFO)
                continue
            else:
                _log_verbose(f"CLI model '{cli_model}' not found or wrong category – falling back", LogType.WARNING)

        if getattr(args, "non_interactive", False):
            default = recommended.get(cat_name)
            if default:
                choices[cat_name] = default
                _log_verbose(f"Non‑interactive: using default {cat_name}: {default}", LogType.INFO)
            else:
                choices[cat_name] = None
                _log_verbose(f"Non‑interactive: no default for {cat_name} – skipping", LogType.WARNING)
            continue

        choice = await _onboard_category_interactive(
            cat_enum,
            title,
            recommended.get(cat_name),
            system,
            required=required,
        )
        choices[cat_name] = choice

    _log_verbose("\n── Onboarding Summary ──", LogType.INFO)
    for cat, key in choices.items():
        if key:
            spec = find_spec(key)
            if spec:
                if spec.requires_provider:
                    provider = spec.requires_provider
                    has_key = is_key_set(provider)
                    status_str = "key ready" if has_key else "  missing key"
                else:
                    downloaded = modelManager.is_downloaded(ModelCategory[cat.upper()], key)
                    status_str = " downloaded" if downloaded else "⏳ to download"
                _log_verbose(f"  {cat}: {key} ({spec.tier}) – {status_str}", LogType.INFO)
        else:
            _log_verbose(f"  {cat}: (none selected)", LogType.INFO)

    preferences = {}
    for cat, key in choices.items():
        if key:
            cat_enum = ModelCategory[cat.upper()]
            preferences[cat_enum] = key

    if preferences:
        _log_verbose("Preparing to download and activate selected models...", LogType.INFO)
        results = modelManager.setup_all(
            preferences=preferences,
            auto_download=args.auto_download,
            required_categories=[ModelCategory.EMBEDDING] if 'embedding' in choices and choices['embedding'] else [],
        )
        for cat, result in results.items():
            if result.success:
                _log_verbose(f" {cat.name}: {result.model_id} ready", LogType.SUCCESS)
            else:
                _log_verbose(f"  {cat.name}: {result.error}", LogType.ERROR)
    else:
        _log_verbose("No models selected – nothing to do.", LogType.WARNING)

    _log_verbose(" Onboarding complete!", LogType.SUCCESS)


async def _onboard_category_interactive(
    category: ModelCategory,
    title: str,
    recommended_key: Optional[str],
    system,
    required: bool = False,
) -> Optional[str]:
    _log_verbose(f"\n── {title} ──", LogType.INFO)

    all_specs = [s for s in MODEL_SPECS if s.category == category]
    if not all_specs:
        _log_verbose("No models registered for this category.", LogType.WARNING)
        return None

    options = []
    for spec in all_specs:
        if spec.requires_provider:
            provider = spec.requires_provider
            has_key = is_key_set(provider)
            status_str = "key " + ("" if has_key else " ")
        else:
            downloaded = modelManager.is_downloaded(category, spec.model_id)
            status_str = "💾 " + (" cached" if downloaded else "⬇️ to download")

        can_run = system.ram_gb >= spec.ram_gb if spec.ram_gb > 0 else True
        if not can_run:
            status_str += "  may not fit"

        options.append((spec.model_id, spec, status_str))

    choice = await _prompt_choice_interactive(
        category,
        options,
        title,
        required,
        recommended_key,
    )

    if choice == "s" and not required:
        return None
    if choice == "r" and recommended_key:
        return recommended_key

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            selected_key = options[idx][0]
        else:
            selected_key = None
    except ValueError:
        selected_key = None

    if selected_key is None and required:
        _log_verbose("No valid selection; using recommended if available.", LogType.WARNING)
        selected_key = recommended_key or (options[0][0] if options else None)

    if selected_key:
        spec = find_spec(selected_key)
        if spec and spec.requires_provider:
            provider = spec.requires_provider
            if not is_key_set(provider):
                _log_verbose(f"Model {selected_key} requires an API key for {provider}.", LogType.WARNING)
                key = await _prompt_api_key(provider)
                if key and key.strip():
                    try:
                        set_api_key(provider, key, validate=True)
                        _log_verbose(f" API key for {provider} saved.", LogType.SUCCESS)
                    except Exception as e:
                        _log_verbose(f"  Failed to save key: {e}", LogType.ERROR)
                else:
                    _log_verbose("No key entered – model will not work.", LogType.WARNING)

    return selected_key


def parse_arguments():
    parser = argparse.ArgumentParser(description="Darwin Model Setup")
    parser.add_argument("--embedding", help="Embedding model ID")
    parser.add_argument("--vision", help="Vision model ID")
    parser.add_argument("--nlp", help="NLP model ID")
    parser.add_argument("--reranker", help="Reranker model ID")
    parser.add_argument("--ocr", help="OCR model ID")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Skip interactive prompts, use defaults")
    parser.add_argument("--auto-download", action="store_true", default=True,
                        help="Auto-download missing models")
    return parser.parse_args()


def main():
    args = parse_arguments()
    asyncio.run(run_onboarding_async(args))


if __name__ == "__main__":
    main()