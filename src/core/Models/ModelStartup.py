# src/core/Models/startup.py

from pathlib import Path
from typing import Optional
from src.core.Models.ModelStore import ModelStore
from src.core.Models.ModelLoader import ModelLoader
from src.core.Models.SystemDetector import detect_system, recommend_defaults, SystemInfo
from src.core.Models.ModelRegistry import (
    REGISTRY, TIER_LABELS, ModelCategory,
    required_models, optional_models, get_spec,
)


def create_loader(
    models_dir: Path,
    preferences: Optional[dict] = None,
    skip_download: bool = False,
) -> tuple[ModelLoader, SystemInfo]:
    """
    Create a ModelLoader. If no preferences given, auto-detect from hardware.

    Only downloads REQUIRED models by default. Optional models (vision, tokenizer)
    are downloaded on-demand when the user passes --caption or uses NLP features.
    """
    store = ModelStore(models_dir)
    system = detect_system()

    if preferences is None:
        preferences = recommend_defaults(system)

    loader = ModelLoader(store, preferences)

    if not skip_download:
        _download_required(store, preferences, system)

    return loader, system


def _download_required(store: ModelStore, preferences: dict, system: SystemInfo) -> None:
    """Download required models. Show clear progress."""
    required_keys = [k for k in preferences.values() if k and REGISTRY.get(k, get_spec(k)).required]
    optional_keys = [k for k in preferences.values() if k and not REGISTRY.get(k, get_spec(k)).required]

    if not required_keys:
        return

    print(f"\n📦 Models directory: {store.root}")
    print(f"💻 System: {system.ram_gb}GB RAM, {system.disk_free_gb:.1f}GB free")
    print()

    # ── Required models ──
    print("── Required models ───────────────────────────────────────────")
    for key in required_keys:
        spec = get_spec(key)
        if store.is_downloaded(key):
            size = store.installed_size(key)
            print(f"   ✅ {key} — {size}")
        else:
            print(f"   ⬇️  {key} ({spec.disk_mb}MB) — {spec.notes}")
            store.download(key)

    # ── Optional models (only if user selected them) ──
    if optional_keys:
        print()
        print("── Optional models ───────────────────────────────────────────")
        for key in optional_keys:
            spec = get_spec(key)
            if store.is_downloaded(key):
                size = store.installed_size(key)
                print(f"   ✅ {key} — {size}")
            else:
                print(f"   ⬇️  {key} ({spec.disk_mb}MB) — {spec.notes}")
                store.download(key)

    print()
    required_sizes = sum(get_spec(k).disk_mb for k in required_keys)
    optional_sizes = sum(get_spec(k).disk_mb for k in optional_keys if not store.is_downloaded(k))
    total_new = sum(
        get_spec(k).disk_mb for k in required_keys + optional_keys
        if not store.is_downloaded(k)
    )
    if total_new > 0:
        print(f"   Total download: ~{total_new / 1024:.1f}GB")
    print()


def run_onboarding(models_dir: Path) -> ModelLoader:
    """
    Interactive first-run setup.

    Flow:
    1. Detect hardware, show specs
    2. Required models: auto-pick best for hardware, let user override
    3. Optional models: ask if they want vision/spaCy now or later
    4. Download selected models
    """
    store = ModelStore(models_dir)
    system = detect_system()
    recommended = recommend_defaults(system)

    print("\n" + "=" * 60)
    print("🧠  Welcome to Darwin — Let's set up your AI models")
    print("=" * 60)
    print(f"\n💻  Your system: {system.ram_gb}GB RAM | {system.cpu_cores} cores | "
          f"{system.disk_free_gb:.0f}GB free disk")
    if system.has_gpu:
        print(f"🎮  GPU: {system.gpu_name}")
    print(f"📊  Best tier for your hardware: {TIER_LABELS[system.tier]}")
    print()

    choices: dict[str, str | None] = {
        "embedding": None,
        "reranker":  "reranker-medium",
        "vision":    None,
        "tokenizer": None,
    }

    # ── STEP 1: Required — Embedding ─────────────────────────────────────
    print("─" * 60)
    print("🔤  Step 1/3: Embedding model (REQUIRED)")
    print("   This turns your documents into searchable vectors.")
    print(f"   Recommended for your hardware: {recommended['embedding']}")
    print()
    print("   Options:")
    opts = [k for k in REGISTRY if REGISTRY[k].category == "embedding"]
    for i, key in enumerate(opts):
        spec = REGISTRY[key]
        star = " ⭐" if key == recommended['embedding'] else ""
        can = "✅" if system.can_run(spec) else "⚠️  needs more RAM"
        req = " (required)" if spec.required else " (experimental)"
        print(f"   [{i+1}] {key} — {TIER_LABELS[spec.tier]} "
              f"| {spec.disk_mb}MB | ~{spec.ram_gb}GB RAM {can}{star}{req}")

    choice = input(f"\n   Pick (1-{len(opts)}) or Enter for recommended: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(opts):
        choices["embedding"] = opts[int(choice) - 1]
    else:
        choices["embedding"] = recommended['embedding']

    # ── STEP 2: Optional — Vision ────────────────────────────────────────
    print("\n" + "─" * 60)
    print("👁️  Step 2/3: Vision model (OPTIONAL)")
    print("   Describes images in your PDFs. Only needed with --caption.")
    print("   Skip this if you don't have image-heavy PDFs.")
    print()
    print("   Options:")
    opts = [k for k in REGISTRY if REGISTRY[k].category == "vision"]
    for i, key in enumerate(opts):
        spec = REGISTRY[key]
        can = "✅" if system.can_run(spec) else "⚠️  needs more RAM"
        print(f"   [{i+1}] {key:<20} — {spec.disk_mb:>4}MB | {spec.notes:<50} {can}")
    print(f"   [s] Skip — no vision model for now (can add later)")

    choice = input(f"\n   Pick (1-{len(opts)}) or [s]kip: ").strip().lower()
    if choice == 's' or choice == '':
        choices["vision"] = None
        print("   ⏭️  Skipped. Use --caption later to auto-download.")
    elif choice.isdigit() and 1 <= int(choice) <= len(opts):
        choices["vision"] = opts[int(choice) - 1]
    else:
        choices["vision"] = None

    # ── STEP 3: Optional — spaCy tokenizer ───────────────────────────────
    print("\n" + "─" * 60)
    print("📝  Step 3/3: NLP tokenizer (OPTIONAL)")
    print("   Better sentence splitting for chunking. Improves retrieval quality.")
    print("   You can use the app without it — we fall back to regex.")
    print()
    print("   Options:")
    opts = [k for k in REGISTRY if REGISTRY[k].category == "tokenizer"]
    for i, key in enumerate(opts):
        spec = REGISTRY[key]
        print(f"   [{i+1}] {key:<20} — {spec.disk_mb:>4}MB | {spec.notes}")
    print(f"   [s] Skip — no tokenizer for now (can add later)")

    choice = input(f"\n   Pick (1-{len(opts)}) or [s]kip: ").strip().lower()
    if choice == 's' or choice == '':
        choices["tokenizer"] = None
        print("   ⏭️  Skipped. Install later from settings.")
    elif choice.isdigit() and 1 <= int(choice) <= len(opts):
        choices["tokenizer"] = opts[int(choice) - 1]
    else:
        choices["tokenizer"] = None

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📦  Summary")
    print("=" * 60)

    total_download = 0
    for cat, key in choices.items():
        if key:
            spec = REGISTRY[key]
            total_download += spec.disk_mb if not store.is_downloaded(key) else 0
            status = "✅ cached" if store.is_downloaded(key) else f"⬇️  {spec.disk_mb}MB"
            req = "REQUIRED" if spec.required else "optional"
            print(f"   {cat:<12} {key:<25} {TIER_LABELS[spec.tier]:<20} {status:<15} {req}")
        else:
            print(f"   {cat:<12} (none)")

    if total_download > 0:
        print(f"\n   Total to download: ~{total_download / 1024:.1f}GB")
        print(f"   Free disk space: {system.disk_free_gb:.1f}GB")

        if total_download / 1024 > system.disk_free_gb:
            print(f"\n   ⚠️  Not enough disk space! Free up {(total_download / 1024 - system.disk_free_gb):.1f}GB.")
            confirm = input("\n   Continue anyway? [y/N]: ").strip().lower()
            if confirm != 'y':
                print("   Setup cancelled. Run again later.")
                return ModelLoader(store, choices)
    else:
        print(f"\n   Nothing to download — all models already cached.")

    print()
    confirm = input("   Download and start? [Y/n]: ").strip().lower()
    if confirm and confirm != 'y':
        print("   Using your selections...")
    else:
        print("\n⬇️  Downloading...")
        for category, key in choices.items():
            if key and not store.is_downloaded(key):
                store.download(key)

    print("✅  All models ready!\n")
    return ModelLoader(store, choices)