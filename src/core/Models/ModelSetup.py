"""
src/core/Models/ModelSetup.py
-------------------------------
Custom model selection with hardware validation.
Checks RAM, disk space, and inference viability before downloading.
"""

from dataclasses import dataclass
from typing import Optional, List
from src.core.Models.ModelRegistry import REGISTRY, get_spec, ModelCategory
from src.core.Models.SystemDetector import SystemInfo
from src.core.Models.ModelStore import ModelStore


@dataclass
class ValidationResult:
    """Result of checking if a model can run on this hardware."""
    model_key: str
    can_run: bool
    ram_ok: bool
    disk_ok: bool
    warnings: List[str]


def validate_model(key: str, system: SystemInfo, store: Optional[ModelStore] = None) -> ValidationResult:
    """
    Check if a model can run on this system.
    Returns a ValidationResult with details about each constraint.
    """
    if key not in REGISTRY:
        return ValidationResult(key, False, False, False, [f"Unknown model: {key}"])
    
    spec = get_spec(key)
    warnings = []
    ram_ok = system.can_run(spec)
    disk_ok = system.has_disk_space(spec) if not (store and store.is_downloaded(key)) else True
    
    if not ram_ok:
        warnings.append(f"Needs {spec.ram_gb}GB RAM (you have {system.ram_gb}GB)")
    if not disk_ok:
        needed_gb = spec.disk_mb / 1024
        warnings.append(f"Needs {needed_gb:.1f}GB disk (you have {system.disk_free_gb:.1f}GB free)")
 
    return ValidationResult(
        model_key=key,
        can_run=(ram_ok and disk_ok),
        ram_ok=ram_ok,
        disk_ok=disk_ok,
        warnings=warnings,
    )


def validate_category(category: str, system: SystemInfo, store: Optional[ModelStore] = None) -> List[ValidationResult]:
    """Validate all models in a category. Returns list of results for each model."""
    return [
        validate_model(key, system, store)
        for key, spec in REGISTRY.items()
        if spec.category == category
    ]


def select_best_for_category(category: str, system: SystemInfo) -> Optional[str]:
    """
    Pick the best model for a category given hardware constraints.
    Returns the best model key that CAN run, or None if nothing fits.
    """
    candidates = [
        (key, spec) for key, spec in REGISTRY.items()
        if spec.category == category and system.can_run(spec)
    ]
    if not candidates:
        return None
    
    # Sort by tier (best first), then by RAM requirement (lowest first as tiebreaker)
    tier_order = {"overkill": 4, "large": 3, "medium": 2, "light": 1}
    candidates.sort(key=lambda x: (tier_order.get(x[1].tier, 0), -x[1].ram_gb), reverse=True)
    return candidates[0][0]


def calculate_download_size(selected_models: dict[str, str], store: ModelStore) -> dict:
    """
    Calculate total download size for selected models.
    Returns {total_mb, per_model_mb, already_cached_mb}.
    """
    total_new = 0
    total_cached = 0
    per_model = {}
    
    for category, key in selected_models.items():
        if not key or key == "none":
            continue
        spec = get_spec(key)
        if store.is_downloaded(key):
            total_cached += spec.disk_mb
            per_model[key] = {"size_mb": spec.disk_mb, "status": "cached"}
        else:
            total_new += spec.disk_mb
            per_model[key] = {"size_mb": spec.disk_mb, "status": "new"}
    
    return {
        "total_mb": total_new,
        "total_gb": total_new / 1024,
        "cached_mb": total_cached,
        "per_model": per_model,
    }


def validate_all_selections(selections: dict[str, str], system: SystemInfo, store: ModelStore) -> dict:
    """
    Validate all user selections. Returns detailed report.
    Used by onboarding and custom setup to show what will work vs. what will fail.
    """
    report = {
        "valid": [],
        "rejected": [],
        "download_info": None,
        "all_valid": True,
    }
    
    for category, key in selections.items():
        if not key or key == "none":
            continue
        
        result = validate_model(key, system, store)
        if result.can_run:
            report["valid"].append(result)
        else:
            report["rejected"].append(result)
            report["all_valid"] = False
    
    if report["valid"]:
        # Only count models that passed validation for download
        valid_selections = {r.model_key: r.model_key for r in report["valid"]}
        report["download_info"] = calculate_download_size(valid_selections, store)
    
    return report





# ═══════════════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════════════

def handle_download_error(error: Exception, key: str) -> None:
    """Handle download errors with actionable recovery suggestions."""
    from src.core.Models.ModelStore import DownloadError, IntegrityError

    print(f"\n❌ Error downloading {key}: {error}")

    # model_path = manager.store.root / key

    if isinstance(error, IntegrityError):
        print("\n🔧 Recovery steps:")
        print(f"   1. Clear corrupted files: python -m src.core.Models.cli uninstall --model {key}")
        print(f"   2. Free up disk space")
        print(f"   3. Retry: python -m src.core.Models.cli install --model or use uv (recomended) uv run Main.py install --model {key}")
        # print(f"\n   Model location: {model_path}")

    elif isinstance(error, DownloadError):
        if error.recoverable:
            print(f"\n💡 {error.hint}")
            print(f"   Partial download saved. Re-run to resume.")
        else:
            print(f"\n💡 {error.hint}")
            print(f"   This is a required dependency. Please install it and try again.")
    else:
        print(f"\n💡 Check your internet connection and disk space.")
        print(f"   Error details: {str(error)}")
