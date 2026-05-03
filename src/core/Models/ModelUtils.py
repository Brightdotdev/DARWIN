

from pathlib import Path
from src.core.Hardware.SystemDetector import  default_models_dir
from src.core.Models.ModelRegistry import get_spec


# ═══════════════════════════════════════════════════════════════════════════
# Config sync
# ═══════════════════════════════════════════════════════════════════════════

def _get_config_path() -> Path:
    """Return the path to the app config file."""
    return default_models_dir().parent / "config.json"


def load_config() -> dict:
    """Load the persisted model preferences from config.json."""
    config_path = _get_config_path()
    if not config_path.exists():
        return {}
    try:
        import json
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️  Could not read config ({config_path}): {e}")
        return {}



def save_config(config: dict) -> None:
    """
    Persist model preferences to config.json.

    Called after every install, uninstall, or switch so that the main
    application always loads the right model on the next startup without
    manual intervention.
    """
    import json
    config_path = _get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Config saved → {config_path}")
    except Exception as e:
        print(f"Could not save config ({config_path}): {e}")



def sync_model_to_config(category: str, key: str) -> None:
    """
    Update the active model for a category in config.json.
    Stores the full model_id (e.g., 'BAAI/bge-m3').
    """
    config = load_config()
    spec = get_spec(key)
    config[category] = spec.model_id  # Store model_id, not key
    save_config(config)

def remove_model_from_config(category: str, key: str) -> None:
    """
    Clear a model from config.json on uninstall.
    """
    config = load_config()
    spec = get_spec(key)
    if config.get(category) == spec.model_id:
        config.pop(category, None)
        save_config(config)
        print(f"   ℹ️  Cleared '{category}' from config (was '{spec.model_id}').")


