"""
src/core/Models/ModelConfig.py
-------------------------------
Persistent configuration for which model is active per category.
Atomic saves with file locking.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from filelock import FileLock
from src.lib.AppPaths import default_models_config_dir


CONFIG_FILENAME = "models.json"


def _config_path() -> Path:
    return default_models_config_dir() / CONFIG_FILENAME


def _lock_path() -> Path:
    return default_models_config_dir() / (CONFIG_FILENAME + ".lock")


def load_active_models() -> Dict[str, Optional[str]]:
    """Return {category: model_id} from config file. Missing keys → None."""
    path = _config_path()
    if not path.exists():
        return {}

    lock = FileLock(str(_lock_path()))
    try:
        with lock:
            try:
                data = json.loads(path.read_text())
                from src.core.Models.ModelRegistry import ModelCategory
                config = {}
                for cat in ModelCategory:
                    config[cat.value] = data.get(cat.value, None)
                return config
            except Exception:
                return {}
    except Exception:
        return {}


def save_active_models(config: Dict[str, Optional[str]]) -> None:
    """Write the full active models config to disk atomically."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lock = FileLock(str(_lock_path()))
    with lock:
        path.write_text(json.dumps(config, indent=2))


def set_active_model(category: str, model_id: Optional[str]) -> None:
    """Update a single category and persist."""
    config = load_active_models()
    config[category] = model_id
    save_active_models(config)


def get_active_model(category: str) -> Optional[str]:
    """Return the currently active model_id for a category, or None."""
    return load_active_models().get(category)


# if __name__ == "__main__":
#     print(load_active_models())