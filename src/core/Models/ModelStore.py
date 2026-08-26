# src/core/Models/Lib/ModelStore.py
"""
Lightweight model store – scans the filesystem to find installed models.
Does not depend on ModelManager.
"""

from pathlib import Path
from typing import List, Optional

from core.Models.Lib.ModelRegistry import _ID_INDEX, find_spec, ModelCategory
from lib.AppPaths import default_models_dir


class ModelStore:
    """Scans the models directory to list installed models and resolve paths."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or default_models_dir()

    def list_installed(self) -> List[str]:
        """
        Return all model IDs that have files on disk.
        Scans: base_dir / category_name / safe_model_id
        """
        installed = []
        for model_id in _ID_INDEX.keys():
            path = self.model_path(model_id)
            if path and path.exists():
                installed.append(model_id)
        return installed

    def model_path(self, model_id: str) -> Optional[Path]:
        """
        Return the directory path for the given model ID, or None if not found.
        The directory name is the model_id with '/' replaced by '--'.
        """
        spec = find_spec(model_id)
        if spec is None:
            return None
        safe_name = model_id.replace('/', '--')
        category_dir = self.base_dir / spec.category.name.lower()
        return category_dir / safe_name

    def list_by_category(self, category: ModelCategory) -> List[str]:
        """List installed model IDs in a specific category."""
        installed = []
        for model_id in _ID_INDEX.keys():
            spec = find_spec(model_id)
            if spec and spec.category == category:
                path = self.model_path(model_id)
                if path and path.exists():
                    installed.append(model_id)
        return installed

    def get_category_path(self, category: ModelCategory) -> Path:
        """Return the base directory for a given category."""
        return self.base_dir / category.name.lower()