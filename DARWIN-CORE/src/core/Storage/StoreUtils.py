"""
src/core/Storage/StoreUtils.py
------------------------------
Constants, helpers, and JSON utilities for DarwinStore.
"""

from pathlib import Path
import hashlib
import re
from datetime import datetime, timezone
import json
from typing import Optional
import time


import os
import tempfile
from pathlib import Path
from typing import Any,Iterable




# ── Model name cache ───────────────────────────────────────────────────────

_model_name_cache: Optional[str] = None


def get_current_model_name() -> str:
    """Resolve the currently active embedding model name (full model_id)."""
    global _model_name_cache
    if _model_name_cache is not None:
        return _model_name_cache
    try:
        from src.core.Models import ModelManager, ModelCategory
        loader = ModelManager.get_loader()
        _model_name_cache = loader.get_model_key(ModelCategory.EMBEDDING)
        return _model_name_cache
    except Exception as e:
        from src.core.Logger.DarwinLogger import darwin, LogType
        darwin.log(f"Failed to detect current model: {e}", LogType.ERROR)
        return "unknown"


def invalidate_model_cache() -> None:
    """Clear the cached model name so the next call re-detects."""
    global _model_name_cache
    _model_name_cache = None


# ── Helpers ────────────────────────────────────────────────────────────────

def sanitize(name: str) -> str:
    """Turn 'biology.pdf' into 'biology_pdf'."""
    name = Path(name).name
    name = re.sub(r"[^\w]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def file_hash(path: str | Path) -> str:
    """SHA-256 hash for quick change detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()



def write_json(path: Path, data: Any, compact: bool = False, max_retries: int = 3) -> None:
    """Write JSON atomically with retry on Windows PermissionError."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    json_str = json.dumps(data, indent=None if compact else 2, ensure_ascii=False)

    for attempt in range(max_retries):
        try:
            tmp.write_text(json_str, encoding="utf-8")
            tmp.replace(path)
            return
        except (PermissionError, OSError) as e:
            if attempt == max_retries - 1:
                # Fallback: direct write (non-atomic but better than failure)
                from src.core.Logger.DarwinLogger import darwin, LogType
                darwin.log(f"Atomic replace failed for {path}, using direct write: {e}", LogType.WARNING)
                path.write_text(json_str, encoding="utf-8")
                if tmp.exists():
                    tmp.unlink()
                return
            time.sleep(0.1 * (attempt + 1))


def read_json(path: Path) -> dict | None:
    """Read JSON, return None if missing or corrupted."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None



