from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sanitize(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def write_json(path: Path, data: Any, compact: bool = False, max_retries: int = 3) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    json_str = json.dumps(data, indent=None if compact else 2, ensure_ascii=False)

    for attempt in range(max_retries):
        try:
            tmp.write_text(json_str, encoding="utf-8")
            tmp.replace(path)
            return
        except (PermissionError, OSError) as e:
            if attempt == max_retries - 1:
                path.write_text(json_str, encoding="utf-8")
                if tmp.exists():
                    tmp.unlink()
                return
            time.sleep(0.1 * (attempt + 1))


def to_dict(obj: Any) -> dict:
    """Convert supported objects (dataclass, dict) to a dict."""
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Cannot convert {type(obj).__name__} to dict")


def get_current_embedding_model() -> str:
    from src.core.Models.ModelConfig import get_active_model
    from src.core.Models.Lib.ModelSpec import ModelCategory
    return get_active_model(ModelCategory.EMBEDDING.name.lower()) or ""


def get_current_model_dimensions() -> int:
    try:
        from src.core.Models.EmbeddingConfig import get_preset
        key = get_current_embedding_model()
        presets = get_preset(key)
        if presets:
            return presets.dimensions
        _DIMENSIONS = {
            "intfloat/e5-small-v2": 384,
            "BAAI/bge-m3": 1024,
            "GritLM/GritLM-7B": 4096,
        }
        return _DIMENSIONS.get(key, 0)
    except Exception:
        return 0