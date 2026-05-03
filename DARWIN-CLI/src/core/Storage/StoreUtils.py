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


# ── Default global config ──────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "version": "0.2.0",

    # ── Model preferences (user choices from onboarding) ──
    "embedding_model": None,       # "bge-m3" | "e5-small-v2" | "gritlm-7b"
    "reranker_model": None,
    "vision_model": None,          # "smolvlm" | "moondream2" | None
    "tokenizer_model": None,       # "spacy-en-core-md" | "spacy-en-core-lg" | None

    # ── LLM settings ──
    "llm_mode": "none",            # "local" | "api" | "none"
    "llm_model": None,
    "llm_api_key": None,

    # ── Graph ──
    "graph_threshold": 0.40,

    # ── Metadata ──
    "created_at": None,
    "updated_at": None,
}


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


def write_json(path: Path, data: dict, compact: bool = False) -> None:
    """Atomic-ish JSON write (write to temp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        if compact:
            json.dump(data, f, separators=(',', ':'), ensure_ascii=False)
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)  # atomic on most OSes


def read_json(path: Path) -> dict | None:
    """Read JSON, return None if missing or corrupted."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None