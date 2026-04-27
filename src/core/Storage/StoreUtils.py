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


# ── Helpers ────────────────────────────────────────────────────────────────

def sanitize(name: str) -> str:
    """Turn 'biology.pdf' into 'biology_pdf'."""
    name = Path(name).name
    name = re.sub(r"[^\w]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def file_hash(path: str | Path) -> str:
    """MD5 hash for quick change detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    """Atomic-ish JSON write (write to temp, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
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