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
import time


from pathlib import Path
from typing import Any


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


def reconstruct_records(rows: list[dict[str, Any]], cls: Any = None) -> list[Any]:
    """
    Reconstruct objects from plain dict rows using duck typing.
    Runtime reconstruction happens OUTSIDE the storage layer, but this helper
    is provided for convenience in the retrieval pipeline.
    """
    if cls is None or not rows:
        return rows

    # Pydantic v2
    if hasattr(cls, "model_validate") and callable(cls.model_validate):
        return [cls.model_validate(row) for row in rows]

    # Pydantic v1
    if hasattr(cls, "parse_obj") and callable(cls.parse_obj):
        return [cls.parse_obj(row) for row in rows]

    # Dataclass
    dc_fields = getattr(cls, "__dataclass_fields__", None)
    if dc_fields is not None:
        valid = set(dc_fields)
        return [
            cls(**{k: v for k, v in row.items() if k in valid})
            for row in rows
        ]

    # attrs
    attrs_attrs = getattr(cls, "__attrs_attrs__", None)
    if attrs_attrs is not None:
        valid = {a.name for a in attrs_attrs}
        return [
            cls(**{k: v for k, v in row.items() if k in valid})
            for row in rows
        ]

    # Generic fallback: inspect __init__ signature to avoid passing unknown kwargs
    import inspect
    try:
        sig = inspect.signature(cls.__init__)
        valid = {
            p.name for p in sig.parameters.values()
            if p.name != "self" and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        return [
            cls(**{k: v for k, v in row.items() if k in valid})
            for row in rows
        ]
    except (ValueError, TypeError):
        pass

    # Last resort: pass all kwargs and hope for the best
    return [cls(**row) for row in rows]




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



ARTIFACT_MANIFEST = {
    "images": {"dir": "images/"},
    "markdown": {"source": "markdown.md", "display": "display-markdown.md"},
    "chunks": {"file": "chunks.parquet"},
    "vector": {"index": "vectors/vector.usearch", "meta": "vectors/vector.meta.json"},
    "embeddings": {"file": "vectors/embeddings.f16.npy", "index_map": "vectors/embeddings.index_map.json"},
    "bm25": {"index": "bm25.npz", "meta": "bm25_meta.json", "tokens": "bm25_tokens.parquet"},
    "stopwords": {"file": "stopwords.json"},
    "graph": {"nodes": "graph/nodes.parquet", "edges": "graph/edges.parquet"},
    "graph_vis": {"file": "graph/graph.html"},
    "wiki": {"file": "wiki/wiki.parquet"},
    "meta": {"file": "meta.json"},
}
