from __future__ import annotations

import hashlib
import inspect
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


def normalize_record(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump()
    if hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    dc_fields = getattr(obj, "__dataclass_fields__", None)
    if dc_fields is not None:
        return {k: getattr(obj, k) for k in dc_fields if not k.startswith("_")}
    attrs_attrs = getattr(obj, "__attrs_attrs__", None)
    if attrs_attrs is not None:
        return {a.name: getattr(obj, a.name) for a in attrs_attrs}
    if hasattr(obj, "__slots__"):
        slots = obj.__slots__
        if isinstance(slots, str):
            slots = [slots]
        result = {}
        for slot in slots:
            if not slot.startswith("_"):
                try:
                    result[slot] = getattr(obj, slot)
                except AttributeError:
                    pass
        if result:
            return result
    if hasattr(obj, "__dict__"):
        return vars(obj)
    raise TypeError(f"Cannot normalize {type(obj).__name__}")


def _deep_serialize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_deep_serialize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _deep_serialize(v) for k, v in obj.items()}
    try:
        return _deep_serialize(normalize_record(obj))
    except TypeError:
        return str(obj)


def reconstruct_records(rows: list[dict[str, Any]], cls: Any = None) -> list[Any]:
    if cls is None or not rows:
        return rows
    if hasattr(cls, "model_validate") and callable(cls.model_validate):
        return [cls.model_validate(row) for row in rows]
    if hasattr(cls, "parse_obj") and callable(cls.parse_obj):
        return [cls.parse_obj(row) for row in rows]
    dc_fields = getattr(cls, "__dataclass_fields__", None)
    if dc_fields is not None:
        valid = set(dc_fields)
        return [cls(**{k: v for k, v in row.items() if k in valid}) for row in rows]
    attrs_attrs = getattr(cls, "__attrs_attrs__", None)
    if attrs_attrs is not None:
        valid = {a.name for a in attrs_attrs}
        return [cls(**{k: v for k, v in row.items() if k in valid}) for row in rows]
    try:
        sig = inspect.signature(cls.__init__)
        valid = {
            p.name for p in sig.parameters.values()
            if p.name != "self" and p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        return [cls(**{k: v for k, v in row.items() if k in valid}) for row in rows]
    except (ValueError, TypeError):
        pass
    return [cls(**row) for row in rows]