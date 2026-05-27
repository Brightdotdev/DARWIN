"""
Recursively extracts schema definitions for dataclass types encountered
in a payload. Caches results per class for performance.

Schema format:
{
    "<ClassName>": {
        "fields": [
            {
                "name": ...,
                "type": ...,
                "default": ...,
                "metadata": { ... }  # includes "sensitive": true if set
            },
            ...
        ],
        "doc": "<class docstring>"
    }
}
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Set

# Module‑level cache: type -> schema definition
_schema_cache: Dict[type, Dict[str, Any]] = {}


def _type_repr(t: Any) -> str:
    """Return a clean string representation for a type annotation."""
    if hasattr(t, "__name__"):
        return t.__name__
    return str(t).replace("typing.", "")


def _build_schema(cls: type) -> Dict[str, Any]:
    """Extract schema for a single dataclass type and cache it."""
    if cls in _schema_cache:
        return _schema_cache[cls]

    fields_list = []
    for f in dataclasses.fields(cls):
        metadata = f.metadata if f.metadata else {}
        fields_list.append({
            "name": f.name,
            "type": _type_repr(f.type),
            "default": f.default if f.default is not dataclasses.MISSING else None,
            "metadata": metadata,
        })
    doc = cls.__doc__
    schema = {
        "fields": fields_list,
        "doc": doc.strip() if doc else "",
    }
    _schema_cache[cls] = schema
    return schema


def extract_schemas(obj: Any, _seen: Set[int] | None = None) -> Dict[str, Any]:
    """
    Recursively traverse *obj* and collect schema definitions for all
    dataclass types encountered.

    Returns a dict mapping class name -> schema definition.
    """
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return {}
    _seen.add(obj_id)

    schemas: Dict[str, Any] = {}

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = obj.__class__
        class_name = cls.__name__
        if class_name not in schemas:
            schemas[class_name] = _build_schema(cls)
        # Recurse into field values to discover nested dataclasses
        for f in dataclasses.fields(cls):
            val = getattr(obj, f.name)
            nested = extract_schemas(val, _seen)
            for k, v in nested.items():
                if k not in schemas:
                    schemas[k] = v

    elif isinstance(obj, list):
        for item in obj:
            nested = extract_schemas(item, _seen)
            for k, v in nested.items():
                if k not in schemas:
                    schemas[k] = v

    elif isinstance(obj, dict):
        for val in obj.values():
            nested = extract_schemas(val, _seen)
            for k, v in nested.items():
                if k not in schemas:
                    schemas[k] = v

    return schemas