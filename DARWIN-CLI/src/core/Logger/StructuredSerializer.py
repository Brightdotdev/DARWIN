"""
Recursive structured serializer that produces a JSON‑safe representation
while preserving schema / type information. Fields marked as sensitive
are replaced with "[REDACTED]".
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Set


def serialize(obj: Any, _seen: Set[int] | None = None) -> Any:
    """
    Convert *obj* into a nested dict/list structure with type markers.

    - dataclass instances   →  { "__darwin_type__": "<class name>", <field>: <value>, … }
    - lists                 →  { "__darwin_type__": "list", "items": [ … ] }
    - plain dicts           →  { <key>: <value>, … }   (no type marker added)
    - other built‑in types  →  returned unchanged
    - sensitive fields      →  value replaced with "[REDACTED]"
    """
    if _seen is None:
        _seen = set()

    obj_id = id(obj)
    if obj_id in _seen:
        return {"__darwin_type__": "cycle", "ref": str(obj_id)}
    _seen.add(obj_id)

    # --- dataclass -----------------------------------------------------------
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = obj.__class__
        fields = dataclasses.fields(obj)
        result: Dict[str, Any] = {"__darwin_type__": cls.__name__}
        for f in fields:
            value = getattr(obj, f.name)
            if f.metadata.get("sensitive"):
                value = "[REDACTED]"
            else:
                value = serialize(value, _seen)
            result[f.name] = value
        return result

    # --- list ----------------------------------------------------------------
    if isinstance(obj, list):
        return {
            "__darwin_type__": "list",
            "items": [serialize(item, _seen) for item in obj],
        }

    # --- dict ----------------------------------------------------------------
    if isinstance(obj, dict):
        return {k: serialize(v, _seen) for k, v in obj.items()}

    # --- primitives ----------------------------------------------------------
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # --- fallback ------------------------------------------------------------
    return str(obj)