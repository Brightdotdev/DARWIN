"""
src/core/Retrival/retrieval_utils.py
-------------------------------------
Shared utility functions for the retrieval module.
Eliminates duplicate implementations across submodules.
"""

from typing import Any, List

from core.Retrival.lib import ChunkLink, OverlapDirection


def normalize_links(links_raw: Any) -> List[ChunkLink]:
    """
    Normalize a raw links field into a list of ChunkLink objects.

    Handles:
    - list of dicts (from storage/JSON)
    - list of ChunkLink objects (already normalized)
    - string values (treated as URL with empty text)
    - None / missing fields
    """
    out = []
    for item in links_raw or []:
        if isinstance(item, dict):
            out.append(ChunkLink(url=item.get("url", ""), text=item.get("text", "")))
        elif isinstance(item, ChunkLink):
            out.append(item)
        else:
            out.append(ChunkLink(url=str(item), text=""))
    return out


def ensure_overlap_direction(val: Any) -> OverlapDirection:
    """
    Safely convert a value to OverlapDirection.
    Returns OverlapDirection.NONE for invalid or missing values.
    """
    if isinstance(val, OverlapDirection):
        return val
    if isinstance(val, str):
        try:
            return OverlapDirection(val)
        except ValueError:
            pass
    return OverlapDirection.NONE
