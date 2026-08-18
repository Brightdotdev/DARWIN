"""
Structured chunker — converts StructuredDocument records into Chunks.

Each record becomes one Chunk. No paragraph splitting.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.core.Pipeline.Ingestion.Chunking.lib import (
    Chunk,
    ChunkLink,
    OverlapDirection,
)
from src.core.Pipeline.Ingestion.Parsing.StructuredParser import StructuredDocument


def _concatenate_fields(record: Dict[str, Any]) -> str:
    """
    Flatten a record into a single string for BM25 tokenization.
    Format: "key: value | key: value | ..."
    Nested objects use dot-notation.
    """
    parts = []
    for key, value in record.items():
        flat_value = _flatten_value(value)
        parts.append(f"{key}: {flat_value}")
    return " | ".join(parts)


def _flatten_value(value: Any, prefix: str = "") -> str:
    """Flatten a value to string, with dot-notation for nested objects."""
    if value is None:
        return ""
    if isinstance(value, dict):
        inner = []
        for k, v in value.items():
            inner.append(f"{prefix}{k}: {_flatten_value(v, prefix + k + '.')}")
        return ", ".join(inner)
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _best_label(record: Dict[str, Any]) -> str:
    """Extract a human-readable label from a record (for heading field)."""
    for key in ("id", "name", "title", "subject", "label", "key"):
        if key in record and record[key] is not None:
            return str(record[key])
    return ""


def chunk_structured(
    doc: StructuredDocument,
    index_mode: str = "structured",
) -> List[Chunk]:
    """
    Convert a StructuredDocument into a list of Chunks.
    Each record becomes one Chunk with fields populated.
    """
    chunks: List[Chunk] = []

    for i, record in enumerate(doc.records):
        # JSON string of the full record (for embedding)
        text = json.dumps(record, ensure_ascii=False, default=str)

        # Concatenated field values (for BM25)
        clean = _concatenate_fields(record)

        # Token count (rough: 1 token ≈ 4 chars for English)
        token_count = max(1, len(clean) // 4)

        # Best-effort label
        heading = _best_label(record)

        chunk = Chunk(
            text=text,
            clean=clean,
            index=i,
            source=doc.filename,
            token_count=token_count,
            heading=heading,
            page=0,
            page_end=0,
            overlap_direction=OverlapDirection.NONE,
            links=[],
            structural_type="record",
            fields=record,
            index_mode=index_mode,
        )

        chunks.append(chunk)

    return chunks
