"""
ContextResolver.py — Reconstruct the full de-duplicated text context for a chunk.

Accepts either a Chunk dataclass or a dict. If dict, lazy-fetches neighbors
and optionally hydrates the central chunk from storage if fields are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from src.features.Ingestion.Chunking.ChunkerUtils import (
    Chunk,
    OverlapDirection,
    split_sentences,
)
from src.core.Logger.DarwinLogger import darwin, LogType


_OVERLAP_SEARCH_CAP = 10


@dataclass
class FullContext:
    text: str
    chunks: List[Chunk]
    contributing_indices: List[int]
    page: int
    page_end: int


def _normalise(s: str) -> str:
    return " ".join(s.split())


def _find_overlap_count(donor_text: str, receiver_text: str) -> int:
    donor_sents = split_sentences(donor_text)
    receiver_sents = split_sentences(receiver_text)
    cap = min(_OVERLAP_SEARCH_CAP, len(donor_sents), len(receiver_sents))
    for k in range(cap, 0, -1):
        donor_tail = [_normalise(s) for s in donor_sents[-k:]]
        receiver_head = [_normalise(s) for s in receiver_sents[:k]]
        if donor_tail == receiver_head:
            return k
    return 0


def _to_chunk(obj: Union[Chunk, Dict[str, Any]]) -> Chunk:
    """Normalize input to a Chunk. If dict, reconstruct via Chunk's own logic."""
    if isinstance(obj, Chunk):
        return obj
    if isinstance(obj, dict):
        # Ensure overlap_direction is an Enum, not a string
        od = obj.get("overlap_direction", "none")
        if isinstance(od, str):
            try:
                od = OverlapDirection(od)
            except ValueError:
                od = OverlapDirection.NONE
        # Build Chunk with defaults for missing fields
        return Chunk(
            text=obj.get("text", ""),
            clean=obj.get("clean", ""),
            index=obj.get("index", 0),
            source=obj.get("source", ""),
            token_count=obj.get("token_count", 0),
            heading=obj.get("heading", ""),
            page=obj.get("page", 0),
            page_end=obj.get("page_end", 0),
            overlap_direction=od,
            links=obj.get("links", []),
        )
    raise TypeError(f"Expected Chunk or dict, got {type(obj).__name__}")


def _fetch_chunks_by_index(
    store,
    filename: str,
    indices: List[int],
    columns: List[str] | None = None,
) -> Dict[int, Chunk]:
    """Fetch specific chunks by index. Returns {index: Chunk}."""
    if not indices:
        return {}
    
    cols = columns or ["index", "text", "page", "page_end", "overlap_direction", "links"]
    
    rows = store.query_records(
        filename=filename,
        columns=cols,
        index__in=indices,
    )
    
    chunks = [Chunk.from_dict(row, default_source=filename) for row in rows]
    return {c.index: c for c in chunks}


def get_full_context(
    chunk: Union[Chunk, Dict[str, Any]],
    store,
    filename: str,
    deduplicate: bool = True,
) -> FullContext:
    """
    Reconstruct full de-duplicated context from a Chunk or a dict.

    If given a dict with partial fields (e.g., from vector search results),
    the central chunk is hydrated. Neighbors are always fetched from storage.
    """
    # Normalize input to Chunk
    center = _to_chunk(chunk)
    direction = center.overlap_direction

    # Fast path — no neighbors needed
    if direction == OverlapDirection.NONE:
        return FullContext(
            text=center.text,
            chunks=[center],
            contributing_indices=[center.index],
            page=center.page,
            page_end=center.page_end,
        )

    needs_prev = direction in (OverlapDirection.LEFT, OverlapDirection.BOTH)
    needs_next = direction in (OverlapDirection.RIGHT, OverlapDirection.BOTH)

    # Determine all indices we need: center + neighbors
    all_indices: List[int] = [center.index]
    if needs_prev:
        all_indices.append(center.index - 1)
    if needs_next:
        all_indices.append(center.index + 1)

    # Fetch everything in one shot — storage handles deduplication of indices
    by_index = _fetch_chunks_by_index(store, filename, all_indices)

    # Hydrate center if the original dict was incomplete (e.g., missing text)
    if center.index in by_index and (
        not center.text or center.text != by_index[center.index].text
    ):
        center = by_index[center.index]

    prev_chunk = by_index.get(center.index - 1) if needs_prev else None
    next_chunk = by_index.get(center.index + 1) if needs_next else None

    # Paranoid warnings
    if needs_prev and prev_chunk is None:
        darwin.log(
            f"ContextResolver: missing prev chunk {center.index - 1} for {filename}",
            LogType.WARNING,
        )
    if needs_next and next_chunk is None:
        darwin.log(
            f"ContextResolver: missing next chunk {center.index + 1} for {filename}",
            LogType.WARNING,
        )

    # Build contributors list
    contributors: List[Chunk] = []
    if prev_chunk is not None:
        contributors.append(prev_chunk)
    contributors.append(center)
    if next_chunk is not None:
        contributors.append(next_chunk)

    if not deduplicate:
        return FullContext(
            text="\n\n".join(c.text for c in contributors),
            chunks=contributors,
            contributing_indices=[c.index for c in contributors],
            page=contributors[0].page,
            page_end=contributors[-1].page_end,
        )

    # Deduplicate seams
    sentence_windows: List[List[str]] = [
        split_sentences(c.text) for c in contributors
    ]
    for i in range(len(sentence_windows) - 1):
        k = _find_overlap_count(contributors[i].text, contributors[i + 1].text)
        if k > 0:
            sentence_windows[i] = sentence_windows[i][:-k]

    all_sentences = [s for window in sentence_windows for s in window]
    stitched = " ".join(all_sentences)

    return FullContext(
        text=stitched,
        chunks=contributors,
        contributing_indices=[c.index for c in contributors],
        page=contributors[0].page,
        page_end=contributors[-1].page_end,
    )