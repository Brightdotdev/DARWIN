"""
ContextResolver.py — Reconstruct the full de-duplicated text context for a chunk.

Why this exists
---------------
_split_with_overlap() intentionally repeats `overlap` sentences between
adjacent chunks so each chunk is retrievable on its own.  When you want
to *show* the full context to an LLM (not embed it), those repeated
sentences must be collapsed back into one continuous passage.

The OverlapDirection on each chunk tells us exactly which neighbours to
pull in:

    NONE   → chunk is self-contained, return as-is
    LEFT   → chunk opens with sentences repeated from chunk[index-1]'s tail
    RIGHT  → chunk closes with sentences repeated at chunk[index+1]'s head
    BOTH   → both neighbours needed

Usage
-----
    from src.features.Storage import DarwinStore
    from src.features.Chunking.ContextResolver import get_full_context

    store  = DarwinStore(project_dir)
    chunk  = ...   # whatever chunk came back from retrieval
    result = get_full_context(chunk, store, filename="biology.pdf")

    print(result.text)                   # de-duplicated full passage
    print(result.page)                   # start page of the passage
    print(result.page_end)               # end page of the passage
    print(result.contributing_indices)   # e.g. [4, 5, 6]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from src.features.Chunking.ChunkerUtils import (
    Chunk,
    OverlapDirection,
    _split_sentences,
)

if TYPE_CHECKING:
    from src.features.Storage import DarwinStore


# How many sentences back we search when detecting the overlap seam.
# The original `overlap` param maxes out at ~8 (base 5 + overlap_add 3),
# so 10 is a safe ceiling without being expensive.
_OVERLAP_SEARCH_CAP = 10


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FullContext:
    """
    The reconstructed passage for a chunk, with duplicated overlap removed.

    Attributes
    ----------
    text:
        De-duplicated plain text of the full passage.
        Pass this directly to the LLM as context.
    chunks:
        The Chunk objects that contributed, in index order.
        Useful if you need .clean, .links, or any other chunk field.
    contributing_indices:
        Convenience list of the chunk indices — same order as `chunks`.
    page:
        First page of the passage (from the earliest contributing chunk).
    page_end:
        Last page of the passage (from the latest contributing chunk).
    """
    text:                 str
    chunks:               List[Chunk]
    contributing_indices: List[int]
    page:                 int
    page_end:             int


# ─────────────────────────────────────────────────────────────────────────────
# Overlap seam detection
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(s: str) -> str:
    """Collapse whitespace for comparison — avoids false mismatches."""
    return " ".join(s.split())


def _find_overlap_count(donor: Chunk, receiver: Chunk) -> int:
    """
    Find how many sentences from the tail of `donor` are duplicated at
    the head of `receiver`.

    _split_with_overlap() carries window_orig[-overlap:] into the next
    window verbatim, so the duplicated sentences are identical in both
    chunks (barring whitespace normalisation).

    Returns 0 if no overlap is detected — safe fallback, just concatenates.
    """
    donor_sents    = _split_sentences(donor.text)
    receiver_sents = _split_sentences(receiver.text)

    cap = min(_OVERLAP_SEARCH_CAP, len(donor_sents), len(receiver_sents))

    for k in range(cap, 0, -1):
        donor_tail    = [_normalise(s) for s in donor_sents[-k:]]
        receiver_head = [_normalise(s) for s in receiver_sents[:k]]
        if donor_tail == receiver_head:
            return k

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_full_context(
    chunk:       Chunk,
    store:       "DarwinStore",
    filename:    str,
    deduplicate: bool = True,
) -> FullContext:
    """
    Return the full de-duplicated passage for `chunk`.

    Parameters
    ----------
    chunk:
        The chunk you want full context for — typically whatever came back
        from vector search or BM25 retrieval.

    store:
        A DarwinStore instance pointed at the right project directory.
        Needed to call store.load_chunks(filename) to get the neighbours.
        You always have one of these in the calling code already — passing
        it here avoids re-opening the store or threading a global.

    filename:
        The source filename the chunk belongs to (e.g. "biology.pdf").
        Needed because the store keys chunk lists by filename, and a single
        store instance can hold chunks for many documents.

    deduplicate:
        Default True — strips the repeated overlap sentences at seam
        boundaries so the final text reads as one continuous passage.
        Set to False for debugging: the raw concatenation shows the overlap
        sentences appearing twice, making the seam visible.

    Returns
    -------
    FullContext
        .text                  — the stitched passage, ready for the LLM
        .chunks                — contributing Chunk objects in index order
        .contributing_indices  — their indices (e.g. [4, 5, 6])
        .page / .page_end      — page span of the full passage
    """
    direction = chunk.overlap_direction

    # Fast path — standalone chunk, nothing to stitch
    if direction == OverlapDirection.NONE:
        return FullContext(
            text                 = chunk.text,
            chunks               = [chunk],
            contributing_indices = [chunk.index],
            page                 = chunk.page,
            page_end             = chunk.page_end,
        )

    # Load the full chunk list for this file.
    # store.load_chunks returns the list directly, ordered by index.
    all_chunks: List[Chunk] = store.load_chunks(filename)

    # Build index → Chunk map for O(1) neighbour lookup.
    # We can't assume list position == chunk.index because merge_small()
    # can collapse chunks and leave gaps in the index sequence.
    by_index = {c.index: c for c in all_chunks}

    prev_chunk: Optional[Chunk] = by_index.get(chunk.index - 1)
    next_chunk: Optional[Chunk] = by_index.get(chunk.index + 1)

    needs_prev = direction in (OverlapDirection.LEFT,  OverlapDirection.BOTH)
    needs_next = direction in (OverlapDirection.RIGHT, OverlapDirection.BOTH)

    # Collect contributing chunks in order
    contributors: List[Chunk] = []
    if needs_prev and prev_chunk is not None:
        contributors.append(prev_chunk)
    contributors.append(chunk)
    if needs_next and next_chunk is not None:
        contributors.append(next_chunk)

    if not deduplicate:
        return FullContext(
            text                 = "\n\n".join(c.text for c in contributors),
            chunks               = contributors,
            contributing_indices = [c.index for c in contributors],
            page                 = contributors[0].page,
            page_end             = contributors[-1].page_end,
        )

    # ── Stitch with deduplication ─────────────────────────────────────────
    #
    # Build a sentence list per contributor, then trim the duplicated seam
    # between each adjacent pair before flattening.
    #
    # Example with overlap=2, LEFT direction:
    #   prev  sentences: [A, B, C, D, E]     ← D,E were donated rightward
    #   chunk sentences: [D, E, F, G, H]     ← D,E are the borrowed head
    #   after trim:      [A, B, C] + [D, E, F, G, H]
    #   stitched:        [A, B, C, D, E, F, G, H]   ✓  no duplication

    sentence_windows: List[List[str]] = [
        _split_sentences(c.text) for c in contributors
    ]

    # Trim the seam between each adjacent pair of windows
    for i in range(len(sentence_windows) - 1):
        k = _find_overlap_count(contributors[i], contributors[i + 1])
        if k > 0:
            # Drop the duplicated tail from the left window.
            # The right window keeps its head — that IS the overlap text,
            # appearing exactly once in the final passage.
            sentence_windows[i] = sentence_windows[i][:-k]

    all_sentences = [s for window in sentence_windows for s in window]
    stitched_text = " ".join(all_sentences)

    return FullContext(
        text                 = stitched_text,
        chunks               = contributors,
        contributing_indices = [c.index for c in contributors],
        page                 = contributors[0].page,
        page_end             = contributors[-1].page_end,
    )