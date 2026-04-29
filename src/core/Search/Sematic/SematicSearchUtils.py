"""
src/core/Search/Sematic/SematicSearchUtils.py
----------------------------------------------
Utilities and argument handling for semantic search.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from src.core.Chunking.ChunkerUtils import ChunkLink, OverlapDirection
from src.core.Chunking.Chunker import Chunk

from enum import Enum


# ============================================================================
# Data model
# ============================================================================


class OverlapDirection(str, Enum):
    """
    Describes which side(s) of a chunk carry overlap sentences.

    NONE   — no overlap; chunk is self-contained
    LEFT   — chunk *opens* with sentences repeated from the tail of the
             previous chunk  (this chunk "looks left")
    RIGHT  — chunk *closes* with sentences repeated at the head of the
             next chunk  (this chunk "donates right")
    BOTH   — interior window: received from the left AND donates to the right

    Use str-Enum so the value serialises cleanly to JSON as a plain string
    ("none", "left", "right", "both") without needing a custom encoder.
    """
    NONE  = "none"
    LEFT  = "left"
    RIGHT = "right"
    BOTH  = "both"



# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkLink:
    """A hyperlink scoped to a chunk."""
    url:  str
    text: str = ""



@dataclass
class SematicSearchResult:
    """Single search result with all metadata needed for RAG."""
    text: str           # Original text (for LLM context)
    clean: str          # Cleaned text (what was embedded)
    score: float        # Cosine similarity (0-1)
    source: str         # Document filename
    page: int           # Page number
    heading: str        # Section heading
    chunk_index: int    # Original chunk position
    is_overlap: bool    # Whether this chunk overlaps with adjacent chunks
    links: List[tuple[str, str]] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return f"SematicSearchResult(score={self.score:.3f}, source='{self.source}', page={self.page})"


# ============================================================================
# Constants
# ============================================================================

DEFAULT_TOP_K =  80
DEFAULT_MIN_SCORE = 0.0


# ============================================================================
# Args
# ============================================================================

@dataclass
class SearchArgs:
    query: str
    file: Optional[str] = None
    project: Optional[str] = None
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    inspect: bool = False
    json_output: bool = False


# ============================================================================
# Chunk reconstruction
# ============================================================================

def reconstruct_chunks(data: dict, source: str) -> List[Chunk]:
    """Reconstruct Chunk objects from stored .npz data."""
    chunks = []

    texts = data.get("text", [])
    headings = data.get("headings", [])
    pages = data.get("pages", [])
    page_ends = data.get("page_ends", pages)
    overlap_directions = data.get("overlap_direction", [])
    links_data = data.get("links", [[] for _ in texts])

    for i, text in enumerate(texts):
        # Restore overlap direction enum safely
        direction_raw = (
            overlap_directions[i]
            if i < len(overlap_directions)
            else "none"
        )

        try:
            direction = OverlapDirection(direction_raw)
        except ValueError:
            direction = OverlapDirection.NONE

        # Restore links
        links = []
        if i < len(links_data):
            for l in links_data[i]:
                links.append(ChunkLink(**l))

        # Build chunk
        chunks.append(
            Chunk(
                text=text,
                clean=text,  # same unless you stored separately
                index=i,
                source=source,
                token_count=0,
                heading=headings[i] if i < len(headings) else "",
                page=int(pages[i]) if i < len(pages) else 0,
                page_end=int(page_ends[i]) if i < len(page_ends) else 0,
                overlap_direction=direction,
                links=links,
            )
        )

    return chunks


# ============================================================================
# Result formatting
# ============================================================================

def format_results(results: list[SematicSearchResult], json_output: bool = False) -> str:
    """Format search results for display or JSON export."""
    import json
    
    if json_output:
        output = []
        for r in results:
            output.append({
                "score": round(r.score, 4),
                "source": r.source,
                "page": r.page,
                "heading": r.heading,
                "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
            })
        return json.dumps(output, indent=2)
    
    # Rich terminal output
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.score:.3f} | {r.source} | page {r.page}")
        if r.heading:
            lines.append(f"    └─ {r.heading}")
        preview = r.text[:200].replace('\n', ' ').strip()
        lines.append(f"       {preview}...")
        lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# Export for LLM context
# ============================================================================

def to_llm_context(results: List[SematicSearchResult], max_chunks: int = 5) -> str:
    """
    Convert search results to LLM-ready context string.
    
    Format:
        [Document: filename, Page: X]
        Heading: section title
        Content: chunk text
        ---
    """
    context_parts = []
    
    for r in results[:max_chunks]:
        part = f"[Document: {r.source}, Page: {r.page}]"
        if r.heading:
            part += f"\nHeading: {r.heading}"
        part += f"\nContent: {r.text}\n---"
        context_parts.append(part)
    
    return "\n".join(context_parts)