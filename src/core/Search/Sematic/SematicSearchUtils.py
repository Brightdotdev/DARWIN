

from src.core.Chunking.Chunker import Chunk
from typing import List

from dataclasses import dataclass,field


# ============================================================================
# Data model
# ============================================================================

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
    is_overlap: bool 
    links: List[tuple[str, str]] = field(default_factory=list)
    
    
    def __repr__(self) -> str:
        return f"SematicSearchResult(score={self.score:.3f}, source='{self.source}', page={self.page})"



def reconstruct_chunks(data: dict, source: str) -> List[Chunk]:
    """Reconstruct Chunk objects from stored .npz data."""
    chunks = []
    texts = data.get("text", [])
    headings = data.get("headings", [])
    pages = data.get("pages", [])
    
    for i, text in enumerate(texts):
        # Note: clean text is what was embedded, so text == clean here
        chunks.append(Chunk(
            text=text,
            clean=text,
            index=i,
            source=source,
            token_count=0,  # Not stored in .npz, optional
            heading=headings[i] if i < len(headings) else "",
            page=int(pages[i]) if i < len(pages) else 0,
            page_end=int(pages[i]) if i < len(pages) else 0,
            is_overlap=False,
        ))
    
    return chunks



"""
src/core/Search/SearchUtils.py
------------------------------
Utilities and argument handling for semantic search.
"""

from dataclasses import dataclass
from typing import Optional



# ============================================================================
# Constants
# ============================================================================

DEFAULT_TOP_K = 5
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
# Result formatting
# ============================================================================

def format_results(results: list, json_output: bool = False) -> str:
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

def to_llm_context(results: List, max_chunks: int = 5) -> str:
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
