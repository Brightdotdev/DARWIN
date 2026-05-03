"""
src/core/Search/Sematic/SematicSearchUtils.py
----------------------------------------------
Utilities and argument handling for semantic search.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from src.core.Chunking.ChunkerUtils import ChunkLink, OverlapDirection
from src.core.Chunking.lib import Chunk
from src.core.Search.lib import (SematicSearchResult)


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
    from src.core.Chunking.ChunkerUtils import Chunk, ChunkLink, OverlapDirection
    import json
    
    chunks = []
    n_chunks = len(data.get("text", []))

    for i in range(n_chunks):
        # Parse links from JSON
        links = []
        if 'links_json' in data and i < len(data['links_json']):
            links_json = data['links_json'][i]
            if links_json and links_json != 'null' and links_json != "[]":
                try:
                    if isinstance(links_json, str):
                        links_data = json.loads(links_json)
                    else:
                        links_data = links_json
                    
                    for l in links_data:
                        if isinstance(l, dict):
                            links.append(ChunkLink(
                                url=l.get('url', ''),
                                text=l.get('text', '')
                            ))
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Get overlap direction as Enum
        overlap_raw = "none"
        if 'overlap_direction' in data and i < len(data['overlap_direction']):
            overlap_raw = str(data['overlap_direction'][i])
        
        try:
            overlap_dir = OverlapDirection(overlap_raw)
        except ValueError:
            overlap_dir = OverlapDirection.NONE
        
        # Get page with fallback
        page = 0
        if 'pages' in data and i < len(data['pages']):
            page = int(data['pages'][i])
        
        # Get page_end with fallback
        page_end = page
        if 'pages_end' in data and i < len(data['pages_end']):
            page_end = int(data['pages_end'][i])
        
        # Get token_count with fallback
        token_count = 0
        if 'token_counts' in data and i < len(data['token_counts']):
            token_count = int(data['token_counts'][i])
        
        chunk = Chunk(
            text=str(data['text'][i]),
            clean=str(data.get('clean', data['text'])[i]),  # Fallback to text if clean missing
            index=i,
            source=source,
            token_count=token_count,
            heading=str(data['headings'][i]) if i < len(data['headings']) else "",
            page=page,
            page_end=page_end,
            overlap_direction=overlap_dir,
            links=links,
        )
        chunks.append(chunk)
    
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