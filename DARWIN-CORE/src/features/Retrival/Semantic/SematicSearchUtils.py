"""
src/core/Search/Sematic/SematicSearchUtils.py
----------------------------------------------
Utilities and argument handling for semantic search.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from src.features.Retrival.lib import (SemanticSearchResult)



# ============================================================================
# Constants
# ============================================================================

DEFAULT_TOP_K =  80
DEFAULT_MIN_SCORE = 0.0
# ============================================================================
# Args
# ============================================================================

@dataclass
class SemanticSearchArgs:
    query: str
    files: list[str] = field(default_factory=list)
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    inspect: bool = False
    json_output: bool = False
    
# ============================================================================
# Result formatting
# ============================================================================

def format_results(results: list[SemanticSearchResult], json_output: bool = False) -> str:
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


