"""
src/core/Search/Reranking/RerankerUtils.py
------------------------------------------
Utilities for cross-encoder reranking: model loading,
argument handling, and result formatting.
"""

from dataclasses import dataclass
from typing import List, Optional, Any
from pathlib import Path


# ============================================================================
# Constants
# ============================================================================

# Model options (trade speed vs accuracy)
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Balanced (80MB)

MODEL_OPTIONS = {
    "fast": "cross-encoder/ms-marco-TinyBERT-L-2-v2",     # 15MB, fastest
    "balanced": "cross-encoder/ms-marco-MiniLM-L-6-v2",   # 80MB, default
    "accurate": "BAAI/bge-reranker-base",                 # 280MB, best
}

DEFAULT_TOP_K = 5
DEFAULT_POOL_SIZE = 50  # Candidates to rerank


# ============================================================================
# Args
# ============================================================================

@dataclass
class RerankingArgs:
    query: str
    model: str = DEFAULT_MODEL
    top_k: int = DEFAULT_TOP_K
    pool_size: int = DEFAULT_POOL_SIZE
    inspect: bool = False
    json_output: bool = False
    file: Optional[str] = None


# ============================================================================
# Model loading (lazy, cached)
# ============================================================================

_cross_encoder_cache: dict = {}


def resolve_model_name(model_name: str) -> str:
    """Resolve shortcut names to full model IDs."""
    return MODEL_OPTIONS.get(model_name.lower(), model_name)


def load_cross_encoder(model_name: str = DEFAULT_MODEL):
    """Lazy-load cross-encoder model."""
    from src.lib.Terminal import TerminalUI
    
    model_name = resolve_model_name(model_name)
    
    if model_name in _cross_encoder_cache:
        return _cross_encoder_cache[model_name]
    
    try:
        from sentence_transformers import CrossEncoder
        
        t = TerminalUI.timer(f"load_reranker:{model_name.split('/')[-1]}")
        TerminalUI.dim(f"Loading {model_name}...")
        
        model = CrossEncoder(model_name)
        _cross_encoder_cache[model_name] = model
        
        t.done()
        return model
        
    except ImportError:
        TerminalUI.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        raise


# ============================================================================
# Result formatting
# ============================================================================

def format_reranked_results(results: list, json_output: bool = False) -> str:
    """Format reranked results for display or JSON export."""
    import json
    
    if json_output:
        output = []
        for r in results:
            output.append({
                "final_score": round(r.final_score, 4),
                "original_score": round(r.original_score, 4) if r.original_score else None,
                "rerank_score": round(r.rerank_score, 4),
                "source": r.source,
                "page": r.page,
                "heading": r.heading,
                "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
            })
        return json.dumps(output, indent=2)
    
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] Final={r.final_score:.3f} | Rerank={r.rerank_score:.3f} | {r.source} | p{r.page}")
        if r.heading:
            lines.append(f"    └─ {r.heading}")
        preview = r.text[:200].replace('\n', ' ').strip()
        lines.append(f"       {preview}...")
        lines.append("")
    
    return "\n".join(lines)


def to_llm_context(results: list, max_chunks: int = 5) -> str:
    """Convert reranked results to LLM-ready context string."""
    context_parts = []
    
    for r in results[:max_chunks]:
        part = f"[Document: {r.source}, Page: {r.page}]"
        if r.heading:
            part += f"\nHeading: {r.heading}"
        part += f"\nContent: {r.text}\n---"
        context_parts.append(part)
    
    return "\n".join(context_parts)


