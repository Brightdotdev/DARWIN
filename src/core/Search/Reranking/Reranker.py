"""
src/core/Search/Reranking/Reranker.py
-------------------------------------
Cross-encoder reranking for improved search precision.

Why reranking?
--------------
Bi-encoders (semantic search) encode query and chunk separately — fast but
misses fine-grained query/chunk interaction.  Cross-encoders process the pair
together with full attention — slower but much more accurate at judging
relevance.  The standard pipeline is:

    fast retrieval (bi-encoder / BM25, pool_size=50-100)
        → cross-encoder rerank
            → top-k final results

Score blending
--------------
Raw cross-encoder logits are unbounded (roughly -10 to +10 for ms-marco
models).  The original retrieval score lives in a completely different range
(cosine 0-1, BM25 0-∞, RRF ~0.01-0.03).  Mixing them directly produces
garbage.

Fix: we sigmoid-normalize the rerank logit to [0,1] and also min-max
normalize the original score across the candidate set before blending:

    final = (1 - w) * norm_rerank + w * norm_original

This way both terms always live in [0,1] and the weight `w` actually means
something.  Set w=0 to use pure cross-encoder ordering.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.core.Search.Sematic.SematicSearchUtils import SematicSearchResult
    from src.core.Search.Keyword.KeywordSearch import KeywordSearchResult
    from src.core.Search.Hybrid.HybridSearchUtils import HybridSearchResult

from src.core.Search.Reranking.RerankerUtils import (
    load_cross_encoder,
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_POOL_SIZE,
)
from src.lib.Terminal import TerminalUI


# ============================================================================
# Data model
# ============================================================================

@dataclass
class RerankedResult:
    """A result that has been reranked by a cross-encoder."""
    text:           str
    clean:          str
    rerank_score:   float                # raw cross-encoder logit
    original_score: Optional[float]      # score from original retriever (pre-norm)
    final_score:    float                # blended normalized score
    source:         str
    page:           int
    page_end:       int
    heading:        str
    chunk_index:    int
    links:          List[Any] = field(default_factory=list)
    is_overlap:     bool = False

    def __repr__(self) -> str:
        return (
            f"RerankedResult(final={self.final_score:.3f}, "
            f"rerank={self.rerank_score:.3f}, "
            f"source='{self.source}', page={self.page})"
        )


# ============================================================================
# Core reranking
# ============================================================================

def _sigmoid(x: float) -> float:
    """Sigmoid to map an unbounded logit into (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def rerank_results(
    query:                    str,
    candidates:               List[Any],   # Any result with .text, .source, .page, etc.
    model_name:               str   = DEFAULT_MODEL,
    top_k:                    int   = DEFAULT_TOP_K,
    original_score_weight:    float = 0.3,
) -> List[RerankedResult]:
    """
    Re-score candidate chunks with a cross-encoder and return top-k.

    Args:
        query:                 Search query.
        candidates:            Results from any retriever (semantic / keyword / hybrid).
        model_name:            CrossEncoder model name or shortcut (fast/balanced/accurate).
        top_k:                 How many results to return.
        original_score_weight: Blend weight for the original retrieval score (0 = pure
                               cross-encoder order, 1 = original order unchanged).

    Returns:
        List[RerankedResult] sorted by final_score descending.
    """
    if not candidates:
        TerminalUI.warn("  no candidates to rerank")
        return []

    TerminalUI.dim(f"  reranking {len(candidates)} candidates...")

    model = load_cross_encoder(model_name)

    # Truncate chunks to 512 tokens worth of chars (cross-encoders have a
    # hard 512-token limit; going over silently truncates inside the model)
    pairs  = [[query, c.text[:1500]] for c in candidates]
    logits = model.predict(pairs, show_progress_bar=False)   # shape: (n,)

    # ── Normalize rerank scores to [0, 1] via sigmoid ────────────────────────
    norm_rerank = np.array([_sigmoid(s) for s in logits])

    # ── Normalize original scores to [0, 1] via min-max ──────────────────────
    # FIX: original scores from different retrievers live in completely
    # different ranges. We must normalize before blending, otherwise the
    # weight has no meaningful interpretation.
    raw_originals = []
    for c in candidates:
        if   hasattr(c, "final_score"):  raw_originals.append(c.final_score)
        elif hasattr(c, "rrf_score"):    raw_originals.append(c.rrf_score)
        elif hasattr(c, "score"):        raw_originals.append(c.score)
        else:                            raw_originals.append(None)

    valid_orig = [s for s in raw_originals if s is not None]
    orig_min   = min(valid_orig) if valid_orig else 0.0
    orig_max   = max(valid_orig) if valid_orig else 1.0
    orig_range = orig_max - orig_min if orig_max != orig_min else 1.0

    def _norm_orig(s: Optional[float]) -> float:
        if s is None:
            return 0.0
        return (s - orig_min) / orig_range

    # ── Build results ──────────────────────────────────────────────────────────
    reranked: List[RerankedResult] = []
    for candidate, logit, nr, raw_orig in zip(candidates, logits, norm_rerank, raw_originals):
        norm_orig  = _norm_orig(raw_orig)
        final      = (1.0 - original_score_weight) * float(nr) + original_score_weight * norm_orig

        reranked.append(RerankedResult(
            text           = candidate.text,
            clean          = candidate.clean,
            rerank_score   = float(logit),
            original_score = raw_orig,
            final_score    = final,
            source         = candidate.source,
            page           = candidate.page,
            page_end       = getattr(candidate, "page_end", candidate.page),
            heading        = candidate.heading,
            chunk_index    = candidate.chunk_index,
            links          = getattr(candidate, "links",       []),
            is_overlap     = getattr(candidate, "is_overlap",  False),
        ))

    reranked.sort(key=lambda x: x.final_score, reverse=True)

    TerminalUI.dim(f"  ✓ reranked — returning top {min(top_k, len(reranked))}")
    return reranked[:top_k]


# ============================================================================
# Convenience wrappers
# ============================================================================

def rerank_semantic(
    query:            str,
    semantic_results: List["SematicSearchResult"],
    model_name:       str = DEFAULT_MODEL,
    top_k:            int = DEFAULT_TOP_K,
) -> List[RerankedResult]:
    """Rerank semantic search results."""
    return rerank_results(query, semantic_results, model_name, top_k)


def rerank_keyword(
    query:           str,
    keyword_results: List["KeywordSearchResult"],
    model_name:      str = DEFAULT_MODEL,
    top_k:           int = DEFAULT_TOP_K,
) -> List[RerankedResult]:
    """Rerank keyword search results."""
    return rerank_results(query, keyword_results, model_name, top_k)


def rerank_hybrid(
    query:          str,
    hybrid_results: List["HybridSearchResult"],
    model_name:     str = DEFAULT_MODEL,
    top_k:          int = DEFAULT_TOP_K,
) -> List[RerankedResult]:
    """Rerank hybrid search results."""
    return rerank_results(query, hybrid_results, model_name, top_k)


# ============================================================================
# Integrated pipeline helpers
# (thin wrappers — retrieval + rerank in one call)
# ============================================================================

def semantic_search_with_rerank(
    query:      str,
    store,
    filename:   Optional[str] = None,
    model_name: str           = DEFAULT_MODEL,
    top_k:      int           = DEFAULT_TOP_K,
    pool_size:  int           = DEFAULT_POOL_SIZE,
) -> List[RerankedResult]:
    """Semantic retrieval → rerank."""
    from src.core.Search.Sematic.SematicSearch import (
        search_single_document  as sem_single,
        search_across_documents as sem_all,
    )
    TerminalUI.dim("  searching semantics...")
    candidates = (
        sem_single(query, filename, store, top_k=pool_size)
        if filename else
        sem_all(query, store, top_k=pool_size)
    )
    TerminalUI.dim(f"  ✓ {len(candidates)} candidates")
    return rerank_semantic(query, candidates, model_name, top_k)


def hybrid_search_with_rerank(
    query:      str,
    store,
    filename:   Optional[str] = None,
    model_name: str           = DEFAULT_MODEL,
    top_k:      int           = DEFAULT_TOP_K,
    pool_size:  int           = DEFAULT_POOL_SIZE,
) -> List[RerankedResult]:
    """Hybrid retrieval → rerank."""
    from src.core.Search.Hybrid.HybridSearch import (
        search_single_document_hybrid  as hyb_single,
        search_across_documents_hybrid as hyb_all,
    )
    candidates = (
        hyb_single(query, filename, store, top_k=pool_size)
        if filename else
        hyb_all(query, store, top_k=pool_size)
    )
    return rerank_hybrid(query, candidates, model_name, top_k)