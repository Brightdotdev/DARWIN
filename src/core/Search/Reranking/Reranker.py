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
from enum import Enum
from typing import List, Optional, Any, TYPE_CHECKING
import numpy as np


DEFAULT_TOP_K = 100
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
    links:          List[ChunkLink] = field(default_factory=list)
    overlap_direction: OverlapDirection = OverlapDirection.NONE
    matched_terms: List[str] | None = None  # For keyword search results, the terms that matched the query

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
    query: str,
    candidates: List[Any],
    top_k: int = DEFAULT_TOP_K,
    original_score_weight: float = 0.3,
) -> List[RerankedResult]:
    """Re-score candidate chunks with a cross-encoder."""
    from src.core.Logger.DarwinLogger import darwin, LogType
    from src.core.Models import get_loader, ModelCategory
    
    if not candidates:
        darwin.log("no candidates to rerank", LogType.WARNING)
        return []

    darwin.log(f"reranking {len(candidates)} candidates...", LogType.DEBUG)

    # Use ModelLoader to get cross-encoder
    loader = get_loader()
    model = loader.get(ModelCategory.RERANKER)

    # Truncate chunks to 512 tokens (approx 1500 chars)
    pairs = [[query, c.text[:1500]] for c in candidates]
    logits = model.predict(pairs, show_progress_bar=False)
    norm_rerank = np.array([1.0 / (1.0 + np.exp(-s)) for s in logits])

    # ── Normalize original scores using global bounds ──────────────────────
    # Use fixed ranges based on score type, not min-max across batch
    def get_normalized_original(candidate: Any) -> float:
        if hasattr(candidate, "final_score"):
            score = candidate.final_score
            # RRF scores are typically 0-0.033, semantic 0-1
            if score <= 0.05:  # Likely RRF score
                return score / 0.033 if score <= 0.033 else min(1.0, score)
            return score  # Already 0-1
        elif hasattr(candidate, "score"):
            score = candidate.score
            # BM25 scores are unbounded, use sigmoid
            return 1.0 / (1.0 + np.exp(-score / 5.0))  # Normalize BM25
        elif hasattr(candidate, "rrf_score"):
            return min(1.0, candidate.rrf_score / 0.033)
        else:
            return 0.0

    # ── Build results ──────────────────────────────────────────────────────
    reranked = []
    for candidate, logit, nr in zip(candidates, logits, norm_rerank):
        norm_orig = get_normalized_original(candidate)
        final = (1.0 - original_score_weight) * nr + original_score_weight * norm_orig

        reranked.append(RerankedResult(
            text=candidate.text,
            clean=candidate.clean,
            rerank_score=float(logit),
            original_score=getattr(candidate, 'score', getattr(candidate, 'final_score', 0.0)),
            final_score=final,
            source=candidate.source,
            page=candidate.page,
            page_end=getattr(candidate, "page_end", candidate.page),
            heading=candidate.heading,
            chunk_index=candidate.chunk_index,
            links=getattr(candidate, "links", []),
            overlap_direction=getattr(candidate, "overlap_direction", OverlapDirection.NONE),
            matched_terms=getattr(candidate, "matched_terms", None),
        ))

    reranked.sort(key=lambda x: x.final_score, reverse=True)
    darwin.log(f"reranked — returning top {min(top_k, len(reranked))}", LogType.SUCCESS)
    return reranked[:top_k]

