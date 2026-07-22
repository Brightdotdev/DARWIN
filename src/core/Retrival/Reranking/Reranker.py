from dataclasses import dataclass, field
from typing import List, Optional, Any
import numpy as np

from src.core.Retrival.lib import OverlapDirection, ChunkLink,DEFAULT_TOP_K,DEFAULT_RERANK_WEIGHT
from src.core.Logger.DarwinLogger import darwin



@dataclass
class RerankedResult:
    """A result that has been reranked by a cross‑encoder."""
    text: str
    clean: str
    rerank_score: Optional[float]
    original_score: Optional[float]
    final_score: float
    source: str
    page: int
    page_end: int
    heading: str
    chunk_index: int
    links: List[ChunkLink] = field(default_factory=list)
    overlap_direction: OverlapDirection = OverlapDirection.NONE
    matched_terms: Optional[List[str]] = None

    def __repr__(self) -> str:
        rerank = f"{self.rerank_score:.3f}" if self.rerank_score is not None else "N/A"
        return (
            f"RerankedResult(final={self.final_score:.3f}, "
            f"rerank={rerank}, "
            f"source='{self.source}', page={self.page})"
        )


def _sigmoid(x: float) -> float:
    """Sigmoid to map an unbounded logit into (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_original_scores(candidates: List[Any]) -> List[float]:
    """Extract and normalize original scores to [0,1] dynamically."""
    raw_scores = []
    for c in candidates:
        # Try to get a score from common attributes
        score = None
        if hasattr(c, 'final_score') and c.final_score is not None:
            score = c.final_score
        elif hasattr(c, 'score') and c.score is not None:
            score = c.score
        elif hasattr(c, 'rrf_score') and c.rrf_score is not None:
            score = c.rrf_score
        else:
            score = 0.0
        raw_scores.append(score if score is not None else 0.0)

    max_score = max(raw_scores) if raw_scores else 1.0
    if max_score <= 0:
        max_score = 1.0

    # Normalize: if scores are already in [0,1], keep as is; else apply sigmoid scaling
    norm_scores = []
    for c, raw in zip(candidates, raw_scores):
        if hasattr(c, 'final_score') and c.final_score is not None and 0 <= c.final_score <= 1:
            norm_scores.append(c.final_score)
        elif hasattr(c, 'score') and c.score is not None and 0 <= c.score <= 1:
            norm_scores.append(c.score)
        else:
            # Scale by max and apply sigmoid to bring into (0,1)
            norm_scores.append(_sigmoid(raw / max_score))
    return norm_scores


def rerank_results(
    query: str,
    candidates: List[Any],
    top_k: int = DEFAULT_TOP_K,
    original_score_weight: float =DEFAULT_RERANK_WEIGHT,
) -> List[RerankedResult]:
    """Re‑score candidate chunks with a cross‑encoder."""
    from src.core.Models.ModelManager import modelManager
    from src.core.Models.Lib.ModelRegistry import ModelCategory

    if not candidates:
        darwin.warning("No candidates to rerank")
        return []

    darwin.info(f"Reranking {len(candidates)} candidates")

    # Attempt to load the reranker model
    try:
        model = modelManager.get(ModelCategory.RERANKER)
        if model is None:
            darwin.warning("Reranking model not set, skipping")
            return _fallback_rerank(candidates, top_k)
    except Exception as e:
        darwin.error(f"Failed to load reranker model: {e}")
        return _fallback_rerank(candidates, top_k)

    # Prepare documents (truncate to avoid token limits)
    documents = [getattr(c, 'text', '')[:1500] for c in candidates]

    # Get cross‑encoder scores
    try:
        result = model.rerank(query, documents)
        scores = result.scores
    except Exception as e:
        darwin.error(f"Reranking failed: {e}")
        return _fallback_rerank(candidates, top_k)

    # Ensure we have a score for each candidate
    if len(scores) != len(candidates):
        darwin.warning(
            f"Reranker returned {len(scores)} scores, expected {len(candidates)}, falling back"
        )
        return _fallback_rerank(candidates, top_k)

    scores_np = np.array(scores)

    # Normalise rerank scores to [0,1]
    if scores_np.max() > 1.0 or scores_np.min() < 0.0:
        norm_rerank = _sigmoid(scores_np)
    else:
        norm_rerank = scores_np

    # Normalise original scores
    norm_orig = _normalize_original_scores(candidates)

    # Build reranked results
    reranked = []
    for candidate, raw_score, norm_score, norm_orig_score in zip(
        candidates, scores, norm_rerank, norm_orig
    ):
        final = (1.0 - original_score_weight) * norm_score + original_score_weight * norm_orig_score

        # Determine original score for storage
        orig_score = None
        if hasattr(candidate, 'final_score') and candidate.final_score is not None:
            orig_score = candidate.final_score
        elif hasattr(candidate, 'score') and candidate.score is not None:
            orig_score = candidate.score
        elif hasattr(candidate, 'rrf_score') and candidate.rrf_score is not None:
            orig_score = candidate.rrf_score

        reranked.append(RerankedResult(
            text=getattr(candidate, 'text', ''),
            clean=getattr(candidate, 'clean', getattr(candidate, 'text', '')),
            rerank_score=float(raw_score),
            original_score=orig_score,
            final_score=float(final),
            source=getattr(candidate, 'source', ''),
            page=getattr(candidate, 'page', 0),
            page_end=getattr(candidate, 'page_end', getattr(candidate, 'page', 0)),
            heading=getattr(candidate, 'heading', ''),
            chunk_index=getattr(candidate, 'chunk_index', 0),
            links=getattr(candidate, 'links', []),
            overlap_direction=getattr(candidate, 'overlap_direction', OverlapDirection.NONE),
            matched_terms=getattr(candidate, 'matched_terms', None),
        ))

    reranked.sort(key=lambda x: x.final_score, reverse=True)
    darwin.success(f"Reranked: returning top {min(top_k, len(reranked))} results")
    return reranked[:top_k]


def _fallback_rerank(candidates: List[Any], top_k: int) -> List[RerankedResult]:
    """Return candidates as RerankedResult when the model is unavailable."""
    out = []
    for c in candidates[:top_k]:
        score = getattr(c, 'final_score', None)
        if score is None:
            score = getattr(c, 'score', 0.0)
        out.append(RerankedResult(
            text=getattr(c, 'text', ''),
            clean=getattr(c, 'clean', getattr(c, 'text', '')),
            rerank_score=None,  # FIX: no reranking occurred, don't set a score
            original_score=score,
            final_score=score,
            source=getattr(c, 'source', ''),
            page=getattr(c, 'page', 0),
            page_end=getattr(c, 'page_end', getattr(c, 'page', 0)),
            heading=getattr(c, 'heading', ''),
            chunk_index=getattr(c, 'chunk_index', 0),
            links=getattr(c, 'links', []),
            overlap_direction=getattr(c, 'overlap_direction', OverlapDirection.NONE),
            matched_terms=getattr(c, 'matched_terms', None),
        ))
    return out