"""
src/features/Retrival/RetrievalResult.py
-----------------------------------------
Universal result type for all retrieval paths.

Every search method (wiki, hybrid, semantic, keyword) returns a
List[RetrievalResult]. Fields that don't apply to a given path are None.

ResultType enum tells you which path produced the result so you know
exactly which fields will be populated without guessing.

    WIKI     → wiki_concept, wiki_score, wiki_centroid_sim,
               wiki_keyword_hits, wiki_frequency, wiki_related,
               wiki_keywords, wiki_summary, wiki_chunk_indices
               (chunk fields all None)

    HYBRID   → text, clean, source, page, page_end, heading,
               chunk_index, score (= final_score), rrf_score,
               semantic_score, keyword_score, matched_terms,
               overlap_direction, links
               wiki_* fields populated only if wiki stage ran first

    SEMANTIC → text, clean, source, page, page_end, heading,
               chunk_index, score, overlap_direction, links
               (rrf_score, keyword_score, matched_terms all None)

    KEYWORD  → text, clean, source, page, page_end, heading,
               chunk_index, score, matched_terms, overlap_direction, links
               (rrf_score, semantic_score all None)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class ResultType(str, Enum):
    WIKI     = "wiki"
    HYBRID   = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD  = "keyword"


class OverlapDirection(str, Enum):
    NONE    = "none"
    FORWARD = "forward"
    BACK    = "back"
    BOTH    = "both"


# ═══════════════════════════════════════════════════════════════════════════
# ChunkLink (mirrors existing ChunkLink — defined here so this module is
# self-contained; downstream code can import from either place)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkLink:
    url:  str
    text: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Universal result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """
    Single result from any retrieval path.

    Always populated:
        result_type   — which path produced this result
        score         — confidence score (normalised where possible)
        source        — filename this result came from

    Chunk paths (HYBRID / SEMANTIC / KEYWORD):
        text, clean, page, page_end, heading, chunk_index,
        overlap_direction, links

    HYBRID only:
        rrf_score, semantic_score, keyword_score

    KEYWORD / HYBRID:
        matched_terms

    HYBRID / SEMANTIC:
        rerank_score, rerank_original_score  (populated only when reranking ran)

    Wiki reference (any path that ran a wiki stage first):
        wiki_concept, wiki_score, wiki_centroid_sim,
        wiki_keyword_hits, wiki_frequency, wiki_related,
        wiki_keywords, wiki_summary, wiki_chunk_indices

    WIKI path (wiki_* are the primary fields; chunk fields are None):
        wiki_concept is the "text" equivalent for wiki results
    """

    # ── universal ──────────────────────────────────────────────────────────
    result_type:            ResultType
    score:                  float               # primary confidence score
    source:                 str                 # filename

    # ── chunk-level fields (HYBRID / SEMANTIC / KEYWORD) ──────────────────
    text:                   Optional[str]       = None   # original passage text
    clean:                  Optional[str]       = None   # cleaned/embedded text
    page:                   Optional[int]       = None
    page_end:               Optional[int]       = None
    heading:                Optional[str]       = None
    chunk_index:            Optional[int]       = None
    overlap_direction:      OverlapDirection    = OverlapDirection.NONE
    links:                  List[ChunkLink]     = field(default_factory=list)

    # ── score breakdown ────────────────────────────────────────────────────
    rrf_score:              Optional[float]     = None   # HYBRID only
    semantic_score:         Optional[float]     = None   # HYBRID only
    keyword_score:          Optional[float]     = None   # HYBRID only
    matched_terms:          Optional[List[str]] = None   # KEYWORD / HYBRID

    # ── reranking (HYBRID / SEMANTIC / KEYWORD when rerank=True) ──────────
    rerank_score:           Optional[float]     = None
    rerank_original_score:  Optional[float]     = None

    # ── wiki reference (populated whenever wiki stage ran) ─────────────────
    wiki_concept:           Optional[str]       = None
    wiki_score:             Optional[float]     = None   # composite wiki score
    wiki_centroid_sim:      Optional[float]     = None
    wiki_keyword_hits:      Optional[int]       = None
    wiki_frequency:         Optional[int]       = None
    wiki_related:           Optional[List[str]] = None
    wiki_keywords:          Optional[List[str]] = None
    wiki_summary:           Optional[str]       = None
    wiki_chunk_indices:     Optional[List[int]] = None

    # ── helpers ────────────────────────────────────────────────────────────

    @property
    def is_wiki(self) -> bool:
        return self.result_type == ResultType.WIKI

    @property
    def is_chunk(self) -> bool:
        return self.result_type in (ResultType.HYBRID, ResultType.SEMANTIC, ResultType.KEYWORD)

    @property
    def has_wiki_context(self) -> bool:
        """True for chunk results that also carry wiki metadata."""
        return self.is_chunk and self.wiki_concept is not None

    @property
    def display_text(self) -> str:
        """Best available text for display / LLM context."""
        if self.text:
            return self.text
        if self.wiki_concept:
            if self.wiki_summary:
                return self.wiki_summary
            kw = ", ".join(self.wiki_keywords) if self.wiki_keywords else "—"
            return f"[concept] {self.wiki_concept} | keywords: {kw}"
        return ""

    def __repr__(self) -> str:
        if self.is_wiki:
            return (
                f"RetrievalResult(type=WIKI, concept='{self.wiki_concept}', "
                f"score={self.score:.3f}, source='{self.source}')"
            )
        return (
            f"RetrievalResult(type={self.result_type.value.upper()}, "
            f"score={self.score:.3f}, source='{self.source}', page={self.page})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Converters — build RetrievalResult from the existing per-type dataclasses
# ═══════════════════════════════════════════════════════════════════════════

def from_hybrid(
    r: Any,
    *,
    wiki_concept:       Optional[str]       = None,
    wiki_score:         Optional[float]     = None,
    wiki_centroid_sim:  Optional[float]     = None,
    wiki_keyword_hits:  Optional[int]       = None,
    wiki_frequency:     Optional[int]       = None,
    wiki_related:       Optional[List[str]] = None,
    wiki_keywords:      Optional[List[str]] = None,
    wiki_summary:       Optional[str]       = None,
    wiki_chunk_indices: Optional[List[int]] = None,
) -> RetrievalResult:
    """Convert a HybridSearchResult → RetrievalResult."""
    score = getattr(r, "final_score", getattr(r, "rrf_score", 0.0))
    return RetrievalResult(
        result_type           = ResultType.HYBRID,
        score                 = score,
        source                = r.source,
        text                  = r.text,
        clean                 = getattr(r, "clean", None),
        page                  = r.page,
        page_end              = getattr(r, "page_end", None),
        heading               = getattr(r, "heading", None),
        chunk_index           = getattr(r, "chunk_index", None),
        overlap_direction     = getattr(r, "overlap_direction", OverlapDirection.NONE),
        links                 = getattr(r, "links", []),
        rrf_score             = getattr(r, "rrf_score", None),
        semantic_score        = getattr(r, "semantic_score", None),
        keyword_score         = getattr(r, "keyword_score", None),
        matched_terms         = getattr(r, "matched_terms", None),
        rerank_score          = getattr(r, "rerank_score", None),
        rerank_original_score = getattr(r, "original_score", None),
        wiki_concept          = wiki_concept,
        wiki_score            = wiki_score,
        wiki_centroid_sim     = wiki_centroid_sim,
        wiki_keyword_hits     = wiki_keyword_hits,
        wiki_frequency        = wiki_frequency,
        wiki_related          = wiki_related,
        wiki_keywords         = wiki_keywords,
        wiki_summary          = wiki_summary,
        wiki_chunk_indices    = wiki_chunk_indices,
    )


def from_semantic(
    r: Any,
    *,
    wiki_concept:       Optional[str]       = None,
    wiki_score:         Optional[float]     = None,
    wiki_centroid_sim:  Optional[float]     = None,
    wiki_keyword_hits:  Optional[int]       = None,
    wiki_frequency:     Optional[int]       = None,
    wiki_related:       Optional[List[str]] = None,
    wiki_keywords:      Optional[List[str]] = None,
    wiki_summary:       Optional[str]       = None,
    wiki_chunk_indices: Optional[List[int]] = None,
) -> RetrievalResult:
    """Convert a SemanticSearchResult → RetrievalResult."""
    return RetrievalResult(
        result_type           = ResultType.SEMANTIC,
        score                 = getattr(r, "score", 0.0),
        source                = r.source,
        text                  = r.text,
        clean                 = getattr(r, "clean", None),
        page                  = r.page,
        page_end              = getattr(r, "page_end", None),
        heading               = getattr(r, "heading", None),
        chunk_index           = getattr(r, "chunk_index", None),
        overlap_direction     = getattr(r, "overlap_direction", OverlapDirection.NONE),
        links                 = getattr(r, "links", []),
        rerank_score          = getattr(r, "rerank_score", None),
        rerank_original_score = getattr(r, "original_score", None),
        wiki_concept          = wiki_concept,
        wiki_score            = wiki_score,
        wiki_centroid_sim     = wiki_centroid_sim,
        wiki_keyword_hits     = wiki_keyword_hits,
        wiki_frequency        = wiki_frequency,
        wiki_related          = wiki_related,
        wiki_keywords         = wiki_keywords,
        wiki_summary          = wiki_summary,
        wiki_chunk_indices    = wiki_chunk_indices,
    )


def from_keyword(
    r: Any,
    *,
    wiki_concept:       Optional[str]       = None,
    wiki_score:         Optional[float]     = None,
    wiki_centroid_sim:  Optional[float]     = None,
    wiki_keyword_hits:  Optional[int]       = None,
    wiki_frequency:     Optional[int]       = None,
    wiki_related:       Optional[List[str]] = None,
    wiki_keywords:      Optional[List[str]] = None,
    wiki_summary:       Optional[str]       = None,
    wiki_chunk_indices: Optional[List[int]] = None,
) -> RetrievalResult:
    """Convert a KeywordSearchResult → RetrievalResult."""
    return RetrievalResult(
        result_type           = ResultType.KEYWORD,
        score                 = getattr(r, "score", 0.0),
        source                = r.source,
        text                  = r.text,
        clean                 = getattr(r, "clean", None),
        page                  = r.page,
        page_end              = getattr(r, "page_end", None),
        heading               = getattr(r, "heading", None),
        chunk_index           = getattr(r, "chunk_index", None),
        overlap_direction     = getattr(r, "overlap_direction", OverlapDirection.NONE),
        links                 = getattr(r, "links", []),
        matched_terms         = getattr(r, "matched_terms", None),
        rerank_score          = getattr(r, "rerank_score", None),
        rerank_original_score = getattr(r, "original_score", None),
        wiki_concept          = wiki_concept,
        wiki_score            = wiki_score,
        wiki_centroid_sim     = wiki_centroid_sim,
        wiki_keyword_hits     = wiki_keyword_hits,
        wiki_frequency        = wiki_frequency,
        wiki_related          = wiki_related,
        wiki_keywords         = wiki_keywords,
        wiki_summary          = wiki_summary,
        wiki_chunk_indices    = wiki_chunk_indices,
    )


def from_wiki(w: Any) -> RetrievalResult:
    """
    Convert a WikiSearchResult → RetrievalResult.
    chunk fields are all None — this is a concept-level hit.
    """
    return RetrievalResult(
        result_type        = ResultType.WIKI,
        score              = w.score,
        source             = w.filename,
        wiki_concept       = w.concept,
        wiki_score         = w.score,
        wiki_centroid_sim  = w.centroid_sim,
        wiki_keyword_hits  = w.keyword_hits,
        wiki_frequency     = w.frequency,
        wiki_related       = w.related,
        wiki_keywords      = w.keywords,
        wiki_summary       = w.summary,
        wiki_chunk_indices = w.chunk_indices,
    )