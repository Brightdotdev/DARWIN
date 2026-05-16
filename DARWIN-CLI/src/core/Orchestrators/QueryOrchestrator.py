"""
src/core/Knowledge/QueryOrchestrator.py
----------------------------------------
Unified query orchestrator for deep (retrieval) and shallow (wiki) search.
Produces per-file, provenance-aware results for cross-document synthesis.

Usage:
    from src.core.Knowledge.QueryOrchestrator import QueryOrchestrator, DeepSearchOptions, ShallowSearchOptions

    orch = QueryOrchestrator(store)

    # Deep search across multiple files
    results = orch.search_deep(
        query="who are the watchers",
        files=["book_of_enoch.pdf", "bible_kjv.pdf"],
        options=DeepSearchOptions(method="hybrid", top_k=5, rerank=True)
    )
    # Returns List[QueryResult], one per file

    # Shallow search
    wiki_results = orch.search_shallow(
        query="fallen angels",
        files=["book_of_enoch.pdf"],
        options=ShallowSearchOptions(top_k=3, expand_graph=False)
    )

    # Both at once
    deep, shallow = orch.search(
        query="who are the watchers",
        files=["book_of_enoch.pdf"],
        return_both=True
    )
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from src.features.Storage.DarwinStore import DarwinStore
from src.core.Logger.DarwinLogger import darwin, LogType

from src.features.Retrival.RetrievalOrchestrator import (
    RetrievalOrchestrator,
    ResolvedPassage,
    RetrievalContext,
)
from src.features.Knowledge.WikiOrchestrator import (
    WikiOrchestrator,
    WikiConcept,
    WikiContext,
)


# ═══════════════════════════════════════════════════════════════════════════
# Search modality enum
# ═══════════════════════════════════════════════════════════════════════════

class SearchModality(Enum):
    DEEP = "deep"
    SHALLOW = "shallow"


# ═══════════════════════════════════════════════════════════════════════════
# Options dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DeepSearchOptions:
    """
    Configuration for deep retrieval search.
    Maps directly to RetrievalOrchestrator parameters + search method tunables.
    """
    method: str = "hybrid"              # "hybrid" | "semantic" | "keyword"
    top_k: int = 5
    pool_size: int = 20               # hybrid: candidate pool size
    rrf_k: int = 60                   # hybrid: RRF fusion parameter
    semantic_weight: float = 0.7      # hybrid: semantic score weight
    keyword_weight: float = 0.3         # hybrid: keyword score weight
    alpha: float = 0.5                # hybrid: semantic normalization
    beta: float = 0.5                 # hybrid: keyword normalization
    rerank: bool = False              # enable post-fusion reranking
    rerank_weight: float = 0.3        # how much rerank overrides original
    rerank_top_k: int = 10
    min_score: float = 0.0            # semantic: minimum score threshold
    k1: float = 1.5                   # keyword: BM25 k1
    b: float = 0.75                   # keyword: BM25 b
    # Passage assembly
    max_chars: int = 6000
    merge_sources: bool = True
    fill_gaps: bool = True
    filter_structural: bool = True
    aggressive_filter: bool = True


@dataclass
class ShallowSearchOptions:
    """
    Configuration for shallow wiki search.
    Maps directly to WikiOrchestrator parameters.
    """
    top_k: int = 5
    max_chars: int = 4000
    merge_sources: bool = True
    fill_gaps: bool = True
    filter_structural: bool = True
    expand_graph: bool = True         # include graph-expanded concepts


# ═══════════════════════════════════════════════════════════════════════════
# Unified result type
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QueryResult:
    """
    Provenance-aware search result from a single source file.
    Normalizes both deep and shallow results into one structure.
    """
    source: str                         # filename that produced this
    query: str
    modality: str                       # "deep" | "shallow"
    
    # Passages — unified interface regardless of modality
    passages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    total_items: int = 0                # passages (deep) or concepts (shallow)
    filtered_count: int = 0
    fallback_used: bool = False
    related_topics: List[str] = field(default_factory=list)
    
    # Score breakdown — modality-agnostic keys
    scores: Dict[str, float] = field(default_factory=dict)
    
    # Raw context for advanced consumers
    _raw_context: Any = None            # RetrievalContext | WikiContext | None

    def to_llm_blocks(self) -> List[str]:
        """Convert passages to LLM-ready text blocks."""
        blocks = []
        for i, p in enumerate(self.passages):
            header = f"[{self.modality.upper()} {i+1}: {self.source}"
            if p.get("page_start"):
                header += f" | Page {p['page_start']}-{p['page_end']}"
            if p.get("score"):
                header += f" | Score: {p['score']:.3f}"
            header += "]"
            
            lines = [header]
            if p.get("concept"):
                lines.append(f"Concept: {p['concept']}")
            if p.get("summary"):
                lines.append(f"Summary: {p['summary']}")
            if p.get("keywords"):
                lines.append(f"Keywords: {', '.join(p['keywords'][:5])}")
            lines.append(f"Passage: {p.get('text', p.get('passage_text', ''))}")
            
            blocks.append("\n".join(lines))
        return blocks


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_deep_scores(passage: ResolvedPassage) -> Dict[str, float]:
    """Extract normalized score components from a deep result."""
    scores = {"avg": passage.avg_score}
    if passage.semantic_score is not None:
        scores["semantic"] = passage.semantic_score
    if passage.keyword_score is not None:
        scores["keyword"] = passage.keyword_score
    if passage.rrf_score is not None:
        scores["rrf"] = passage.rrf_score
    if passage.rerank_score is not None:
        scores["rerank"] = passage.rerank_score
    return scores


def _normalize_shallow_scores(concept: WikiConcept) -> Dict[str, float]:
    """Extract normalized score from a shallow result."""
    return {"final": concept.score}


def _deep_passage_to_dict(p: ResolvedPassage) -> Dict[str, Any]:
    """Convert ResolvedPassage to modality-agnostic dict."""
    return {
        "text": p.passage_text,
        "passage_text": p.passage_text,  # alias for compatibility
        "page_start": p.page_start,
        "page_end": p.page_end,
        "chunk_indices": p.chunk_indices,
        "source_chunks": p.source_chunks,
        "score": p.avg_score,
        "semantic_score": p.semantic_score,
        "keyword_score": p.keyword_score,
        "rrf_score": p.rrf_score,
        "rerank_score": p.rerank_score,
        "matched_terms": p.matched_terms,
        "gap_filled": len(p.chunk_indices) > len(p.original_results),
    }


def _shallow_concept_to_dict(c: WikiConcept) -> Dict[str, Any]:
    """Convert WikiConcept to modality-agnostic dict."""
    return {
        "concept": c.concept,
        "text": c.passage_text,
        "passage_text": c.passage_text,  # alias
        "summary": c.summary,
        "keywords": c.keywords,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "chunk_indices": c.chunk_indices,
        "source_count": c.source_count,
        "score": c.score,
    }


def _deep_context_to_result(ctx: RetrievalContext, source: str) -> QueryResult:
    """Normalize RetrievalContext into QueryResult."""
    passages = [_deep_passage_to_dict(p) for p in ctx.passages]
    
    # Aggregate scores from best passage
    best_scores = {}
    if ctx.passages:
        best_scores = _normalize_deep_scores(ctx.passages[0])
    
    return QueryResult(
        source=source,
        query=ctx.query,
        modality=SearchModality.DEEP.value,
        passages=passages,
        total_items=len(passages),
        filtered_count=ctx.filtered_count,
        fallback_used=ctx.fallback_used,
        scores=best_scores,
        _raw_context=ctx,
    )


def _shallow_context_to_result(ctx: WikiContext, source: str) -> QueryResult:
    """Normalize WikiContext into QueryResult."""
    passages = [_shallow_concept_to_dict(c) for c in ctx.concepts]
    
    # Aggregate scores from best concept
    best_scores = {}
    if ctx.concepts:
        best_scores = _normalize_shallow_scores(ctx.concepts[0])
    
    return QueryResult(
        source=source,
        query=ctx.query,
        modality=SearchModality.SHALLOW.value,
        passages=passages,
        total_items=len(passages),
        filtered_count=ctx.filtered_count,
        related_topics=ctx.related_topics,
        scores=best_scores,
        _raw_context=ctx,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Core orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class QueryOrchestrator:
    """
    Unified entry point for deep (retrieval) and shallow (wiki) search.
    Produces per-file, provenance-aware QueryResult objects.
    """

    def __init__(self, store: DarwinStore):
        self.store = store
        self._retrieval = RetrievalOrchestrator(store)
        self._wiki = WikiOrchestrator(store)

    # ── Deep search ─────────────────────────────────────────────────────────

    def search_deep(
        self,
        query: str,
        files: List[str],
        options: Optional[DeepSearchOptions] = None,
    ) -> List[QueryResult]:
        """
        Deep retrieval across one or more files.
        Returns one QueryResult per file.
        """
        if not files:
            raise ValueError("At least one file required for deep search")

        opts = options or DeepSearchOptions()
        results: List[QueryResult] = []

        for filename in files:
            darwin.log(f"[QueryOrchestrator] DEEP '{query}' → {filename}", LogType.INFO)

            ctx = self._retrieval.retrieve_and_resolve(
                query=query,
                filename=filename,
                method=opts.method,
                top_k=opts.top_k,
                max_chars=opts.max_chars,
                merge_sources=opts.merge_sources,
                fill_gaps=opts.fill_gaps,
                filter_structural=opts.filter_structural,
                aggressive_filter=opts.aggressive_filter,
                # Method-specific tunables passed through **kwargs
                pool_size=opts.pool_size,
                rrf_k=opts.rrf_k,
                semantic_weight=opts.semantic_weight,
                keyword_weight=opts.keyword_weight,
                alpha=opts.alpha,
                beta=opts.beta,
                rerank=opts.rerank,
                rerank_weight=opts.rerank_weight,
                rerank_top_k=opts.rerank_top_k,
                min_score=opts.min_score,
                k1=opts.k1,
                b=opts.b,
            )

            result = _deep_context_to_result(ctx, filename)
            results.append(result)

        return results

    # ── Shallow search ────────────────────────────────────────────────────────

    def search_shallow(
        self,
        query: str,
        files: List[str],
        options: Optional[ShallowSearchOptions] = None,
    ) -> List[QueryResult]:
        """
        Shallow wiki search across one or more files.
        Returns one QueryResult per file.
        """
        if not files:
            raise ValueError("At least one file required for shallow search")

        opts = options or ShallowSearchOptions()
        results: List[QueryResult] = []

        for filename in files:
            darwin.log(f"[QueryOrchestrator] SHALLOW '{query}' → {filename}", LogType.INFO)

            ctx = self._wiki.search(
                query=query,
                filename=filename,
                top_k=opts.top_k,
                merge_sources=opts.merge_sources,
                fill_gaps=opts.fill_gaps,
                filter_structural=opts.filter_structural,
            )

            result = _shallow_context_to_result(ctx, filename)
            results.append(result)

        return results

    # ── Unified search ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        files: List[str],
        deep_options: Optional[DeepSearchOptions] = None,
        shallow_options: Optional[ShallowSearchOptions] = None,
        return_both: bool = False,
    ) -> QueryResult | Tuple[List[QueryResult], List[QueryResult]]:
        """
        Search with automatic modality selection or explicit dual search.

        If return_both=True: returns (deep_results, shallow_results)
        If return_both=False: returns best single result per file
                              (deep if available, shallow fallback)
        """
        if not files:
            raise ValueError("At least one file required")

        if return_both:
            deep = self.search_deep(query, files, deep_options)
            shallow = self.search_shallow(query, files, shallow_options)
            return deep, shallow

        # Single modality: prefer deep, fallback to shallow
        results: List[QueryResult] = []
        for filename in files:
            deep_opts = deep_options or DeepSearchOptions()
            shallow_opts = shallow_options or ShallowSearchOptions()

            # Try deep first
            deep_result = self.search_deep(query, [filename], deep_opts)[0]
            if deep_result.total_items > 0:
                results.append(deep_result)
                continue

            # Fallback to shallow
            darwin.log(f"[QueryOrchestrator] Deep empty for {filename}, trying shallow", LogType.INFO)
            shallow_result = self.search_shallow(query, [filename], shallow_opts)[0]
            results.append(shallow_result)

        # Return first result if single file, list if multiple
        return results[0] if len(results) == 1 else results

    # ── LLM prompt builder ──────────────────────────────────────────────────

    def build_llm_prompt(
        self,
        results: List[QueryResult],
        query: str,
        include_related: bool = True,
    ) -> str:
        """
        Build a unified LLM prompt from mixed deep/shallow results.
        Groups by source file for cross-document synthesis.
        """
        if not results:
            return f"Question: {query}\n\nNo relevant sources found."

        # Group by source
        by_source: Dict[str, List[QueryResult]] = {}
        for r in results:
            by_source.setdefault(r.source, []).append(r)

        blocks = []
        for source, source_results in by_source.items():
            source_blocks = []
            for r in source_results:
                modality_label = "DEEP RETRIEVAL" if r.modality == "deep" else "SHALLOW WIKI"
                source_blocks.append(f"\n--- {modality_label} ---\n")
                source_blocks.extend(r.to_llm_blocks())

            blocks.append(f"\n{'='*60}\nSOURCE: {source}\n{'='*60}\n")
            blocks.extend(source_blocks)

        # Related topics from shallow results
        related = []
        if include_related:
            for r in results:
                if r.related_topics:
                    related.extend(r.related_topics)
            if related:
                related = list(dict.fromkeys(related))[:10]
                blocks.append(f"\n{'='*60}\nRELATED TOPICS: {', '.join(related)}\n{'='*60}\n")

        source_text = "\n".join(blocks)

        return f"""Answer based on the following document sources and passages.

Question: {query}

{source_text}

Answer:"""

    # ── Convenience one-liners ────────────────────────────────────────────────

    def ask_deep(
        self,
        query: str,
        files: List[str],
        options: Optional[DeepSearchOptions] = None,
    ) -> Tuple[str, List[QueryResult]]:
        """One-liner: deep search + prompt."""
        results = self.search_deep(query, files, options)
        prompt = self.build_llm_prompt(results, query)
        return prompt, results

    def ask_shallow(
        self,
        query: str,
        files: List[str],
        options: Optional[ShallowSearchOptions] = None,
    ) -> Tuple[str, List[QueryResult]]:
        """One-liner: shallow search + prompt."""
        results = self.search_shallow(query, files, options)
        prompt = self.build_llm_prompt(results, query)
        return prompt, results

    def ask(
        self,
        query: str,
        files: List[str],
        deep_options: Optional[DeepSearchOptions] = None,
        shallow_options: Optional[ShallowSearchOptions] = None,
    ) -> Tuple[str, List[QueryResult]]:
        """One-liner: both modalities + unified prompt."""
        deep_results = self.search_deep(query, files, deep_options)
        shallow_results = self.search_shallow(query, files, shallow_options)
        all_results = deep_results + shallow_results
        prompt = self.build_llm_prompt(all_results, query)
        return prompt, all_results