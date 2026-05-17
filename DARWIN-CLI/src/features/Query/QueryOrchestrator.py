"""
src/core/Knowledge/QueryOrchestrator.py
----------------------------------------
Unified query orchestrator for deep (retrieval) and shallow (wiki) search.
Produces per-file, provenance-aware results with optional diagnostics.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Any, Callable
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

from src.features.Query.RetrievalDiagnostics import (
    RetrievalDiagnostics,
    DiagnosticCollector,
    IntentClassifier,
    FailureClassifier,
    ConfidenceScorer,
    RetrievalIntent,
    FailureType,
    ConfidenceLevel,
    ResultTrace,
    PresentationBlock,
    IntentPresentationMapper,
)
from src.features.Query.StructuralAudit import StructuralAudit


class SearchModality(Enum):
    DEEP = "deep"
    SHALLOW = "shallow"


@dataclass
class DeepSearchOptions:
    method: str = "hybrid"
    top_k: int = 5
    pool_size: int = 20
    rrf_k: int = 60
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    alpha: float = 0.5
    beta: float = 0.5
    rerank: bool = False
    rerank_weight: float = 0.3
    rerank_top_k: int = 10
    min_score: float = 0.0
    k1: float = 1.5
    b: float = 0.75
    max_chars: int = 6000
    merge_sources: bool = True
    fill_gaps: bool = True
    filter_structural: bool = True
    aggressive_filter: bool = True


@dataclass
class ShallowSearchOptions:
    top_k: int = 5
    max_chars: int = 4000
    merge_sources: bool = True
    fill_gaps: bool = True
    filter_structural: bool = True
    expand_graph: bool = True


@dataclass
class QueryResult:
    source: str
    query: str
    modality: str
    passages: List[Dict[str, Any]] = field(default_factory=list)
    total_items: int = 0
    filtered_count: int = 0
    fallback_used: bool = False
    related_topics: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    _raw_context: Any = None

    def to_llm_blocks(self) -> List[str]:
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


def _normalize_deep_scores(passage: ResolvedPassage) -> Dict[str, float]:
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
    return {"final": concept.score}


def _deep_passage_to_dict(p: ResolvedPassage) -> Dict[str, Any]:
    return {
        "text": p.passage_text,
        "passage_text": p.passage_text,
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
    return {
        "concept": c.concept,
        "text": c.passage_text,
        "passage_text": c.passage_text,
        "summary": c.summary,
        "keywords": c.keywords,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "chunk_indices": c.chunk_indices,
        "source_count": c.source_count,
        "score": c.score,
    }


def _deep_context_to_result(ctx: RetrievalContext, source: str) -> QueryResult:
    passages = [_deep_passage_to_dict(p) for p in ctx.passages]
    best_scores = _normalize_deep_scores(ctx.passages[0]) if ctx.passages else {}
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
    passages = [_shallow_concept_to_dict(c) for c in ctx.concepts]
    best_scores = _normalize_shallow_scores(ctx.concepts[0]) if ctx.concepts else {}
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


class QueryOrchestrator:
    def __init__(self, store: DarwinStore, llm_client: Optional[Callable] = None):
        self.store = store
        self._retrieval = RetrievalOrchestrator(store)
        self._wiki = WikiOrchestrator(store)
        self._intent = IntentClassifier(llm_client) if llm_client else None

    def _classify_intent(self, query: str, files: List[str]) -> tuple[str, float]:
        if self._intent is None:
            return RetrievalIntent.UNKNOWN.value, 0.0
        return self._intent.classify(query, files)

    def _empty_diagnostic(self, query: str, source: str, modality: str) -> RetrievalDiagnostics:
        return RetrievalDiagnostics(
            query=query,
            file=source,
            modality=modality,
            intent=RetrievalIntent.UNKNOWN.value,
            intent_confidence=0.0,
            steps=[],
            result_traces=[],
            total_results=0,
            filtered_count=0,
            fallback_used=False,
            failure_type=FailureType.NONE.value,
            confidence_level=ConfidenceLevel.UNCERTAIN.value,
            confidence_score=0.0,
        )

    def _diagnose_deep(self, result: QueryResult, query: str, files: List[str], intent: Optional[Any] = None) -> RetrievalDiagnostics:
        ctx = result._raw_context
        if not isinstance(ctx, RetrievalContext):
            return self._empty_diagnostic(query, result.source, "deep")

        collector = DiagnosticCollector(query, result.source, "deep")
        classified_intent, intent_conf = self._classify_intent(query, files)

        collector.start_step("retrieve")
        collector.end_step("retrieve", len(ctx.raw_results), len(ctx.raw_results))

        collector.start_step("filter_structural")
        after_filter = len(ctx.passages) + ctx.filtered_count
        collector.end_step("filter_structural", len(ctx.raw_results), after_filter, {
            "filtered": ctx.filtered_count,
            "fallback_used": ctx.fallback_used,
        })

        collector.start_step("merge_and_resolve")
        collector.end_step("merge_and_resolve", after_filter, len(ctx.passages), {
            "passages": len(ctx.passages),
            "total_chars": getattr(ctx, "total_chars", 0),
        })

        for p in ctx.passages:
            trace = ResultTrace(
                chunk_index=p.chunk_indices[0] if p.chunk_indices else None,
                concept_name=None,
                structural_score=0.0,
                merge_reason=f"merged_from={len(p.original_results)} gap_filled={len(p.chunk_indices) - len(p.original_results)}",
                gap_filled=len(p.chunk_indices) > len(p.original_results),
                gap_indices=[],
                score_breakdown={
                    "avg": p.avg_score,
                    "semantic": p.semantic_score or 0,
                    "keyword": p.keyword_score or 0,
                    "rrf": p.rrf_score or 0,
                    "rerank": p.rerank_score or 0,
                },
                structural_penalty=0.0,
                final_score=p.avg_score,
            )
            collector.add_result_trace(trace)

        best_score = max((p.avg_score for p in ctx.passages), default=0.0)
        scores = [p.avg_score for p in ctx.passages]
        variance = max(scores) - min(scores) if len(scores) > 1 else 0.0

        failure = FailureClassifier.classify(
            total_results=result.total_items,
            filtered_count=ctx.filtered_count,
            fallback_used=ctx.fallback_used,
            best_score=best_score,
            score_variance=variance,
            modality="deep",
        )

        conf_score, conf_level = ConfidenceScorer.score(
            deep_results=ctx.passages,
            shallow_results=[],
            best_deep_score=best_score,
            best_shallow_score=0.0,
            score_variance=variance,
        )

        suggested_formats = []
        if intent and hasattr(intent, "suggested_formats"):
            suggested_formats = [f.value for f in intent.suggested_formats]

        audit = StructuralAudit.audit(ctx.passages, suggested_formats)
        suggested = IntentPresentationMapper.map(
            intent=classified_intent,
            suggested_formats=suggested_formats,
            query=query,
            traces=collector.result_traces,
            audit=audit,
        )

        return collector.build(
            intent=classified_intent,
            intent_confidence=intent_conf,
            total_results=result.total_items,
            filtered_count=ctx.filtered_count,
            fallback_used=ctx.fallback_used,
            failure_type=failure,
            confidence_score=conf_score,
            confidence_level=conf_level,
            suggested_blocks=suggested,
            audit_result={
                "scores": audit.scores,
                "best_format": audit.best_format,
                "best_score": audit.best_score,
                "missing": audit.missing,
            },
        )

    def _diagnose_shallow(self, result: QueryResult, query: str, files: List[str], intent: Optional[Any] = None) -> RetrievalDiagnostics:
        ctx = result._raw_context
        if not isinstance(ctx, WikiContext):
            return self._empty_diagnostic(query, result.source, "shallow")

        collector = DiagnosticCollector(query, result.source, "shallow")
        classified_intent, intent_conf = self._classify_intent(query, files)

        collector.start_step("wiki_search")
        collector.end_step("wiki_search", len(ctx.concepts), len(ctx.concepts))

        collector.start_step("filter_structural")
        collector.end_step("filter_structural", len(ctx.concepts), len(ctx.concepts), {
            "filtered": ctx.filtered_count,
        })

        for c in ctx.concepts:
            trace = ResultTrace(
                chunk_index=c.chunk_indices[0] if c.chunk_indices else None,
                concept_name=c.concept,
                structural_score=0.0,
                merge_reason=None,
                gap_filled=False,
                gap_indices=[],
                score_breakdown={"final": c.score},
                structural_penalty=0.0,
                final_score=c.score,
            )
            collector.add_result_trace(trace)

        best_score = max((c.score for c in ctx.concepts), default=0.0)
        scores = [c.score for c in ctx.concepts]
        variance = max(scores) - min(scores) if len(scores) > 1 else 0.0

        failure = FailureClassifier.classify(
            total_results=result.total_items,
            filtered_count=ctx.filtered_count,
            fallback_used=False,
            best_score=best_score,
            score_variance=variance,
            modality="shallow",
        )

        conf_score, conf_level = ConfidenceScorer.score(
            deep_results=[],
            shallow_results=ctx.concepts,
            best_deep_score=0.0,
            best_shallow_score=best_score,
            score_variance=variance,
        )

        suggested_formats = []
        if intent and hasattr(intent, "suggested_formats"):
            suggested_formats = [f.value for f in intent.suggested_formats]

        audit = StructuralAudit.audit(ctx.concepts, suggested_formats)
        suggested = IntentPresentationMapper.map(
            intent=classified_intent,
            suggested_formats=suggested_formats,
            query=query,
            traces=collector.result_traces,
            audit=audit,
        )

        return collector.build(
            intent=classified_intent,
            intent_confidence=intent_conf,
            total_results=result.total_items,
            filtered_count=ctx.filtered_count,
            fallback_used=False,
            failure_type=failure,
            confidence_score=conf_score,
            confidence_level=conf_level,
            suggested_blocks=suggested,
            audit_result={
                "scores": audit.scores,
                "best_format": audit.best_format,
                "best_score": audit.best_score,
                "missing": audit.missing,
            },
        )

    def search_deep(
        self,
        query: str,
        files: List[str],
        options: Optional[DeepSearchOptions] = None,
        diagnose: bool = False,
        intent: Optional[Any] = None,
    ) -> Tuple[List[QueryResult], Optional[List[RetrievalDiagnostics]]]:
        if not files:
            raise ValueError("At least one file required for deep search")

        opts = options or DeepSearchOptions()
        results: List[QueryResult] = []
        diagnostics: List[RetrievalDiagnostics] = []

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

            if diagnose:
                diag = self._diagnose_deep(result, query, files, intent=intent)
                diagnostics.append(diag)

        return results, diagnostics if diagnose else None

    def search_shallow(
        self,
        query: str,
        files: List[str],
        options: Optional[ShallowSearchOptions] = None,
        diagnose: bool = False,
        intent: Optional[Any] = None,
    ) -> Tuple[List[QueryResult], Optional[List[RetrievalDiagnostics]]]:
        if not files:
            raise ValueError("At least one file required for shallow search")

        opts = options or ShallowSearchOptions()
        results: List[QueryResult] = []
        diagnostics: List[RetrievalDiagnostics] = []

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

            if diagnose:
                diag = self._diagnose_shallow(result, query, files, intent=intent)
                diagnostics.append(diag)

        return results, diagnostics if diagnose else None

    def search(
        self,
        query: str,
        files: List[str],
        deep_options: Optional[DeepSearchOptions] = None,
        shallow_options: Optional[ShallowSearchOptions] = None,
        return_both: bool = False,
        diagnose: bool = False,
        intent: Optional[Any] = None,
    ) -> Tuple[Any, Optional[List[RetrievalDiagnostics]]]:
        if not files:
            raise ValueError("At least one file required")

        all_diagnostics: List[RetrievalDiagnostics] = []

        if return_both:
            deep_results, deep_diag = self.search_deep(query, files, deep_options, diagnose=diagnose, intent=intent)
            shallow_results, shallow_diag = self.search_shallow(query, files, shallow_options, diagnose=diagnose, intent=intent)
            if diagnose:
                all_diagnostics.extend(deep_diag or [])
                all_diagnostics.extend(shallow_diag or [])
            return (deep_results, shallow_results), all_diagnostics if diagnose else None

        results: List[QueryResult] = []
        for filename in files:
            deep_opts = deep_options or DeepSearchOptions()
            shallow_opts = shallow_options or ShallowSearchOptions()

            deep_results, deep_diag = self.search_deep(query, [filename], deep_opts, diagnose=diagnose, intent=intent)
            deep_result = deep_results[0]

            if diagnose and deep_diag:
                all_diagnostics.extend(deep_diag)

            if deep_result.total_items > 0:
                results.append(deep_result)
                continue

            darwin.log(f"[QueryOrchestrator] Deep empty for {filename}, trying shallow", LogType.INFO)
            shallow_results, shallow_diag = self.search_shallow(query, [filename], shallow_opts, diagnose=diagnose, intent=intent)
            shallow_result = shallow_results[0]

            if diagnose and shallow_diag:
                all_diagnostics.extend(shallow_diag)

            results.append(shallow_result)

        return (results[0] if len(results) == 1 else results), all_diagnostics if diagnose else None

    def build_llm_prompt(
        self,
        results: List[QueryResult],
        query: str,
        include_related: bool = True,
    ) -> str:
        if not results:
            return f"Question: {query}\n\nNo relevant sources found."

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

    def ask_deep(
        self,
        query: str,
        files: List[str],
        options: Optional[DeepSearchOptions] = None,
    ) -> Tuple[str, List[QueryResult]]:
        results, _ = self.search_deep(query, files, options, diagnose=False)
        prompt = self.build_llm_prompt(results, query)
        return prompt, results

    def ask_shallow(
        self,
        query: str,
        files: List[str],
        options: Optional[ShallowSearchOptions] = None,
    ) -> Tuple[str, List[QueryResult]]:
        results, _ = self.search_shallow(query, files, options, diagnose=False)
        prompt = self.build_llm_prompt(results, query)
        return prompt, results

    def ask(
        self,
        query: str,
        files: List[str],
        deep_options: Optional[DeepSearchOptions] = None,
        shallow_options: Optional[ShallowSearchOptions] = None,
    ) -> Tuple[str, List[QueryResult]]:
        deep_results, _ = self.search_deep(query, files, deep_options, diagnose=False)
        shallow_results, _ = self.search_shallow(query, files, shallow_options, diagnose=False)
        all_results = deep_results + shallow_results
        prompt = self.build_llm_prompt(all_results, query)
        return prompt, all_results