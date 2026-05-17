"""
src/core/Knowledge/RetrievalDiagnostics.py
-------------------------------------------
Retrieval observability, intent classification, failure taxonomy,
confidence scoring, and honest presentation block generation.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time

from src.core.Logger.DarwinLogger import darwin, LogType


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class RetrievalIntent(Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    TIMELINE = "timeline"
    EXPLORATION = "exploration"
    SUMMARIZATION = "summary"
    CONTRADICTION = "contradiction"
    UNKNOWN = "unknown"


class FailureType(Enum):
    NONE = "none"
    NO_SEMANTIC_MATCH = "no_semantic_match"
    SPARSE_EVIDENCE = "sparse_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    OVERLY_BROAD_QUERY = "overly_broad_query"
    OVER_FRAGMENTED = "over_fragmented"
    LOW_CONFIDENCE = "low_confidence"
    ALL_STRUCTURAL = "all_structural"
    RERANK_FAILURE = "rerank_failure"


class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


# ═══════════════════════════════════════════════════════════════════════════
# Presentation block — honest visual suggestion
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PresentationBlock:
    id: str
    type: str
    title: str
    description: str
    presentation_goal: str = "inform"
    interaction_intents: List[str] = field(default_factory=list)
    priority: int = 1
    payload: Dict[str, Any] = field(default_factory=dict)
    semantic_blocks: List[Dict[str, Any]] = field(default_factory=list)
    audit_score: float = 0.0
    fallback_from: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "presentation_goal": self.presentation_goal,
            "interaction_intents": self.interaction_intents,
            "priority": self.priority,
            "payload": self.payload,
            "semantic_blocks": self.semantic_blocks,
            "audit_score": self.audit_score,
            "fallback_from": self.fallback_from,
        }

    def __str__(self) -> str:
        downgrade = f" (downgraded from {self.fallback_from})" if self.fallback_from else ""
        return f"  [{self.type}] {self.title} (audit={self.audit_score:.2f}){downgrade}"


# ═══════════════════════════════════════════════════════════════════════════
# Presentation patch — for progressive disclosure
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PresentationPatch:
    op: str
    target_block_id: Optional[str]
    path: str
    value: Any
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "target_block_id": self.target_block_id,
            "path": self.path,
            "value": self.value,
            "reason": self.reason,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Step trace
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineStep:
    name: str
    input_count: int
    output_count: int
    dropped: int
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        drop_info = f" | dropped={self.dropped}" if self.dropped > 0 else ""
        return f"  [{self.name}] {self.input_count} → {self.output_count}{drop_info} ({self.duration_ms:.1f}ms)"


# ═══════════════════════════════════════════════════════════════════════════
# Per-result trace
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResultTrace:
    chunk_index: Optional[int]
    concept_name: Optional[str]
    structural_score: float
    merge_reason: Optional[str]
    gap_filled: bool
    gap_indices: List[int]
    score_breakdown: Dict[str, float]
    structural_penalty: float
    final_score: float

    def __str__(self) -> str:
        lines = [
            f"    Result: {self.concept_name or f'chunk[{self.chunk_index}]'}",
            f"      structural_score={self.structural_score:.2f}",
            f"      merge_reason={self.merge_reason or 'none'}",
            f"      gap_filled={self.gap_filled} gaps={self.gap_indices}",
            f"      scores={self.score_breakdown}",
            f"      final={self.final_score:.3f}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main diagnostics container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalDiagnostics:
    query: str
    file: str
    modality: str

    intent: str
    intent_confidence: float

    steps: List[PipelineStep]
    result_traces: List[ResultTrace]

    total_results: int
    filtered_count: int
    fallback_used: bool
    failure_type: str
    confidence_level: str
    confidence_score: float

    suggested_blocks: List[PresentationBlock] = field(default_factory=list)
    audit_result: Optional[Dict[str, Any]] = None
    raw_scores: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"DIAGNOSTICS: {self.modality.upper()} | {self.file}",
            f"Query: '{self.query}'",
            f"Intent: {self.intent} (confidence={self.intent_confidence:.2f})",
            f"{'='*60}",
            "Pipeline:",
        ]
        for step in self.steps:
            lines.append(str(step))
        lines.extend(["\nResult Traces:"])
        for trace in self.result_traces:
            lines.append(str(trace))
        lines.extend([
            f"\nSuggested Blocks:",
        ])
        for block in self.suggested_blocks:
            lines.append(str(block))
        lines.extend([
            f"\nOutcome: {self.total_results} results",
            f"  filtered={self.filtered_count} fallback={self.failure_type if self.fallback_used else 'no'}",
            f"  confidence={self.confidence_level} ({self.confidence_score:.3f})",
            f"{'='*60}",
        ])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "file": self.file,
            "modality": self.modality,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "steps": [
                {
                    "name": s.name,
                    "input_count": s.input_count,
                    "output_count": s.output_count,
                    "dropped": s.dropped,
                    "duration_ms": s.duration_ms,
                    "details": s.details,
                }
                for s in self.steps
            ],
            "result_traces": [
                {
                    "chunk_index": t.chunk_index,
                    "concept_name": t.concept_name,
                    "structural_score": t.structural_score,
                    "merge_reason": t.merge_reason,
                    "gap_filled": t.gap_filled,
                    "gap_indices": t.gap_indices,
                    "score_breakdown": t.score_breakdown,
                    "structural_penalty": t.structural_penalty,
                    "final_score": t.final_score,
                }
                for t in self.result_traces
            ],
            "suggested_blocks": [b.to_dict() for b in self.suggested_blocks],
            "audit_result": self.audit_result,
            "total_results": self.total_results,
            "filtered_count": self.filtered_count,
            "fallback_used": self.fallback_used,
            "failure_type": self.failure_type,
            "confidence_level": self.confidence_level,
            "confidence_score": self.confidence_score,
            "raw_scores": self.raw_scores,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Intent classifier
# ═══════════════════════════════════════════════════════════════════════════

_INTENT_PROMPT = """Classify the user's research query intent.

Query: "{query}"
Files available: {files}

Choose ONE intent:
- factual: specific entity lookup
- comparison: comparing across sources
- timeline: chronological sequence
- exploration: broad topic exploration
- summarization: document overview
- contradiction: detecting agreement/conflict

Return ONLY JSON: {{"intent": "...", "confidence": 0.0-1.0}}"""


class IntentClassifier:
    def __init__(self, llm_client: Callable):
        self._call = llm_client

    def classify(self, query: str, files: List[str]) -> tuple[str, float]:
        prompt = _INTENT_PROMPT.format(
            query=query,
            files=", ".join(files) if files else "none",
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            resp = self._call(messages)
            if resp.error or not resp.text:
                darwin.log(f"Intent classification failed: {resp.error}", LogType.WARNING)
                return RetrievalIntent.UNKNOWN.value, 0.0
            text = resp.text.strip()
            if "```" in text:
                for part in text.split("```"):
                    clean = part.replace("json", "").strip()
                    if clean.startswith("{"):
                        text = clean
                        break
            parsed = json.loads(text)
            intent = parsed.get("intent", "unknown")
            confidence = float(parsed.get("confidence", 0.5))
            valid = {i.value for i in RetrievalIntent}
            if intent not in valid:
                intent = RetrievalIntent.UNKNOWN.value
            return intent, confidence
        except Exception as e:
            darwin.log(f"Intent classification error: {e}", LogType.WARNING)
            return RetrievalIntent.UNKNOWN.value, 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Failure classifier
# ═══════════════════════════════════════════════════════════════════════════

class FailureClassifier:
    @staticmethod
    def classify(
        total_results: int,
        filtered_count: int,
        fallback_used: bool,
        best_score: float,
        score_variance: float,
        modality: str,
    ) -> FailureType:
        if total_results == 0:
            if filtered_count > 0 and not fallback_used:
                return FailureType.ALL_STRUCTURAL
            if best_score < 0.1:
                return FailureType.NO_SEMANTIC_MATCH
            return FailureType.SPARSE_EVIDENCE
        if total_results > 0 and best_score < 0.3:
            return FailureType.LOW_CONFIDENCE
        if score_variance > 0.5:
            return FailureType.CONFLICTING_EVIDENCE
        if total_results > 5 and best_score < 0.5:
            return FailureType.OVERLY_BROAD_QUERY
        if total_results > 0 and filtered_count > total_results * 2:
            return FailureType.OVER_FRAGMENTED
        return FailureType.NONE


# ═══════════════════════════════════════════════════════════════════════════
# Confidence scorer
# ═══════════════════════════════════════════════════════════════════════════

class ConfidenceScorer:
    @staticmethod
    def score(
        deep_results: List[Any],
        shallow_results: List[Any],
        best_deep_score: float,
        best_shallow_score: float,
        score_variance: float,
    ) -> tuple[float, ConfidenceLevel]:
        factors = []
        if deep_results and shallow_results:
            factors.append(0.9)
        elif deep_results or shallow_results:
            factors.append(0.6)
        else:
            factors.append(0.0)
        best = max(best_deep_score, best_shallow_score, 0.0)
        factors.append(min(best * 1.2, 1.0))
        consistency = 1.0 - min(score_variance, 1.0)
        factors.append(consistency)
        score = sum(factors) / len(factors)
        if score > 0.8:
            level = ConfidenceLevel.HIGH
        elif score > 0.5:
            level = ConfidenceLevel.MEDIUM
        elif score > 0.3:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.UNCERTAIN
        return score, level


# ═══════════════════════════════════════════════════════════════════════════
# Intent → Presentation mapper (AUDITED)
# ═══════════════════════════════════════════════════════════════════════════

class IntentPresentationMapper:
    FALLBACK_CHAIN = {
        "comparison_table": "pros_cons",
        "pros_cons": "markdown",
        "network_graph": "relationship_map",
        "relationship_map": "profile_card",
        "profile_card": "markdown",
        "timeline": "stats_grid",
        "stats_grid": "markdown",
        "parameter_sandbox": "stats_grid",
        "summary_card": "markdown",
        "flowchart": "stepper",
        "stepper": "markdown",
        "diagnostic_tree": "markdown",
        "bar_chart": "stats_grid",
    }

    @classmethod
    def map(
        cls,
        intent: str,
        suggested_formats: List[str],
        query: str,
        traces: List[ResultTrace],
        audit: Any,
    ) -> List[PresentationBlock]:
        blocks: List[PresentationBlock] = []
        intent_enum = RetrievalIntent(intent) if any(intent == i.value for i in RetrievalIntent) else RetrievalIntent.UNKNOWN

        formats = suggested_formats or []
        if not formats:
            formats = [cls._default_format_for_intent(intent_enum)]

        chosen_format = None
        chosen_score = 0.0
        for fmt in formats:
            score = audit.scores.get(fmt, 0.0)
            if score >= 0.7:
                chosen_format = fmt
                chosen_score = score
                break
            elif score >= 0.4:
                fallback = cls.FALLBACK_CHAIN.get(fmt, "markdown")
                chosen_format = fallback
                chosen_score = audit.scores.get(fallback, 1.0)
                break

        if not chosen_format:
            chosen_format = "markdown"
            chosen_score = 1.0

        primary = cls._build_block(
            chosen_format, query, traces, chosen_score,
            fallback_from=(formats[0] if formats and chosen_format != formats[0] else None)
        )
        blocks.append(primary)

        # Stats grid secondary
        stats = PresentationBlock(
            id=f"stats_{hash(query) & 0xFFFFFF}",
            type="stats_grid",
            title="Retrieval Stats",
            description="Diagnostic overview",
            presentation_goal="inform",
            priority=2,
            audit_score=1.0,
            payload={
                "results_found": len(traces),
                "best_score": round(max((t.final_score for t in traces), default=0.0), 3),
                "avg_score": round(sum(t.final_score for t in traces) / len(traces), 3) if traces else 0.0,
                "audit_scores": {k: round(v, 2) for k, v in audit.scores.items()},
            },
        )
        blocks.append(stats)

        # Downgrade hint
        if primary.fallback_from and audit.best_score < 0.7:
            hint = PresentationBlock(
                id=f"hint_{hash(query) & 0xFFFFFF}",
                type="summary_card",
                title="Could be richer",
                description=f"A {primary.fallback_from} needs: {', '.join(audit.missing.get(primary.fallback_from, []))}",
                presentation_goal="inform",
                priority=3,
                audit_score=1.0,
                payload={"missing_for_rich_render": audit.missing.get(primary.fallback_from, [])},
            )
            blocks.append(hint)

        return blocks

    @classmethod
    def _default_format_for_intent(cls, intent: RetrievalIntent) -> str:
        return {
            RetrievalIntent.FACTUAL: "profile_card",
            RetrievalIntent.COMPARISON: "comparison_table",
            RetrievalIntent.TIMELINE: "timeline",
            RetrievalIntent.EXPLORATION: "network_graph",
            RetrievalIntent.SUMMARIZATION: "summary_card",
            RetrievalIntent.CONTRADICTION: "pros_cons",
            RetrievalIntent.UNKNOWN: "markdown",
        }.get(intent, "markdown")

    @classmethod
    def _build_block(
        cls,
        fmt: str,
        query: str,
        traces: List[ResultTrace],
        score: float,
        fallback_from: Optional[str] = None,
    ) -> PresentationBlock:
        block_id = f"{fmt}_{hash(query) & 0xFFFFFF}"
        base_payload = {
            "query": query,
            "source_chunks": [t.chunk_index for t in traces if t.chunk_index is not None][:5],
            "scores": [round(t.final_score, 3) for t in traces],
        }

        if fmt == "comparison_table" or fmt == "pros_cons":
            return PresentationBlock(
                id=block_id,
                type=fmt,
                title=f"Comparing: {query}",
                description="Side-by-side evidence" if fmt == "comparison_table" else "For and against",
                presentation_goal="compare",
                interaction_intents=["compare", "drill_down"],
                audit_score=score,
                fallback_from=fallback_from,
                payload={
                    **base_payload,
                    "rows": [
                        {"source": f"chunk[{t.chunk_index}]", "evidence": t.merge_reason or "retrieved", "score": round(t.final_score, 3)}
                        for t in traces[:5]
                    ],
                },
            )

        if fmt == "timeline":
            return PresentationBlock(
                id=block_id,
                type=fmt,
                title="Chronological Evidence",
                description=f"Events related to: {query}",
                presentation_goal="explore",
                interaction_intents=["explore", "drill_down"],
                audit_score=score,
                fallback_from=fallback_from,
                payload={
                    **base_payload,
                    "events": [
                        {"label": f"chunk[{t.chunk_index}]", "detail": t.merge_reason or "retrieved", "confidence": round(t.final_score, 3)}
                        for t in traces[:5]
                    ],
                },
            )

        if fmt == "network_graph" or fmt == "relationship_map":
            nodes = [{"id": "query", "label": query, "type": "root"}]
            edges = []
            for i, t in enumerate(traces[:6]):
                nid = f"chunk_{t.chunk_index or i}"
                nodes.append({"id": nid, "label": f"chunk[{t.chunk_index}]", "type": "evidence", "score": round(t.final_score, 3)})
                edges.append({"source": "query", "target": nid, "label": "relates_to"})
            return PresentationBlock(
                id=block_id,
                type=fmt,
                title=f"Exploring: {query}",
                description="Relationship map of evidence",
                presentation_goal="explore",
                interaction_intents=["explore", "filter", "drill_down"],
                audit_score=score,
                fallback_from=fallback_from,
                payload={"nodes": nodes, "edges": edges},
            )

        if fmt == "profile_card":
            entity = query.strip("?").split()[-1] if query else "entity"
            return PresentationBlock(
                id=block_id,
                type=fmt,
                title=f"About: {entity.title()}",
                description=f"Factual lookup for '{query}'",
                presentation_goal="inform",
                interaction_intents=["drill_down", "verify"],
                audit_score=score,
                fallback_from=fallback_from,
                payload={
                    **base_payload,
                    "entity_name": entity,
                    "top_evidence": [t.chunk_index for t in traces[:3]],
                },
            )

        if fmt == "summary_card":
            return PresentationBlock(
                id=block_id,
                type=fmt,
                title="Summary",
                description=f"Overview for: {query}",
                presentation_goal="summarize",
                interaction_intents=["expand", "drill_down"],
                audit_score=score,
                fallback_from=fallback_from,
                payload={
                    **base_payload,
                    "source_count": len(traces),
                    "top_score": round(max((t.final_score for t in traces), default=0.0), 3),
                },
            )

        if fmt == "stats_grid":
            return PresentationBlock(
                id=block_id,
                type=fmt,
                title="Retrieval Stats",
                description="Diagnostic numbers",
                presentation_goal="inform",
                audit_score=score,
                fallback_from=fallback_from,
                payload={
                    "results_found": len(traces),
                    "best_score": round(max((t.final_score for t in traces), default=0.0), 3),
                    "avg_score": round(sum(t.final_score for t in traces) / len(traces), 3) if traces else 0.0,
                },
            )

        if fmt == "markdown":
            return PresentationBlock(
                id=block_id,
                type="markdown",
                title=f"Results for: {query}",
                description="Retrieved passages and evidence",
                presentation_goal="inform",
                interaction_intents=["expand", "drill_down"],
                audit_score=score,
                fallback_from=fallback_from,
                payload={
                    **base_payload,
                    "passages": [
                        {
                            "index": t.chunk_index,
                            "score": round(t.final_score, 3),
                            "text": f"**chunk[{t.chunk_index}]** (score: {round(t.final_score, 3)}): {t.merge_reason or 'retrieved'}",
                        }
                        for t in traces
                    ],
                    "raw_query": query,
                },
            )

        # Ultimate fallback — markdown
        return PresentationBlock(
            id=block_id,
            type="markdown",
            title="Retrieved Evidence",
            description=f"Results for: {query}",
            presentation_goal="inform",
            audit_score=score,
            fallback_from=fallback_from,
            payload={
                **base_payload,
                "passages": [
                    {
                        "index": t.chunk_index,
                        "score": round(t.final_score, 3),
                        "text": f"**chunk[{t.chunk_index}]** (score: {round(t.final_score, 3)}): {t.merge_reason or 'retrieved'}",
                    }
                    for t in traces
                ],
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic collector
# ═══════════════════════════════════════════════════════════════════════════

class DiagnosticCollector:
    def __init__(self, query: str, file: str, modality: str):
        self.query = query
        self.file = file
        self.modality = modality
        self.steps: List[PipelineStep] = []
        self.result_traces: List[ResultTrace] = []
        self._step_start: Optional[float] = None

    def start_step(self, name: str):
        self._step_start = time.perf_counter() * 1000

    def end_step(self, name: str, input_count: int, output_count: int, details: Optional[Dict] = None):
        duration = (time.perf_counter() * 1000) - self._step_start if self._step_start else 0
        self.steps.append(PipelineStep(
            name=name,
            input_count=input_count,
            output_count=output_count,
            dropped=input_count - output_count,
            duration_ms=duration,
            details=details or {},
        ))

    def add_result_trace(self, trace: ResultTrace):
        self.result_traces.append(trace)

    def build(
        self,
        intent: str,
        intent_confidence: float,
        total_results: int,
        filtered_count: int,
        fallback_used: bool,
        failure_type: FailureType,
        confidence_score: float,
        confidence_level: ConfidenceLevel,
        suggested_blocks: Optional[List[PresentationBlock]] = None,
        audit_result: Optional[Dict] = None,
        raw_scores: Optional[Dict] = None,
    ) -> RetrievalDiagnostics:
        return RetrievalDiagnostics(
            query=self.query,
            file=self.file,
            modality=self.modality,
            intent=intent,
            intent_confidence=intent_confidence,
            steps=self.steps,
            result_traces=self.result_traces,
            total_results=total_results,
            filtered_count=filtered_count,
            fallback_used=fallback_used,
            failure_type=failure_type.value,
            confidence_level=confidence_level.value,
            confidence_score=confidence_score,
            suggested_blocks=suggested_blocks or [],
            audit_result=audit_result,
            raw_scores=raw_scores or {},
        )