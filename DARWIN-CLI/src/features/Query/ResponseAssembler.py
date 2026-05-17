"""
src/core/Agent/ResponseAssembler.py
------------------------------------
Assembles the final response envelope from:
  - QueryIntent (from ClarificationGate)
  - QueryResults (from retrieval)
  - RetrievalDiagnostics (from audit)

Implements progressive disclosure:
  1. Return honest primary blocks immediately
  2. If data supports richer visualization, include a patch
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.features.Query.ClarificationGate import QueryIntent
from src.features.Query.QueryOrchestrator import QueryResult
from src.features.Query.RetrievalDiagnostics import (
    RetrievalDiagnostics,
    PresentationBlock,
    PresentationPatch,
)


@dataclass
class ResponseEnvelope:
    query: str
    primary_blocks: List[PresentationBlock] = field(default_factory=list)
    patches: List[PresentationPatch] = field(default_factory=list)
    status_events: List[Dict[str, Any]] = field(default_factory=list)
    conversation_state: Dict[str, Any] = field(default_factory=dict)
    pending_expansion: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "primary_blocks": [b.to_dict() for b in self.primary_blocks],
            "patches": [p.to_dict() for p in self.patches],
            "status_events": self.status_events,
            "pending_expansion": self.pending_expansion,
            "conversation_state": self.conversation_state,
        }


class ResponseAssembler:
    def assemble(
        self,
        intent: QueryIntent,
        results: List[QueryResult],
        diagnostics: List[RetrievalDiagnostics],
    ) -> ResponseEnvelope:
        events: List[Dict[str, Any]] = []
        events.append({
            "event": "intent_locked",
            "intent": intent.question_types[0].value if intent.question_types else "unknown"
        })

        all_blocks: List[PresentationBlock] = []
        all_patches: List[PresentationPatch] = []

        for result, diag in zip(results, diagnostics):
            for block in diag.suggested_blocks:
                block.payload["source_file"] = result.source
                all_blocks.append(block)

            primary = next((b for b in diag.suggested_blocks if b.priority == 1), None)
            if primary and primary.fallback_from:
                cross = self._cross_file_boost(primary.fallback_from, results, diagnostics)
                if cross:
                    patch = self._build_upgrade_patch(primary, cross, result.source)
                    all_patches.append(patch)

        seen = set()
        deduped = []
        for b in all_blocks:
            if b.id not in seen:
                seen.add(b.id)
                deduped.append(b)

        deduped.sort(key=lambda b: (b.priority, -b.audit_score))

        return ResponseEnvelope(
            query=intent.clean_query,
            primary_blocks=deduped,
            patches=all_patches,
            status_events=events,
            pending_expansion=bool(all_patches),
        )

    def _cross_file_boost(
        self,
        target_format: str,
        results: List[QueryResult],
        diagnostics: List[RetrievalDiagnostics],
    ) -> Optional[Dict[str, Any]]:
        if len(results) < 2:
            return None
        if target_format == "comparison_table":
            sources = [r.source for r in results if r.total_items > 0]
            if len(set(sources)) >= 2:
                return {"reason": "multi_file_evidence_found", "sources": sources}
        return None

    def _build_upgrade_patch(
        self,
        current: PresentationBlock,
        boost: Dict[str, Any],
        source: str,
    ) -> PresentationPatch:
        rich = {
            "id": f"{current.id}_upgraded",
            "type": current.fallback_from or "comparison_table",
            "title": current.title.replace("Comparing", "Full Comparison"),
            "description": f"Cross-source {current.fallback_from} (upgraded from single-file accordion)",
            "presentation_goal": current.presentation_goal,
            "audit_score": 0.75,
            "payload": {
                **current.payload,
                "upgraded": True,
                "upgrade_reason": boost.get("reason"),
                "sources": boost.get("sources"),
            },
        }
        return PresentationPatch(
            op="add",
            target_block_id=current.id,
            path="blocks",
            value=rich,
            reason=f"cross_file_boost: {boost.get('reason')}",
        )