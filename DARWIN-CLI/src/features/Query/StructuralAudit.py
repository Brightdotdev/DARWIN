"""
src/core/Knowledge/StructuralAudit.py
--------------------------------------
Data-structure audit for retrieved passages.
Returns a richness score (0.0–1.0) per visualization type.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class AuditResult:
    scores: Dict[str, float]          # format -> 0.0-1.0
    missing: Dict[str, List[str]]     # format -> [what's lacking]
    best_format: str
    best_score: float


class StructuralAudit:
    """
    Audits retrieved passages to see if they actually contain
    the semantic primitives required for rich visualizations.
    """

    CONTRAST_MARKERS = {
        "unlike", "whereas", "however", "but", "yet", "although",
        "in contrast", "on the other hand", "differs from", "compared to",
        "more than", "less than", "rather than", "instead of",
        "versus", "vs", "while", "though", "nevertheless",
    }

    TEMPORAL_MARKERS = {
        "before", "after", "then", "first", "second", "third",
        "finally", "later", "earlier", "previously", "subsequently",
        "in the days of", "during", "when", "until", "since",
        "chapter", "book", "verse", "generation",
    }

    RELATIONAL_VERBS = {
        "teaches", "leads to", "associated with", "son of", "father of",
        "born from", "descended from", "created", "gave birth to",
        "appointed", "sent", "fell", "rebelled", "watched over",
        "ruled", "commanded", "taught", "revealed", "showed",
    }

    @staticmethod
    def _extract_entities(text: str) -> set[str]:
        found = set(re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text))
        found.update(re.findall(r'"([^"]{2,40})"', text))
        return {e for e in found if len(e) > 2}

    @classmethod
    def audit(cls, passages: List[Any], target_formats: List[str]) -> AuditResult:
        all_text = " ".join([
            getattr(p, "passage_text", getattr(p, "text", ""))
            for p in passages
        ]).lower()

        entities = cls._extract_entities(all_text)
        scores: Dict[str, float] = {}
        missing: Dict[str, List[str]] = {}

        # Markdown is the canonical honest format — LLMs render it natively
        scores["markdown"] = 1.0
        missing["markdown"] = []

        # Comparison
        if any(f in target_formats for f in ("comparison_table", "pros_cons")):
            contrast_hits = sum(1 for m in cls.CONTRAST_MARKERS if m in all_text)
            entity_count = len(entities)
            score = 0.0
            lacks = []
            if entity_count >= 2:
                score += 0.4
            else:
                lacks.append("need 2+ distinct entities")
            if contrast_hits > 0:
                score += 0.4
            else:
                lacks.append("no contrastive language")
            per_passage_entities = [
                cls._extract_entities(getattr(p, "passage_text", getattr(p, "text", "")))
                for p in passages
            ]
            if len(per_passage_entities) > 1:
                union = len(set.union(*per_passage_entities)) if per_passage_entities else 0
                overlap = len(set.intersection(*per_passage_entities)) if per_passage_entities else 0
                if union > overlap:
                    score += 0.2
                else:
                    lacks.append("passages overlap too much")
            scores["comparison_table"] = min(score, 1.0)
            scores["pros_cons"] = min(score + 0.1, 1.0)
            missing["comparison_table"] = lacks
            missing["pros_cons"] = lacks.copy()

        # Timeline
        if "timeline" in target_formats:
            temporal_hits = sum(1 for m in cls.TEMPORAL_MARKERS if m in all_text)
            event_count = len(re.findall(r'\b(?:then|after|before|first|second|finally)\b', all_text))
            score = 0.0
            lacks = []
            if temporal_hits >= 2:
                score += 0.5
            else:
                lacks.append("need temporal markers")
            if event_count >= 3:
                score += 0.5
            else:
                lacks.append("need 3+ sequential events")
            scores["timeline"] = min(score, 1.0)
            missing["timeline"] = lacks

        # Network / Graph
        if any(f in target_formats for f in ("network_graph", "relationship_map")):
            rel_hits = sum(1 for m in cls.RELATIONAL_VERBS if m in all_text)
            entity_count = len(entities)
            score = 0.0
            lacks = []
            if entity_count >= 3:
                score += 0.5
            else:
                lacks.append("need 3+ named entities")
            if rel_hits >= 2:
                score += 0.5
            else:
                lacks.append("need relational verbs")
            scores["network_graph"] = min(score, 1.0)
            scores["relationship_map"] = min(score + 0.1, 1.0)
            missing["network_graph"] = lacks
            missing["relationship_map"] = lacks.copy()

        # Always possible
        scores["stats_grid"] = 1.0 if passages else 0.0
        scores["summary_card"] = 1.0 if passages else 0.0
        scores["accordion"] = 1.0 if passages else 0.0
        missing["stats_grid"] = []
        missing["summary_card"] = []
        missing["accordion"] = []

        # Parameter sandbox
        if "parameter_sandbox" in target_formats:
            quantities = re.findall(
                r'\b\d+(?:\.\d+)?\s*(?:cubits|feet|meters|years|days|miles|km|kg|lbs|tons|degrees|percent|%)\b',
                all_text, re.IGNORECASE
            )
            score = min(len(quantities) / 2, 1.0)
            scores["parameter_sandbox"] = score
            missing["parameter_sandbox"] = ["need measurable quantities"] if not quantities else []

        # Pick best honest format
        best_format = "markdown"
        best_score = 1.0
        for fmt in target_formats:
            s = scores.get(fmt, 0.0)
            if s > best_score:
                best_score = s
                best_format = fmt

        if best_score < 0.3 and target_formats:
            best_format = "markdown"
            best_score = 1.0

        return AuditResult(
            scores=scores,
            missing=missing,
            best_format=best_format,
            best_score=best_score,
        )