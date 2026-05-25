"""
Darwin Query Module — Timeline Skill
Generates chronological event structures with zoom levels.
"""

from typing import Any, Dict, List, Optional

from src.features.QueryModule.types import (
    BlobHandle,
    ConversationContext,
    RepresentationSkill,
    ResponsePresentation,
    RetrievalContext,
    SkillResult,
)
from src.features.QueryModule.schemas import TIMELINE_SCHEMA


class TimelineSkill(RepresentationSkill):
    @property
    def name(self) -> str:
        return "timeline"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["query", "passages"],
            "properties": {
                "query": {"type": "string"},
                "passages": {"type": "array", "items": {"type": "string"}},
            },
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return TIMELINE_SCHEMA

    def execute(
        self,
        query: str,
        context: RetrievalContext,
        conversation: ConversationContext,
        blob: Optional[BlobHandle] = None,
    ) -> SkillResult:
        passages = [getattr(p, "content", str(p)) for p in context.passages]
        events = self._extract_events(passages)
        if not events:
            events = [
                {"date": "Unknown", "title": "No dated events found", "description": "Context lacks chronological data.", "category": "general", "importance": "low"},
            ]

        categories = sorted({e.get("category", "general") for e in events})
        data = {
            "events": events,
            "zoom_levels": ["all", "year", "decade", "century"],
            "categories": categories,
        }
        valid, errors = self.validate_output(data)
        if not valid:
            return SkillResult(success=False, errors=errors, fallback_text="")

        return SkillResult(
            success=True,
            data=data,
            fallback_text=self._generate_fallback(data, query),
        )

    def _extract_events(self, passages: List[str]) -> List[Dict[str, Any]]:
        import re
        events = []
        date_patterns = [
            (re.compile(r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b"), "numeric"),
            (re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b", re.I), "month_day_year"),
            (re.compile(r"\b(\d{4})\b"), "year_only"),
            (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", re.I), "day_month_year"),
        ]

        for text in passages:
            sentences = re.split(r"[.!?]\s+", text)
            for sent in sentences:
                for pattern, ptype in date_patterns:
                    match = pattern.search(sent)
                    if match:
                        date_str = match.group(0)
                        # Extract a title from the sentence
                        words = sent.split()
                        title = " ".join(words[:6]) + ("..." if len(words) > 6 else "")
                        events.append({
                            "date": date_str,
                            "title": title,
                            "description": sent.strip(),
                            "category": "general",
                            "importance": "medium",
                        })
                        break  # One event per sentence max
        # Deduplicate by date+title
        seen = set()
        unique = []
        for e in events:
            key = (e["date"], e["title"])
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique[:20]

    def _generate_fallback(self, data: Dict[str, Any], query: str) -> str:
        lines = [f"## Timeline: {query}", ""]
        for e in data.get("events", []):
            lines.append(f"**{e['date']}** — {e['title']}")
            lines.append(f"  {e['description']}")
            lines.append("")
        return "\n".join(lines)