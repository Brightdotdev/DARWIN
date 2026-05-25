"""
Darwin Query Module — Comparison Table Skill
Generates structured comparison table data with sortable/filterable schema.
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
from src.features.QueryModule.schemas import COMPARISON_TABLE_SCHEMA


class ComparisonTableSkill(RepresentationSkill):
    @property
    def name(self) -> str:
        return "comparison_table"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["query", "passages", "entities"],
            "properties": {
                "query": {"type": "string"},
                "passages": {"type": "array", "items": {"type": "string"}},
                "entities": {"type": "array", "items": {"type": "string"}},
            },
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return COMPARISON_TABLE_SCHEMA

    def execute(
        self,
        query: str,
        context: RetrievalContext,
        conversation: ConversationContext,
        blob: Optional[BlobHandle] = None,
    ) -> SkillResult:
        # Extract entities from passages using simple heuristics
        entities = self._extract_entities(context.passages)
        if len(entities) < 2:
            return SkillResult(
                success=False,
                errors=["Need at least 2 entities for comparison table"],
                fallback_text="",
            )

        # Build table structure
        headers = ["Attribute"] + entities[:6]  # Cap at 6 entities
        rows = []
        # Simple attribute extraction from passage keywords
        attributes = self._extract_attributes(context.passages, entities)
        for attr in attributes[:10]:
            row = {"entity": attr, "values": [], "highlight": False}
            for ent in entities[:6]:
                val = self._find_value_for_entity_attr(context.passages, ent, attr)
                row["values"].append(val)
            rows.append(row)

        data = {
            "headers": headers,
            "rows": rows,
            "sortable": True,
            "filters": entities,
        }

        valid, errors = self.validate_output(data)
        if not valid:
            return SkillResult(success=False, errors=errors, fallback_text="")

        return SkillResult(
            success=True,
            data=data,
            fallback_text=self._generate_fallback(data, query),
        )

    def _extract_entities(self, passages: List[Any]) -> List[str]:
        import re
        candidates = []
        for p in passages:
            text = getattr(p, "content", str(p))
            # Look for capitalized phrases (naive NER)
            found = re.findall(r"\b[A-Z][a-zA-Z\s]{2,20}\b", text)
            candidates.extend(found)
        # Deduplicate preserving order
        seen = set()
        result = []
        for c in candidates:
            c = c.strip()
            if c and c not in seen and len(c) > 2:
                seen.add(c)
                result.append(c)
        return result[:8]

    def _extract_attributes(self, passages: List[Any], entities: List[str]) -> List[str]:
        import re
        # Extract nouns/adjectives that appear near entity mentions
        attrs = ["Overview", "Pros", "Cons", "Key Feature", "Limitation", "Use Case"]
        for p in passages:
            text = getattr(p, "content", str(p))
            # Find sentences mentioning comparison keywords
            if any(kw in text.lower() for kw in ["has", "supports", "provides", "offers", "includes"]):
                words = re.findall(r"\b[a-z]{4,12}\b", text.lower())
                attrs.extend(words)
        seen = set()
        result = []
        for a in attrs:
            if a not in seen:
                seen.add(a)
                result.append(a.title())
        return result[:12]

    def _find_value_for_entity_attr(self, passages: List[Any], entity: str, attr: str) -> Optional[str]:
        for p in passages:
            text = getattr(p, "content", str(p))
            if entity.lower() in text.lower():
                # Return a snippet
                idx = text.lower().find(entity.lower())
                start = max(0, idx - 40)
                end = min(len(text), idx + 100)
                return text[start:end].strip()
        return None

    def _generate_fallback(self, data: Dict[str, Any], query: str) -> str:
        lines = [f"## Comparison: {query}", ""]
        headers = data.get("headers", [])
        rows = data.get("rows", [])
        for row in rows:
            lines.append(f"**{row['entity']}:**")
            for i, val in enumerate(row.get("values", [])):
                if i + 1 < len(headers) and val:
                    lines.append(f"  - {headers[i+1]}: {val}")
            lines.append("")
        return "\n".join(lines)