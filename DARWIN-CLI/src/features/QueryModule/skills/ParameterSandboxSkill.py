"""
Darwin Query Module — Parameter Sandbox Skill
Generates interactive variable sliders + outcome matrices for what-if analysis.
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
from src.features.QueryModule.schemas import PARAMETER_SANDBOX_SCHEMA


class ParameterSandboxSkill(RepresentationSkill):
    @property
    def name(self) -> str:
        return "parameter_sandbox"

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
        return PARAMETER_SANDBOX_SCHEMA

    def execute(
        self,
        query: str,
        context: RetrievalContext,
        conversation: ConversationContext,
        blob: Optional[BlobHandle] = None,
    ) -> SkillResult:
        passages = [getattr(p, "content", str(p)) for p in context.passages]
        variables = self._extract_variables(passages)
        if len(variables) < 1:
            variables = [
                {"id": "var_a", "label": "Parameter A", "type": "slider", "min": 0, "max": 100, "step": 1, "default": 50},
                {"id": "var_b", "label": "Parameter B", "type": "toggle", "default": True},
            ]

        outcomes = [
            {"id": "out_1", "label": "Primary Outcome", "formula": "var_a * 2", "description": "Depends on Parameter A"},
            {"id": "out_2", "label": "Secondary Outcome", "formula": "var_b ? 100 : 0", "description": "Depends on Parameter B"},
        ]

        scenarios = [
            {"name": "Default", "values": {v["id"]: v.get("default", 0) for v in variables}},
            {"name": "High Performance", "values": {v["id"]: v.get("max", 100) for v in variables if "max" in v}},
        ]

        data = {
            "variables": variables,
            "outcomes": outcomes,
            "scenarios": scenarios,
        }
        valid, errors = self.validate_output(data)
        if not valid:
            return SkillResult(success=False, errors=errors, fallback_text="")

        return SkillResult(
            success=True,
            data=data,
            fallback_text=self._generate_fallback(data, query),
        )

    def _extract_variables(self, passages: List[str]) -> List[Dict[str, Any]]:
        import re
        variables = []
        # Look for numeric patterns and percentages
        for text in passages:
            # Find patterns like "X%" or "between 10 and 50"
            pct_matches = re.findall(r"(\d+(?:\.\d+)?)%", text)
            range_matches = re.findall(r"between (\d+) and (\d+)", text)
            for m in pct_matches[:2]:
                variables.append({
                    "id": f"pct_{len(variables)}",
                    "label": f"Percentage {len(variables)+1}",
                    "type": "slider",
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "default": int(float(m)),
                })
            for lo, hi in range_matches[:2]:
                variables.append({
                    "id": f"range_{len(variables)}",
                    "label": f"Range {len(variables)+1}",
                    "type": "slider",
                    "min": int(lo),
                    "max": int(hi),
                    "step": max(1, (int(hi) - int(lo)) // 10),
                    "default": (int(lo) + int(hi)) // 2,
                })
        return variables[:6]

    def _generate_fallback(self, data: Dict[str, Any], query: str) -> str:
        lines = [f"## Parameter Sandbox: {query}", ""]
        lines.append("**Variables:**")
        for v in data.get("variables", []):
            lines.append(f"- {v['label']} ({v['type']}): default={v.get('default')}, range={v.get('min')}–{v.get('max')}")
        lines.append("")
        lines.append("**Outcomes:**")
        for o in data.get("outcomes", []):
            lines.append(f"- {o['label']}: {o['description']} (formula: {o.get('formula', 'n/a')})")
        return "\n".join(lines)