"""
Darwin Query Module — Diagnostic Tree Skill
Generates branching diagnostic trees for troubleshooting queries.
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
from src.features.QueryModule.schemas import DIAGNOSTIC_TREE_SCHEMA


class DiagnosticTreeSkill(RepresentationSkill):
    @property
    def name(self) -> str:
        return "diagnostic_tree"

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
        return DIAGNOSTIC_TREE_SCHEMA

    def execute(
        self,
        query: str,
        context: RetrievalContext,
        conversation: ConversationContext,
        blob: Optional[BlobHandle] = None,
    ) -> SkillResult:
        # Build a simple diagnostic tree from passages
        root_id = "root"
        nodes = {}
        passages = [getattr(p, "content", str(p)) for p in context.passages]

        # Extract symptoms/issues and solutions
        issues = self._extract_issues(passages)
        solutions = self._extract_solutions(passages)

        branches = []
        for i, issue in enumerate(issues[:4]):
            node_id = f"issue_{i}"
            branches.append({
                "label": issue[:60],
                "next": node_id,
                "condition": f"If {issue[:40]}",
            })
            # Link to solution or sub-issue
            if i < len(solutions):
                nodes[node_id] = {
                    "id": node_id,
                    "type": "question",
                    "content": issue,
                    "branches": [{
                        "label": "Apply fix",
                        "next": f"sol_{i}",
                    }],
                }
                nodes[f"sol_{i}"] = {
                    "id": f"sol_{i}",
                    "type": "solution",
                    "solution": solutions[i],
                    "severity": "warning",
                }
            else:
                nodes[node_id] = {
                    "id": node_id,
                    "type": "info",
                    "content": f"Investigate: {issue}",
                }

        root = {
            "id": root_id,
            "question": f"Diagnosing: {query[:80]}",
            "description": "Select the symptom that matches your situation.",
            "branches": branches,
        }
        nodes[root_id] = root

        data = {"root": root, "nodes": nodes}
        valid, errors = self.validate_output(data)
        if not valid:
            return SkillResult(success=False, errors=errors, fallback_text="")

        return SkillResult(
            success=True,
            data=data,
            fallback_text=self._generate_fallback(data, query),
        )

    def _extract_issues(self, passages: List[str]) -> List[str]:
        import re
        markers = ["error", "fail", "issue", "problem", "bug", "crash", "broken", "not working",
                   "slow", "timeout", "exception", "warning"]
        issues = []
        for text in passages:
            sentences = re.split(r"[.!?]\s+", text)
            for sent in sentences:
                if any(m in sent.lower() for m in markers):
                    issues.append(sent.strip())
        return issues[:6] if issues else ["General issue detected"]

    def _extract_solutions(self, passages: List[str]) -> List[str]:
        import re
        markers = ["fix", "solution", "resolve", "workaround", "try", "check", "ensure",
                   "verify", "update", "restart", "configure"]
        solutions = []
        for text in passages:
            sentences = re.split(r"[.!?]\s+", text)
            for sent in sentences:
                if any(m in sent.lower() for m in markers):
                    solutions.append(sent.strip())
        return solutions[:6] if solutions else ["Consult documentation for resolution steps."]

    def _generate_fallback(self, data: Dict[str, Any], query: str) -> str:
        lines = [f"## Diagnostic Tree: {query}", ""]
        nodes = data.get("nodes", {})
        root = data.get("root", {})
        lines.append(f"**Start:** {root.get('question', '')}")
        for b in root.get("branches", []):
            lines.append(f"- {b['label']} → {b['next']}")
        lines.append("")
        for nid, node in nodes.items():
            if nid == "root":
                continue
            t = node.get("type", "info")
            if t == "solution":
                lines.append(f"**Solution [{nid}]:** {node.get('solution', '')}")
            else:
                lines.append(f"**Step [{nid}]:** {node.get('content', '')}")
        return "\n".join(lines)