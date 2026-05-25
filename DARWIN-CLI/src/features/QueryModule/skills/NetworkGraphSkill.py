"""
Darwin Query Module — Network Graph Skill
Generates node/edge graph structures for relationship visualization.
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
from src.features.QueryModule.schemas import NETWORK_GRAPH_SCHEMA


class NetworkGraphSkill(RepresentationSkill):
    @property
    def name(self) -> str:
        return "network_graph"

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
        return NETWORK_GRAPH_SCHEMA

    def execute(
        self,
        query: str,
        context: RetrievalContext,
        conversation: ConversationContext,
        blob: Optional[BlobHandle] = None,
    ) -> SkillResult:
        passages = [getattr(p, "content", str(p)) for p in context.passages]
        entities = self._extract_entities(passages)
        if len(entities) < 3:
            return SkillResult(
                success=False,
                errors=["Need at least 3 entities for network graph"],
                fallback_text="",
            )

        nodes = []
        for i, ent in enumerate(entities[:15]):
            nodes.append({
                "id": f"n{i}",
                "label": ent,
                "type": "entity",
                "metadata": {},
            })

        edges = []
        # Create edges based on co-occurrence in passages
        for i, n1 in enumerate(nodes):
            for j, n2 in enumerate(nodes):
                if i >= j:
                    continue
                weight = self._cooccurrence_weight(passages, n1["label"], n2["label"])
                if weight > 0:
                    edges.append({
                        "source": n1["id"],
                        "target": n2["id"],
                        "label": "related",
                        "type": "association",
                        "weight": weight,
                    })

        data = {
            "nodes": nodes,
            "edges": edges,
            "layout": "force",
        }
        valid, errors = self.validate_output(data)
        if not valid:
            return SkillResult(success=False, errors=errors, fallback_text="")

        return SkillResult(
            success=True,
            data=data,
            fallback_text=self._generate_fallback(data, query),
        )

    def _extract_entities(self, passages: List[str]) -> List[str]:
        import re
        candidates = []
        for text in passages:
            found = re.findall(r"\b[A-Z][a-zA-Z\s]{2,20}\b", text)
            candidates.extend(found)
        seen = set()
        result = []
        for c in candidates:
            c = c.strip()
            if c and c not in seen and len(c) > 2:
                seen.add(c)
                result.append(c)
        return result

    def _cooccurrence_weight(self, passages: List[str], e1: str, e2: str) -> float:
        count = 0
        for text in passages:
            if e1.lower() in text.lower() and e2.lower() in text.lower():
                count += 1
        return min(count / len(passages), 1.0) if passages else 0.0

    def _generate_fallback(self, data: Dict[str, Any], query: str) -> str:
        lines = [f"## Network Graph: {query}", ""]
        lines.append(f"**Nodes ({len(data.get('nodes', []))}):**")
        for n in data.get("nodes", []):
            lines.append(f"- {n['label']} ({n['type']})")
        lines.append("")
        lines.append(f"**Edges ({len(data.get('edges', []))}):**")
        for e in data.get("edges", []):
            lines.append(f"- {e['source']} → {e['target']} [{e['label']}, weight={e.get('weight', 0):.2f}]")
        return "\n".join(lines)