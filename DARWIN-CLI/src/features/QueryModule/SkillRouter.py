"""
Darwin Query Module — Skill Router
Routes generation requests to specialized representation skills.
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


class SkillRouter:
    """Routes generation requests to specialized representation skills."""

    def __init__(self):
        self._skills: Dict[str, RepresentationSkill] = {}
        self._register_builtin_skills()

    def _register_builtin_skills(self) -> None:
        """Lazy-import and register built-in skills to avoid circular imports."""
        from src.features.QueryModule.skills.ComparisonTableSkill import ComparisonTableSkill
        from src.features.QueryModule.skills.DiagnosticTreeSkill import DiagnosticTreeSkill
        from src.features.QueryModule.skills.NetworkGraphSkill import NetworkGraphSkill
        from src.features.QueryModule.skills.ParameterSandboxSkill import ParameterSandboxSkill
        from src.features.QueryModule.skills.TimelineSkill import TimelineSkill

        self.register(ComparisonTableSkill())
        self.register(DiagnosticTreeSkill())
        self.register(NetworkGraphSkill())
        self.register(ParameterSandboxSkill())
        self.register(TimelineSkill())

    def register(self, skill: RepresentationSkill) -> None:
        self._skills[skill.name] = skill

    def get_skill(self, representation: ResponsePresentation) -> Optional[RepresentationSkill]:
        return self._skills.get(representation.value)

    def execute(
        self,
        representation: ResponsePresentation,
        query: str,
        context: RetrievalContext,
        conversation: ConversationContext,
        blob: Optional[BlobHandle] = None,
    ) -> SkillResult:
        skill = self.get_skill(representation)
        if skill is None:
            return SkillResult(
                success=False,
                errors=[f"No skill registered for representation: {representation.value}"],
                fallback_text="",
            )
        return skill.execute(query, context, conversation, blob)

    def list_skills(self) -> List[str]:
        return list(self._skills.keys())