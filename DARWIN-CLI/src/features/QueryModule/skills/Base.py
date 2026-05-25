"""
Darwin Query Module — Skill System Base
Abstract base class for representation skills.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


from src.features.QueryModule.types import (
    BlobHandle,
    ConversationContext,
    RetrievalContext,
    SkillResult,
)


class RepresentationSkill(ABC):
    """Abstract base for all representation skills."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        ...

    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def execute(
        self,
        query: str,
        context: RetrievalContext,
        conversation: ConversationContext,
        blob: Optional[BlobHandle] = None,
    ) -> SkillResult:
        ...

    def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Basic schema validation. Override for stricter checks."""
        errors = []
        schema = self.input_schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for key in required:
            if key not in data:
                errors.append(f"Missing required field: {key}")
            elif key in properties:
                expected_type = properties[key].get("type")
                if expected_type and not self._check_type(data[key], expected_type):
                    errors.append(f"Field '{key}' expected {expected_type}, got {type(data[key]).__name__}")
        return len(errors) == 0, errors

    def validate_output(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Basic schema validation. Override for stricter checks."""
        errors = []
        schema = self.output_schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for key in required:
            if key not in data:
                errors.append(f"Missing required output field: {key}")
            elif key in properties:
                expected_type = properties[key].get("type")
                if expected_type and not self._check_type(data[key], expected_type):
                    errors.append(f"Output field '{key}' expected {expected_type}, got {type(data[key]).__name__}")
        return len(errors) == 0, errors

    @staticmethod
    def _check_type(value: Any, expected: str) -> bool:
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected_cls = type_map.get(expected)
        if expected_cls is None:
            return True
        return isinstance(value, expected_cls)

