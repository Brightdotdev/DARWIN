"""
Darwin Query Module — Response Builder
Packages generation output into frontend-ready response structures.
"""

from typing import Any, Dict, List, Optional

from src.features.QueryModule.types import (
    DeferredQuestion,
    DeferredResponse,
    DirectResponse,
    FollowupProcessedResponse,
    GenerationResult,
    ResponsePackage,
)


class ResponseBuilder:
    """
    Converts GenerationResult into one of three response modes:
      A. Direct Answer
      B. Deferred Question
      C. Follow-up Processed
    """

    def build(
        self,
        result: GenerationResult,
        deferred_question: Optional[DeferredQuestion] = None,
        original_deferred: Optional[Any] = None,
    ) -> ResponsePackage:
        """Build the appropriate response package."""
        
        # Mode B: Deferred Question
        if deferred_question or result.deferred_question:
            dq = deferred_question or result.deferred_question
            return self._build_deferred(dq, result)
        
        # Mode C: Follow-up Processed (if original_deferred provided)
        if original_deferred:
            return self._build_followup_processed(result, original_deferred)
        
        # Mode A: Direct Answer (default)
        return self._build_direct(result)

    def _build_direct(self, result: GenerationResult) -> DirectResponse:
        """Build Mode A: Direct Answer."""
        # Try to parse primary_output as JSON if it looks like JSON
        data = self._safe_parse_json(result.primary_output)
        
        return DirectResponse(
            mode="direct",
            representation=result.representation.value,
            data=data if isinstance(data, dict) else {"content": result.primary_output},
            interactive_schema=result.interactive_schema,
            fallback_text=result.fallback_text,
            citations=[self._citation_to_dict(c) for c in result.citations],
            sources=result.sources,
        )

    def _build_deferred(
        self,
        deferred: DeferredQuestion,
        result: GenerationResult,
    ) -> DeferredResponse:
        """Build Mode B: Deferred Question."""
        return DeferredResponse(
            mode="deferred",
            message=deferred.question,
            options=deferred.options,
            context_preview=deferred.context_preview,
            fallback_text=result.fallback_text if result.fallback_text else deferred.question,
            fallback_available=True,
        )

    def _build_followup_processed(
        self,
        result: GenerationResult,
        original_deferred: Any,
    ) -> FollowupProcessedResponse:
        """Build Mode C: Follow-up Processed."""
        data = self._safe_parse_json(result.primary_output)
        
        return FollowupProcessedResponse(
            mode="followup_processed",
            representation=result.representation.value,
            data=data if isinstance(data, dict) else {"content": result.primary_output},
            interactive_schema=result.interactive_schema,
            fallback_text=result.fallback_text,
            citations=[self._citation_to_dict(c) for c in result.citations],
            sources=result.sources,
            original_deferred=original_deferred,
        )

    def _safe_parse_json(self, text: str) -> Any:
        """Safely parse JSON from text."""
        import json
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        return text

    def _citation_to_dict(self, citation) -> Dict[str, Any]:
        """Convert Citation object to dict."""
        return {
            "source": citation.source,
            "page": citation.page,
            "passage_index": citation.passage_index,
            "text": citation.text,
        }

    def to_json(self, package: ResponsePackage) -> Dict[str, Any]:
        """Convert any response package to a plain JSON-serializable dict."""
        import dataclasses
        if dataclasses.is_dataclass(package):
            result = {}
            for field in dataclasses.fields(package):
                value = getattr(package, field.name)
                if value is None:
                    continue
                if isinstance(value, list):
                    result[field.name] = value
                elif isinstance(value, dict):
                    result[field.name] = value
                elif hasattr(value, "value"):  # Enum
                    result[field.name] = value.value
                elif dataclasses.is_dataclass(value):
                    result[field.name] = self.to_json(value)
                else:
                    result[field.name] = value
            return result
        return dict(package)