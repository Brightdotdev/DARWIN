"""
Darwin Query Module — Follow-Up Skill
Processes user choice from deferred question and re-runs pipeline.
"""

from typing import Any, Dict, Optional

from src.features.QueryModule.types import (
    ClassificationResult,
    DeferredChoice,
    DeferredQuestion,
    DeferredState,
    GenerationResult,
    QuestionType,
    ResponsePresentation,
    RetrievalMode,
    RetrievalPolicy,
)


class FollowUpSkill:
    """
    Processes deferred choices:
      - Maps option id to query_rewrite
      - Handles custom input
      - Injects resolved intent into session state
      - Re-runs retrieval + generation with narrowed query
    """

    def __init__(
        self,
        retriever=None,
        generation_orchestrator=None,
        content_validator=None,
    ):
        self.retriever = retriever
        self.generation_orchestrator = generation_orchestrator
        self.content_validator = content_validator
        self._deferred_states: Dict[str, DeferredState] = {}  # session_id -> state

    def register_deferred(
        self,
        session_id: str,
        original_query: str,
        deferred_question: DeferredQuestion,
    ) -> None:
        """Register a deferred question for a session."""
        self._deferred_states[session_id] = DeferredState(
            original_query=original_query,
            deferred_question=deferred_question.question,
            chosen_option="",
            resolved_query="",
            resolved_type=QuestionType.FACT,
        )

    def process_choice(
        self,
        session_id: str,
        choice: DeferredChoice,
        deferred_question: DeferredQuestion,
        conversation_id: Optional[str] = None,
        blob=None,
    ) -> GenerationResult:
        """Process user choice and regenerate answer."""
        
        # Resolve query
        resolved_query = self._resolve_query(choice, deferred_question)
        
        # Determine question type (skip classifier — intent now known)
        resolved_type = self._infer_type_from_options(choice, deferred_question)
        resolved_policy = RetrievalPolicy(
            retrieval=RetrievalMode.HYBRID,
            top_k=5,
            requires_rag=True,
        )
        
        # Update state
        state = self._deferred_states.get(session_id)
        if state:
            state.chosen_option = choice.option_id or "custom"
            state.resolved_query = resolved_query
            state.resolved_type = resolved_type
            state.resolved_policy = resolved_policy
        else:
            state = DeferredState(
                original_query=deferred_question.question,
                deferred_question=deferred_question.question,
                chosen_option=choice.option_id or "custom",
                resolved_query=resolved_query,
                resolved_type=resolved_type,
                resolved_policy=resolved_policy,
            )
            self._deferred_states[session_id] = state
        
        # Re-run retrieval
        context = self.retriever.retrieve(
            query=resolved_query,
            policy=resolved_policy,
            blob=blob,
        )
        
        # Determine representation
        suggested_rep = self._suggest_representation(resolved_type, choice)
        
        # Validate
        if self.content_validator:
            validation = self.content_validator.validate(context, suggested_rep)
            final_rep = validation.representation
        else:
            final_rep = suggested_rep
        
        # Re-generate
        classification = ClassificationResult(
            clean_query=resolved_query,
            question_types=[resolved_type],
            suggested_representations=[final_rep],
            retrieval_policy=resolved_policy,
            needs_clarification=False,
            followup_blocks=None,
            confidence=0.85,
        )
        
        result = self.generation_orchestrator.generate(
            query=resolved_query,
            classification=classification,
            context=context,
            representation=final_rep,
            conversation_id=conversation_id,
            blob=blob,
        )
        
        return result

    def _resolve_query(
        self,
        choice: DeferredChoice,
        deferred_question: DeferredQuestion,
    ) -> str:
        """Map choice to rewritten query."""
        if choice.custom_text:
            return choice.custom_text
        
        if choice.option_id:
            for opt in deferred_question.options:
                if opt.get("id") == choice.option_id:
                    rewrite = opt.get("query_rewrite")
                    if rewrite:
                        return rewrite
                    return opt.get("label", "")
        
        return deferred_question.question

    def _infer_type_from_options(
        self,
        choice: DeferredChoice,
        deferred_question: DeferredQuestion,
    ) -> QuestionType:
        """Infer question type from the option chosen."""
        # Default to fact, but can be extended with option metadata
        if choice.option_id:
            for opt in deferred_question.options:
                if opt.get("id") == choice.option_id:
                    # Check option metadata for type hint
                    meta = opt.get("metadata", {})
                    type_hint = meta.get("question_type", "fact")
                    try:
                        return QuestionType(type_hint)
                    except ValueError:
                        pass
        return QuestionType.FACT

    def _suggest_representation(
        self,
        qtype: QuestionType,
        choice: DeferredChoice,
    ) -> ResponsePresentation:
        """Suggest representation based on resolved type."""
        mapping = {
            QuestionType.FACT: ResponsePresentation.SUMMARY_CARD,
            QuestionType.INSTRUCTION: ResponsePresentation.STEPPER,
            QuestionType.REASONING: ResponsePresentation.ACCORDION,
            QuestionType.COMPARISON: ResponsePresentation.COMPARISON_TABLE,
            QuestionType.OPINION: ResponsePresentation.PROS_CONS,
            QuestionType.HYPOTHETICAL: ResponsePresentation.PARAMETER_SANDBOX,
            QuestionType.CLARIFICATION: ResponsePresentation.MARKDOWN,
            QuestionType.MULTI_PART: ResponsePresentation.ACCORDION,
            QuestionType.YES_NO: ResponsePresentation.SUMMARY_CARD,
            QuestionType.STATEMENT: ResponsePresentation.MARKDOWN,
        }
        return mapping.get(qtype, ResponsePresentation.SUMMARY_CARD)

    def get_state(self, session_id: str) -> Optional[DeferredState]:
        """Get deferred state for a session."""
        return self._deferred_states.get(session_id)

    def clear_state(self, session_id: str) -> None:
        """Clear deferred state for a session."""
        self._deferred_states.pop(session_id, None)