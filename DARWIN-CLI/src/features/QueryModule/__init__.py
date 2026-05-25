"""
Darwin Query Module
===================
The brain of the RAG pipeline. Handles all user input from raw text
to structured frontend-ready response.

Modules:
  - InputParser: Blob detection, URL extraction, query cleaning
  - Classifier: Fast regex + slow LLM cascade for intent classification
  - Retriever: Thin wrapper around Darwin retrieval infrastructure
  - ContentValidator: Representation requirement checking + downgrade chains
  - GenerationOrchestrator: Single LLM call for primary + fallback output
  - ResponseBuilder: Packages output for frontend (direct/deferred/follow-up)
  - FollowUpSkill: Processes deferred choices and re-runs pipeline
  - SkillRouter: Routes heavy interactive representations to typed skills

Usage:
    from src.core.Agent.QueryModule import QueryPipeline
    pipeline = QueryPipeline(llm_manager=..., retrieval_orchestrator=...)
    response = pipeline.process("Compare Python and JavaScript")
"""

from src.features.QueryModule.Classifier import Classifier
from src.features.QueryModule.ContentValidator import ContentValidator
from src.features.QueryModule.FollowUpSkill import FollowUpSkill
from src.features.QueryModule.GenerationOrchestrator import GenerationOrchestrator
from src.features.QueryModule.InputParser import InputParser
from src.features.QueryModule.ResponseBuilder import ResponseBuilder
from src.features.QueryModule.Retriever import Retriever
from src.features.QueryModule.SkillRouter import SkillRouter
from src.features.QueryModule.types import (
    BlobHandle,
    ClassificationResult,
    ConfidenceLevel,
    ConversationContext,
    ConversationContextProvider,
    DeferredChoice,
    DeferredQuestion,
    DeferredResponse,
    DeferredState,
    DirectResponse,
    FollowupBlock,
    FollowupProcessedResponse,
    GenerationResult,
    ParsedInput,
    QuestionType,
    RepresentationSkill,
    ResponsePackage,
    ResponsePresentation,
    RetrievalContext,
    RetrievalMode,
    RetrievalPolicy,
    SkillResult,
    ValidationResult,
)

__all__ = [
    "Classifier",
    "ContentValidator",
    "FollowUpSkill",
    "GenerationOrchestrator",
    "InputParser",
    "ResponseBuilder",
    "Retriever",
    "SkillRouter",
    "BlobHandle",
    "ClassificationResult",
    "ConfidenceLevel",
    "ConversationContext",
    "ConversationContextProvider",
    "DeferredChoice",
    "DeferredQuestion",
    "DeferredResponse",
    "DeferredState",
    "DirectResponse",
    "FollowupBlock",
    "FollowupProcessedResponse",
    "GenerationResult",
    "ParsedInput",
    "QuestionType",
    "RepresentationSkill",
    "ResponsePackage",
    "ResponsePresentation",
    "RetrievalContext",
    "RetrievalMode",
    "RetrievalPolicy",
    "SkillResult",
    "ValidationResult",
]


class QueryPipeline:
    """
    High-level orchestrator that wires all QueryModule components together.
    This is the main entry point for the frontend/backend integration.
    """

    def __init__(
        self,
        llm_manager=None,
        retrieval_orchestrator=None,
        wiki_orchestrator=None,
        darwin_store=None,
        conversation_provider=None,
    ):
        self.input_parser = InputParser(llm_manager=llm_manager)
        self.classifier = Classifier(llm_manager=llm_manager)
        self.retriever = Retriever(
            retrieval_orchestrator=retrieval_orchestrator,
            wiki_orchestrator=wiki_orchestrator,
            darwin_store=darwin_store,
        )
        self.validator = ContentValidator()
        self.skill_router = SkillRouter()
        self.generation_orchestrator = GenerationOrchestrator(
            llm_manager=llm_manager,
            skill_router=self.skill_router,
            conversation_provider=conversation_provider,
        )
        self.response_builder = ResponseBuilder()
        self.follow_up_skill = FollowUpSkill(
            retriever=self.retriever,
            generation_orchestrator=self.generation_orchestrator,
            content_validator=self.validator,
        )

    def process(
        self,
        raw_input,
        session_id=None,
        conversation_id=None,
        pinned_files=None,
    ):
        """
        Main pipeline entry point.
        Returns a JSON-serializable response dict.
        """
        pinned_files = pinned_files or []
        
        # 1. Parse input
        parsed = self.input_parser.parse(raw_input, pinned_files=pinned_files)

        # 2. Classify
        classification = self.classifier.classify(
            query=parsed.clean_query,
            session_id=session_id,
            file_registry=pinned_files,
        )

        # 3. Retrieve
        context = self.retriever.retrieve(
            query=classification.clean_query,
            policy=classification.retrieval_policy,
            blob=parsed.blob,
            pinned_files=parsed.pinned_files,
        )

        # 4. Validate representation
        suggested_rep = (
            classification.explicit_format_lock
            or (classification.suggested_representations[0] if classification.suggested_representations else ResponsePresentation.MARKDOWN)
        )
        validation = self.validator.validate(
            context=context,
            suggested=suggested_rep,
            explicit_lock=classification.explicit_format_lock,
        )
        final_rep = validation.representation

        # 5. Handle deferred path
        if classification.needs_clarification and classification.followup_blocks:
            deferred_q = DeferredQuestion(
                question=classification.followup_blocks[0].question,
                options=classification.followup_blocks[0].options,
                allow_custom=classification.followup_blocks[0].allow_custom,
            )
            # Register for follow-up processing
            if session_id:
                self.follow_up_skill.register_deferred(session_id, parsed.clean_query, deferred_q)

            # Generate a minimal result for fallback text
            gen_result = GenerationResult(
                primary_output="",
                fallback_text=f"I need more information: {deferred_q.question}",
                representation=ResponsePresentation.MARKDOWN,
                citations=[],
                sources=[],
                confidence=ConfidenceLevel.UNCERTAIN,
                used_retrieval=False,
                deferred_question=deferred_q,
            )
            package = self.response_builder.build(gen_result, deferred_question=deferred_q)
            return self.response_builder.to_json(package)

        # 6. Generate
        gen_result = self.generation_orchestrator.generate(
            query=classification.clean_query,
            classification=classification,
            context=context,
            representation=final_rep,
            conversation_id=conversation_id,
            blob=parsed.blob,
        )

        # 7. Build response
        package = self.response_builder.build(gen_result)
        return self.response_builder.to_json(package)

    def process_deferred_choice(
        self,
        session_id,
        choice,
        conversation_id=None,
        blob=None,
    ):
        """
        Process a user choice from a deferred question.
        """
        state = self.follow_up_skill.get_state(session_id)
        if not state:
            return {
                "mode": "error",
                "message": "No deferred state found for this session.",
            }

        # Reconstruct the deferred question from state
        deferred_q = DeferredQuestion(
            question=state.deferred_question,
            options=[],
        )

        gen_result = self.follow_up_skill.process_choice(
            session_id=session_id,
            choice=choice,
            deferred_question=deferred_q,
            conversation_id=conversation_id,
            blob=blob,
        )

        package = self.response_builder.build(gen_result, original_deferred=state)
        return self.response_builder.to_json(package)

    def get_fallback_text(self, response_package):
        """Extract fallback text from any response package."""
        return response_package.get("fallback_text", "")