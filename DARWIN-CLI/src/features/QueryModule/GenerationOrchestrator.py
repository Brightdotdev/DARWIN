"""
Darwin Query Module — Generation Orchestrator
The "big guy" — one LLM call that generates primary output + fallback text.
Routes heavy sandboxing to SkillSystem with strict types.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.features.QueryModule.types import (
    BlobHandle,
    Citation,
    ClassificationResult,
    ConfidenceLevel,
    ConversationContext,
    ConversationContextProvider,
    DeferredQuestion,
    GenerationResult,
    HEAVY_SANDBOX_REPRESENTATIONS,
    ResponsePresentation,
    RetrievalContext,
    SkillResult,
)
from src.features.QueryModule.schemas import get_schema


class GenerationOrchestrator:
    """
    Single LLM call generates both structured primary output and fallback markdown.
    For heavy interactive representations, routes to SkillRouter.
    """

    def __init__(
        self,
        llm_manager=None,
        skill_router=None,
        conversation_provider: Optional[ConversationContextProvider] = None,
    ):
        self.llm_manager = llm_manager
        self.skill_router = skill_router
        self.conversation_provider = conversation_provider

    def generate(
        self,
        query: str,
        classification: ClassificationResult,
        context: RetrievalContext,
        representation: ResponsePresentation,
        conversation_id: Optional[str] = None,
        blob: Optional[BlobHandle] = None,
    ) -> GenerationResult:
        """Main generation entry point."""
        
        # Build conversation context
        conv_context = self._build_conversation_context(conversation_id, query)
        
        # Handle heavy sandboxing via skills
        if representation in HEAVY_SANDBOX_REPRESENTATIONS and self.skill_router:
            skill_result = self.skill_router.execute(
                representation=representation,
                query=query,
                context=context,
                conversation=conv_context,
                blob=blob,
            )
            if skill_result.success:
                return self._wrap_skill_result(skill_result, representation, context)
            else:
                # Skill failed — downgrade to markdown with error note
                return self._generate_fallback_response(
                    query, context, representation,
                    f"Skill error: {'; '.join(skill_result.errors)}"
                )
        
        # Standard LLM generation
        if not self.llm_manager:
            return self._generate_fallback_response(
                query, context, representation,
                "No LLM manager available"
            )
        
        return self._llm_generate(
            query, classification, context, representation, conv_context, blob
        )

    def _llm_generate(
        self,
        query: str,
        classification: ClassificationResult,
        context: RetrievalContext,
        representation: ResponsePresentation,
        conv_context: ConversationContext,
        blob: Optional[BlobHandle],
    ) -> GenerationResult:
        """Execute the single LLM call for primary + fallback generation."""
        
        system_prompt = self._build_system_prompt(representation)
        user_prompt = self._build_user_prompt(
            query, classification, context, representation, conv_context, blob
        )
        
        try:
            raw_output = self.llm_manager.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=2500,
            )
            
            # Parse dual output: PRIMARY and FALLBACK sections
            primary, fallback = self._parse_dual_output(raw_output)
            
            # Extract citations
            citations = self._extract_citations(raw_output, context)
            sources = list({c.source for c in citations})
            
            # Build interactive schema if applicable
            interactive_schema = self._build_interactive_schema(representation, primary)
            
            # Determine confidence
            confidence = self._assess_confidence(context, primary)
            
            return GenerationResult(
                primary_output=primary,
                fallback_text=fallback,
                representation=representation,
                citations=citations,
                sources=sources,
                confidence=confidence,
                used_retrieval=not context.is_empty(),
                interactive_schema=interactive_schema,
            )
            
        except Exception as e:
            return self._generate_fallback_response(
                query, context, representation,
                f"Generation error: {str(e)}"
            )

    def _build_system_prompt(self, representation: ResponsePresentation) -> str:
        """Build system prompt with representation schema."""
        schema = get_schema(representation.value)
        
        prompt = f"""You are a precise research assistant. Answer using ONLY provided context.
Cite sources with [Source: filename, page X].
Choose the best presentation format based on content.
If context is insufficient, say so explicitly.

Generate TWO outputs in your response:

1. PRIMARY: Structured data for {representation.value} format
   - Follow this schema: {json.dumps(schema, indent=2)}
   - Include all necessary data for frontend rendering
   - Wrap in <PRIMARY> ... </PRIMARY> tags

2. FALLBACK: Plain markdown summary of the same answer
   - No special formatting, just clean text with citations
   - Wrap in <FALLBACK> ... </FALLBACK> tags

Rules:
- Be concise but complete
- Always include citations for factual claims
- If you cannot answer from context, state "I don't have enough information"
- The fallback must be readable as standalone text"""
        
        return prompt

    def _build_user_prompt(
        self,
        query: str,
        classification: ClassificationResult,
        context: RetrievalContext,
        representation: ResponsePresentation,
        conv_context: ConversationContext,
        blob: Optional[BlobHandle],
    ) -> str:
        """Build the user prompt with all context."""
        
        parts = []
        parts.append(f"QUESTION: {query}")
        parts.append(f"QUESTION TYPE: {classification.question_types[0].value if classification.question_types else 'fact'}")
        parts.append(f"REPRESENTATION HINT: {representation.value}")
        
        # Retrieved context
        if context.passages:
            parts.append("\nCONTEXT:")
            for i, p in enumerate(context.passages[:10]):
                page_str = f", page {p.page}" if p.page else ""
                parts.append(f"[{i+1}] [Source: {p.source}{page_str}] {p.content[:400]}")
        else:
            parts.append("\nCONTEXT: No document context available.")
        
        # Wiki concepts
        if context.concepts:
            parts.append("\nCONCEPTS:")
            for c in context.concepts[:5]:
                parts.append(f"- {c.name}: {c.description[:200]}")
        
        # Conversation history
        if conv_context.summary:
            parts.append(f"\nCONVERSATION SUMMARY: {conv_context.summary}")
        if conv_context.turns:
            parts.append("\nRECENT TURNS:")
            for turn in conv_context.turns[-4:]:
                role = turn.get("role", "unknown")
                content = turn.get("content", "")[:150]
                parts.append(f"{role}: {content}")
        
        # Blob summary
        if blob:
            parts.append(f"\nATTACHED BLOB ({blob.blob_type.value}): {blob.summary}")
            # Include most relevant chunks
            relevant = blob.get_relevant_chunks(query, top_k=2)
            for chunk in relevant:
                parts.append(f"[Chunk {chunk.index}] {chunk.content[:300]}")
        
        parts.append("\nGenerate PRIMARY and FALLBACK outputs now.")
        
        return "\n".join(parts)

    def _parse_dual_output(self, raw: str) -> Tuple[str, str]:
        """Extract PRIMARY and FALLBACK sections from LLM output."""
        primary_match = re.search(r"<PRIMARY>(.*?)</PRIMARY>", raw, re.DOTALL)
        fallback_match = re.search(r"<FALLBACK>(.*?)</FALLBACK>", raw, re.DOTALL)
        
        if primary_match and fallback_match:
            return primary_match.group(1).strip(), fallback_match.group(1).strip()
        
        # Fallback parsing: try to split by headers
        if "PRIMARY" in raw and "FALLBACK" in raw:
            parts = re.split(r"(?:PRIMARY|FALLBACK)[\s:]*", raw, flags=re.I)
            if len(parts) >= 3:
                return parts[1].strip(), parts[2].strip()
        
        # Ultimate fallback: use full as fallback, try to extract JSON as primary
        # Try to find JSON block for primary
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            return json_match.group(0), raw
        
        return raw, raw

    def _extract_citations(self, raw: str, context: RetrievalContext) -> List[Citation]:
        """Extract citations from LLM output and map to sources."""
        citations = []
        citation_pattern = re.compile(r"\[Source:\s*([^,\]]+)(?:,\s*page\s*(\d+))?\]", re.I)
        
        for match in citation_pattern.finditer(raw):
            source_name = match.group(1).strip()
            page = int(match.group(2)) if match.group(2) else None
            
            # Find matching passage
            passage_idx = 0
            passage_text = ""
            for i, p in enumerate(context.passages):
                if source_name.lower() in p.source.lower():
                    passage_idx = i
                    passage_text = p.content[:100]
                    break
            
            citations.append(Citation(
                source=source_name,
                page=page,
                passage_index=passage_idx,
                text=passage_text,
            ))
        
        # Deduplicate
        seen = set()
        unique = []
        for c in citations:
            key = (c.source, c.page)
            if key not in seen:
                seen.add(key)
                unique.append(c)
        
        return unique

    def _build_interactive_schema(
        self,
        representation: ResponsePresentation,
        primary_output: str,
    ) -> Optional[Dict[str, Any]]:
        """Build interaction schema for interactive representations."""
        schemas = {
            ResponsePresentation.COMPARISON_TABLE: {"sortable": True, "filters": [], "selectable_rows": True},
            ResponsePresentation.TIMELINE: {"zoomable": True, "filterable_categories": True},
            ResponsePresentation.PARAMETER_SANDBOX: {"live_update": True, "shareable": True},
            ResponsePresentation.DIAGNOSTIC_TREE: {"clickable_branches": True, "track_path": True},
            ResponsePresentation.STEPPER: {"navigable": True, "show_progress": True},
            ResponsePresentation.ACCORDION: {"expandable": True, "multi_expand": False},
        }
        return schemas.get(representation)

    def _assess_confidence(self, context: RetrievalContext, primary: str) -> ConfidenceLevel:
        """Assess confidence level based on context coverage and output quality."""
        if not context.passages:
            if "don't have enough information" in primary.lower() or "insufficient" in primary.lower():
                return ConfidenceLevel.UNCERTAIN
            return ConfidenceLevel.LOW
        
        avg_score = sum(p.score for p in context.passages) / len(context.passages) if context.passages else 0
        
        if avg_score > 0.85 and len(context.passages) >= 3:
            return ConfidenceLevel.HIGH
        elif avg_score > 0.70 and len(context.passages) >= 2:
            return ConfidenceLevel.MEDIUM
        elif avg_score > 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN

    def _build_conversation_context(
        self,
        conversation_id: Optional[str],
        query: str,
    ) -> ConversationContext:
        """Build conversation context from provider."""
        if not self.conversation_provider or not conversation_id:
            return ConversationContext()
        
        try:
            turns = self.conversation_provider.get_context(conversation_id, query)
            summary = self.conversation_provider.get_summary(conversation_id)
            return ConversationContext(
                turns=turns,
                summary=summary,
                conversation_id=conversation_id,
            )
        except Exception:
            return ConversationContext()

    def _wrap_skill_result(
        self,
        skill_result: SkillResult,
        representation: ResponsePresentation,
        context: RetrievalContext,
    ) -> GenerationResult:
        """Wrap a successful skill result into GenerationResult."""
        primary = json.dumps(skill_result.data) if isinstance(skill_result.data, dict) else str(skill_result.data)
        
        return GenerationResult(
            primary_output=primary,
            fallback_text=skill_result.fallback_text,
            representation=representation,
            citations=[],
            sources=list({p.source for p in context.passages}),
            confidence=ConfidenceLevel.HIGH if skill_result.success else ConfidenceLevel.MEDIUM,
            used_retrieval=not context.is_empty(),
            interactive_schema=self._build_interactive_schema(representation, primary),
        )

    def _generate_fallback_response(
        self,
        query: str,
        context: RetrievalContext,
        attempted_representation: ResponsePresentation,
        error_note: str,
    ) -> GenerationResult:
        """Generate a graceful fallback when generation fails."""
        if context.is_empty():
            fallback = f"I don't have enough information to answer your question about: {query}\n\n{error_note}"
        else:
            # Build a simple summary from passages
            parts = [f"Based on the available context, here's what I found about: {query}", ""]
            for i, p in enumerate(context.passages[:5]):
                parts.append(f"- {p.content[:300]}... [Source: {p.source}]")
            parts.append("")
            parts.append(error_note)
            fallback = "\n".join(parts)
        
        return GenerationResult(
            primary_output=fallback,
            fallback_text=fallback,
            representation=ResponsePresentation.MARKDOWN,
            citations=[],
            sources=list({p.source for p in context.passages}),
            confidence=ConfidenceLevel.LOW,
            used_retrieval=not context.is_empty(),
        )