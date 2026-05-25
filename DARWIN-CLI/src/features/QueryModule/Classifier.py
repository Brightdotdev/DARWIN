"""
Darwin Query Module — Classifier
Two-tier cascade: fast regex path (<1ms) + slow LLM path (~500ms).
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.features.QueryModule.types import (
    ClassificationResult,
    FAST_PATH_PATTERNS,
    FORMAT_LOCK_PATTERNS,
    FollowupBlock,
    QuestionType,
    ResponsePresentation,
    RetrievalMode,
    RetrievalPolicy,
)


class Classifier:
    """
    Classifies user queries into QuestionTypes, suggests representations,
    and determines retrieval policy.
    
    Two-tier architecture:
      1. Fast path: regex patterns, <1ms
      2. Slow path: LLM call, ~500ms, only for vague/ambiguous/novel queries
    """

    FAST_CONFIDENCE_THRESHOLD = 0.85

    def __init__(self, llm_manager=None):
        self.llm_manager = llm_manager
        self._followup_history: Dict[str, Dict[str, Any]] = {}  # session_id -> last classification

    def classify(
        self,
        query: str,
        session_id: Optional[str] = None,
        file_registry: Optional[List[str]] = None,
        conversation_summary: str = "",
    ) -> ClassificationResult:
        """Main classification entry point."""
        file_registry = file_registry or []
        
        # Step 1: Check for explicit format lock
        explicit_format = self._detect_explicit_format(query)
        
        # Step 2: Check for follow-up pattern
        if self._is_followup(query, session_id):
            return self._handle_followup(query, session_id, explicit_format)
        
        # Step 3: Fast path regex classification
        fast_result = self._fast_classify(query)
        
        if fast_result and fast_result.confidence >= self.FAST_CONFIDENCE_THRESHOLD:
            result = fast_result
            result.explicit_format_lock = explicit_format
            if session_id:
                self._followup_history[session_id] = {
                    "type": result.question_types[0] if result.question_types else QuestionType.FACT,
                    "policy": result.retrieval_policy,
                    "query": query,
                }
            return result
        
        # Step 4: Slow path (LLM) for vague/ambiguous/novel queries
        slow_result = self._slow_classify(query, file_registry, conversation_summary)
        slow_result.explicit_format_lock = explicit_format
        
        if session_id:
            self._followup_history[session_id] = {
                "type": slow_result.question_types[0] if slow_result.question_types else QuestionType.FACT,
                "policy": slow_result.retrieval_policy,
                "query": query,
            }
        
        return slow_result

    def _fast_classify(self, query: str) -> Optional[ClassificationResult]:
        """Regex-based fast classification. Returns None if no strong match."""
        lowered = query.lower()
        
        # Check for vague indicator first
        if "?" in query and len(query.split()) > 50:
            # Check if any pattern matches; if not, it's vague
            has_match = any(p.search(query) for p, _, _ in FAST_PATH_PATTERNS)
            if not has_match:
                return ClassificationResult(
                    clean_query=query,
                    question_types=[QuestionType.VAGUE],
                    suggested_representations=[ResponsePresentation.MARKDOWN],
                    retrieval_policy=RetrievalPolicy(
                        retrieval=RetrievalMode.HYBRID,
                        top_k=5,
                        requires_rag=True,
                    ),
                    needs_clarification=True,
                    followup_blocks=[
                        FollowupBlock(
                            question="Could you clarify what you're looking for?",
                            options=[
                                {"id": "more_specific", "label": "Be more specific", "value": "specific"},
                                {"id": "examples", "label": "Show examples", "value": "examples"},
                            ],
                            allow_custom=True,
                            reason="ambiguous_query",
                        )
                    ],
                    confidence=0.60,
                )
        
        # Pattern matching in priority order
        for pattern, qtype, confidence in FAST_PATH_PATTERNS:
            if pattern.search(query):
                rep = self._default_representation_for_type(qtype)
                policy = self._default_policy_for_type(qtype)
                return ClassificationResult(
                    clean_query=query,
                    question_types=[qtype],
                    suggested_representations=[rep],
                    retrieval_policy=policy,
                    needs_clarification=False,
                    followup_blocks=None,
                    confidence=confidence,
                )
        
        # Default fallback if no pattern matched but query is short
        if len(query) < 100 and "?" in query:
            return ClassificationResult(
                clean_query=query,
                question_types=[QuestionType.FACT],
                suggested_representations=[ResponsePresentation.SUMMARY_CARD],
                retrieval_policy=RetrievalPolicy(
                    retrieval=RetrievalMode.HYBRID,
                    top_k=3,
                    requires_rag=True,
                ),
                needs_clarification=False,
                followup_blocks=None,
                confidence=0.70,
            )
        
        return None

    def _slow_classify(
        self,
        query: str,
        file_registry: List[str],
        conversation_summary: str,
    ) -> ClassificationResult:
        """LLM-based slow classification for ambiguous queries."""
        if not self.llm_manager:
            # No LLM available — return vague with deferred
            return ClassificationResult(
                clean_query=query,
                question_types=[QuestionType.VAGUE],
                suggested_representations=[ResponsePresentation.MARKDOWN],
                retrieval_policy=RetrievalPolicy(
                    retrieval=RetrievalMode.HYBRID,
                    top_k=5,
                    requires_rag=True,
                ),
                needs_clarification=True,
                followup_blocks=[
                    FollowupBlock(
                        question="I need a bit more context to help you best.",
                        options=[
                            {"id": "clarify", "label": "Clarify my question", "value": "clarify"},
                        ],
                        allow_custom=True,
                        reason="missing_context",
                    )
                ],
                confidence=0.50,
            )
        
        system_prompt = """You are a query classifier for a research assistant system. 
Analyze the user query and output a JSON object with these fields:
- clean_query: rewritten/cleaned version of the query
- question_types: array of types from [fact, instruction, reasoning, comparison, opinion, hypothetical, clarification, multi_part, yes_no, statement]
- suggested_representations: array from [summary_card, comparison_table, timeline, network_graph, flowchart, parameter_sandbox, diagnostic_tree, stepper, accordion, markdown]
- retrieval_policy: object with {retrieval: "semantic"|"keyword"|"hybrid", top_k: number, requires_rag: boolean, searchable_topics: [string]}
- needs_clarification: boolean
- followup_blocks: null or array of {question, options: [{id, label, value}], allow_custom, reason}
- confidence: float 0.0-1.0

Rules:
- If the query is vague or ambiguous, set needs_clarification=true and provide followup_blocks
- If the query asks for a specific format (table, graph, etc.), include that in suggested_representations
- For comparisons, set searchable_topics to include both entities being compared
- Confidence should reflect how clear the intent is"""

        user_prompt = f"""Query: {query}
Available files: {file_registry}
Conversation context: {conversation_summary}

Classify this query."""

        try:
            raw = self.llm_manager.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_tokens=800,
            )
            
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(raw)
            
            # Map string types to enums
            qtypes = []
            for qt in parsed.get("question_types", ["fact"]):
                try:
                    qtypes.append(QuestionType(qt.lower()))
                except ValueError:
                    qtypes.append(QuestionType.FACT)
            
            reps = []
            for r in parsed.get("suggested_representations", ["summary_card"]):
                try:
                    reps.append(ResponsePresentation(r.lower()))
                except ValueError:
                    reps.append(ResponsePresentation.SUMMARY_CARD)
            
            policy_dict = parsed.get("retrieval_policy", {"retrieval": "hybrid", "top_k": 5, "requires_rag": True})
            policy = RetrievalPolicy(
                retrieval=RetrievalMode(policy_dict.get("retrieval", "hybrid")),
                top_k=policy_dict.get("top_k", 5),
                requires_rag=policy_dict.get("requires_rag", True),
                searchable_topics=policy_dict.get("searchable_topics", []),
            )
            
            followup_blocks = None
            if parsed.get("followup_blocks"):
                followup_blocks = [
                    FollowupBlock(
                        question=fb.get("question", ""),
                        options=fb.get("options", []),
                        allow_custom=fb.get("allow_custom", True),
                        reason=fb.get("reason", "missing_context"),
                    )
                    for fb in parsed["followup_blocks"]
                ]
            
            return ClassificationResult(
                clean_query=parsed.get("clean_query", query),
                question_types=qtypes if qtypes else [QuestionType.FACT],
                suggested_representations=reps if reps else [ResponsePresentation.SUMMARY_CARD],
                retrieval_policy=policy,
                needs_clarification=parsed.get("needs_clarification", False),
                followup_blocks=followup_blocks,
                confidence=parsed.get("confidence", 0.75),
            )
            
        except Exception as e:
            # LLM failure — graceful fallback
            return ClassificationResult(
                clean_query=query,
                question_types=[QuestionType.FACT],
                suggested_representations=[ResponsePresentation.SUMMARY_CARD],
                retrieval_policy=RetrievalPolicy(
                    retrieval=RetrievalMode.HYBRID,
                    top_k=5,
                    requires_rag=True,
                ),
                needs_clarification=False,
                followup_blocks=None,
                confidence=0.60,
            )

    def _detect_explicit_format(self, query: str) -> Optional[ResponsePresentation]:
        """Check if user explicitly requested a representation format."""
        for pattern, presentation in FORMAT_LOCK_PATTERNS:
            if pattern.search(query):
                return presentation
        return None

    def _is_followup(self, query: str, session_id: Optional[str]) -> bool:
        """Detect follow-up queries that should inherit previous state."""
        if not session_id or session_id not in self._followup_history:
            return False
        
        followup_patterns = [
            r"what\s+about",
            r"tell\s+me\s+more",
            r"and\s+then",
            r"go\s+on",
            r"continue",
            r"elaborate",
            r"expand",
            r"more\s+details",
            r"what\s+else",
        ]
        
        lowered = query.lower().strip()
        # Very short queries are likely follow-ups
        if len(lowered) < 30 and lowered not in ["hi", "hello", "hey"]:
            return True
        
        for pat in followup_patterns:
            if re.search(pat, lowered):
                return True
        
        return False

    def _handle_followup(
        self,
        query: str,
        session_id: str,
        explicit_format: Optional[ResponsePresentation],
    ) -> ClassificationResult:
        """Inherit QuestionType and RetrievalPolicy from previous turn."""
        history = self._followup_history.get(session_id, {})
        prev_type = history.get("type", QuestionType.FACT)
        prev_policy = history.get("policy", RetrievalPolicy(
            retrieval=RetrievalMode.HYBRID, top_k=5, requires_rag=True
        ))
        
        # Merge query
        prev_query = history.get("query", "")
        merged_query = f"{prev_query} {query}" if prev_query else query
        
        reps = [explicit_format] if explicit_format else [self._default_representation_for_type(prev_type)]
        
        return ClassificationResult(
            clean_query=merged_query,
            question_types=[prev_type],
            suggested_representations=reps,
            retrieval_policy=prev_policy,
            needs_clarification=False,
            followup_blocks=None,
            confidence=0.80,
            is_followup=True,
            inherited_type=prev_type,
            explicit_format_lock=explicit_format,
        )

    def _default_representation_for_type(self, qtype: QuestionType) -> ResponsePresentation:
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
            QuestionType.VAGUE: ResponsePresentation.MARKDOWN,
            QuestionType.FOLLOWUP: ResponsePresentation.SUMMARY_CARD,
        }
        return mapping.get(qtype, ResponsePresentation.SUMMARY_CARD)

    def _default_policy_for_type(self, qtype: QuestionType) -> RetrievalPolicy:
        from .types import QUESTION_RETRIEVAL_MAP
        cfg = QUESTION_RETRIEVAL_MAP.get(qtype, {
            "retrieval": RetrievalMode.HYBRID,
            "top_k": 5,
            "requires_rag": True,
        })
        return RetrievalPolicy(
            retrieval=cfg["retrieval"],
            top_k=cfg["top_k"],
            requires_rag=cfg["requires_rag"],
            searchable_topics=cfg.get("searchable_topics", []),
        )

    def clear_session(self, session_id: str) -> None:
        """Clear follow-up history for a session."""
        self._followup_history.pop(session_id, None)