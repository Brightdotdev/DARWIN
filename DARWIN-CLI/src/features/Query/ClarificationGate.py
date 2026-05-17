"""
src/core/Agent/ClarificationGate.py
------------------------------------
Determines if a query is ready for retrieval or needs clarification.

- Fast rule-based guards (empty query, comparison without 2 files, etc.)
- Small LLM call for intent classification + format suggestion
- Post-LLM heuristic override for vague references the model missed
- Returns either: clean intent strategy, OR follow-up blocks to render
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum
import re
from src.features.Query.InputPreprocessor import RawQuery
from src.core.Logger.DarwinLogger import darwin, LogType


class QuestionType(Enum):
    FACT = "fact"
    INSTRUCTION = "instruction"
    REASONING = "reasoning"
    COMPARISON = "comparison"
    OPINION = "opinion"
    HYPOTHETICAL = "hypothetical"
    CLARIFICATION = "clarification"
    MULTI_PART = "multi_part"
    YES_NO = "yes_no"
    STATEMENT = "statement"


class ResponseFormat(Enum):
    MARKDOWN = "markdown"
    SUMMARY_CARD = "summary_card"
    PROFILE_CARD = "profile_card"
    STATS_GRID = "stats_grid"
    PROS_CONS = "pros_cons"
    COMPARISON_TABLE = "comparison_table"
    ACCORDION = "accordion"
    TIMELINE = "timeline"
    FLOWCHART = "flowchart"
    NETWORK_GRAPH = "network_graph"
    RELATIONSHIP_MAP = "relationship_map"
    BAR_CHART = "bar_chart"
    STEPPER = "stepper"
    PARAMETER_SANDBOX = "parameter_sandbox"
    DIAGNOSTIC_TREE = "diagnostic_tree"


@dataclass
class FollowUpOption:
    id: str
    label: str
    value: str


@dataclass
class FollowUpBlock:
    question: str
    options: List[FollowUpOption] = field(default_factory=list)
    allow_custom: bool = False
    answer: Optional[str] = None
    reason: str = "other"


@dataclass
class QueryIntent:
    clean_query: str
    question_types: List[QuestionType]
    suggested_formats: List[ResponseFormat]
    needs_clarification: bool
    followup_blocks: List[FollowUpBlock]
    missing_requirements: List[str]
    retrieval_mode: str = "hybrid"
    top_k: int = 5
    requires_rag: bool = True


class ClarificationGate:
    # Tokens that are vague without prior conversation context
    VAGUE_TOKENS = {
        "them", "those", "these", "it", "that", "this", "they",
        "the big ones", "the small ones", "the other one", "the first one",
        "the last one", "the thing", "the stuff", "the person", "the place",
        "he", "she", "his", "her", "its",
    }

    def __init__(self, llm_client: Callable):
        self._call = llm_client  # call_small(messages) -> response with .text and .error

    def process(self, raw: RawQuery, file_registry: List[str]) -> QueryIntent:
        # 1. Fast rule-based guards (cheap, deterministic)
        fast = self._rule_based_check(raw, file_registry)
        if fast.needs_clarification:
            return fast

        # 2. LLM classification (understands nuance)
        llm = self._llm_classify(raw, file_registry)

        # 3. Merge with rules + post-LLM heuristics
        return self._merge_with_rules(llm, raw, file_registry)

    # ── Rule-based guards ─────────────────────────────────────────────────

    def _rule_based_check(self, raw: RawQuery, file_registry: List[str]) -> QueryIntent:
        missing: List[str] = []
        followups: List[FollowUpBlock] = []

        # Empty or garbage query
        if not raw.query or len(raw.query.strip()) < 3:
            return QueryIntent(
                clean_query=raw.query,
                question_types=[QuestionType.CLARIFICATION],
                suggested_formats=[ResponseFormat.DIAGNOSTIC_TREE],
                needs_clarification=True,
                followup_blocks=[FollowUpBlock(
                    question="What would you like to know?",
                    allow_custom=True,
                    reason="missing_context"
                )],
                missing_requirements=["query_too_short"],
            )

        # Obvious comparison intent but not enough files
        compare_keywords = ["compare", "vs", "versus", "difference between", "how do", "contrast"]
        if any(k in raw.query.lower() for k in compare_keywords) and len(raw.pinned_files) < 2:
            missing.append("comparison requires 2+ files")
            followups.append(FollowUpBlock(
                question=f"You want to compare sources, but only have {len(raw.pinned_files)} file(s) pinned. Add another file?",
                options=[
                    FollowUpOption(id="add_file", label="Add another file", value="add_file"),
                    FollowUpOption(id="continue_single", label="Just search this one", value="continue_single"),
                ],
                allow_custom=True,
                reason="missing_parameter"
            ))

        # Pasted blob with no clear directive
        if raw.pasted_texts and len(raw.query.strip()) < 30:
            missing.append("pasted_text_without_question")
            followups.append(FollowUpBlock(
                question="I see you've pasted content. What would you like me to do with it?",
                options=[
                    FollowUpOption(id="summarize", label="Summarize", value="summarize"),
                    FollowUpOption(id="extract", label="Extract key points", value="extract"),
                    FollowUpOption(id="compare", label="Compare with my files", value="compare"),
                ],
                allow_custom=True,
                reason="missing_context"
            ))

        # Referenced pinned file not in registry
        for pf in raw.pinned_files:
            if pf not in file_registry:
                missing.append(f"file_not_found: {pf}")
                followups.append(FollowUpBlock(
                    question=f"File '{pf}' is not in the registry. Please upload or select it.",
                    allow_custom=True,
                    reason="file_not_searchable"
                ))

        if missing:
            return QueryIntent(
                clean_query=raw.query,
                question_types=[QuestionType.CLARIFICATION],
                suggested_formats=[ResponseFormat.DIAGNOSTIC_TREE],
                needs_clarification=True,
                followup_blocks=followups,
                missing_requirements=missing,
            )

        # Pass through to LLM
        return QueryIntent(
            clean_query=raw.query,
            question_types=[],
            suggested_formats=[],
            needs_clarification=False,
            followup_blocks=[],
            missing_requirements=[],
        )

    # ── LLM classification ────────────────────────────────────────────────

    def _llm_classify(self, raw: RawQuery, file_registry: List[str]) -> QueryIntent:
        prompt = self._build_prompt(raw, file_registry)
        messages = [{"role": "user", "content": prompt}]

        try:
            resp = self._call(messages)
            if resp.error or not resp.text:
                darwin.log(f"Clarification gate LLM failed: {resp.error}", LogType.WARNING)
                return self._fallback_intent(raw)

            # Extract JSON
            text = resp.text.strip()
            if "```" in text:
                for part in text.split("```"):
                    clean = part.replace("json", "").strip()
                    if clean.startswith("{"):
                        text = clean
                        break

            parsed = json.loads(text)

            # Safely parse enums
            qtypes = []
            for qt in parsed.get("question_types", []):
                try:
                    qtypes.append(QuestionType(qt))
                except ValueError:
                    pass

            formats = []
            for fmt in parsed.get("suggested_formats", []):
                try:
                    formats.append(ResponseFormat(fmt))
                except ValueError:
                    pass

            # Parse follow-up blocks
            followups = []
            for fb in parsed.get("followup_blocks", []):
                opts = [FollowUpOption(**o) for o in fb.get("options", [])]
                followups.append(FollowUpBlock(
                    question=fb.get("question", ""),
                    options=opts,
                    allow_custom=fb.get("allow_custom", False),
                    reason=fb.get("reason", "other")
                ))

            return QueryIntent(
                clean_query=parsed.get("clean_query", raw.query),
                question_types=qtypes or [QuestionType.FACT],
                suggested_formats=formats or [ResponseFormat.MARKDOWN],
                needs_clarification=parsed.get("needs_clarification", False),
                followup_blocks=followups,
                missing_requirements=parsed.get("missing_requirements", []),
                retrieval_mode=parsed.get("retrieval_mode", "hybrid"),
                top_k=parsed.get("top_k", 5),
                requires_rag=parsed.get("requires_rag", True),
            )

        except Exception as e:
            darwin.log(f"Clarification gate error: {e}", LogType.WARNING)
            return self._fallback_intent(raw)

    def _build_prompt(self, raw: RawQuery, file_registry: List[str]) -> str:
        # Truncate huge inputs for the small model
        q_display = raw.query[:2000] + "..." if len(raw.query) > 2000 else raw.query

        pasted_desc = ""
        for i, pt in enumerate(raw.pasted_texts):
            preview = pt.content[:200].replace("\n", " ")
            pasted_desc += f"\n- Blob {i+1} ({pt.type.value}): {preview}..."

        url_desc = ""
        for u in raw.extracted_urls:
            url_desc += f"\n- {u.placeholder}: {u.url}"

        return f"""You are a query classifier for a research assistant.

USER INPUT:
{q_display}

PASTED BLOBS: {pasted_desc or "none"}
EXTRACTED URLS: {url_desc or "none"}
PINNED FILES: {', '.join(raw.pinned_files) if raw.pinned_files else 'none'}
AVAILABLE REGISTRY: {', '.join(file_registry) if file_registry else 'none'}

Task:
1. Identify question type(s)
2. Suggest best response format(s)
3. Check if pinned files satisfy the query requirements
4. Detect vague references (pronouns like "them", "those", "it", "the big ones", "that thing")
5. Decide if clarification is needed

Question types: fact, instruction, reasoning, comparison, opinion, hypothetical, clarification, multi_part, yes_no, statement

Response formats: markdown, summary_card, profile_card, stats_grid, pros_cons, comparison_table, accordion, timeline, flowchart, network_graph, relationship_map, bar_chart, stepper, parameter_sandbox, diagnostic_tree

Rules:
- comparison needs 2+ files
- yes_no and statement can work with 0 files
- fact/reasoning/instruction/multi_part need at least 1 file if RAG required
- pasted blob without clear question → needs clarification
- VAGUE REFERENCES: If the query uses pronouns or unclear nouns ("the big ones", "them", "that", "it") with NO prior conversation context, set needs_clarification=true and ask what they mean
- If the query is so broad that any answer would be a guess, ask for specificity

Return ONLY JSON:
{{
  "clean_query": "core question",
  "question_types": ["fact"],
  "suggested_formats": ["summary_card"],
  "needs_clarification": false,
  "missing_requirements": [],
  "followup_blocks": [],
  "retrieval_mode": "hybrid",
  "top_k": 5,
  "requires_rag": true
}}

If needs_clarification=true, include followup_blocks:
{{
  "followup_blocks": [
    {{
      "question": "What would you like to know?",
      "options": [{{"id": "opt1", "label": "Summarize", "value": "summarize"}}],
      "allow_custom": true,
      "reason": "missing_context"
    }}
  ]
}}
"""

    # ── Post-LLM merge + heuristic override ───────────────────────────────

    def _merge_with_rules(self, llm_intent: QueryIntent, raw: RawQuery, file_registry: List[str]) -> QueryIntent:
        # Override LLM if it missed hard constraints
        if QuestionType.COMPARISON in llm_intent.question_types and len(raw.pinned_files) < 2:
            llm_intent.needs_clarification = True
            llm_intent.missing_requirements.append("comparison requires 2+ files")
            if not any(fb.reason == "missing_parameter" for fb in llm_intent.followup_blocks):
                llm_intent.followup_blocks.append(FollowUpBlock(
                    question="Comparison requires at least 2 files. Please add another.",
                    options=[FollowUpOption(id="add_file", label="Add file", value="add_file")],
                    allow_custom=True,
                    reason="missing_parameter"
                ))

        # File existence check
        for pf in raw.pinned_files:
            if pf not in file_registry:
                llm_intent.needs_clarification = True
                llm_intent.missing_requirements.append(f"file_not_found: {pf}")

        # POST-LLM HEURISTIC: catch vague references the LLM missed
        if not llm_intent.needs_clarification and self._has_vague_references(llm_intent.clean_query):
            llm_intent.needs_clarification = True
            llm_intent.missing_requirements.append("vague_reference_detected")
            llm_intent.followup_blocks.append(FollowUpBlock(
                question=f"Your query mentions '{llm_intent.clean_query}'. Could you be more specific?",
                allow_custom=True,
                reason="vague_reference"
            ))

        return llm_intent

    def _has_vague_references(self, query: str) -> bool:
        """Fast regex scan for vague pronouns and unclear nouns."""
        if not query:
            return False
        lowered = query.lower()
        # Check exact vague phrases
        for token in self.VAGUE_TOKENS:
            # Use word boundaries for short tokens, substring for phrases
            if len(token) <= 3:
                pattern = r'\b' + re.escape(token) + r'\b'
                if re.search(pattern, lowered):
                    return True
            else:
                if token in lowered:
                    return True
        return False

    def _fallback_intent(self, raw: RawQuery) -> QueryIntent:
        return QueryIntent(
            clean_query=raw.query,
            question_types=[QuestionType.FACT],
            suggested_formats=[ResponseFormat.PLAIN_TEXT],
            needs_clarification=False,
            followup_blocks=[],
            missing_requirements=[],
            retrieval_mode="hybrid",
            top_k=5,
            requires_rag=True,
        )