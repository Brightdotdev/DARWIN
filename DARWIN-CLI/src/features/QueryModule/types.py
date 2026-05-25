"""
Darwin Query Module — Core Types
All dataclasses, enums, and type aliases for the QueryModule pipeline.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union


# ───────────────────────────────────────────────
# Enums
# ───────────────────────────────────────────────

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
    VAGUE = "vague"
    FOLLOWUP = "followup"


class ResponsePresentation(Enum):
    SUMMARY_CARD = "summary_card"
    COMPARISON_TABLE = "comparison_table"
    TIMELINE = "timeline"
    NETWORK_GRAPH = "network_graph"
    FLOWCHART = "flowchart"
    PARAMETER_SANDBOX = "parameter_sandbox"
    DIAGNOSTIC_TREE = "diagnostic_tree"
    STEPPER = "stepper"
    ACCORDION = "accordion"
    MARKDOWN = "markdown"
    RELATIONSHIP_MAP = "relationship_map"
    PROS_CONS = "pros_cons"
    STATS_GRID = "stats_grid"
    PROFILE_CARD = "profile_card"


class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class BlobType(Enum):
    CODE = "code_blob"
    DATA = "data_blob"
    DOCUMENT = "document_blob"
    LONG_QUESTION = "long_question"
    NONE = "none"


class RetrievalMode(Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


# ───────────────────────────────────────────────
# Blob & Input Types
# ───────────────────────────────────────────────

@dataclass
class BlobChunk:
    index: int
    content: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BlobHandle:
    blob_id: str
    blob_type: BlobType
    summary: str
    chunk_count: int
    chunks: List[BlobChunk] = field(default_factory=list)
    raw_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_relevant_chunks(self, query: str, top_k: int = 3) -> List[BlobChunk]:
        """Simple keyword relevance scoring for chunk retrieval."""
        query_terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for chunk in self.chunks:
            chunk_terms = set(re.findall(r"\w+", chunk.content.lower()))
            score = len(query_terms & chunk_terms)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]


@dataclass
class URLRef:
    url: str
    title: Optional[str] = None
    domain: Optional[str] = None


@dataclass
class ParsedInput:
    clean_query: str
    blob: Optional[BlobHandle]
    extracted_urls: List[URLRef] = field(default_factory=list)
    pinned_files: List[str] = field(default_factory=list)
    raw_input: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ───────────────────────────────────────────────
# Classification Types
# ───────────────────────────────────────────────

@dataclass
class FollowupBlock:
    question: str
    options: List[Dict[str, str]]  # {"id": str, "label": str, "value": str}
    allow_custom: bool = False
    reason: str = "missing_context"  # missing_context | ambiguous_query | missing_parameter


@dataclass
class RetrievalPolicy:
    retrieval: RetrievalMode
    top_k: int
    requires_rag: bool
    searchable_topics: List[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    clean_query: str
    question_types: List[QuestionType]
    suggested_representations: List[ResponsePresentation]
    retrieval_policy: RetrievalPolicy
    needs_clarification: bool
    followup_blocks: Optional[List[FollowupBlock]]
    confidence: float  # 0.0–1.0
    explicit_format_lock: Optional[ResponsePresentation] = None  # Set when user asks for specific format
    is_followup: bool = False
    inherited_type: Optional[QuestionType] = None


# ───────────────────────────────────────────────
# Retrieval Types
# ───────────────────────────────────────────────

@dataclass
class RetrievedPassage:
    content: str
    source: str
    score: float
    page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedConcept:
    name: str
    description: str
    related: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class RetrievalContext:
    passages: List[RetrievedPassage] = field(default_factory=list)
    concepts: List[RetrievedConcept] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    total_found: int = 0
    query_used: str = ""
    policy: Optional[RetrievalPolicy] = None

    def is_empty(self) -> bool:
        return len(self.passages) == 0 and len(self.concepts) == 0


# ───────────────────────────────────────────────
# Validation Types
# ───────────────────────────────────────────────

@dataclass
class ValidationResult:
    representation: ResponsePresentation
    status: str  # "PASS" | "PARTIAL" | "FAIL"
    reason: str = ""
    downgrade_chain: List[ResponsePresentation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ───────────────────────────────────────────────
# Generation Types
# ───────────────────────────────────────────────

@dataclass
class Citation:
    source: str
    page: Optional[int] = None
    passage_index: int = 0
    text: str = ""


@dataclass
class DeferredQuestion:
    question: str
    options: List[Dict[str, Any]]  # {"id": str, "label": str, "query_rewrite": str | None}
    context_preview: str = ""
    allow_custom: bool = True


@dataclass
class GenerationResult:
    primary_output: str
    fallback_text: str
    representation: ResponsePresentation
    confidence: ConfidenceLevel
    used_retrieval: bool

    citations: List[Citation] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    interactive_schema: Optional[Dict[str, Any]] = None
    deferred_question: Optional[DeferredQuestion] = None

# ───────────────────────────────────────────────
# Response Types
# ───────────────────────────────────────────────

@dataclass
class DirectResponse:
    mode: str = "direct"
    representation: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    interactive_schema: Optional[Dict[str, Any]] = None
    fallback_text: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


@dataclass
class DeferredResponse:
    mode: str = "deferred"
    message: str = ""
    options: List[Dict[str, Any]] = field(default_factory=list)
    context_preview: str = ""
    fallback_text: str = ""
    fallback_available: bool = True


@dataclass
class FollowupProcessedResponse:
    mode: str = "followup_processed"
    representation: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    interactive_schema: Optional[Dict[str, Any]] = None
    fallback_text: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    original_deferred: Optional[DeferredState] = None


ResponsePackage = Union[DirectResponse, DeferredResponse, FollowupProcessedResponse]


# ───────────────────────────────────────────────
# Follow-up & Deferred Types
# ───────────────────────────────────────────────

@dataclass
class DeferredState:
    original_query: str
    deferred_question: str
    chosen_option: str
    resolved_query: str
    resolved_type: QuestionType
    resolved_policy: RetrievalPolicy = field(default_factory=lambda: RetrievalPolicy(
        retrieval=RetrievalMode.HYBRID, top_k=5, requires_rag=True
    ))


@dataclass
class DeferredChoice:
    option_id: Optional[str] = None
    custom_text: Optional[str] = None


# ───────────────────────────────────────────────
# Conversation Plug-in Interface
# ───────────────────────────────────────────────

class ConversationContextProvider(ABC):
    """Plug-in interface for conversation memory. This module does NOT implement persistence."""

    @abstractmethod
    def get_context(self, conversation_id: str, current_query: str) -> List[Dict[str, Any]]:
        """Return recent conversation turns as list of {role, content, timestamp} dicts."""
        ...

    @abstractmethod
    def add_turn(self, conversation_id: str, role: str, content: str) -> None:
        """Record a turn in the conversation."""
        ...

    @abstractmethod
    def get_summary(self, conversation_id: str) -> str:
        """Return a summarized state of the conversation for LLM context injection."""
        ...


@dataclass
class ConversationContext:
    turns: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    conversation_id: Optional[str] = None


# ───────────────────────────────────────────────
# Skill System Types
# ───────────────────────────────────────────────

@dataclass
class SkillResult:
    success: bool
    fallback_text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


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


# ───────────────────────────────────────────────
# Utility / Mapping Constants
# ───────────────────────────────────────────────

QUESTION_RETRIEVAL_MAP: Dict[QuestionType, Dict[str, Any]] = {
    QuestionType.FACT:        {"retrieval": RetrievalMode.HYBRID, "top_k": 3,  "requires_rag": True},
    QuestionType.INSTRUCTION: {"retrieval": RetrievalMode.HYBRID, "top_k": 5,  "requires_rag": True},
    QuestionType.REASONING:   {"retrieval": RetrievalMode.HYBRID, "top_k": 8,  "requires_rag": True},
    QuestionType.COMPARISON:  {"retrieval": RetrievalMode.HYBRID, "top_k": 10, "requires_rag": True, "searchable_topics": []},
    QuestionType.OPINION:     {"retrieval": RetrievalMode.HYBRID, "top_k": 5,  "requires_rag": False},
    QuestionType.HYPOTHETICAL:{"retrieval": RetrievalMode.SEMANTIC, "top_k": 6, "requires_rag": False},
    QuestionType.CLARIFICATION:{"retrieval": RetrievalMode.KEYWORD, "top_k": 3, "requires_rag": False},
    QuestionType.MULTI_PART:  {"retrieval": RetrievalMode.HYBRID, "top_k": 10, "requires_rag": True},
    QuestionType.YES_NO:      {"retrieval": RetrievalMode.KEYWORD, "top_k": 3, "requires_rag": False},
    QuestionType.STATEMENT:   {"retrieval": RetrievalMode.KEYWORD, "top_k": 2, "requires_rag": False},
    QuestionType.VAGUE:       {"retrieval": RetrievalMode.HYBRID, "top_k": 5,  "requires_rag": True},
    QuestionType.FOLLOWUP:    {"retrieval": RetrievalMode.HYBRID, "top_k": 5,  "requires_rag": True},
}


REPRESENTATION_REQUIREMENTS: Dict[ResponsePresentation, Dict[str, Any]] = {
    ResponsePresentation.COMPARISON_TABLE:   {"min_passages": 4, "min_entities": 2, "requires_structure": True},
    ResponsePresentation.TIMELINE:           {"min_passages": 3, "requires_dates": True},
    ResponsePresentation.NETWORK_GRAPH:      {"min_passages": 5, "min_entities": 3, "requires_relationships": True},
    ResponsePresentation.FLOWCHART:          {"min_passages": 3, "requires_steps": True},
    ResponsePresentation.PARAMETER_SANDBOX:  {"min_passages": 3, "requires_variables": True},
    ResponsePresentation.DIAGNOSTIC_TREE:    {"min_passages": 3, "requires_branches": True},
    ResponsePresentation.STEPPER:            {"min_passages": 2, "requires_steps": True},
    ResponsePresentation.SUMMARY_CARD:       {"min_passages": 1},
    ResponsePresentation.ACCORDION:          {"min_passages": 2},
    ResponsePresentation.MARKDOWN:           {"min_passages": 0},
    ResponsePresentation.RELATIONSHIP_MAP: {"min_passages": 3, "min_entities": 2, "requires_relationships": True},
    ResponsePresentation.PROS_CONS:          {"min_passages": 2, "min_entities": 2},
    ResponsePresentation.STATS_GRID:       {"min_passages": 2, "requires_variables": True},
    ResponsePresentation.PROFILE_CARD:       {"min_passages": 1, "min_entities": 1},
}


DOWNGRADE_CHAIN: Dict[ResponsePresentation, List[ResponsePresentation]] = {
    ResponsePresentation.NETWORK_GRAPH: [
        ResponsePresentation.RELATIONSHIP_MAP,
        ResponsePresentation.SUMMARY_CARD,
        ResponsePresentation.MARKDOWN,
    ],
    ResponsePresentation.COMPARISON_TABLE: [
        ResponsePresentation.PROS_CONS,
        ResponsePresentation.SUMMARY_CARD,
        ResponsePresentation.MARKDOWN,
    ],
    ResponsePresentation.TIMELINE: [
        ResponsePresentation.ACCORDION,
        ResponsePresentation.SUMMARY_CARD,
        ResponsePresentation.MARKDOWN,
    ],
    ResponsePresentation.PARAMETER_SANDBOX: [
        ResponsePresentation.STATS_GRID,
        ResponsePresentation.SUMMARY_CARD,
        ResponsePresentation.MARKDOWN,
    ],
    ResponsePresentation.DIAGNOSTIC_TREE: [
        ResponsePresentation.STEPPER,
        ResponsePresentation.ACCORDION,
        ResponsePresentation.MARKDOWN,
    ],
    ResponsePresentation.FLOWCHART: [
        ResponsePresentation.STEPPER,
        ResponsePresentation.ACCORDION,
        ResponsePresentation.MARKDOWN,
    ],
    ResponsePresentation.STEPPER: [
        ResponsePresentation.ACCORDION,
        ResponsePresentation.MARKDOWN,
    ],
    ResponsePresentation.ACCORDION: [
        ResponsePresentation.SUMMARY_CARD,
        ResponsePresentation.MARKDOWN,
    ],
    ResponsePresentation.SUMMARY_CARD: [
        ResponsePresentation.MARKDOWN,
    ],
    ResponsePresentation.PROS_CONS: [
        ResponsePresentation.SUMMARY_CARD,
        ResponsePresentation.MARKDOWN,
    ],
    ResponsePresentation.STATS_GRID: [
        ResponsePresentation.SUMMARY_CARD,
        ResponsePresentation.MARKDOWN,
    ],
    ResponsePresentation.RELATIONSHIP_MAP: [
        ResponsePresentation.SUMMARY_CARD,
        ResponsePresentation.MARKDOWN,
    ],
    ResponsePresentation.PROFILE_CARD: [
        ResponsePresentation.SUMMARY_CARD,
        ResponsePresentation.MARKDOWN,
    ],
}

# Fast-path regex patterns → QuestionType
FAST_PATH_PATTERNS: List[Tuple[re.Pattern, QuestionType, float]] = [
    (re.compile(r"compare|vs\b|difference|versus", re.I), QuestionType.COMPARISON, 0.90),
    (re.compile(r"how\s+(do|can|to|would)", re.I), QuestionType.INSTRUCTION, 0.88),
    (re.compile(r"why|because|reason|cause", re.I), QuestionType.REASONING, 0.85),
    (re.compile(r"who|what\s+is|when|where|how\s+(many|much)", re.I), QuestionType.FACT, 0.90),
    (re.compile(r"is\s+it\s+true|can\s+you|are\s+they|was\s+he", re.I), QuestionType.YES_NO, 0.85),
    (re.compile(r"what\s+if|imagine|suppose|hypothetically", re.I), QuestionType.HYPOTHETICAL, 0.88),
    (re.compile(r"clarify|explain\s+what\s+you\s+mean|what\s+do\s+you", re.I), QuestionType.CLARIFICATION, 0.82),
    (re.compile(r"what\s+about|tell\s+me\s+more|and\s+then|go\s+on", re.I), QuestionType.FOLLOWUP, 0.80),
]


# Explicit format lock patterns → ResponsePresentation
FORMAT_LOCK_PATTERNS: List[Tuple[re.Pattern, ResponsePresentation]] = [
    (re.compile(r"show\s+me\s+a\s+(table|comparison\s+table|compare\s+table)", re.I), ResponsePresentation.COMPARISON_TABLE),
    (re.compile(r"show\s+me\s+a\s+(graph|network\s+graph|node\s+graph)", re.I), ResponsePresentation.NETWORK_GRAPH),
    (re.compile(r"show\s+me\s+a\s+(timeline|chronology)", re.I), ResponsePresentation.TIMELINE),
    (re.compile(r"show\s+me\s+a\s+(flowchart|flow\s+chart)", re.I), ResponsePresentation.FLOWCHART),
    (re.compile(r"show\s+me\s+a\s+(tree|diagnostic\s+tree|decision\s+tree)", re.I), ResponsePresentation.DIAGNOSTIC_TREE),
    (re.compile(r"show\s+me\s+(a\s+)?sandbox|parameter\s+sandbox|interactive\s+model", re.I), ResponsePresentation.PARAMETER_SANDBOX),
    (re.compile(r"show\s+me\s+(a\s+)?stepper|step\s+by\s+step", re.I), ResponsePresentation.STEPPER),
    (re.compile(r"show\s+me\s+(a\s+)?accordion", re.I), ResponsePresentation.ACCORDION),
    (re.compile(r"show\s+me\s+(a\s+)?summary\s+card", re.I), ResponsePresentation.SUMMARY_CARD),
]


HEAVY_SANDBOX_REPRESENTATIONS: List[ResponsePresentation] = [
    ResponsePresentation.PARAMETER_SANDBOX,
    ResponsePresentation.DIAGNOSTIC_TREE,
    ResponsePresentation.NETWORK_GRAPH,
]