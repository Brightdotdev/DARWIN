"""
Core dataclasses for the Query pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal


# ── Question Classification ────────────────────────────────────────────────
QuestionType = Literal[
    "fact",
    "instruction",
    "reasoning",
    "comparison",
    "opinion",
    "hypothetical",
    "clarification",
    "multi_part",
    "yes_no",
    "statement",
]


# ── Retrieval ───────────────────────────────────────────────────────────────
RetrievalMode = Literal["hybrid", "semantic", "keyword"]


@dataclass
class RetrievalPolicy:
    retrieval: RetrievalMode
    top_k: int
    searchable_topics: List[str] = field(default_factory=list)
    requires_tools: bool = False
    requires_rag: bool = True


# ── Artifact Presentation ──────────────────────────────────────────────────
ResponsePresentation = Literal[
    "summary_card",
    "profile_card",
    "stats_grid",
    "pros_cons",
    "comparison_table",
    "accordion",
    "timeline",
    "flowchart",
    "network_graph",
    "relationship_map",
    "bar_chart",
    "stepper",
    "parameter_sandbox",
    "diagnostic_tree",
]


# ── Pasted Text / URLs ─────────────────────────────────────────────────────
PastedTextType = Literal["document", "code", "data", "long_question"]


@dataclass
class PastedText:
    content: str
    type: PastedTextType
    id: Optional[str] = None


@dataclass
class URLRef:
    url: str
    domain: str
    placeholder: str


@dataclass
class URLContent:
    url: str
    fetched_text: str
    title: Optional[str] = None
    fetched_at: Optional[float] = None


# ── Follow‑up ───────────────────────────────────────────────────────────────
FollowUpReason = Literal[
    "missing_context",
    "ambiguous_query",
    "missing_parameter",
    "tool_selection",
    "confirmation_required",
    "vague_reference",
    "no_scope",
    "url_no_search",
    "file_not_searchable",
    "ambiguous_file_match",
    "other",
]


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
    reason: FollowUpReason = "other"


# ── Raw Query ───────────────────────────────────────────────────────────────
@dataclass
class RawQuery:
    query: str
    pasted_texts: List[PastedText] = field(default_factory=list)
    pinned_files: List[str] = field(default_factory=list)
    extracted_urls: List[URLRef] = field(default_factory=list)


# ── Query Context (after preprocessing) ─────────────────────────────────────
@dataclass
class QueryContext:
    query: str                                    # cleaned query
    blobs: List[PastedText] = field(default_factory=list)
    searchable_files: List[str] = field(default_factory=list)
    non_searchable_files: List[str] = field(default_factory=list)
    extracted_urls: List[URLRef] = field(default_factory=list)
    fetched_url_contents: List[URLContent] = field(default_factory=list)


# ── Indexing Suggestion (short‑circuit) ─────────────────────────────────────
@dataclass
class IndexingSuggestion:
    message: str
    files: List[str]
    action: str = "index_now"
    next_step: str = "Would you like to index these files so they can be searched?"


# ── LLM Response (from our client) ──────────────────────────────────────────
@dataclass
class LLMResponse:
    text: str = ""
    json: Optional[Dict[str, Any]] = None
    error: Optional[str] = None




TurnType = Literal[
    "new_query",
    "follow_up_answer",
    "elaboration",
    "correction",
    "re_render",
    "chitchat",
]