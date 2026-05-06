"""
src/core/Agent/ResponseTypes.py
--------------------------------
Registry of all Darwin response types.
Maps each type to its spaCy/regex triggers, LLM JSON schema,
and frontend component name.

Usage:
    from src.core.Agent.ResponseTypes import RESPONSE_TYPES, detect_response_type

    response_type = detect_response_type(doc, query)
    schema        = RESPONSE_TYPES[response_type].schema
    component     = RESPONSE_TYPES[response_type].component
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import re


# ═══════════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════════

ResponseTypeName = Literal[
    "plain_text",
    "accordion",
    "comparison_table",
    "stats_grid",
    "summary_card",
    "flowchart",
    "timeline",
    "entity_map",
    "bar_chart",
    "stepper",
    "parametric",
    "knowledge_graph",
]


@dataclass
class ResponseType:
    name: ResponseTypeName

    # what the frontend React component is called
    component: str

    # plain english description
    description: str

    # difficulty tier — used to decide if we attempt it
    tier: Literal["easy", "medium", "hard"]

    # keyword triggers (checked against lowercased query)
    keyword_triggers: list[str] = field(default_factory=list)

    # spaCy tag triggers — checked against doc token tags
    # e.g. "WP" = who/what, "WRB" = where/when/how
    pos_triggers: list[str] = field(default_factory=list)

    # regex patterns — any match triggers this type
    regex_triggers: list[str] = field(default_factory=list)

    # minimum number of spaCy entities needed to trigger
    min_entities: int = 0

    # the JSON schema the LLM must return for this type
    # sent as part of the system prompt
    schema: str = ""

    # how many chunks to retrieve for this type
    top_k: int = 5

    # whether to run search at all (False for followups)
    requires_search: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

RESPONSE_TYPES: dict[ResponseTypeName, ResponseType] = {

    # ── Tier 1: Easy ──────────────────────────────────────────────────────────

    "plain_text": ResponseType(
        name="plain_text",
        component="PlainTextResponse",
        description="Default fallback. Biographical, simple lookups, followups.",
        tier="easy",
        schema="""
        {
            "answer": "string",
            "sources": [{"file": "string", "page": "number"}]
        }
        """,
        top_k=5,
    ),

    "accordion": ResponseType(
        name="accordion",
        component="AccordionResponse",
        description="Multiple questions detected. Each gets its own collapsible card.",
        tier="easy",
        keyword_triggers=["and why", "and how", "and what", "also"],
        schema="""
        {
            "questions": [
                {
                    "question": "string",
                    "answer": "string",
                    "sources": [{"file": "string", "page": "number"}]
                }
            ]
        }
        """,
        top_k=5,
    ),

    "comparison_table": ResponseType(
        name="comparison_table",
        component="ComparisonTable",
        description="Two subjects compared across attributes. Side by side table.",
        tier="easy",
        keyword_triggers=[
            "compare", "vs", "versus", "difference between",
            "contrast", "similarities", "how do they differ",
            "what is the difference",
        ],
        schema="""
        {
            "title": "string",
            "subjects": ["subject_a", "subject_b"],
            "attributes": [
                {
                    "name": "string",
                    "subject_a": "string",
                    "subject_b": "string"
                }
            ],
            "sources": [{"file": "string", "page": "number"}]
        }
        """,
        top_k=10,
    ),

    "stats_grid": ResponseType(
        name="stats_grid",
        component="StatsGrid",
        description="Big number cards. Counts, frequencies, quantitative answers.",
        tier="easy",
        keyword_triggers=[
            "how many", "count", "number of", "total",
            "how often", "frequency", "statistics", "how much",
        ],
        schema="""
        {
            "title": "string",
            "metrics": [
                {
                    "label": "string",
                    "value": "number",
                    "unit": "string",
                    "note": "string"
                }
            ],
            "sources": [{"file": "string", "page": "number"}]
        }
        """,
        top_k=10,
    ),

    "summary_card": ResponseType(
        name="summary_card",
        component="SummaryCard",
        description="Title, key points, source list. Richer than plain text.",
        tier="easy",
        keyword_triggers=[
            "summarize", "summary", "overview", "tldr",
            "brief", "in short", "key points", "main points",
            "what is this about", "give me a summary",
        ],
        schema="""
        {
            "title": "string",
            "overview": "string",
            "key_points": ["string"],
            "sources": [{"file": "string", "page": "number"}]
        }
        """,
        top_k=8,
    ),

    # ── Tier 2: Medium ────────────────────────────────────────────────────────

    "flowchart": ResponseType(
        name="flowchart",
        component="FlowchartViz",
        description="Process steps, cause and effect, decision trees.",
        tier="medium",
        keyword_triggers=[
            "how does", "how do", "process", "steps", "flow",
            "what happens when", "what happens to", "procedure",
            "mechanism", "how is", "how are",
        ],
        regex_triggers=[
            r"how does .+ work",
            r"what (are|is) the steps",
            r"walk me through .+ process",
        ],
        schema="""
        {
            "title": "string",
            "nodes": [
                {
                    "id": "string",
                    "label": "string",
                    "sub": "string",
                    "color": "purple | teal | coral | amber | gray"
                }
            ],
            "edges": [
                {
                    "from": "string",
                    "to": "string",
                    "label": "string"
                }
            ],
            "sources": [{"file": "string", "page": "number"}]
        }
        """,
        top_k=10,
    ),

    "timeline": ResponseType(
        name="timeline",
        component="TimelineViz",
        description="Chronological events. History, sequence, order of events.",
        tier="medium",
        keyword_triggers=[
            "timeline", "chronological", "chronology", "history of",
            "order of events", "sequence of", "when did",
            "what happened first", "events in",
        ],
        regex_triggers=[
            r"(history|story|events) of .+",
            r"what (happened|occurs) (in|during|after|before)",
            r"chronolog(y|ical)",
        ],
        schema="""
        {
            "title": "string",
            "events": [
                {
                    "era": "string",
                    "label": "string",
                    "description": "string",
                    "significance": "string",
                    "source": {"file": "string", "page": "number"}
                }
            ]
        }
        """,
        top_k=15,
    ),

    "entity_map": ResponseType(
        name="entity_map",
        component="EntityMapViz",
        description="Relationship graph between people, places, concepts.",
        tier="medium",
        keyword_triggers=[
            "who is related", "relationship", "connected to",
            "map the", "network of", "who are the", "characters in",
            "people in", "figures in",
        ],
        regex_triggers=[
            r"(who|what) is (related|connected) to",
            r"relationship between .+ and .+",
            r"map .+ (characters|figures|entities|people)",
        ],
        min_entities=1,
        schema="""
        {
            "title": "string",
            "nodes": [
                {
                    "id": "string",
                    "label": "string",
                    "type": "person | place | concept | divine | creature",
                    "color": "purple | teal | coral | amber | red | gray"
                }
            ],
            "edges": [
                {
                    "from": "string",
                    "to": "string",
                    "label": "string"
                }
            ],
            "sources": [{"file": "string", "page": "number"}]
        }
        """,
        top_k=15,
    ),

    "bar_chart": ResponseType(
        name="bar_chart",
        component="BarChartViz",
        description="Distribution, frequency across sections or chapters.",
        tier="medium",
        keyword_triggers=[
            "frequency", "distribution", "across chapters",
            "chart", "graph", "breakdown", "per chapter",
            "how often does", "how frequently",
        ],
        regex_triggers=[
            r"(frequency|distribution|breakdown) of .+",
            r"how (often|many times) .+ (appears?|occurs?|mentioned)",
            r"across (all )?(chapters|sections|documents|books)",
        ],
        schema="""
        {
            "title": "string",
            "x_label": "string",
            "y_label": "string",
            "bars": [
                {
                    "label": "string",
                    "value": "number"
                }
            ],
            "sources": [{"file": "string", "page": "number"}]
        }
        """,
        top_k=20,
    ),

    # ── Tier 3: Hard ──────────────────────────────────────────────────────────

    "stepper": ResponseType(
        name="stepper",
        component="StepperViz",
        description="Guided walkthrough, one stage at a time. Click through.",
        tier="hard",
        keyword_triggers=[
            "walk me through", "step by step", "guide me",
            "take me through", "explain each", "one by one",
        ],
        regex_triggers=[
            r"walk (me )?through .+",
            r"(step|stage) by (step|stage)",
            r"explain .+ step by step",
        ],
        schema="""
        {
            "title": "string",
            "stages": [
                {
                    "label": "string",
                    "answer": "string",
                    "source": {"file": "string", "page": "number"}
                }
            ]
        }
        """,
        top_k=20,
    ),

    "parametric": ResponseType(
        name="parametric",
        component="ParametricViz",
        description="User controls variables, values update live. What-if queries.",
        tier="hard",
        keyword_triggers=[
            "what if", "if they were", "suppose", "assume",
            "hypothetically", "if it was", "if there were",
        ],
        regex_triggers=[
            r"(if|what if|suppose|assume)\s.+\s(were?|is|had|has)\s[\d,]+",
            r"hypothetically.+\d+",
            r"if .+ (was|were) [\d,]+ (feet|meters|times|percent|years)",
        ],
        schema="""
        {
            "title": "string",
            "variable": {
                "name": "string",
                "unit": "string",
                "min": "number",
                "max": "number",
                "default": "number"
            },
            "formula": "string",
            "metrics": [
                {
                    "label": "string",
                    "expression": "string",
                    "unit": "string"
                }
            ],
            "context": "string",
            "sources": [{"file": "string", "page": "number"}]
        }
        """,
        top_k=5,
    ),

    "knowledge_graph": ResponseType(
        name="knowledge_graph",
        component="KnowledgeGraphViz",
        description="Full document concept map. Force-directed, interactive.",
        tier="hard",
        keyword_triggers=[
            "map everything", "show all", "full overview",
            "all connections", "everything related",
            "complete map", "entire document",
        ],
        regex_triggers=[
            r"map (everything|all (concepts|entities|connections))",
            r"(full|complete|entire) (overview|map|graph)",
            r"show (me )?all (the )?(connections|relationships|concepts)",
        ],
        min_entities=3,
        schema="""
        {
            "title": "string",
            "nodes": [
                {
                    "id": "string",
                    "label": "string",
                    "type": "string",
                    "weight": "number"
                }
            ],
            "edges": [
                {
                    "from": "string",
                    "to": "string",
                    "label": "string",
                    "weight": "number"
                }
            ]
        }
        """,
        top_k=30,
        requires_search=True,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_response_type(
    doc,                          # spaCy doc
    query: str,
    multi_question: bool = False,
    is_followup: bool = False,
    enabled_tiers: list[str] = None,
) -> ResponseTypeName:
    """
    Detect the best response type for a query.

    Priority order:
    1. followup           → plain_text (reuse context, no search)
    2. multi_question     → accordion
    3. hard tier matches  → parametric, stepper, knowledge_graph
    4. medium tier        → flowchart, timeline, entity_map, bar_chart
    5. easy tier          → comparison_table, stats_grid, summary_card
    6. fallback           → plain_text

    Args:
        doc:             spaCy processed document
        query:           raw query string
        multi_question:  whether spaCy detected 2+ questions
        is_followup:     whether this is a conversational followup
        enabled_tiers:   which tiers to allow — defaults to all
                         pass ["easy"] for v1, add more as you build
    """

    if enabled_tiers is None:
        enabled_tiers = ["easy", "medium", "hard"]

    text = query.lower().strip()
    entity_count = len(doc.ents)

    # followup → always plain text, skip search
    if is_followup:
        return "plain_text"

    # multi question → accordion
    if multi_question and "easy" in enabled_tiers:
        return "accordion"

    # check each type in priority order
    priority_order: list[ResponseTypeName] = [
        # hard first — most specific triggers
        "parametric",
        "knowledge_graph",
        "stepper",
        # medium
        "timeline",
        "entity_map",
        "flowchart",
        "bar_chart",
        # easy
        "comparison_table",
        "stats_grid",
        "summary_card",
    ]

    for type_name in priority_order:
        rt = RESPONSE_TYPES[type_name]

        # skip if tier not enabled
        if rt.tier not in enabled_tiers:
            continue

        # check entity minimum
        if entity_count < rt.min_entities:
            continue

        # check keyword triggers
        if any(kw in text for kw in rt.keyword_triggers):
            return type_name

        # check regex triggers
        if any(re.search(pattern, text) for pattern in rt.regex_triggers):
            return type_name

    return "plain_text"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_schema(type_name: ResponseTypeName) -> str:
    """Get the LLM JSON schema for a response type."""
    return RESPONSE_TYPES[type_name].schema.strip()


def get_top_k(type_name: ResponseTypeName) -> int:
    """Get the recommended top_k for a response type."""
    return RESPONSE_TYPES[type_name].top_k


def get_component(type_name: ResponseTypeName) -> str:
    """Get the frontend React component name for a response type."""
    return RESPONSE_TYPES[type_name].component


def list_by_tier(tier: Literal["easy", "medium", "hard"]) -> list[ResponseType]:
    """List all response types in a given tier."""
    return [rt for rt in RESPONSE_TYPES.values() if rt.tier == tier]


def get_enabled(enabled_tiers: list[str]) -> list[ResponseTypeName]:
    """Get all type names that are currently enabled."""
    return [
        name for name, rt in RESPONSE_TYPES.items()
        if rt.tier in enabled_tiers
    ]