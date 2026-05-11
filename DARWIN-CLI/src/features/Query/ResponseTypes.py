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

from typing import Literal, Optional,TypedDict
from dataclasses import dataclass, field


RetrievalMode = Literal["semantic", "keyword", "hybrid"]

@dataclass
class RetrievalPolicy(TypedDict, total=False):
    retrieval: RetrievalMode
    top_k: int
    searchable_topics: list[str]
    requires_tools: bool
    requires_rag: bool


QuestionType = Literal[
    "fact" ,
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

question_retrieval_map: dict[str, RetrievalPolicy] = {

    "fact": {
        "retrieval": "hybrid",
        "top_k": 3,
        "requires_rag": True,
    },

    "instruction": {
        "retrieval": "hybrid",
        "top_k": 5,
        "requires_rag": True,
    },

    "reasoning": {
        "retrieval": "hybrid",
        "top_k": 8,
        "requires_rag": True,
    },

    "comparison": {
        "retrieval": "hybrid",
        "top_k": 10,
        "searchable_topics": ["both_entities"],
        "requires_rag": True,
    },

    "opinion": {
        "retrieval": "hybrid",
        "top_k": 5,
        "requires_rag": False,
    },

    "hypothetical": {
        "retrieval": "semantic",
        "top_k": 6,
        "requires_rag": False,
    },

    "clarification": {
        "retrieval": "keyword",
        "top_k": 3,
        "requires_rag": False,
    },

    "multi_part": {
        "retrieval": "hybrid",
        "top_k": 10,
        "requires_rag": True,
    },

    "yes_no": {
        "retrieval": "keyword",
        "top_k": 3,
        "requires_rag": False,
    },

    "statement": {
        "retrieval": "keyword",
        "top_k": 2,
        "requires_rag": False,
    },
}


ResponsePresentation = Literal[
    # Structural
    "summary_card",       # TL;DR — condensed overview of a topic
    "profile_card",       # Fact sheet for a single entity (person, place, product)
    "stats_grid",         # Key numbers/metrics at a glance
    "pros_cons",          # Two-column evaluation of a single subject
    "comparison_table",   # N×M grid comparing multiple subjects across attributes
    "accordion",          # Expandable sections for long, structured content
    "timeline",           # Chronological sequence of events

    # Relational
    "flowchart",          # Process or decision flow (directional steps)
    "network_graph",      # Nodes + edges where relationships are the point
    "relationship_map",   # Concept cards with labeled connections between them

    # Data
    "bar_chart",          # Categorical comparisons
    "stepper",            # Guided multi-step walkthrough (linear)

    # Interactive
    "parameter_sandbox",  # Sliders/inputs to explore "what if" scenarios live
    "diagnostic_tree",    # Branching yes/no to reach a conclusion
    "code_walkthrough",   # Annotated code with step-by-step highlights
]

PastedTextType = Literal[
    "document",
    "code",
    "data",
    "long_question",
]


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
    "other"
]


@dataclass
class PastedText:
    content: str
    type: PastedTextType

@dataclass
class ExtractedURL:
    content: str
    type: PastedTextType



@dataclass
class URLRef:
    url: str
    domain: str
    placeholder: str



@dataclass
class FollowUpOption:
    id: str
    label: str
    value: str


@dataclass
class FollowUpBlock:
    question: str
    options: list[FollowUpOption] = field(default_factory=list),
    allow_custom: bool = False,
    answer: Optional[str] = None,
    reason: FollowUpReason = "other",
    

@dataclass
class RawQuery:
    query: str
    pasted_texts: list[PastedText] = field(default_factory=list)
    pinned_files : list[str] = field(default_factory=list)
    extracted_urls: list[URLRef] = field(default_factory=list) 





@dataclass
class OrganizedQuery:
    clean_query: str
    questionType : list[QuestionType]
    suggestedRepresentation: list[ResponsePresentation]
    pasted_texts: list[PastedText]
    extracted_urls: list[URLRef]
    pinned_file_ids: list[str]
    is_followup: bool
    needs_clarification: bool
    followup: list[FollowUpBlock] | None




prompt = """"

BELOW IS A QUERY FROM A USER:
{query}
HERE IS EXTRA METATATA FROM THE USER
PASTED TEXTS: {pasted_texts}
extracted_urls: {extracted_urls}

your job is to clasify this query and respond accordingly.

return only json of this class

class OrganizedQuery:
    clean_query: str
    questionType : list[QuestionType]
    suggestedRepresentation: list[ResponsePresentation]
    pasted_texts: list[PastedText]
    extracted_urls: list[URLRef]
    pinned_file_ids: list[str]
    is_followup: bool
    needs_clarification: bool
    followup: list[FollowUpBlock] | None

clean_query :: the pure query that the user asked
questionType :: the type of question the user is asking 
[
    "fact" ,
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
suggestedRepresentation :: this is a list of possible response types and you are to return a list of the best ones that fit the users query
also based on the question type you set for the query
[
   "plain_text",
    "markdown",
    "accordion",
    "comparison_table",
    "stats_grid",
    "summary_card",
    "timeline",
    "entity_map",
    "flowchart",
    "bar_chart",
    "stepper",
    "graph",
]

pasted_texts :: list of pasted text from the user
[
    {
        "content": "the content of the pasted text",
        "type": "document" | "code" | "data" | "long_question"
    }
]
this is just to be passed down do not touch this
extracted_urls :: list of urls extracted from the user query
[
    {
        "url": "the url",
        "domain": "the domain",
        "placeholder": "the placeholder"
    }
]
this is just to be passed down do not touch this



pinned_file_ids :: compare this to the {file_registry} list
{pinned_file_ids}....
and compare the  question type to
 "fact": {
        "retrieval": "hybrid",
        "top_k": 3,
        "requires_rag": True,
    },

    "instruction": {
        "retrieval": "hybrid",
        "top_k": 5,
        "requires_rag": True,
    },

    "reasoning": {
        "retrieval": "hybrid",
        "top_k": 8,
        "requires_rag": True,
    },

    "comparison": {
        "retrieval": "hybrid",
        "top_k": 10,
        "searchable_topics": ["both_entities"],
        "requires_rag": True,
    },

    "opinion": {
        "retrieval": "hybrid",
        "top_k": 5,
        "requires_rag": False,
    },

    "hypothetical": {
        "retrieval": "semantic",
        "top_k": 6,
        "requires_rag": False,
    },

    "clarification": {
        "retrieval": "keyword",
        "top_k": 3,
        "requires_rag": False,
    },

    "multi_part": {
        "retrieval": "hybrid",
        "top_k": 10,
        "requires_rag": True,
    },

    "yes_no": {
        "retrieval": "keyword",
        "top_k": 3,
        "requires_rag": False,
    },

    "statement": {
        "retrieval": "keyword",
        "top_k": 2,
        "requires_rag": False,
    },
ths map....to either push back on where the files pinned or refrenced in the query are not in the file registry or dont meet the requirements of the question type

is_followup :: true if the user has asked a followup question
needs_clarification :: if this is true then you have processed this query and skip to using the followup block to properly prepare the response
followup :: list of followup blocks creeated by you if you need to ask followup questions based on the users query or retrival flow....if tthis is none then you can skip to the response


"""


