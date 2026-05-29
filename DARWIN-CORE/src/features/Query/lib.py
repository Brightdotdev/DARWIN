from dataclasses import dataclass, field
from typing import Dict, List,  Any, Optional
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------

class PresentationType(str):
    """Enumeration of supported presentation types."""
    MARKDOWN = "markdown"
    SUMMARY_CARD = "summary_card"
    PROFILE_CARD = "profile_card"
    PROS_CONS = "pros_cons"
    SWOT = "swot"
    COMPARISON_TABLE = "comparison_table"
    TIMELINE = "timeline"
    WALKTHROUGH = "walkthrough"
    FLOWCHART = "flowchart"
    GRAPH = "graph"
    VISUALIZATION = "visualization"
    FORMULA = "formula"
    QA = "qa"
    CLARIFICATION = "clarification"
    HOW_TO = "how_to"
    NARRATIVE = "narrative"
    EVIDENCE_SYNTHESIS = "evidence_synthesis"
    FLASHCARD = "flashcard"
    MINI_TEST = "mini_test"
    PARAMETER_SANDBOX = "parameter_sandbox"
    DIAGNOSTIC_TREE = "diagnostic_tree"
    GLOSSARY = "glossary"





stopwords =  {"the","is","a","an","in","on","at","to","for","of","and","or","but","with"}



@dataclass
class ResolvedPlaceholder:
    placeholder: str
    original: str                      # Original URL or blob content
    full_content: str                  # Fetched full content
    summary: Optional[str]             # Summary if full was too long
    was_truncated: bool
    token_count: int

@dataclass
class ReconstructedContext:
    final_text: str                    # The fully resolved context for the LLM
    total_tokens: int
    was_truncated: bool
    truncation_log: List[str]          # What was cut and why
    resolved_placeholders: Dict[str, ResolvedPlaceholder]




@dataclass
class ExtractedLink:
    raw_url: str
    domain: str
    placeholder: str

@dataclass
class ExtractedBlob:
    blob_id: str
    content_preview: str
    placeholder: str
    token_count: int
    is_truncated: bool

@dataclass
class PresentationTypeWithScore:
    type: str
    role: str  # starter, elaborator, finisher
    confidence: float



@dataclass
class RawQuery:
    text: str
    blobs : List[ExtractedBlob] = field(default_factory=list)
    pinned_files: List[str] = field(default_factory=list)
    extracted_urls: List[ExtractedLink] = field(default_factory=list)


@dataclass
class PreprocessedQuery:
    original_text: str
    cleaned_text: str
    rag_query: str
    presentation_types: List[PresentationTypeWithScore] = field(default_factory=list)
    requires_llm_classification: bool = False
    desperation_score: float = 0.0
    extracted_links: List[ExtractedLink] = field(default_factory=list)
    extracted_blobs: List[ExtractedBlob] = field(default_factory=list)
    placeholders: Dict[str, str] = field(default_factory=dict)
    sub_questions: List[str] = field(default_factory=list)
    named_entities: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 1.0
    quality_flags: List[str] = field(default_factory=list)
    tier: str = "complex"  # "simple", "moderate", "complex"
    is_multi_intent: bool = False
