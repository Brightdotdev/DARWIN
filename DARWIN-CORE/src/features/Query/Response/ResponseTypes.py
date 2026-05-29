"""
src/core/Response/ResponseTypes.py
-----------------------------------
Canonical dataclasses for the Response layer.
Unifies the fragmented ResolvedQuery / SearchContext definitions from
upstream modules so ResponseRouter can import one truth.
"""

from typing import List, Dict, Optional, Any, Literal
from dataclasses import dataclass, field
import json


# ═══════════════════════════════════════════════════════════════════════════
# Upstream-compatible types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    text: str
    type: str
    start: int = 0
    end: int = 0
    label: str = ""


@dataclass
class WikiConcept:
    raw_concept: str = ""
    concept: str = ""
    score: float = 0.0
    source_file: str = ""
    keywords: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    source_count: int = 1

    def get_page_range(self) -> str:
        return ""


@dataclass
class ResolvedPassage:
    passage_text: str = ""
    source: str = ""
    page_start: int = 0
    page_end: int = 0
    avg_score: float = 0.0
    primary_concept: Optional[str] = None
    related_concepts: List[str] = field(default_factory=list)
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    chunk_indices: List[int] = field(default_factory=list)
    original_results: List[Any] = field(default_factory=list)


@dataclass
class SearchContext:
    query: str = ""
    intent: str = ""
    files: List[str] = field(default_factory=list)
    concepts: List[WikiConcept] = field(default_factory=list)
    passages: List[ResolvedPassage] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    merged_prompt: str = ""


@dataclass
class ResolvedQuery:
    intent: Literal["broad", "dense", "both"] = "broad"
    original_query: str = ""
    expanded_terms: List[str] = field(default_factory=list)
    query_embedding: List[float] = field(default_factory=list)

    # File routing (from QueryExpansionLayer)
    pinned_files: List[str] = field(default_factory=list)
    inferred_files: List[str] = field(default_factory=list)
    excluded_files: List[str] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    requires_comparison: bool = False
    decomposed_queries: List['ResolvedQuery'] = field(default_factory=list)

    # Response routing
    explicit_response_type: Optional[str] = None

    @property
    def all_files(self) -> List[str]:
        combined = list(dict.fromkeys(self.pinned_files + self.inferred_files))
        return [f for f in combined if f not in self.excluded_files]


# ═══════════════════════════════════════════════════════════════════════════
# Response types (new Artifacts architecture)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ResponseBlock:
    """
    A single presentation block.
    Visual blocks carry markdown_breakdown + markdown_fallback.
    Markdown blocks carry only data.content.
    """
    block_type: str
    title: str
    data: Dict[str, Any] = field(default_factory=dict)
    schema: Dict[str, Any] = field(default_factory=dict)
    markdown_breakdown: Optional[str] = None
    markdown_fallback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponsePlan:
    """
    Container for the full response.
    Simple queries: 1 block. Complex queries: N blocks.
    """
    script_key: Optional[str] = None
    blocks: List[ResponseBlock] = field(default_factory=list)
    is_complex: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "script_key": self.script_key,
            "is_complex": self.is_complex,
            "blocks": [
                {
                    "type": b.block_type,
                    "title": b.title,
                    "data": b.data,
                    "schema": b.schema,
                    "markdown_breakdown": b.markdown_breakdown,
                    "markdown_fallback": b.markdown_fallback,
                    "metadata": b.metadata,
                }
                for b in self.blocks
            ],
            "metadata": self.metadata,
        }, indent=2, default=str)

    def to_markdown(self) -> str:
        lines = []
        for block in self.blocks:
            if block.block_type == "markdown":
                content = block.data.get("content", "")
                if content:
                    lines.append(content)
            else:
                if block.title:
                    lines.append(f"## {block.title}")
                if block.markdown_breakdown:
                    lines.append(block.markdown_breakdown)
                if block.markdown_fallback:
                    lines.append("\n**Plain Text Version:**")
                    lines.append(block.markdown_fallback)
            lines.append("\n---\n")
        return "\n".join(lines)