"""
src/core/Knowledge/Wiki/WikiUtils.py
-------------------------------------
Data models and CLI args for wiki generation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict

from src.features.Retrival.lib import SearchAudit


DEFAULT_CHUNKS_PER_CONCEPT = 8


@dataclass
class WikiPage:
    concept:       str
    chunk_indices: List[int]
    related:       List[str]
    centroid:      List[float]
    frequency:     int
    keywords:      List[str]           = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "concept":       self.concept,
            "chunk_indices": self.chunk_indices,
            "related":       self.related,
            "centroid":      self.centroid,
            "frequency":     self.frequency,
            "keywords":      self.keywords,
        }

    @staticmethod
    def from_dict(d: Dict) -> "WikiPage":
        return WikiPage(
            concept       = d.get("concept", ""),
            chunk_indices = d.get("chunk_indices", []),
            related       = d.get("related", []),
            centroid      = d.get("centroid", []),
            frequency     = d.get("frequency", 0),
            keywords      = d.get("keywords", []),
        )


@dataclass
class WikiSearchResult:
    """
    A scored search result with full component breakdown.
    Passed to the agent so it can reason about *why* a page ranked.
    """
    page:                  WikiPage
    score:                 float
    vector_score:          float
    keyword_score:         float
    concept_score:         float
    content_quality_score: float
    structural_penalty:    float
    is_expanded:           bool = False
    audit: Optional[SearchAudit] = None


@dataclass
class DocumentWiki:
    filename: str
    pages:    List[WikiPage]           = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "filename": self.filename,
            "pages":    [p.to_dict() for p in self.pages],
        }

    @staticmethod
    def from_dict(d: Dict) -> "DocumentWiki":
        return DocumentWiki(
            filename = d.get("filename", ""),
            pages    = [WikiPage.from_dict(p) for p in d.get("pages", [])],
        )


@dataclass
class WikiGenerationArgs:
    files: list[str] = field(default_factory=list)
    inspect:            bool           = False
    rerun:              bool           = False





