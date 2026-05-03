"""
src/core/Knowledge/Wiki/WikiUtils.py
-------------------------------------
Data models and CLI args for wiki generation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


DEFAULT_MAX_CONCEPTS = 20
DEFAULT_CHUNKS_PER_CONCEPT = 8


@dataclass
class WikiPage:
    concept:       str
    chunk_indices: List[int]
    related:       List[str]
    centroid:      List[float]
    frequency:     int
    summary:       Optional[str]       = None
    keywords:      List[str]           = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "concept":       self.concept,
            "chunk_indices": self.chunk_indices,
            "related":       self.related,
            "centroid":      self.centroid,
            "frequency":     self.frequency,
            "summary":       self.summary,
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
            summary       = d.get("summary"),
            keywords      = d.get("keywords", []),
        )


@dataclass
class DocumentWiki:
    filename: str
    pages:    List[WikiPage]           = field(default_factory=list)
    manifest: Optional[str]           = None

    def to_dict(self) -> Dict:
        return {
            "filename": self.filename,
            "manifest": self.manifest,
            "pages":    [p.to_dict() for p in self.pages],
        }

    @staticmethod
    def from_dict(d: Dict) -> "DocumentWiki":
        return DocumentWiki(
            filename = d.get("filename", ""),
            manifest = d.get("manifest"),
            pages    = [WikiPage.from_dict(p) for p in d.get("pages", [])],
        )


@dataclass
class WikiGenerationArgs:
    target:             str
    file:               Optional[str]  = None
    project:            Optional[str]  = None
    mode:               str            = "index"   # "index" | "summary"
    llm_model:          Optional[str]  = None
    max_concepts:       int            = DEFAULT_MAX_CONCEPTS
    chunks_per_concept: int            = DEFAULT_CHUNKS_PER_CONCEPT
    inspect:            bool           = False
    rerun:              bool           = False