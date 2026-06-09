from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Source:
    """A user-facing file entry. Mutable name/path, immutable UUID."""
    uuid: str
    display_name: str
    current_path: Optional[str]
    content_hash: str
    file_size: int
    mtime: float
    created_at: str
    updated_at: str


@dataclass
class Content:
    """Immutable content blob. Pipeline state lives here."""
    hash: str
    has_markdown: bool = False
    has_chunks: bool = False
    has_embeddings: bool = False
    has_vector: bool = False
    has_bm25: bool = False
    has_stopwords: bool = False
    has_graph: bool = False
    has_graph_vis: bool = False
    has_wiki: bool = False
    embedding_model: Optional[str] = None
    chunk_count: int = 0
    page_count: int = 0
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ContentStatus:
    """Runtime computed view of a Content row + artifacts."""
    hash: str
    stages: dict[str, bool] = field(default_factory=dict)
    progress_percent: int = 0
    is_core_complete: bool = False
    missing_stages: list[str] = field(default_factory=list)
    embedding_model: Optional[str] = None
    chunk_count: int = 0
    page_count: int = 0

    @property
    def status_summary(self) -> str:
        if self.is_core_complete:
            return f"complete — {self.chunk_count} chunks"
        if self.stages.get("markdown"):
            return "parsed"
        return "registered"

    @property
    def is_kw_searchable(self) -> bool:
        return self.stages.get("bm25", False) and self.stages.get("stopwords", False)

    @property
    def is_embedding_searchable(self) -> bool:
        return self.stages.get("embeddings", False) and self.stages.get("vector", False)

    @property
    def is_hybrid_searchable(self) -> bool:
        return self.is_kw_searchable and self.is_embedding_searchable