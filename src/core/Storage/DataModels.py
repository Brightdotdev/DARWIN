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
    stages: set[str] = field(default_factory=set)
    embedding_model: Optional[str] = None
    chunk_count: Optional[int] = None
    page_count: Optional[int] = None
    created_at: str = ""
    updated_at: str = ""

    # Backward-compat properties (read-only) – derived from stages
    @property
    def has_markdown(self) -> bool:
        return "markdown" in self.stages

    @property
    def has_chunks(self) -> bool:
        return "chunks" in self.stages

    @property
    def has_embeddings(self) -> bool:
        return "embeddings" in self.stages

    @property
    def has_vector(self) -> bool:
        return "vector" in self.stages

    @property
    def has_bm25(self) -> bool:
        return "bm25" in self.stages

    @property
    def has_stopwords(self) -> bool:
        return "stopwords" in self.stages

    @property
    def has_graph(self) -> bool:
        return "graph" in self.stages

    @property
    def has_graph_vis(self) -> bool:
        return "graph_vis" in self.stages

    @property
    def has_wiki(self) -> bool:
        return "wiki" in self.stages
