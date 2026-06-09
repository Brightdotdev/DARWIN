from __future__ import annotations

from pathlib import Path

from src.core.Storage.Artifacts import (
    StoreCore,
    StoreSources,
    StoreStatus,
    StoreHealth,
    StoreDashboard,
    ArtifactMarkdown,
    ArtifactChunks,
    ArtifactEmbeddings,
    ArtifactBM25,
    ArtifactAuxMixin,
    StoreSearch,
)


class DarwinStore(
    StoreCore,
    StoreSources,
    StoreStatus,
    StoreHealth,
    StoreDashboard,
    ArtifactMarkdown,
    ArtifactChunks,
    ArtifactEmbeddings,
    ArtifactBM25,
    ArtifactAuxMixin,
    StoreSearch,
):
    """
    Thin facade over composed artifact mixins.
    Preserves the public API surface of the original DarwinStore.
    """

    def __init__(self, darwin_dir: Path | None = None):
        super().__init__(darwin_dir)