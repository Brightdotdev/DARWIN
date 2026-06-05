"""
src/core/Storage/StorePaths.py
------------------------------
Content-addressed storage paths. Same PDF → same index folder, always.
Nothing here touches file content — it only knows where things live.
Directory creation is opt-in via ensure_* methods.
"""

from __future__ import annotations

import hashlib
from functools import cached_property
from pathlib import Path

from src.core.Storage.StoreUtils import ARTIFACT_MANIFEST

# ── Free functions ────────────────────────────────────────────────────────

def hash_file(file_path: str | Path) -> str:
    """SHA-256 of file contents. Same file → same hash, always."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def is_file_hash(value: str) -> bool:
    """True if value looks like a SHA-256 hex string (64 lowercase hex chars)."""
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())




# ── Class ─────────────────────────────────────────────────────────────────

class StorePaths:

    def __init__(self, darwin_dir: str | Path):
        if darwin_dir is None:
            raise ValueError("darwin_dir cannot be None")
        if isinstance(darwin_dir, str) and darwin_dir.strip() == "":
            raise ValueError("darwin_dir cannot be an empty string")

        darwin_dir = Path(darwin_dir)

        if darwin_dir.exists() and not darwin_dir.is_dir():
            raise NotADirectoryError(f"{darwin_dir} exists and is not a directory")

        darwin_dir.mkdir(parents=True, exist_ok=True)
        self.root = darwin_dir.resolve()

    # ── Top-level paths ───────────────────────────────────────────────────

    @property
    def file_registry_path(self) -> Path:
        """Global tracker of ALL indexed files."""
        return self.root / "file_registry.json"

    @cached_property
    def indexes_dir(self) -> Path:
        """Directory containing all hash-indexed subdirectories."""
        return self.root / "indexes"




    @cached_property
    def checkpoints_dir(self) -> Path:
        """Directory for pipeline checkpoint files."""
        return self.root / "checkpoints"

    @cached_property
    def conversations_dir(self) -> Path:
        """Directory for conversation history storage."""
        return self.root / "conversations"

    # ── Path existence helper (convenience) ───────────────────────────────

    def exists(self, file_hash: str) -> bool:
        """Check if an index directory exists for a given file hash."""
        return self.dir_for_hash(file_hash).exists()
    # ── Core resolution (pure) ────────────────────────────────────────────

    def dir_for_hash(self, file_hash: str) -> Path:
        """
        Pure path resolution for a known file hash.
        Does NOT touch the filesystem.
        """
        if not is_file_hash(file_hash):
            raise ValueError(f"Invalid file hash: {file_hash}")
        return self.indexes_dir / file_hash

    def file_hash(self, file_path: str | Path) -> str:
        """SHA-256 of the file. Deterministic, content-based."""
        return hash_file(file_path)

    def file_dir(self, file_path: str | Path) -> Path:
        """Return the index directory path for a file, hashing its contents."""
        fh = self.file_hash(file_path)
        return self.dir_for_hash(fh)

    def file_dir_by_hash(self, file_hash: str) -> Path:
        """Alias for dir_for_hash — kept for back-compat."""
        return self.dir_for_hash(file_hash)

    # ── Core resolution (side effects) ────────────────────────────────────

    def ensure_dir_for_hash(self, file_hash: str) -> Path:
        """Create and return the index directory for a known hash."""
        path = self.dir_for_hash(file_hash)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_file_dir(self, file_hash: str) -> Path:
        """Create and return index directory for a known hash."""
        return self.ensure_dir_for_hash(file_hash)

    def ensure_image_dir(self, file_hash: str) -> Path:
        """Create and return the image subdirectory for a known hash."""
        path = self.ensure_dir_for_hash(file_hash) / ARTIFACT_MANIFEST["images"]["dir"]
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── Per-file artifact paths (pure, hash-based) ────────────────────────

    def meta_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["meta"]["file"]

    def markdown_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["markdown"]["source"]

    def display_markdown_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["markdown"]["display"]

    def chunks_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["chunks"]["file"]

    def embedding_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["embeddings"]["file"]

    def embedding_index_map_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["embeddings"]["index_map"]

    def vector_index_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["vector"]["index"]

    def vector_json_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["vector"]["meta"]

    def bm25_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["bm25"]["index"]

    def bm25_meta_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["bm25"]["meta"]

    def bm25_tokens_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["bm25"]["tokens"]

    def stopwords_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["stopwords"]["file"]

    def graph_nodes_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["graph"]["nodes"]

    def graph_edges_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["graph"]["edges"]

    def graph_html_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["graph_vis"]["file"]

    def wiki_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / ARTIFACT_MANIFEST["wiki"]["file"]