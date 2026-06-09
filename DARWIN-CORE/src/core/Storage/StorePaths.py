from __future__ import annotations

import hashlib
from functools import cached_property
from pathlib import Path

from src.core.Storage.Constants import ARTIFACT_MANIFEST


def hash_file(file_path: str | Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def is_file_hash(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


class StorePaths:
    def __init__(self, darwin_dir: str | Path):
        if darwin_dir is None:
            raise ValueError("darwin_dir cannot be None")
        darwin_dir = Path(darwin_dir)
        if darwin_dir.exists() and not darwin_dir.is_dir():
            raise NotADirectoryError(f"{darwin_dir} exists and is not a directory")
        darwin_dir.mkdir(parents=True, exist_ok=True)
        self.root = darwin_dir.resolve()

    @property
    def file_registry_path(self) -> Path:
        return self.root / "file_registry.json"

    @property
    def registry_db_path(self) -> Path:
        return self.root / "registry.db"

    @cached_property
    def indexes_dir(self) -> Path:
        return self.root / "indexes"

    @cached_property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @cached_property
    def conversations_dir(self) -> Path:
        return self.root / "conversations"

    def dir_for_hash(self, file_hash: str) -> Path:
        if not is_file_hash(file_hash):
            raise ValueError(f"Invalid file hash: {file_hash}")
        return self.indexes_dir / file_hash

    def ensure_dir_for_hash(self, file_hash: str) -> Path:
        path = self.dir_for_hash(file_hash)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_image_dir(self, file_hash: str) -> Path:
        path = self.ensure_dir_for_hash(file_hash) / ARTIFACT_MANIFEST["images"]["dir"]
        path.mkdir(parents=True, exist_ok=True)
        return path

    # --- Per-file artifact paths (pure, hash-based) ---
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