"""
src/core/Storage/StorePaths.py
------------------------------
Content-addressed storage paths. Same PDF → same index folder, always.
Nothing here touches file content — it only knows where things live.
"""

from pathlib import Path
import hashlib
import platform
import os


def default_darwin_dir() -> Path:
    """Platform-appropriate global Darwin directory for ALL data."""
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "Darwin"



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


class StorePaths:
    
    def __init__(self, darwin_dir: Path = None):
        self.root = darwin_dir or default_darwin_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Top-level paths ───────────────────────────────────────────────────

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def file_registry_path(self) -> Path:
        """Global tracker of ALL indexed files."""
        return self.root / "file_registry.json"

    @property
    def indexes_dir(self) -> Path:
        path = self.root / "indexes"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def checkpoints_dir(self) -> Path:
        path = self.root / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def conversations_dir(self) -> Path:
        """Directory for conversation storage."""
        path = self.root / "conversations"
        path.mkdir(parents=True, exist_ok=True)
        return path
    # ── Core resolution ───────────────────────────────────────────────────

    def file_hash(self, file_path: str | Path) -> str:
        """SHA-256 of the file. Deterministic, content-based."""
        return hash_file(file_path)

    def dir_for_hash(self, file_hash: str) -> Path:
        """
        Return (and create) the index directory for a known file hash.
        This is the single canonical way to get the index dir from a hash.
        """
        path = self.indexes_dir / file_hash
        path.mkdir(parents=True, exist_ok=True)
        return path

    def file_dir(self, file_path: str | Path) -> Path:
        """
        Return the index directory for a file, hashing its contents.
        Only use this when you have the actual file on disk.
        """
        fh = self.file_hash(file_path)
        return self.dir_for_hash(fh)

    def file_dir_by_hash(self, file_hash: str) -> Path:
        """Alias for dir_for_hash — kept for back-compat."""
        return self.dir_for_hash(file_hash)

    # ── ensure_* helpers (all hash-based) ────────────────────────────────

    def ensure_file_dir(self, file_hash: str) -> Path:
        """
        Create and return index directory for a known hash.
        Mixins call this — they always pass a hash, never a raw file path.
        """
        return self.dir_for_hash(file_hash)

    def ensure_vector_dir(self, file_hash: str) -> Path:
        """Vector index lives alongside other artifacts in the same dir."""
        return self.dir_for_hash(file_hash)

    def ensure_graph_dir(self, file_hash: str) -> Path:
        d = self.dir_for_hash(file_hash) / "graph"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ensure_wiki_dir(self, file_hash: str) -> Path:
        d = self.dir_for_hash(file_hash) / "wiki"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ensure_image_dir(self, file_hash: str) -> Path:
        d = self.dir_for_hash(file_hash) / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Per-file artifact paths (all hash-based) ──────────────────────────
    # These all take a file_hash string, not a file path.

    def meta_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / "meta.json"

    def markdown_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / "markdown.md"

    def chunks_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / "chunks.json"

    def npz_path(self, file_hash: str, model_name: str) -> Path:
        model_slug = model_name.split("/")[-1].replace("-", "_")
        return self.dir_for_hash(file_hash) / f"embeddings_{model_slug}.npz"

    def vector_index_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / "vector.usearch"

    def vector_json_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / "vector.json"

    def bm25_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / "bm25.npz"

    def stopwords_path(self, file_hash: str) -> Path:
        return self.dir_for_hash(file_hash) / "stopwords.json"

    def graph_dir(self, file_hash: str) -> Path:
        d = self.dir_for_hash(file_hash) / "graph"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def graph_json_path(self, file_hash: str) -> Path:
        return self.graph_dir(file_hash) / "graph.json"

    def graph_html_path(self, file_hash: str) -> Path:
        return self.graph_dir(file_hash) / "graph.html"

    def wiki_dir(self, file_hash: str) -> Path:
        d = self.dir_for_hash(file_hash) / "wiki"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def wiki_json_path(self, file_hash: str) -> Path:
        return self.wiki_dir(file_hash) / "wiki.json"

    def image_dir(self, file_hash: str) -> Path:
        d = self.dir_for_hash(file_hash) / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Checkpoint paths (also hash-based) ───────────────────────────────

    def parsing_checkpoint_path(self, file_hash: str) -> Path:
        return self.checkpoints_dir / f"{file_hash}.parse_checkpoint.json"

    def embed_checkpoint_path(self, file_hash: str) -> Path:
        return self.checkpoints_dir / f"{file_hash}.embed_checkpoint.npz"

    def raw_extraction_path(self, file_hash: str) -> Path:
        return self.checkpoints_dir / f"{file_hash}.raw_extraction.json"

    def raw_parsing_extraction(self, file_hash: str) -> Path:
        """Alias for raw_extraction_path — used by CheckpointMixin."""
        return self.raw_extraction_path(file_hash)

    # ── Back-compat: old *_by_hash methods ───────────────────────────────
    # Kept so any existing callers don't break.

    def meta_path_by_hash(self, file_hash: str) -> Path:
        return self.meta_path(file_hash)

    def markdown_path_by_hash(self, file_hash: str) -> Path:
        return self.markdown_path(file_hash)

    def chunks_path_by_hash(self, file_hash: str) -> Path:
        return self.chunks_path(file_hash)

    def npz_path_by_hash(self, file_hash: str, model_name: str) -> Path:
        return self.npz_path(file_hash, model_name)

    def vector_index_path_by_hash(self, file_hash: str) -> Path:
        return self.vector_index_path(file_hash)

    def vector_json_path_by_hash(self, file_hash: str) -> Path:
        return self.vector_json_path(file_hash)

    def bm25_path_by_hash(self, file_hash: str) -> Path:
        return self.bm25_path(file_hash)

    def stopwords_path_by_hash(self, file_hash: str) -> Path:
        return self.stopwords_path(file_hash)