"""
src/core/Storage/DarwinStore.py
--------------------------------
Persistence layer for Darwin.

DarwinStore is the single public interface for all on-disk state inside a
project's `.darwin/` directory.  It is composed from focused mixin classes
(each in StoreArtifacts.py) so every artifact family is easy to find,
read, and test in isolation.

Directory layout
----------------
    your-notes/
    ├── biology.pdf
    └── .darwin/
        ├── config.json                      ← project-level settings
        └── biology_pdf/                     ← one dir per source file
            ├── meta.json                    ← audit trail + freshness hash
            ├── biology_pdf.md               ← embedding_text from parser
            ├── links.json                   ← hyperlinks extracted by parser
            ├── chunks.json                  ← chunker output
            ├── e5_small_biology_pdf.npz     ← vectors + chunk metadata
            ├── bm25.pkl                     ← BM25 search index
            ├── stopwords.json
            ├── graph/
            │   ├── biology_pdf_graph.json
            │   └── biology_pdf_graph.html
            └── wiki/
                └── biology_pdf_wiki.json

Embedding storage (.npz)
------------------------
Vectors and chunk metadata live in one compressed numpy archive per
(file, model) pair.  They are always in sync — the file either contains
both or it doesn't exist.

    data     = np.load("e5_small_biology_pdf.npz", allow_pickle=True)
    vectors  = data["vectors"]    # (n_chunks, dim)  float32
    text     = data["text"]       # original text per chunk
    clean    = data["clean"]      # stripped text per chunk
    headings = data["headings"]   # nearest heading per chunk
    pages    = data["pages"]      # int page numbers
    source   = data["source"]     # filename repeated n times
"""

import shutil
from pathlib import Path

from src.core.Storage.StorePaths import StorePaths
from src.core.Storage.StoreArtifacts import (
    MetaMixin,
    ConfigMixin,
    MarkdownMixin,
    LinksMixin,
    ChunksMixin,
    EmbeddingsMixin,
    BM25Mixin,
    StopwordsMixin,
    GraphMixin,
    WikiMixin,
)
from src.core.Storage.StoreUtils import (
    DEFAULT_CONFIG, DARWIN_DIR, now, write_json,
)
from src.lib.Terminal import TerminalUI


class DarwinStore(
    MetaMixin,
    ConfigMixin,
    MarkdownMixin,
    LinksMixin,
    ChunksMixin,
    EmbeddingsMixin,
    BM25Mixin,
    StopwordsMixin,
    GraphMixin,
    WikiMixin,
):
    """
    One instance per project directory.

    All filesystem access for Darwin goes through here — nothing else
    touches the .darwin/ folder directly.

    Args:
        project_dir: folder containing the user's source documents,
                     e.g. "/home/user/biology-notes"
    """

    def __init__(self, project_dir: str | Path):
        self.project_dir = Path(project_dir).resolve()
        self.darwin_dir  = self.project_dir / DARWIN_DIR
        self.paths       = StorePaths(self.project_dir, self.darwin_dir)

    # ── Initialisation ─────────────────────────────────────────────────────────

    def init(self) -> bool:
        """
        Create the .darwin folder and write default config.json.
        Safe to call repeatedly — does nothing if already initialised.

        Returns True on first initialisation, False if already existed.
        """
        self.darwin_dir.mkdir(parents=True, exist_ok=True)

        if not self.paths.config_path.exists():
            config = {**DEFAULT_CONFIG, "created_at": now()}
            write_json(self.paths.config_path, config)
            return True

        return False

    def is_initialized(self) -> bool:
        return self.darwin_dir.exists()

    def _require_init(self) -> None:
        """Lazily initialise the store on first access."""
        if not self.darwin_dir.exists():
            self.init()
        elif not self.paths.config_path.exists():
            write_json(self.paths.config_path, {**DEFAULT_CONFIG, "created_at": now()})

    # ── Convenience path proxies (keep callers unchanged) ──────────────────────
    # These forward to StorePaths so existing call-sites don't break.

    def image_dir(self, filename: str) -> Path:
        return self.paths.image_dir(filename)

    def ensure_image_dir(self, filename: str) -> Path:
        return self.paths.ensure_image_dir(filename)

    def ensure_image_dir_and_check(self, filename: str) -> tuple[Path, bool]:
        return self.paths.ensure_image_dir_checked(filename)

    # ── File removal ───────────────────────────────────────────────────────────

    def remove_file(self, filename: str) -> bool:
        """Delete all .darwin artifacts for a source file. Returns True if anything was removed."""
        d = self.paths.file_dir(filename)
        if not d.exists():
            return False
        shutil.rmtree(d)
        return True

    # ── Pipeline status ────────────────────────────────────────────────────────

    def _component_flags(self, filename: str) -> dict[str, bool]:
        return {
            "markdown":   self.has_markdown(filename),
            "chunks":     self.has_chunks(filename),
            "embeddings": self.has_embeddings(filename),
            "bm25":       self.has_bm25(filename),
            "stopwords":  self.has_stopwords(filename),
            "graph":      self.has_graph(filename),
        }

    def _pipeline_flags(self, filename: str) -> dict:
        meta    = self.load_meta(filename)
        updates = meta.get("updates", {})
        return {
            "is_parsed":        "markdown"   in updates,
            "is_vectorized":    "embeddings" in updates,
            "has_search_index": "bm25"       in updates,
            "has_graph":        self.has_graph(filename),
            "last_activity":    meta.get("updated_at", "never"),
            "has_markdown":     self.has_markdown(filename),
            "has_chunks":       self.has_chunks(filename),
            "has_embeddings":   self.has_embeddings(filename),
        }

    def _meta_summary(self, filename: str) -> dict:
        meta = self.load_meta(filename)
        return {
            "chunk_count":      meta.get("chunk_count"),
            "page_count":       meta.get("page_count"),
            "parser":           meta.get("parser"),
            "chunk_max_tokens": meta.get("chunk_max_tokens"),
            "chunk_model":      meta.get("chunk_model"),
            "updated_at":       meta.get("updated_at"),
            "created_at":       meta.get("created_at"),
        }

    def file_status(self, filename: str) -> dict:
        """Full status report for a single source file."""
        self._require_init()
        components = self._component_flags(filename)
        return {
            "filename":         filename,
            "exists":           (self.project_dir / filename).exists(),
            "fresh":            self.is_fresh(filename),
            "components":       components,
            "pipeline":         self._pipeline_flags(filename),
            "meta":             self._meta_summary(filename),
            "ready_for_search": components["chunks"] and components["embeddings"],
            "fully_processed":  all(components.values()),
        }

    def status(self) -> dict:
        """Full status report for every source file in the project directory."""
        self._require_init()

        source_files = [
            f for f in self.project_dir.iterdir()
            if f.is_file() and f.name != DARWIN_DIR
        ]

        return {
            "project_dir": str(self.project_dir),
            "initialized": self.is_initialized(),
            "files": {
                f.name: self.file_status(f.name)
                for f in source_files
            },
        }