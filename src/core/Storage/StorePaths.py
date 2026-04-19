"""
src/core/Storage/StorePaths.py
------------------------------
All path computation and directory management for the .darwin workspace.

Nothing here touches file content — it only knows where things live.
Import this wherever you need a path without pulling in all of DarwinStore.
"""

from pathlib import Path

from src.core.Storage.StoreUtils import sanitize, DARWIN_DIR


class StorePaths:
    """
    Computes and creates all paths inside a .darwin workspace.

    All methods are pure (just return Paths) unless they carry an
    `ensure=True` style — those also create the directory.
    The `ensure_*` variants return the path after mkdir.
    """

    def __init__(self, project_dir: Path, darwin_dir: Path):
        self.project_dir = project_dir
        self.darwin_dir  = darwin_dir

    # ── Top-level ──────────────────────────────────────────────────────────────

    @property
    def config_path(self) -> Path:
        return self.darwin_dir / "config.json"

    # ── Per-file directories ───────────────────────────────────────────────────

    def file_dir(self, filename: str) -> Path:
        """Sanitized per-file directory: .darwin/<sanitized_name>/"""
        return self.darwin_dir / sanitize(filename)

    def ensure_file_dir(self, filename: str) -> Path:
        d = self.file_dir(filename)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Artifact paths ─────────────────────────────────────────────────────────

    def meta_path(self, filename: str) -> Path:
        return self.file_dir(filename) / "meta.json"

    def markdown_path(self, filename: str) -> Path:
        return self.file_dir(filename) / f"{sanitize(filename)}.md"

    def chunks_path(self, filename: str) -> Path:
        return self.file_dir(filename) / "chunks.json"

    def links_path(self, filename: str) -> Path:
        return self.file_dir(filename) / "links.json"

    def bm25_path(self, filename: str) -> Path:
        return self.file_dir(filename) / "bm25.pkl"

    def stopwords_path(self, filename: str) -> Path:
        return self.file_dir(filename) / "stopwords.json"

    def npz_path(self, filename: str, model_name: str) -> Path:
        """
        Embedding bundle path — one per (file, model) pair.
        e.g. .darwin/biology_pdf/e5_small_biology_pdf.npz
        """
        file_clean  = sanitize(filename)
        model_clean = sanitize(model_name)
        return self.file_dir(filename) / f"{model_clean}_{file_clean}.npz"

    # ── Image directories ──────────────────────────────────────────────────────

    def image_dir(self, filename: str) -> Path:
        return self.file_dir(filename) / "extracted_images"

    def ensure_image_dir(self, filename: str) -> Path:
        d = self.image_dir(filename)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ensure_image_dir_checked(self, filename: str) -> tuple[Path, bool]:
        """Returns (image_dir, has_existing_images)."""
        d          = self.ensure_image_dir(filename)
        has_images = any(d.glob("*.png"))
        return d, has_images

    # ── Graph paths ────────────────────────────────────────────────────────────

    def graph_dir(self, filename: str) -> Path:
        return self.file_dir(filename) / "graph"

    def ensure_graph_dir(self, filename: str) -> Path:
        d = self.graph_dir(filename)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def graph_json_path(self, filename: str) -> Path:
        return self.graph_dir(filename) / f"{sanitize(filename)}_graph.json"

    def graph_html_path(self, filename: str) -> Path:
        return self.graph_dir(filename) / f"{sanitize(filename)}_graph.html"

    # ── Wiki paths ─────────────────────────────────────────────────────────────

    def wiki_dir(self, filename: str) -> Path:
        return self.file_dir(filename) / "wiki"

    def ensure_wiki_dir(self, filename: str) -> Path:
        d = self.wiki_dir(filename)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def wiki_json_path(self, filename: str) -> Path:
        return self.wiki_dir(filename) / f"{sanitize(filename)}_wiki.json"