"""
src/core/Storage/DarwinStore.py
--------------------------------
Global persistence layer for Darwin. Content-addressed, model-aware.

Usage:
    store = DarwinStore()

    # Always register the file first — this is what lets the store
    # resolve bare filenames to content hashes everywhere else.
    record = store.registry.register(Path("myfile.pdf"))

    # Then save artifacts using just the filename
    store.save_markdown("myfile.pdf", doc)
    store.save_chunks("myfile.pdf", chunks)
"""

import shutil
from pathlib import Path
from typing import Optional

from src.core.Storage.StorePaths import StorePaths, default_darwin_dir
from src.core.Storage.FileRegistry import FileRegistry, FileRecord
from src.core.Storage.StoreArtifacts import (
    MetaMixin,
    ConfigMixin,
    MarkdownMixin,
    VectorMixin,
    ChunksMixin,
    EmbeddingsMixin,
    BM25Mixin,
    StopwordsMixin,
    GraphMixin,
    WikiMixin,
    CheckpointMixin,
)
from src.core.Storage.StoreUtils import DEFAULT_CONFIG, now, write_json, read_json
from src.lib.Terminal import TerminalUI


class DarwinStore(
    MetaMixin,
    ConfigMixin,
    MarkdownMixin,
    VectorMixin,
    ChunksMixin,
    EmbeddingsMixin,
    BM25Mixin,
    StopwordsMixin,
    GraphMixin,
    WikiMixin,
    CheckpointMixin,
):
    """
    Global content-addressed store. One instance per app session.

    No per-project directories. Everything lives in ~/.darwin/
    keyed by content hash so files can move freely.

    Calling convention
    ------------------
    Before saving any artifact for a file, call:

        store.registry.register(full_path)   # or get_or_register

    After that, all artifact methods accept just the bare filename:

        store.save_markdown("myfile.pdf", doc)
        store.load_chunks("myfile.pdf")

    The registry is the bridge — it maps filename → file hash → index dir.
    """

    def __init__(self, darwin_dir: Path = None):
        self._darwin_dir = darwin_dir or default_darwin_dir()
        self.paths       = StorePaths(self._darwin_dir)
        self.registry    = FileRegistry(self.paths)
        self._init_config()

    def _init_config(self) -> None:
        """Ensure config.json exists with defaults."""
        if not self.paths.config_path.exists():
            config = {**DEFAULT_CONFIG, "created_at": now()}
            write_json(self.paths.config_path, config)

    # ── Registration helpers ──────────────────────────────────────────────
    # Convenience wrappers so callers don't need to import FileRegistry.

    def register(
        self,
        file_path: str | Path,
        page_count: int = 0,
        file_size: int = 0,
    ) -> FileRecord:
        """
        Register a file (or refresh its last-seen path).
        Must be called before any artifact save for this file.
        Returns the FileRecord.
        """
        return self.registry.register(file_path, page_count=page_count, file_size=file_size)

    def get_or_register(self, file_path: str | Path) -> FileRecord:
        """Get existing record or create a new one."""
        return self.registry.get_or_register(file_path)

    def update_pipeline(self, file_path: str | Path, stage: str, **extras) -> FileRecord:
        """Mark a pipeline stage as completed for a file."""
        return self.registry.update_pipeline(file_path, stage, **extras)

    # ── File removal ──────────────────────────────────────────────────────

    def remove_file(self, file_path: str | Path) -> bool:
        """Delete all indexes for a file. Returns True if anything removed."""
        file_path = Path(file_path).resolve()
        removed   = self.registry.remove(file_path)

        if file_path.exists():
            fh        = self.paths.file_hash(file_path)
            index_dir = self.paths.dir_for_hash(fh)
            if index_dir.exists():
                shutil.rmtree(index_dir)
                return True

        return removed

    # ── Image directories ─────────────────────────────────────────────────

    def image_dir(self, filename: str) -> Path:
        """Directory for extracted images, keyed by file hash."""
        fh = self._hash_for(filename)
        return self.paths.dir_for_hash(fh) / "images"

    def ensure_image_dir(self, filename: str) -> Path:
        """Create and return the image directory for a file."""
        d = self.image_dir(filename)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def image_dir_for_path(self, file_path: str | Path) -> Path:
        """
        Get image dir when you have the full file path rather than
        a registered filename. Useful during initial parsing before
        the filename is fully registered.
        """
        fh = self.paths.file_hash(file_path)
        d  = self.paths.dir_for_hash(fh) / "images"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Status ────────────────────────────────────────────────────────────

    def file_status(self, file_path: str | Path) -> dict:
        """Full status for one file, suitable for the frontend."""
        file_path = Path(file_path).resolve()
        record    = self.registry.get_or_register(file_path)

        if file_path.exists():
            record = self.registry.register(file_path)

        return {
            "file_hash":       record.file_hash,
            "original_name":   record.original_name,
            "last_seen_path":  record.last_seen_path,
            "file_size_mb":    round(record.file_size_mb, 1),
            "page_count":      record.page_count,
            "is_parsed":       record.is_parsed,
            "is_chunked":      record.is_chunked,
            "is_embedded":     record.is_embedded,
            "is_searchable":   record.is_searchable,
            "progress_percent": record.progress_percent(),
            "status_summary":  record.status_summary(),
            "missing_stages":  record.missing_stages(),
            "embedding_model": record.embedding_model,
            "chunk_count":     record.chunk_count,
            "created_at":      record.created_at,
            "updated_at":      record.updated_at,
        }

    def status(self) -> dict:
        """Full status for all registered files."""
        stats = self.registry.stats()
        files = {}
        for record in self.registry.list_all():
            ref = Path(record.last_seen_path) if record.last_seen_path else record.original_name
            files[record.original_name] = self.file_status(ref)
        return {
            "darwin_dir": str(self._darwin_dir),
            "stats":      stats,
            "files":      files,
        }

    def dashboard(self) -> str:
        """Pretty dashboard for the terminal."""
        stats    = self.registry.stats()
        needing  = self.registry.list_needing_work()

        lines = [
            "╔══════════════════════════════════════════╗",
            "║           📚 DARWIN DASHBOARD             ║",
            "╠══════════════════════════════════════════╣",
            f"║  Files indexed:    {stats['total_files']:<3}                       ║",
            f"║  Searchable:       {stats['searchable_files']:<3}                       ║",
            f"║  Needs work:       {stats['files_needing_work']:<3}                       ║",
            f"║  Total chunks:     {stats['total_chunks']:<5}                    ║",
            f"║  Total pages:      {stats['total_pages']:<5}                    ║",
            "╚══════════════════════════════════════════╝",
        ]

        if needing:
            lines.append("\n📋 Files needing attention:")
            for r in needing[:5]:
                lines.append(f"   • {r.original_name} — {r.status_summary()}")
            if len(needing) > 5:
                lines.append(f"   ... and {len(needing) - 5} more")

        return "\n".join(lines)