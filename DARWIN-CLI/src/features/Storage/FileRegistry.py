"""
src/core/Storage/FileRegistry.py
--------------------------------
Global file tracker — one JSON file that knows every indexed document.

This is the single source of truth for "what files do we know about?"
It answers questions like:
  - Which files are fully searchable right now?
  - Which files need parsing/embedding?
  - What model was used to embed each file?
  - Where is the source file on disk?

All metadata lives here so the frontend can show beautiful status cards
without touching any per-file directories.
"""

from pathlib import Path
from src.features.Storage.StorePaths import StorePaths
from src.features.Storage.StoreUtils import read_json, write_json, now


class FileRecord:
    """Lightweight status object for one indexed file."""

    def __init__(self, data: dict):
        self._data = data

    # ── Identity ──────────────────────────────────────────────────────────
    @property
    def file_hash(self) -> str:
        return self._data.get("file_hash", "")

    @property
    def original_name(self) -> str:
        return self._data.get("original_name", "unknown")

    @property
    def last_seen_path(self) -> str | None:
        """Last known location on disk (may be stale if file moved)."""
        return self._data.get("last_seen_path")

    @property
    def file_size_mb(self) -> float:
        return self._data.get("file_size", 0) / (1024 * 1024)

    @property
    def page_count(self) -> int:
        return self._data.get("page_count", 0)

    # ── Pipeline status ───────────────────────────────────────────────────
    @property
    def is_parsed(self) -> bool:
        return "markdown" in self._data.get("updates", {})

    @property
    def is_chunked(self) -> bool:
        return "chunks" in self._data.get("updates", {})

    @property
    def is_embedded(self) -> bool:
        return "embeddings" in self._data.get("updates", {})

    @property
    def is_hybrid_searchable(self) -> bool:
        """Fully ready for search: parsed + chunked + embedded + BM25 + vector."""
        return self.is_kw_searchable and self.is_embedding_searchable

    @property
    def is_kw_searchable(self) -> bool:
        """Ready for keyword search: parsed + chunked + BM25."""
        return all(
            self._data.get("updates", {}).get(stage)
            for stage in ["markdown", "chunks", "bm25", "stopwords"]
        )

    @property
    def is_embedding_searchable(self) -> bool:
        """Ready for embedding search: parsed + chunked + embedded."""
        return all(
            self._data.get("updates", {}).get(stage)
            for stage in ["markdown", "chunks", "embeddings", "vector"]
        )

    @property
    def embedding_model(self) -> str:
        return self._data.get("embedding_model", "unknown")

    @property
    def embedding_dimensions(self) -> int:
        return self._data.get("embedding_dimensions", 0)

    @property
    def chunk_count(self) -> int:
        return self._data.get("chunk_count", 0)

    # ── Timestamps ────────────────────────────────────────────────────────
    @property
    def created_at(self) -> str:
        return self._data.get("created_at", "")

    @property
    def updated_at(self) -> str:
        return self._data.get("updated_at", "")

    @property
    def updates(self) -> dict:
        return self._data.get("updates", {})

    # ── Status helpers ────────────────────────────────────────────────────
    def stage_completed(self, stage: str) -> bool:
        return bool(self._data.get("updates", {}).get(stage))

    def missing_stages(self) -> list[str]:
        """Which pipeline stages are missing? Returns empty list if complete."""
        required = ["markdown", "chunks", "embeddings", "bm25", "vector"]
        return [s for s in required if not self.stage_completed(s)]

    def progress_percent(self) -> int:
        """0-100% pipeline completion."""
        stages = ["markdown", "chunks", "embeddings", "bm25", "vector"]
        done = sum(1 for s in stages if self.stage_completed(s))
        return int(done / len(stages) * 100)

    def status_summary(self) -> str:
        """Human-readable one-liner."""
        if self.is_hybrid_searchable:
            return f"✅ Ready — {self.chunk_count} chunks, {self.embedding_model}"
        missing = self.missing_stages()
        if not self.is_parsed:
            return f"📄 Needs parsing — {self.page_count} pages"
        if not self.is_embedded:
            return f"🔤 Needs embedding — {self.chunk_count} chunks waiting"
        return f"⏳ In progress — {len(missing)} stages remaining"

    def to_dict(self) -> dict:
        return dict(self._data)


class FileRegistry:
    """
    Global tracker stored at ~/.darwin/file_registry.json.

    Usage:
        registry = FileRegistry()
        registry.register("C:/docs/bio.pdf")
        record = registry.get("C:/docs/bio.pdf")
        print(record.status_summary())
    """

    def __init__(self, paths: "StorePaths"):
        self.paths = paths
        self._cache: dict[str, FileRecord] | None = None
        self._name_index: dict[str, FileRecord] | None = None
        self._hash_cache: dict[str, str] = {}  # file_path -> file_hash

    @property
    def path(self) -> Path:
        return self.paths.file_registry_path

    # ── Internal ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        """Load the raw registry JSON, using cache when valid."""
        if self._cache is not None and self._all_data is not None:
            return self._all_data
        self._all_data = read_json(self.path) or {"version": "1.0", "files": {}}
        self._cache = {
            fh: FileRecord(rec)
            for fh, rec in self._all_data.get("files", {}).items()
        }
        self._name_index = {}
        for fh, rec in self._all_data.get("files", {}).items():
            name = rec.get("original_name", "")
            if name and name not in self._name_index:
                self._name_index[name] = FileRecord(rec)
        return self._all_data

    def _save(self, data: dict) -> None:
        data["updated_at"] = now()
        write_json(self.path, data)
        self._all_data = data
        self._cache = {
            fh: FileRecord(rec)
            for fh, rec in data.get("files", {}).items()
        }
        self._name_index = {}
        for fh, rec in data.get("files", {}).items():
            name = rec.get("original_name", "")
            if name and name not in self._name_index:
                self._name_index[name] = FileRecord(rec)

    def _all_raw(self) -> dict:
        return self._load()

    # ── Public API ────────────────────────────────────────────────────────

    def _get_hash(self, file_path: str | Path) -> str:
        """Get file hash, using cache to avoid re-hashing large files."""
        file_path_str = str(Path(file_path).resolve()) if Path(file_path).exists() else str(file_path)
        if file_path_str in self._hash_cache:
            return self._hash_cache[file_path_str]
        if Path(file_path).exists():
            fh = self.paths.file_hash(file_path)
            self._hash_cache[file_path_str] = fh
            return fh
        # File doesn't exist, try to find by name
        record = self._find_by_name(Path(file_path).name)
        if record:
            return record.file_hash
        return ""

    def register(
        self,
        file_path: str | Path,
        page_count: int = 0,
        file_size: int = 0,
        file_hash: str | None = None,
    ) -> "FileRecord":
        """
        Register a file (or update its last-seen path).
        Does NOT overwrite pipeline progress — only updates metadata.
        Returns the FileRecord.
        """
        file_path = Path(file_path).resolve()
        fh = file_hash or self.paths.file_hash(file_path)
        self._hash_cache[str(file_path)] = fh
        data = self._load()

        if fh not in data["files"]:
            # New file
            data["files"][fh] = {
                "file_hash": fh,
                "original_name": file_path.name,
                "last_seen_path": str(file_path),
                "file_size": file_size or file_path.stat().st_size,
                "page_count": page_count,
                "created_at": now(),
                "updated_at": now(),
                "updates": {},
                "embedding_model": None,
                "embedding_dimensions": None,
                "chunk_count": 0,
            }
        else:
            # Update last-seen path (file may have moved)
            data["files"][fh]["last_seen_path"] = str(file_path)
            data["files"][fh]["original_name"] = file_path.name
            data["files"][fh]["file_size"] = file_size or file_path.stat().st_size
            if page_count:
                data["files"][fh]["page_count"] = page_count
            data["files"][fh]["updated_at"] = now()

        self._save(data)
        return FileRecord(data["files"][fh])

    def update_pipeline(
        self,
        file_path: str | Path,
        stage: str,
        file_hash: str | None = None,
        **extras,
    ) -> "FileRecord":
        """
        Mark a pipeline stage as completed for a file.
        e.g. registry.update_pipeline("bio.pdf", "embeddings", embedding_model="bge-m3")
        """
        data = self._load()
        
        if file_hash:
            fh = file_hash
        else:
            file_path = Path(file_path).resolve()
            fh = self._get_hash(file_path) if file_path.exists() else None
            if not fh:
                record = self._find_by_name(file_path.name)
                if record:
                    fh = record.file_hash
                else:
                    raise KeyError(f"File not registered: {file_path}")

        if fh not in data["files"]:
            raise KeyError(f"File not registered: {file_path}")

        data["files"][fh].setdefault("updates", {})
        data["files"][fh]["updates"][stage] = now()
        data["files"][fh]["updated_at"] = now()

        for key, value in extras.items():
            data["files"][fh][key] = value

        self._save(data)
        return FileRecord(data["files"][fh])

    def get(self, file_path: str | Path) -> "FileRecord | None":
        """Get a file's record. Returns None if never registered."""
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            # File deleted — search by name in registry
            return self._find_by_name(file_path.name)

        fh = self._get_hash(file_path)
        data = self._load()
        raw = data.get("files", {}).get(fh)
        return FileRecord(raw) if raw else None

    def get_by_hash(self, file_hash: str) -> "FileRecord | None":
        """Get a file's record by its SHA-256 hash."""
        data = self._load()
        raw = data.get("files", {}).get(file_hash)
        return FileRecord(raw) if raw else None

    def _find_by_name(self, filename: str) -> "FileRecord | None":
        """O(1) lookup by original name."""
        self._load()  # Ensure name index is built
        return self._name_index.get(filename)

    def get_or_register(self, file_path: str | Path) -> "FileRecord":
        """Get existing record or create a new one."""
        existing = self.get(file_path)
        if existing:
            return existing
        return self.register(file_path)

    def remove(self, file_path: str | Path) -> bool:
        """Remove a file from the registry. Returns True if it existed."""
        import shutil

        file_path = Path(file_path).resolve()
        fh = self._get_hash(file_path) if file_path.exists() else None

        data = self._load()

        # Try by hash first
        if fh and fh in data.get("files", {}):
            del data["files"][fh]
            self._save(data)
            # Also remove index directory
            index_dir = self.paths.file_dir_by_hash(fh)
            if index_dir.exists():
                shutil.rmtree(index_dir)
            return True

        # Try by name fallback
        for key, rec in list(data.get("files", {}).items()):
            if rec.get("original_name") == file_path.name:
                del data["files"][key]
                self._save(data)
                index_dir = self.paths.file_dir_by_hash(key)
                if index_dir.exists():
                    shutil.rmtree(index_dir)
                return True

        return False

    def list_all(self) -> list["FileRecord"]:
        """All registered files."""
        data = self._load()
        return list(self._cache.values())

    def list_kw_searchable(self) -> list["FileRecord"]:
        """Files that are keyword-searchable (parsed + chunked + BM25)."""
        return [r for r in self.list_all() if r.is_kw_searchable]

    def list_needing_work(self) -> list["FileRecord"]:
        """Files that are not yet fully keyword-searchable."""
        return [r for r in self.list_all() if not r.is_kw_searchable]

    def list_by_model(self, model_name: str) -> list["FileRecord"]:
        """Files embedded with a specific model."""
        return [r for r in self.list_all() if r.embedding_model == model_name]

    def count_kw_searchable(self) -> int:
        return len(self.list_kw_searchable())

    def total_chunks(self) -> int:
        return sum(r.chunk_count for r in self.list_all())

    def stats(self) -> dict:
        """Aggregate statistics for the dashboard — single pass over cached data."""
        all_records = self.list_all()
        total_files = len(all_records)
        kw_searchable = 0
        total_chunks = 0
        total_pages = 0
        models_used = set()

        for r in all_records:
            total_chunks += r.chunk_count
            total_pages += r.page_count
            if r.is_kw_searchable:
                kw_searchable += 1
            if r.embedding_model:
                models_used.add(r.embedding_model)

        return {
            "total_files": total_files,
            "keyword_searchable_files": kw_searchable,
            "files_needing_work": total_files - kw_searchable,
            "total_chunks": total_chunks,
            "total_pages": total_pages,
            "models_used": list(models_used),
        }

    def find_stale(self) -> list["FileRecord"]:
        """Files whose source has been deleted or moved."""
        stale = []
        for record in self.list_all():
            path = record.last_seen_path
            if path and not Path(path).exists():
                stale.append(record)
        return stale

    def refresh_paths(self) -> int:
        """
        Scan for moved files and update paths.
        Returns number of paths updated.
        """
        updated = 0
        data = self._load()
        for fh, rec in data.get("files", {}).items():
            stored_path = rec.get("last_seen_path")
            if stored_path and not Path(stored_path).exists():
                # File moved — try to find it by hash in common locations
                # For now, just mark it; user can manually relink
                rec["last_seen_path"] = None
                updated += 1
        if updated:
            self._save(data)
        return updated