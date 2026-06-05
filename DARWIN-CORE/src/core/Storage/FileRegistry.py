# src/core/Storage/FileRegistry.py
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from src.core.Storage.StorePaths import StorePaths
from src.core.Storage.StoreUtils import read_json, write_json, now

PipelineStage = Literal[
    "markdown", "chunks", "embeddings", "bm25", "vector",
    "graph", "graph_vis", "wiki",
]

CORE_STAGES = ["markdown", "chunks", "embeddings", "bm25", "vector"]


class FileRecord:
    """Lightweight identity + stage-timestamp object. No content metadata."""

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
        return self._data.get("last_seen_path")

    @property
    def file_size(self) -> int:
        return self._data.get("file_size", 0)

    @property
    def file_size_mb(self) -> float:
        return self.file_size / (1024 * 1024)

    @property
    def page_count(self) -> int:
        return self._data.get("page_count", 0)

    @property
    def chunk_count(self) -> int:
        return self._data.get("chunk_count", 0)

    @property
    def embedding_model(self) -> str | None:
        return self._data.get("embedding_model")

    # ── Pointers (registry does NOT read these files) ─────────────────────
    @property
    def meta_path(self) -> str | None:
        """Path to per-file metadata JSON (heavy content lives here)."""
        return self._data.get("meta_path")

    @property
    def blob_path(self) -> str | None:
        """Path to original source file blob."""
        return self._data.get("blob_path")

    # ── Timestamps ────────────────────────────────────────────────────────
    @property
    def created_at(self) -> str:
        return self._data.get("created_at", "")

    @property
    def updated_at(self) -> str:
        return self._data.get("updated_at", "")

    @property
    def updates(self) -> dict[str, str]:
        """Stage name → ISO timestamp. Registry makes no claims about artifact health."""
        return self._data.get("updates", {})

    # ── Simple stage checks (timestamp only, no artifact inspection) ────────
    def stage_completed(self, stage: str) -> bool:
        return bool(self.updates.get(stage))

    def stages_completed(self) -> list[str]:
        return [s for s in self.updates if self.updates[s]]

    # ── Derived helpers (used by DarwinStore) ─────────────────────────────
    @property
    def is_parsed(self) -> bool:
        return self.stage_completed("markdown")

    @property
    def is_chunked(self) -> bool:
        return self.stage_completed("chunks")

    @property
    def is_embedded(self) -> bool:
        return self.stage_completed("embeddings")

    @property
    def is_kw_searchable(self) -> bool:
        return self.stage_completed("bm25") and self.stage_completed("stopwords")

    @property
    def is_embedding_searchable(self) -> bool:
        return self.stage_completed("embeddings") and self.stage_completed("vector")

    @property
    def is_hybrid_searchable(self) -> bool:
        return self.is_kw_searchable and self.is_embedding_searchable

    @property
    def has_graph(self) -> bool:
        return self.stage_completed("graph")

    @property
    def has_wiki(self) -> bool:
        return self.stage_completed("wiki")

    def pipeline_progress(self) -> int:
        """
        Coarse 0-100% based on core stage timestamps only.
        For detailed progress (chunk counts, etc.), inspect per-file Meta.
        """
        done = sum(1 for s in CORE_STAGES if self.stage_completed(s))
        return int(done / len(CORE_STAGES) * 100)

    def progress_percent(self) -> int:
        """Back-compat alias for pipeline_progress."""
        return self.pipeline_progress()

    def status_summary(self) -> str:
        if self.is_core_complete():
            return "complete"
        if self.is_parsed:
            return "parsed"
        if self.original_name and self.original_name != "unknown":
            return "registered"
        return "unknown"

    def missing_stages(self) -> list[str]:
        """Core pipeline stages that do not have timestamps yet."""
        return [s for s in CORE_STAGES if not self.stage_completed(s)]

    def is_core_complete(self) -> bool:
        """All core stages have timestamps. Does NOT verify artifacts exist."""
        return all(self.stage_completed(s) for s in CORE_STAGES)

    def to_dict(self) -> dict:
        return dict(self._data)


class FileRegistry:
    """
    Lightweight tracker at ~/.darwin/file_registry.json.
    Schema: {version, files: {hash: FileRecord}}
    """

    def __init__(self, paths: StorePaths):
        self.paths = paths
        self._cache: dict[str, FileRecord] | None = None
        self._all_data: dict | None = None
        self._name_index: dict[str, list[FileRecord]] | None = None
        self._hash_cache: dict[str, str] = {}

    @property
    def registry_path(self) -> Path:
        return self.paths.file_registry_path

    # ── Internal data loading ─────────────────────────────────────────────
    def load_data(self) -> dict:
        if self._cache is not None and self._all_data is not None:
            return self._all_data

        self._all_data = read_json(self.registry_path) or {"version": "1.0", "files": {}}
        self._cache = {
            fh: FileRecord(rec)
            for fh, rec in self._all_data.get("files", {}).items()
        }
        self._rebuild_name_index()
        return self._all_data

    def _rebuild_name_index(self) -> None:
        self._name_index = {}
        for rec in self._cache.values():
            name = rec.original_name
            if name:
                self._name_index.setdefault(name, []).append(rec)

    def save_data(self, data: dict) -> None:
        data["updated_at"] = now()
        write_json(self.paths.file_registry_path, data)
        self._all_data = data
        self._cache = {
            fh: FileRecord(rec)
            for fh, rec in data.get("files", {}).items()
        }
        self._rebuild_name_index()

    def get_hash(self, file_path: str | Path) -> str:
        file_path = Path(file_path)
        file_path_str = str(file_path.resolve()) if file_path.exists() else str(file_path)

        if file_path_str in self._hash_cache:
            return self._hash_cache[file_path_str]

        if file_path.exists():
            fh = self.paths.file_hash(file_path)
            self._hash_cache[file_path_str] = fh
            return fh

        record = self.find_by_name(file_path.name)
        if record:
            return record.file_hash
        return ""

    # ── Public API ────────────────────────────────────────────────────────
    def register(
        self,
        file_path: str | Path,
        page_count: int = 0,
        file_size: int = 0,
        file_hash: str | None = None,
    ) -> FileRecord:
        file_path = Path(file_path).resolve()
        fh = file_hash or self.paths.file_hash(file_path)
        self._hash_cache[str(file_path)] = fh
        data = self.load_data()

        if fh not in data["files"]:
            data["files"][fh] = {
                "file_hash": fh,
                "original_name": file_path.name,
                "last_seen_path": str(file_path),
                "file_size": file_size or (file_path.stat().st_size if file_path.exists() else 0),
                "page_count": page_count,
                "created_at": now(),
                "updated_at": now(),
                "updates": {},
                "meta_path": str(self.paths.dir_for_hash(fh) / f"{file_path.stem}.meta.json"),
                "blob_path": str(self.paths.dir_for_hash(fh) / file_path.name),
            }
        else:
            data["files"][fh]["last_seen_path"] = str(file_path)
            data["files"][fh]["original_name"] = file_path.name
            data["files"][fh]["file_size"] = file_size or (
                file_path.stat().st_size if file_path.exists() else 0
            )
            if page_count:
                data["files"][fh]["page_count"] = page_count
            data["files"][fh]["updated_at"] = now()

        self.save_data(data)
        return FileRecord(data["files"][fh])

    def update_pipeline(
        self,
        file_path: str | Path | None = None,
        stage: str = "",
        file_hash: str | None = None,
    ) -> FileRecord:
        """
        Mark a pipeline stage as completed. Stores ONLY a timestamp.
        Does NOT accept **extras — content metadata belongs in per-file Meta.
        """
        data = self.load_data()

        if file_hash:
            fh = file_hash
        elif file_path:
            file_path = Path(file_path).resolve()
            fh = self.get_hash(file_path) if file_path.exists() else None
            if not fh:
                record = self.find_by_name(file_path.name)
                if record:
                    fh = record.file_hash
        else:
            raise ValueError("Must provide file_path or file_hash")

        if not fh or fh not in data["files"]:
            raise KeyError(f"File not registered: {file_path or file_hash}")

        data["files"][fh].setdefault("updates", {})
        data["files"][fh]["updates"][stage] = now()
        data["files"][fh]["updated_at"] = now()

        self.save_data(data)
        return FileRecord(data["files"][fh])

    def get(self, file_path: str | Path) -> FileRecord | None:
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            return self.find_by_name(file_path.name)

        fh = self.get_hash(file_path)
        data = self.load_data()
        raw = data.get("files", {}).get(fh)
        return FileRecord(raw) if raw else None

    def get_by_hash(self, file_hash: str) -> FileRecord | None:
        data = self.load_data()
        raw = data.get("files", {}).get(file_hash)
        return FileRecord(raw) if raw else None

    def find_by_name(self, filename: str) -> FileRecord | None:
        self.load_data()
        matches = self._name_index.get(filename, [])
        return matches[0] if matches else None

    def get_or_register(self, file_path: str | Path) -> FileRecord:
        existing = self.get(file_path)
        if existing:
            return existing
        return self.register(file_path)

    def remove_by_name(self, filename: str) -> int:
        data = self.load_data()
        files = data.get("files", {})
        to_delete = [fh for fh, rec in files.items() if rec.get("original_name") == filename]

        if not to_delete:
            return 0

        for fh in to_delete:
            del files[fh]
            index_dir = self.paths.file_dir_by_hash(fh)
            if index_dir.exists():
                shutil.rmtree(index_dir)
            stale_keys = [k for k, v in self._hash_cache.items() if v == fh]
            for k in stale_keys:
                del self._hash_cache[k]

        self.save_data(data)
        return len(to_delete)

    def list_all(self) -> list[FileRecord]:
        self.load_data()
        return list(self._cache.values())

    def list_core_complete(self) -> list[FileRecord]:
        """Files with all core stage timestamps (coarse check)."""
        return [r for r in self.list_all() if r.is_core_complete()]

    def list_needing_work(self) -> list[FileRecord]:
        """Files missing at least one core stage timestamp."""
        return [r for r in self.list_all() if not r.is_core_complete()]

    def find_stale(self) -> list[FileRecord]:
        stale = []
        for record in self.list_all():
            path = record.last_seen_path
            if path and not Path(path).exists():
                stale.append(record)
        return stale

    def refresh_paths(self) -> int:
        updated = 0
        data = self.load_data()
        for fh, rec in data.get("files", {}).items():
            stored_path = rec.get("last_seen_path")
            if stored_path and not Path(stored_path).exists():
                rec["last_seen_path"] = None
                updated += 1
        if updated:
            self.save_data(data)
        return updated

    def count(self) -> int:
        self.load_data()
        return len(self._cache)

    def count_core_complete(self) -> int:
        return len(self.list_core_complete())

    def stats(self) -> dict:
        """Dashboard-friendly aggregate stats."""
        files = self.list_all()
        needing = self.list_needing_work()
        kw_searchable = [f for f in files if f.is_kw_searchable]
        return {
            "total_files": len(files),
            "files_needing_work": len(needing),
            "keyword_searchable_files": len(kw_searchable),
            "total_chunks": sum(f.chunk_count for f in files),
            "total_pages": sum(f.page_count for f in files),
        }