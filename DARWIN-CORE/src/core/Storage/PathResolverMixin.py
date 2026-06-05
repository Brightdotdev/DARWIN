# src/core/Storage/PathResolverMixin.py
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path

from src.core.Storage.FileRegistry import FileRecord
from src.core.Storage.StorePaths import ARTIFACT_MANIFEST


class PathResolverMixin:
    """
    Bridge between calling code (bare filenames) and StorePaths (file hashes).

    Responsibilities:
      - Resolve filename → hash → directory
      - Locate source file paths
      - Low-level artifact path existence (no semantic interpretation)
      - List known files (from registry or disk)

    Does NOT:
      - Inspect artifact health or compare against registry timestamps
      - Know about embedding models, chunk counts, or searchable status
    """

    # ── Path resolution ──────────────────────────────────────────────────

    def _record_for(self, filename: str) -> FileRecord | None:
        """Return the FileRecord for a bare filename. Returns None if missing."""
        return self.registry.find_by_name(filename)

    def _resolve(self, filename: str) -> tuple[str, Path | None]:
        """Resolve a bare filename to (file_hash, index_dir)."""
        record = self._record_for(filename)
        if record is None:
            return "", None
        return record.file_hash, self.paths.dir_for_hash(record.file_hash)

    def _hash_for(self, filename: str) -> str:
        """Resolve a bare filename to its content hash."""
        record = self.registry.find_by_name(filename)
        if not record:
            raise KeyError(f"File not registered: {filename}")
        return record.file_hash

    def _dir_for(self, filename: str) -> Path | None:
        """Return the index directory path for a bare filename. Pure resolution, no side effects."""
        _, d = self._resolve(filename)
        return d

    def _ensure_dir_for(self, filename: str) -> Path | None:
        """Create and return the index directory for a bare filename."""
        d = self._dir_for(filename)
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def _source_path_for(self, filename: str) -> Path | None:
        """Return the last-known source Path for a file (may not exist if moved)."""
        record = self._record_for(filename)
        if record and record.last_seen_path:
            return Path(record.last_seen_path)
        return None

    # ── File enumeration ─────────────────────────────────────────────────

    def _list_known_files(self, from_registry: bool = True) -> list[str]:
        """
        Return all known filenames.

        from_registry=True  → Fast, uses cached registry (may be stale if desynced)
        from_registry=False → Slow, scans meta.json on disk (ground truth)
        """
        if from_registry:
            return [r.original_name for r in self.registry.list_all()]

        known = []
        indexes_dir = self.paths.indexes_dir
        if not indexes_dir.exists():
            return known

        from src.core.Storage.StoreUtils import read_json
        for hash_dir in indexes_dir.iterdir():
            if not hash_dir.is_dir():
                continue
            meta_path = hash_dir / "meta.json"
            if meta_path.exists():
                meta = read_json(meta_path)
                if meta and meta.get("filename"):
                    known.append(meta["filename"])
        return known

    def _list_known_hashes(self) -> list[str]:
        """Return all file hashes that have an index directory on disk."""
        indexes_dir = self.paths.indexes_dir
        if not indexes_dir.exists():
            return []
        return [d.name for d in indexes_dir.iterdir() if d.is_dir()]

    # ── Low-level artifact paths ─────────────────────────────────────────

    def _artifact_paths(self, filename: str, stage: str) -> list[tuple[str, Path]]:
        """Return (key, path) pairs for every filesystem entry of a pipeline stage. Does not check existence."""
        _, d = self._resolve(filename)
        if d is None:
            return []
        spec = ARTIFACT_MANIFEST.get(stage)
        if not spec:
            return []
        return [(key, d / rel_path) for key, rel_path in spec.items()]

    def _has_artifact(self, filename: str, stage: str) -> bool:
        """
        Low-level: do the expected files for this stage exist on disk?
        """
        entries = self._artifact_paths(filename, stage)
        if not entries:
            return False
        for key, p in entries:
            if key == "dir":
                if not p.exists() or not p.is_dir():
                    return False
            else:
                if not p.exists():
                    return False
        return True