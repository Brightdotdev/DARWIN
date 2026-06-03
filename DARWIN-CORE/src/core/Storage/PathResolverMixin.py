
from src.core.Logger.DarwinLogger import darwin, LogType
from pathlib import Path
from typing import Optional




ARTIFACT_MANIFEST = {
    "markdown": ["markdown.md", "display-markdown.md"],
    "chunks": ["chunks.parquet"],           # upgraded from chunks.json
    "vector": ["vector.usearch", "vector.meta.json"],
    "embeddings": ["embeddings.f16.npy"],   # upgraded from *.npz
    "bm25": ["bm25.npz", "bm25_meta.json", "bm25_tokens.parquet"],
    "stopwords": ["stopwords.json"],
    "graph": ["graph/nodes.parquet", "graph/edges.parquet"],  # upgraded from graph.json
    "graph_vis": ["graph/graph.html"],
    "wiki": ["wiki/wiki.parquet"],          # upgraded from wiki.json
    "meta" : ["meta.json"],
}




class PathResolverMixin:
    """Bridge between calling code (bare filenames) and StorePaths (file hashes)."""


    # ── Path resolution ──────────────────────────────────────────────────

    def _record_for(self, filename: str):
        """Return the FileRecord for a bare filename. Raises KeyError if missing."""
        record = self.registry._find_by_name(filename)
        if record is None:
            darwin.log(
                f"File '{filename}' is not registered. Register it before saving artifacts.",
                LogType.ERROR,
            )
            return None
        return record

    def _resolve(self, filename: str) -> tuple[str, Path]:
        """Resolve a bare filename to (file_hash, index_dir) in one registry lookup."""
        record = self._record_for(filename)
        if record is None:
            return "", None
        fh = record.file_hash
        return fh, self.paths.dir_for_hash(fh)

    def _hash_for(self, filename: str) -> str:
        """Resolve a bare filename to its file hash."""
        fh, _ = self._resolve(filename)
        return fh

    def _dir_for(self, filename: str) -> Path | None:
        """Return (and create) the index directory for a bare filename."""
        _, d = self._resolve(filename)
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)
        return d

    def _source_path_for(self, filename: str) -> Optional[Path]:
        """Return the last-known source Path for a file (may not exist if moved)."""
        record = self._record_for(filename)
        if record and record.last_seen_path:
            return Path(record.last_seen_path)
        return None

    # ── Unified artifact existence check ─────────────────────────────────

    def _has_artifact(self, filename: str, stage: str) -> bool:
        """
        Check whether all required artifact files for a pipeline stage exist.
        Uses the ARTIFACT_MANIFEST as the single source of truth.
        """
        dir_path = self._dir_for(filename)
        if dir_path is None:
            return False

        # Embeddings: static filename in the new layout
        if stage == "embeddings":
            return (dir_path / "embeddings.f16.npy").exists()

        file_list = ARTIFACT_MANIFEST.get(stage, [])
        if not file_list:
            return True
        return all((dir_path / fname).exists() for fname in file_list)

    def validate_artifact_integrity(self, filename: str) -> dict:
        """
        Full integrity check for one file.
        """
        record = self._record_for(filename)
        if record is None:
            return {
                "filename": filename,
                "error": "Not registered",
                "is_healthy": False,
            }

        registry_done = []
        registry_missing = []
        actually_missing = []
        actually_present = []

        for stage in ARTIFACT_MANIFEST:
            registry_says_done = record.stage_completed(stage)
            files_exist = self._has_artifact(filename, stage)

            if registry_says_done:
                registry_done.append(stage)
                if not files_exist:
                    actually_missing.append(stage)
            else:
                registry_missing.append(stage)
                if files_exist:
                    actually_present.append(stage)

        return {
            "filename": filename,
            "registry_says_done": registry_done,
            "actually_missing": actually_missing,
            "registry_says_missing": registry_missing,
            "actually_present": actually_present,
            "is_healthy": len(actually_missing) == 0,
        }


