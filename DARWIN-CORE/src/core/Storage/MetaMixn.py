# src/core/Storage/MetaMixin.py
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path

from src.core.Storage.FileRegistry import FileRecord
from src.core.Storage.StoreUtils import now, read_json, write_json,ARTIFACT_MANIFEST


class MetaMixin:
    """
    Handles meta.json read/write, content metadata, and artifact health.
    Owns everything the registry refuses to know about.
    """

    # ── Internal helpers (mirror DarwinStore pattern) ────────────────────

    def _hash_for(self, filename: str) -> str:
        """Resolve a registered filename to its content hash."""
        record = self.registry.find_by_name(filename)
        if not record:
            raise KeyError(f"File not registered: {filename}")
        return record.file_hash

    def _dir_for(self, filename: str) -> Path:
        """Return the index directory path for a registered file."""
        return self.paths.dir_for_hash(self._hash_for(filename))

    def _source_path_for(self, filename: str) -> Path | None:
        """Return the last known source path for a registered file."""
        record = self.registry.find_by_name(filename)
        if not record:
            return None
        path = record.last_seen_path
        return Path(path) if path else None

    def _artifact_paths(self, filename: str, stage: str) -> list[tuple[str, Path]]:
        """Return (key, path) pairs for every filesystem entry of a pipeline stage."""
        fh = self._hash_for(filename)
        d = self.paths.dir_for_hash(fh)
        spec = ARTIFACT_MANIFEST.get(stage)
        if not spec:
            return []
        return [(key, d / rel_path) for key, rel_path in spec.items()]

    def _has_artifact(self, filename: str, artifact: str) -> bool:
        """Check if all files for a pipeline stage actually exist on disk."""
        try:
            entries = self._artifact_paths(filename, artifact)
        except KeyError:
            return False
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

    def _record_for(self, filename: str) -> FileRecord | None:
        """Fetch the registry record for a file by its original name."""
        return self.registry.find_by_name(filename)

    def _list_known_files(self, from_registry: bool = True) -> list[str]:
        """Return a list of all registered original filenames."""
        if not from_registry:
            return []
        return [r.original_name for r in self.registry.list_all()]

    # ── Core meta I/O ─────────────────────────────────────────────────────

    def _update_meta(self, filename: str, component: str, extras: dict | None = None) -> None:
        fh = self._hash_for(filename)
        meta_path = self.paths.meta_path(fh)

        meta = read_json(meta_path) or {
            "filename": filename,
            "file_hash": fh,
            "created_at": now(),
            "updates": {},
        }

        meta.setdefault("updates", {})
        timestamp = now()
        meta["updates"][component] = timestamp
        meta["updated_at"] = timestamp

        if extras:
            for key, value in extras.items():
                if isinstance(value, dict) and isinstance(meta.get(key), dict):
                    meta[key].update(value)
                else:
                    meta[key] = value

        write_json(meta_path, meta)

    def _batch_meta_update(self, filename: str, updates: dict[str, dict]) -> None:
        fh = self._hash_for(filename)
        meta_path = self.paths.meta_path(fh)

        meta = read_json(meta_path) or {
            "filename": filename,
            "file_hash": fh,
            "created_at": now(),
            "updates": {},
        }
        meta.setdefault("updates", {})
        timestamp = now()

        for component, extras in updates.items():
            meta["updates"][component] = timestamp
            if extras:
                for key, value in extras.items():
                    if isinstance(value, dict) and isinstance(meta.get(key), dict):
                        meta[key].update(value)
                    else:
                        meta[key] = value

        meta["updated_at"] = timestamp
        write_json(meta_path, meta)

    def load_meta(self, filename: str) -> dict:
        try:
            fh = self._hash_for(filename)
        except KeyError:
            return {}
        return read_json(self.paths.meta_path(fh)) or {}

    def is_fresh(self, filename: str) -> bool:
        record = self.registry.find_by_name(filename)
        if not record:
            return False

        meta_path = self.paths.meta_path(record.file_hash)
        if not meta_path.exists():
            return False

        source = self._source_path_for(filename)
        if source is None or not source.exists():
            return False

        meta = read_json(meta_path) or {}
        return meta.get("file_hash", "") == self.paths.file_hash(source)

    # ── Content metadata accessors ────────────────────────────────────────

    def get_embedding_model(self, filename: str) -> str:
        """Runtime model key used for embeddings (e.g., 'intfloat/e5-small-v2-onnx')."""
        return self.load_meta(filename).get("embedding_model", "unknown")

    def get_embedding_dimensions(self, filename: str) -> int:
        return self.load_meta(filename).get("embedding_dimensions", 0)

    def get_chunk_count(self, filename: str) -> int:
        return self.load_meta(filename).get("chunk_count", 0)

    def get_vocab_size(self, filename: str) -> int:
        return self.load_meta(filename).get("vocab_size", 0)

    def get_node_count(self, filename: str) -> int:
        return self.load_meta(filename).get("node_count", 0)

    def get_edge_count(self, filename: str) -> int:
        return self.load_meta(filename).get("edge_count", 0)

    # ── Artifact health (actual file inspection, not just timestamps) ─────

    def has_artifact(self, filename: str, artifact: str) -> bool:
        """Check if artifact file actually exists on disk."""
        return self._has_artifact(filename, artifact)

    def is_kw_searchable(self, filename: str) -> bool:
        """BM25 index and stopwords exist and are loadable."""
        return self.has_artifact(filename, "bm25") and self.has_artifact(filename, "stopwords")

    def is_embedding_searchable(self, filename: str) -> bool:
        """Vector embeddings and usearch index exist and are loadable."""
        return self.has_artifact(filename, "embeddings") and self.has_artifact(filename, "vector")

    def is_hybrid_searchable(self, filename: str) -> bool:
        """Both BM25 and vector search ready."""
        return self.is_kw_searchable(filename) and self.is_embedding_searchable(filename)

    def has_graph(self, filename: str) -> bool:
        return self.has_artifact(filename, "graph") and self.has_artifact(filename, "graph_vis")

    def has_wiki(self, filename: str) -> bool:
        return self.has_artifact(filename, "wiki")

    def missing_stages(self, filename: str) -> list[str]:
        """Which core stages lack actual artifact files (not just timestamps)."""
        required = ["markdown", "chunks", "embeddings", "bm25", "vector"]
        return [s for s in required if not self.has_artifact(filename, s)]

    # ── Detailed progress (content-aware, not just timestamps) ────────────

    def progress_percent(self, filename: str) -> int:
        """0-100% based on actual artifact presence."""
        meta = self.load_meta(filename)
        if not meta:
            return 0

        stages = ["markdown", "chunks", "embeddings", "bm25", "vector"]
        weights = {"markdown": 10, "chunks": 15, "embeddings": 25, "bm25": 25, "vector": 25}

        score = 0
        for stage in stages:
            if self.has_artifact(filename, stage):
                score += weights.get(stage, 0)

        return min(score, 100)

    def status_summary(self, filename: str) -> str:
        """Human-readable one-liner from content metadata."""
        meta = self.load_meta(filename)
        if not meta:
            return "❓ Unknown — no metadata"

        if self.is_hybrid_searchable(filename):
            return f"✅ Ready — {meta.get('chunk_count', 0)} chunks, {meta.get('embedding_model', 'unknown')}"

        if not self.has_artifact(filename, "markdown"):
            return f"📄 Needs parsing — {meta.get('page_count', 0)} pages"

        if not self.has_artifact(filename, "embeddings"):
            return f"🔤 Needs embedding — {meta.get('chunk_count', 0)} chunks waiting"

        missing = self.missing_stages(filename)
        return f"⏳ In progress — {len(missing)} stages remaining"

    # ── Graph/Wiki specifics ──────────────────────────────────────────────

    def get_graph_stats(self, filename: str) -> dict:
        meta = self.load_meta(filename)
        graph = meta.get("graph", {})
        total_entities = graph.get("total_entities_extracted", 1)
        return {
            "node_count": graph.get("node_count", 0),
            "edge_count": graph.get("edge_count", 0),
            "entities_extracted": total_entities,
            "relations_extracted": graph.get("total_relations_extracted", 0),
            "dedup_ratio": graph.get("node_count", 0) / max(total_entities, 1),
        }

    def get_wiki_stats(self, filename: str) -> dict:
        meta = self.load_meta(filename)
        return meta.get("wiki", {})

    # ── Stats aggregation (across files) ──────────────────────────────────

    def total_chunks(self, filenames: list[str] | None = None) -> int:
        """Sum chunk counts. If no filenames provided, scans all known files."""
        if filenames is None:
            filenames = self._list_known_files(from_registry=True)
        return sum(self.get_chunk_count(f) for f in filenames)

    def stats(self, filenames: list[str] | None = None) -> dict:
        """Aggregate statistics for dashboard."""
        if filenames is None:
            filenames = self._list_known_files(from_registry=True)

        total_files = len(filenames)
        kw_searchable = sum(1 for f in filenames if self.is_kw_searchable(f))
        total_chunks = 0
        total_pages = 0
        models_used = set()

        for f in filenames:
            meta = self.load_meta(f)
            total_chunks += meta.get("chunk_count", 0)
            total_pages += meta.get("page_count", 0)
            if meta.get("embedding_model"):
                models_used.add(meta["embedding_model"])

        return {
            "total_files": total_files,
            "keyword_searchable_files": kw_searchable,
            "files_needing_work": total_files - kw_searchable,
            "total_chunks": total_chunks,
            "total_pages": total_pages,
            "models_used": list(models_used),
        }

    def list_by_model(self, model_name: str, filenames: list[str] | None = None) -> list[str]:
        """Return filenames embedded with a specific model."""
        if filenames is None:
            filenames = self._list_known_files(from_registry=True)
        return [f for f in filenames if self.get_embedding_model(f) == model_name]

    def list_kw_searchable(self, filenames: list[str] | None = None) -> list[str]:
        """Return filenames ready for keyword search."""
        if filenames is None:
            filenames = self._list_known_files(from_registry=True)
        return [f for f in filenames if self.is_kw_searchable(f)]

    def list_embedding_searchable(self, filenames: list[str] | None = None) -> list[str]:
        """Return filenames ready for embedding search."""
        if filenames is None:
            filenames = self._list_known_files(from_registry=True)
        return [f for f in filenames if self.is_embedding_searchable(f)]

    def list_hybrid_searchable(self, filenames: list[str] | None = None) -> list[str]:
        """Return filenames ready for hybrid search."""
        if filenames is None:
            filenames = self._list_known_files(from_registry=True)
        return [f for f in filenames if self.is_hybrid_searchable(f)]

    # ── Artifact integrity (cross-layer validation) ─────────────────────

    def validate_artifact_integrity(self, filename: str) -> dict:
        """
        Full integrity check: registry timestamps vs actual artifact files.
        Detects desync (registry says done but files missing, or files present
        but registry has no timestamp).
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
            "is_healthy": len(actually_missing) == 0 and len(actually_present) == 0,
        }

    def repair_registry_from_artifacts(self, filename: str) -> list[str]:
        """
        Scan artifacts and backfill missing registry timestamps.
        Returns list of stages that were repaired.
        """
        record = self._record_for(filename)
        if record is None:
            return []

        repaired = []
        for stage in ARTIFACT_MANIFEST:
            if not record.stage_completed(stage) and self._has_artifact(filename, stage):
                paths = self._artifact_paths(filename, stage)
                file_paths = [p for key, p in paths if key != "dir" and p.exists()]
                if file_paths:
                    self.registry.update_pipeline(
                        file_hash=record.file_hash,
                        stage=stage,
                    )
                    repaired.append(stage)
        return repaired