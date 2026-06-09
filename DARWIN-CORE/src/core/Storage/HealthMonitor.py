from __future__ import annotations

from typing import Optional

from src.core.Storage.ArtifactStore import ArtifactStore
from src.core.Storage.Constants import ARTIFACT_MANIFEST, CORE_STAGES, OPTIONAL_ARTIFACT_KEYS
from src.core.Storage.StoreRegistry import ContentIndex, Registry, SourceManager
from src.core.Storage.StorePaths import StorePaths
from src.core.Storage.Transanctions import delete_tree


class HealthMonitor:
    """Cross-layer integrity: registry flags vs actual files on disk."""

    def __init__(
        self,
        registry: Registry,
        sources: SourceManager,
        contents: ContentIndex,
        artifacts: ArtifactStore,
        paths: StorePaths,
    ):
        self._reg = registry
        self._sources = sources
        self._contents = contents
        self._artifacts = artifacts
        self._paths = paths

    
    def check_integrity(self, content_hash: str) -> dict:
        content = self._contents.get(content_hash)
        if not content:
            return {
                "hash": content_hash,
                "error": "Not registered in contents table",
                "is_healthy": False,
            }

        registry_done = []
        actually_missing = []
        actually_present = []
        optional = OPTIONAL_ARTIFACT_KEYS

        for stage in ARTIFACT_MANIFEST:
            # Skip artifacts not tracked in the contents table
            # (e.g. images — managed by StorePaths, not a pipeline stage)
            if not hasattr(content, f"has_{stage}"):
                continue

            registry_says = getattr(content, f"has_{stage}", False)
            files_exist = self._artifacts.has_artifact(content_hash, stage)

            if registry_says and not files_exist:
                actually_missing.append(stage)
            elif not registry_says and files_exist:
                actually_present.append(stage)
            elif registry_says and files_exist:
                registry_done.append(stage)

        return {
            "hash": content_hash,
            "is_healthy": len(actually_missing) == 0 and len(actually_present) == 0,
            "registry_done": registry_done,
            "actually_missing": actually_missing,
            "actually_present": actually_present,
        }

    def repair(
        self,
        content_hash: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict:
        hashes = [content_hash] if content_hash else [c.hash for c in self._contents.list_all()]
        result = {
            "hashes_checked": len(hashes),
            "healthy": 0,
            "repaired": 0,
            "removed": 0,
            "repairs": [],
        }

        for h in hashes:
            report = self.check_integrity(h)
            if "error" in report:
                continue

            content = self._contents.get(h)
            if not content:
                continue

            if report["is_healthy"]:
                result["healthy"] += 1
                continue

            missing = report["actually_missing"]
            present = report["actually_present"]

            # CASE 1: All core artifacts missing → content is dead
            core_missing = [s for s in CORE_STAGES if s in missing]
            if len(core_missing) == len(CORE_STAGES):
                if not dry_run:
                    self._contents._reg.execute("DELETE FROM contents WHERE hash = ?", (h,))
                    self._contents._reg.commit()
                    delete_tree(self._paths.dir_for_hash(h))
                    # Also delete sources that pointed here
                    self._contents._reg.execute(
                        "DELETE FROM sources WHERE content_hash = ?", (h,)
                    )
                    self._contents._reg.commit()
                result["removed"] += 1
                result["repairs"].append({"hash": h, "action": "removed", "reason": "all core artifacts missing"})
                continue

            # CASE 2: Some missing → clear flags
            if missing and not dry_run:
                for stage in missing:
                    if hasattr(content, f"has_{stage}"):
                        self._contents.set_stage(h, stage, complete=False)
                result["repaired"] += 1
                result["repairs"].append({"hash": h, "action": "cleared_flags", "stages": missing})

            # CASE 3: Orphans present → set flags
            if present and not dry_run:
                for stage in present:
                    if hasattr(content, f"has_{stage}"):
                        self._contents.set_stage(h, stage, complete=True)
                if not missing:
                    result["repaired"] += 1
                result["repairs"].append({"hash": h, "action": "set_flags", "stages": present})

        return result

    def full_audit(self) -> dict:
        report = self.repair(dry_run=True)
        all_hashes = [c.hash for c in self._contents.list_all()]
        broken = []
        for h in all_hashes:
            integrity = self.check_integrity(h)
            if not integrity.get("is_healthy", True):
                broken.append({
                    "hash": h,
                    "missing": integrity.get("actually_missing", []),
                    "orphans": integrity.get("actually_present", []),
                })

        return {
            "summary": {
                "total_hashes": len(all_hashes),
                "healthy": report["healthy"],
                "broken": len(broken),
                "would_repair": len(report["repairs"]),
            },
            "broken_hashes": broken,
        }