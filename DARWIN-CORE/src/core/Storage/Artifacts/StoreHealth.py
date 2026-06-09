from __future__ import annotations
from src.core.Logger.DarwinLogger import darwin, LogType
class StoreHealth:
    """Registry integrity checks and embedding-model health."""

    def repair_registry(self, filename: str | None = None, dry_run: bool = False) -> dict:
        if filename:
            h = self._resolve_hash(filename)
            if h is None:
                darwin.log("File not registered and operation aborted...try parsing the file first", LogType.ERROR)
                return {}
            return self._health.repair(h, dry_run=dry_run)
        return self._health.repair(dry_run=dry_run)

    def full_registry_health_check(self) -> dict:
        return self._health.full_audit()

    def get_embedding_health(self, model_id: str) -> dict:
        from src.core.Models.ModelRegistry import REGISTRY
        from src.core.Models.ModelStore import get_model_store

        model_store = get_model_store()
        health = {
            "active_model": model_id,
            "up_to_date": [],
            "needs_reembed": [],
            "model_missing": [],
            "never_embedded": [],
        }

        id_to_key = {spec.model_id: key for key, spec in REGISTRY.items()}

        for content in self._contents.list_all():
            h = content.hash
            names = [s.display_name for s in self._sources.get_by_hash(h)]
            name = names[0] if names else h[:8]

            if not content.embedding_model:
                health["never_embedded"].append(name)
                continue
            if content.embedding_model == model_id:
                health["up_to_date"].append(name)
                continue

            old_key = id_to_key.get(content.embedding_model)
            if old_key and model_store.is_downloaded(old_key):
                health["needs_reembed"].append(name)
            else:
                health["model_missing"].append(name)

        return health