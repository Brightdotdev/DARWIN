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
        from src.core.Models.ModelManager import modelManager
        from src.core.Models.Lib.ModelRegistry import ModelCategory, find_spec

        health = {
            "active_model": model_id,
            "up_to_date": [],
            "needs_reembed": [],
            "model_missing": [],
            "never_embedded": [],
        }

        # Quick check: does the active model exist in registry?
        spec = find_spec(model_id)
        if spec is None or spec.category != ModelCategory.EMBEDDING:
            # The active model is not even registered – treat everything as "model_missing"
            # but we can still iterate content to mark missing.
            pass  # we'll still check per content

        for content in self._contents.list_all():
            h = content.hash
            names = [s.display_name for s in self._sources.get_by_hash(h)]
            name = names[0] if names else h[:8]

            if not content.embedding_model:
                health["never_embedded"].append(name)
                continue

            old_model = content.embedding_model

            if old_model == model_id:
                health["up_to_date"].append(name)
            else:
                # Check if the old model is still available
                if modelManager.is_available(ModelCategory.EMBEDDING, old_model):
                    health["needs_reembed"].append(name)
                else:
                    health["model_missing"].append(name)

        return health