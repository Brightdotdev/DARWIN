# src/mcp/tools/models.py
"""
ModelsOrchestrator MCP tool registration.
Refactored from standalone server to unified registration pattern.
"""

import json
import traceback
from typing import Any, Dict, List, Optional

from src.core.Orchestrators.ModelOrchestrator import (
    ModelsOrchestrator,
    modelsOrchestrator,
    ModelCategory,
    ModelTier,
)


def _ok(data: Any) -> str:
    return json.dumps({"success": True, "data": data}, default=str)


def _err(message: str, detail: Optional[str] = None) -> str:
    return json.dumps({
        "success": False,
        "error": message,
        "detail": detail,
    }, default=str)


def _category(cat: str) -> ModelCategory:
    return ModelCategory[cat.upper()]


def _tier(tier: str) -> ModelTier:
    return ModelTier[tier.upper()]


def register_model_tools(mcp):
    """Register all model management tools on the shared FastMCP instance."""

    orch = modelsOrchestrator

    # ── Status & Discovery ──

    @mcp.tool()
    def get_model_status(auto_setup: bool = False) -> str:
        try:
            status = orch.status(auto_setup=auto_setup)
            return _ok({
                "all_ready": status.all_ready,
                "summary": status.summary,
                "missing_required": status.missing_required,
                "warnings": status.warnings,
                "categories": {
                    name: {
                        "available": cs.available,
                        "active_model_id": cs.active_model_id,
                        "available_models": cs.available_models,
                        "error": cs.error,
                        "is_required": cs.is_required,
                    }
                    for name, cs in status.categories.items()
                },
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def list_available_models(category: str) -> str:
        try:
            return _ok(orch.list_available(_category(category)))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def list_registered_models(category: Optional[str] = None) -> str:
        try:
            cat = _category(category) if category else None
            return _ok(orch.list_registered(category=cat))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def get_model_categories() -> str:
        try:
            return _ok(orch.list_categories())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def get_model_providers() -> str:
        try:
            return _ok(orch.list_providers())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def get_active_model(category: str) -> str:
        try:
            return _ok({"active_model_id": orch.get_active(_category(category))})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Installation ──

    @mcp.tool()
    def install_model(
        model_id: str,
        auto_download: bool = True,
        force: bool = False,
        set_active: bool = True,
    ) -> str:
        try:
            r = orch.install(model_id, auto_download=auto_download, force=force, set_active=set_active)
            return _ok({
                "success": r.success,
                "model_id": r.model_id,
                "category": r.category,
                "message": r.message,
                "error": r.error,
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def install_batch_models(
        model_ids: List[str],
        auto_download: bool = True,
        force: bool = False,
        set_active: bool = True,
    ) -> str:
        try:
            batch = orch.install_batch(model_ids, auto_download=auto_download, force=force, set_active=set_active)
            return _ok({
                "all_ok": batch.all_ok,
                "summary": batch.summary,
                "succeeded": batch.succeeded,
                "failed": batch.failed,
                "results": {
                    mid: {"success": r.success, "message": r.message, "error": r.error}
                    for mid, r in batch.results.items()
                },
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Switching ──

    @mcp.tool()
    def can_switch_model(category: str, model_id: str) -> str:
        try:
            r = orch.can_switch(_category(category), model_id)
            return _ok({
                "can_switch": r.success,
                "model_id": r.model_id,
                "category": r.category,
                "message": r.message,
                "error": r.error,
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def switch_model(category: str, model_id: str, auto_download: bool = True) -> str:
        try:
            r = orch.switch(_category(category), model_id, auto_download=auto_download)
            return _ok({
                "success": r.success,
                "model_id": r.model_id,
                "category": r.category,
                "message": r.message,
                "error": r.error,
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Upgrade / Uninstall ──

    @mcp.tool()
    def upgrade_model(model_id: str, auto_download: bool = True) -> str:
        try:
            r = orch.upgrade(model_id, auto_download=auto_download)
            return _ok({
                "success": r.success,
                "model_id": r.model_id,
                "category": r.category,
                "message": r.message,
                "error": r.error,
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def uninstall_model(model_id: str, delete_files: bool = True) -> str:
        try:
            r = orch.uninstall(model_id, delete_files=delete_files)
            return _ok({
                "success": r.success,
                "model_id": r.model_id,
                "category": r.category,
                "message": r.message,
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Validation ──

    @mcp.tool()
    def validate_model(model_id: str) -> str:
        try:
            r = orch.validate(model_id)
            return _ok({
                "valid": r.success,
                "model_id": r.model_id,
                "category": r.category,
                "message": r.message,
                "error": r.error,
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Profiles & Onboarding ──

    @mcp.tool()
    def setup_model_profile(
        profile: str = "required",
        auto_download: bool = True,
        force_download: bool = False,
    ) -> str:
        try:
            results = orch.setup_profile(profile=profile, auto_download=auto_download, force_download=force_download)
            return _ok({
                cat.value: {
                    "success": res.success,
                    "model_id": res.model_id,
                    "error": res.error,
                }
                for cat, res in results.items()
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def onboard_models(
        preferences: Optional[Dict[str, str]] = None,
        auto_download: bool = True,
        detect_system: bool = True,
    ) -> str:
        try:
            report = orch.onboard(
                preferences=preferences,
                auto_download=auto_download,
                detect_system=detect_system,
            )
            return _ok(report)
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def detect_system_info() -> str:
        try:
            info = orch.detect_system()
            return _ok({
                "ram_gb": info.ram_gb,
                "disk_free_gb": info.disk_free_gb,
                "cpu_cores": info.cpu_cores,
                "has_gpu": info.has_gpu,
                "gpu_name": info.gpu_name,
                "tier": info.tier,
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def get_model_recommendations() -> str:
        try:
            recs = orch.get_recommendations()
            return _ok({
                "embedding": recs.embedding,
                "nlp": recs.nlp,
                "ocr": recs.ocr,
                "reranker": recs.reranker,
                "vision": recs.vision,
            })
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Key Management ──

    @mcp.tool()
    def set_api_key(provider: str, key: str, validate: bool = True) -> str:
        try:
            ok = orch.set_key(provider, key, validate=validate)
            return _ok({"success": ok})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def list_api_key_status() -> str:
        try:
            return _ok(orch.list_key_status())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Resource Management ──

    @mcp.tool()
    def unload_all_models() -> str:
        try:
            count = orch.unload_all()
            return _ok({"unloaded_count": count})
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    # ── Registry Introspection ──

    @mcp.tool()
    def get_registry_stats() -> str:
        try:
            return _ok(orch.registry_stats())
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def get_models_by_provider(provider: str, category: Optional[str] = None) -> str:
        try:
            cat = _category(category) if category else None
            return _ok(orch.get_models_by_provider(provider, category=cat))
        except Exception as e:
            return _err(str(e), traceback.format_exc())

    @mcp.tool()
    def get_models_by_tier(tier: str) -> str:
        try:
            return _ok(orch.get_models_by_tier(_tier(tier)))
        except Exception as e:
            return _err(str(e), traceback.format_exc())