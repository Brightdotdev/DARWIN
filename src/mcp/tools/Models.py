import traceback
from typing import Optional

from mcp.types import ToolAnnotations

from src.core.Orchestrators.ModelOrchestrator import modelsOrchestrator, ModelCategory

_VALID_CATEGORIES = {c.value for c in ModelCategory}


def _validate_model_id(model_id: str) -> str:
    model_id = model_id.strip()
    if not model_id:
        raise ValueError("model_id must not be empty")
    if len(model_id) > 256:
        raise ValueError("model_id too long")
    if any(c in model_id for c in '<>"|;`'):
        raise ValueError("model_id contains invalid characters")
    return model_id


def _validate_category(category: str) -> str:
    category = category.strip().lower()
    if category not in _VALID_CATEGORIES:
        raise ValueError(f"unknown category '{category}'; valid: {sorted(_VALID_CATEGORIES)}")
    return category


def register(mcp):
    @mcp.tool(
        annotations=ToolAnnotations(
            title="Model Status",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def model_status(category: Optional[str] = None) -> dict:
        try:
            if category:
                category = _validate_category(category)
            status = modelsOrchestrator.status()
            data = {
                "all_ready": status.all_ready,
                "summary": status.summary,
            }
            if category:
                cat = ModelCategory(category)
                data["active"] = modelsOrchestrator.get_active(cat)
                data["available"] = modelsOrchestrator.list_available(cat)
                return {"success": True, "data": data}

            data["categories"] = {
                name: {
                    "available": cs.available,
                    "active": cs.active_model_id,
                    "error": cs.error,
                    "required": cs.is_required,
                }
                for name, cs in status.categories.items()
            }
            return {"success": True, "data": data}
        except Exception as e:
            return {"success": False, "error": str(e), "detail": traceback.format_exc()}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Install Model",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        ),
    )
    def install_model(
        model_id: str,
        auto_download: bool = True,
        set_active: bool = True,
    ) -> dict:
        try:
            model_id = _validate_model_id(model_id)
            r = modelsOrchestrator.install(
                model_id, auto_download=auto_download, set_active=set_active
            )
            return {
                "success": r.success,
                "data": {
                    "model_id": r.model_id,
                    "category": r.category,
                    "message": r.message,
                    "error": r.error,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e), "detail": traceback.format_exc()}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Switch Model",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def switch_model(
        category: str,
        model_id: str,
        auto_download: bool = True,
    ) -> dict:
        try:
            category = _validate_category(category)
            model_id = _validate_model_id(model_id)
            cat = ModelCategory(category)
            r = modelsOrchestrator.switch(cat, model_id, auto_download=auto_download)
            return {
                "success": r.success,
                "data": {
                    "model_id": r.model_id,
                    "category": r.category,
                    "message": r.message,
                    "error": r.error,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e), "detail": traceback.format_exc()}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Recommend Models",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def recommend_models() -> dict:
        try:
            recs = modelsOrchestrator.get_recommendations()
            return {
                "success": True,
                "data": {
                    "embedding": recs.embedding,
                    "nlp": recs.nlp,
                    "ocr": recs.ocr,
                    "reranker": recs.reranker,
                    "vision": recs.vision,
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e), "detail": traceback.format_exc()}
