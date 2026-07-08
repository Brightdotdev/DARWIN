import traceback
from typing import Optional

from src.core.Orchestrators.ModelOrchestrator import modelsOrchestrator, ModelCategory


def register(mcp):
    @mcp.tool()
    def model_status(category: Optional[str] = None) -> dict:
        try:
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

    @mcp.tool()
    def install_model(
        model_id: str,
        auto_download: bool = True,
        set_active: bool = True,
    ) -> dict:
        try:
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

    @mcp.tool()
    def switch_model(
        category: str,
        model_id: str,
        auto_download: bool = True,
    ) -> dict:
        try:
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

    @mcp.tool()
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
