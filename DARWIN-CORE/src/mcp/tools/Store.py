import traceback

from src.core.Orchestrators.StoreOrchestrator import StoreOrchestrator


def register(mcp, store):
    @mcp.tool()
    def create_store(
        path: str,
        name: str | None = None,
    ) -> dict:
        """Create a new DarwinStore at the given path for isolating data.

        Use this to separate work, research, or personal documents into their
        own store. Returns store_id — pass it to run_pipeline or search_darwin
        to operate on this store specifically.

        Args:
            path: Writable filesystem path where the store will live.
            name: Optional human-readable label (defaults to directory name).
        """
        try:
            meta = StoreOrchestrator.create_store(path, name)
            return {"success": True, "data": meta}
        except Exception as e:
            return {"success": False, "error": str(e), "detail": traceback.format_exc()}

    @mcp.tool()
    def list_files() -> dict:
        try:
            report = store.get_pipeline_report()
            return {"success": True, "data": report}
        except Exception as e:
            return {"success": False, "error": str(e), "detail": traceback.format_exc()}

    @mcp.tool()
    def file_status(filename: str) -> dict:
        try:
            status = store.file_status(filename)
            return {"success": True, "data": status}
        except Exception as e:
            return {"success": False, "error": str(e), "detail": traceback.format_exc()}
