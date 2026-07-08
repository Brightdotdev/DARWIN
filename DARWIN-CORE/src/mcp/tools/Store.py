import traceback


def register(mcp, store):
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
