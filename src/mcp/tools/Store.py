import re
import traceback
from typing import Optional

from mcp.types import ToolAnnotations

from src.core.Orchestrators.StoreOrchestrator import StoreOrchestrator

_MAX_PATH_LEN = 2048
_MAX_NAME_LEN = 256


def _validate_path(path: str) -> str:
    path = path.strip()
    if not path:
        raise ValueError("path must not be empty")
    if len(path) > _MAX_PATH_LEN:
        raise ValueError(f"path exceeds {_MAX_PATH_LEN} chars")
    if ".." in path:
        raise ValueError("path must not contain '..'")
    if re.search(r'[<>"|`]', path):
        raise ValueError("path contains invalid characters")
    return path


def _validate_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    name = name.strip()
    if not name:
        return None
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"name exceeds {_MAX_NAME_LEN} chars")
    if re.search(r'[<>"|`;]', name):
        raise ValueError("name contains invalid characters")
    return name


def _validate_filename(filename: str) -> str:
    filename = filename.strip()
    if not filename:
        raise ValueError("filename must not be empty")
    if len(filename) > 2048:
        raise ValueError("filename exceeds 2048 chars")
    if ".." in filename:
        raise ValueError("filename must not contain '..'")
    return filename


def register(mcp, store):
    @mcp.tool(
        annotations=ToolAnnotations(
            title="Create Store",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def create_store(
        path: str,
        name: Optional[str] = None,
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
            path = _validate_path(path)
            name = _validate_name(name)
            meta = StoreOrchestrator.create_store(path, name)
            return {"success": True, "data": meta}
        except Exception as e:
            return {"success": False, "error": str(e), "detail": traceback.format_exc()}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="List Files",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def list_files() -> dict:
        try:
            report = store.get_pipeline_report()
            return {"success": True, "data": report}
        except Exception as e:
            return {"success": False, "error": str(e), "detail": traceback.format_exc()}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="File Status",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    def file_status(filename: str) -> dict:
        try:
            filename = _validate_filename(filename)
            status = store.file_status(filename)
            return {"success": True, "data": status}
        except Exception as e:
            return {"success": False, "error": str(e), "detail": traceback.format_exc()}
