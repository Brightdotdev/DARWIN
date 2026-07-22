import re
from typing import Optional

from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations

from src.core.Orchestrators.StoreOrchestrator import StoreOrchestrator
from src.core.Orchestrators.lib import resolve_store
from src.core.Logger.DarwinLogger import darwin
from src.mcp.logging_bridge import darwin_bridge

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
    default_store = store  # capture for list_files closure
    @mcp.tool(
        annotations=ToolAnnotations(
            title="Create Store",
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def create_store(
        ctx: Context,
        path: str,
        name: Optional[str] = None,
    ) -> dict:
        """Create a new DarwinStore at the given path for isolating data.

        Usage tips:
        - Use this to create project-specific stores before running pipeline.
        - Pass the returned store ID to other tools via the 'store' parameter.
        - name is optional; if omitted, the store is identified by path only.
        """
        async with darwin_bridge(ctx):
            try:
                path = _validate_path(path)
                name = _validate_name(name)

                import asyncio
                meta = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: StoreOrchestrator.create_store(path, name)
                )
                return {"success": True, "data": meta}
            except Exception as e:
                darwin.error(f"create_store failed: {type(e).__name__}: {e}")
                return {"success": False, "error": str(e)}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="List Files",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def list_files(ctx: Context, filename: Optional[str] = None, store: Optional[str] = None) -> dict:
        """List all files tracked by a store with their pipeline status.

        This is the primary tool for checking pipeline status.

        Usage tips:
        - Returns the pipeline report — each file shows completed stages.
        - Pass filename to filter to a single file.
        - Pass store (name/path/ID) to list files from a specific store.
        - Use this to find exact filenames before running purge_artifacts.
        - Call without arguments to see all files from the default store.
        """
        async with darwin_bridge(ctx):
            try:
                target_store = resolve_store(store) if store else default_store
                import asyncio
                report = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: target_store.get_pipeline_report(filename)
                )
                return {"success": True, "data": report}
            except Exception as e:
                darwin.error(f"list_files failed: {type(e).__name__}: {e}")
                return {"success": False, "error": str(e)}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="File Status",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def file_status(ctx: Context, filename: str, store: Optional[str] = None) -> dict:
        """Get detailed status for a single file across all pipeline stages.

        Usage tips:
        - Use this to check what artifacts exist for a specific file.
        - filename must match the original filename used during ingestion.
        - Pass store (name/path/ID) to check a file in a specific store.
        """
        async with darwin_bridge(ctx):
            try:
                filename = _validate_filename(filename)
                target_store = resolve_store(store) if store else default_store
                import asyncio
                status = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: target_store.file_status(filename)
                )
                return {"success": True, "data": status}
            except Exception as e:
                darwin.error(f"file_status failed: {type(e).__name__}: {e}")
                return {"success": False, "error": str(e)}
