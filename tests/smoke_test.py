"""
End-to-end smoke test for Darwin.
Verifies: CLI, init, store, models, search, pipeline orchestration, MCP.

Run:  python -m pytest tests/smoke_test.py -v
"""

import sys
import json
from pathlib import Path

import pytest


# ── Helpers ──




def _cli(*args, expect_fail=False):
    """Run darwin CLI with args, return (exit_code, stdout, stderr)."""
    import io
    from src.cli import main

    old_argv = sys.argv
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.argv = ["darwin"] + list(args)
    out = io.StringIO()
    err = io.StringIO()
    sys.stdout = out
    sys.stderr = err
    try:
        main()
        code = 0
    except SystemExit as e:
        code = e.code or 0
    except Exception as e:
        err.write(f"{type(e).__name__}: {e}")
        code = 1
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    if not expect_fail and code != 0:
        pytest.fail(f"CLI {' '.join(args)} exited {code}: {err.getvalue()}")
    return code, out.getvalue(), err.getvalue()


def _admin(*args, expect_fail=False):
    """Run darwin-admin CLI with args, return (exit_code, stdout, stderr)."""
    import io
    from src.admin.cli import main

    old_argv = sys.argv
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.argv = ["darwin-admin"] + list(args)
    out = io.StringIO()
    err = io.StringIO()
    sys.stdout = out
    sys.stderr = err
    try:
        main()
        code = 0
    except SystemExit as e:
        code = e.code or 0
    except Exception as e:
        err.write(f"{type(e).__name__}: {e}")
        code = 1
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    if not expect_fail and code != 0:
        pytest.fail(f"Admin CLI {' '.join(args)} exited {code}: {err.getvalue()}")
    return code, out.getvalue(), err.getvalue()


# ── Tests ──


class TestCLI:
    def test_help(self):
        code, out, err = _cli("--help")
        assert "darwin" in out or "usage" in out

    def test_config_help(self):
        code, out, err = _cli("config", "--help")
        assert code == 0
        assert "claude" in out or "config" in out


class TestStore:
    def test_list_files(self):
        code, out, err = _admin("store", "files")
        assert code == 0

    def test_store_status(self):
        code, out, err = _admin("store", "status")
        assert code == 0


class TestModels:
    def test_models_status(self):
        code, out, err = _admin("models", "status")
        assert code == 0

    def test_models_list(self):
        code, out, err = _admin("models", "list")
        assert code == 0
        # Should contain known models
        assert "intfloat/multilingual-e5-small" in out or "en_core_web" in out

    def test_system_info(self):
        code, out, err = _admin("system", "info")
        assert code == 0
        assert "RAM" in out
        assert "Tier" in out


class TestSearch:
    def test_search_empty(self):
        """Search with no ingested docs should return empty gracefully."""
        from src.core.Storage.DarwinStore import DarwinStore
        from src.core.Orchestrators.RetrievalOrchestrator import RetrievalOrchestrator
        from src.core.Orchestrators.lib import RetrievalParameters

        store = DarwinStore()
        store.bootstrap()
        orch = RetrievalOrchestrator(store=store)
        params = RetrievalParameters(query="test", top_k=3)
        ctx = orch.retrieve_and_resolve(params)
        assert ctx is not None
        assert ctx.total_passages >= 0  # 0 is valid with no data


class TestMCP:
    @staticmethod
    def _mcp():
        from src.mcp.app_factory import create_mcp_app
        mcp, _orch = create_mcp_app()
        return mcp

    def test_mcp_imports(self):
        """MCP module initialises without crashing."""
        mcp = self._mcp()
        assert mcp.name == "darwin-mcp"

    def test_mcp_7_tools(self):
        mcp = self._mcp()
        tools = mcp._tool_manager._tools
        assert len(tools) >= 7

    def test_mcp_create_store_tool(self):
        """create_store tool exists and produces store metadata."""
        mcp = self._mcp()
        tool = mcp._tool_manager._tools.get("create_store")
        assert tool is not None
        assert "path" in tool.parameters["properties"]
        assert tool.parameters["properties"]["path"]["type"] == "string"
        from src.core.Orchestrators.StoreOrchestrator import StoreOrchestrator
        test_dir = Path("C:\\Users\\ADMIN\\AppData\\Local\\Temp\\opencode\\test_create_store")
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            meta = StoreOrchestrator.create_store(str(test_dir), name="test_create")
            assert "store_id" in meta
            assert "store_name" in meta
            assert meta["store_name"] == "test_create"
            assert "darwin_dir" in meta
            assert "created_at" in meta
            assert meta["status"] == "created"
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_mcp_search_tool_schema(self):
        """Search tool accepts expected params."""
        mcp = self._mcp()
        tool = mcp._tool_manager._tools.get("search_darwin")
        assert tool is not None
        assert "query" in tool.parameters["properties"]
        assert "method" in tool.parameters["properties"]


class TestPipeline:
    def test_pipeline_schema(self):
        """Admin pipeline commands parse correctly."""
        code, out, err = _admin("pipeline", "--help")
        assert code == 0
        for cmd in ["run", "ingest", "index", "purge"]:
            assert cmd in out


class TestEdgeCases:
    def test_unknown_command(self):
        code, out, err = _cli("nonexistent", expect_fail=True)
        assert code != 0

    def test_admin_standalone(self):
        """Admin CLI runs standalone."""
        code, out, err = _admin("system", "info")
        assert code == 0
        assert "RAM" in out
