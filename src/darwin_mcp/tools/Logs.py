"""
MCP tool — get_logs: Query Darwin session logs.

Sessions are the 4 production orchestrator sessions.
Agent never sees logtype.INFO — it's noise for AI consumers.
"""

from enum import Enum
from typing import Optional

from mcp.server.fastmcp import Context
from mcp.types import ToolAnnotations

from core.Logger.DarwinLogger import darwin, LogType
from core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator
from darwin_mcp.logging_bridge import darwin_bridge


class SessionName(str, Enum):
    """Production session names — the only sessions that exist in normal operation."""

    PIPELINE = "PIPELINE"
    RETRIEVAL = "RETRIEVAL"
    INGESTION = "Ingestion"
    INDEXING = "Indexing"


_AGENT_LEVELS_VERBOSE = {LogType.SUCCESS, LogType.WARNING, LogType.ERROR}
_AGENT_LEVELS_QUIET = {LogType.SUCCESS, LogType.ERROR}

_DEFAULT_LIMIT = 100
_MAX_LIMIT = 1000


def register(mcp):
    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Logs",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def get_logs(
        ctx: Context,
        session: Optional[str] = None,
        level: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """Query Darwin session logs (oldest first for LLM readability).

        Parameters:
          session: Filter by session name. Exact values:
                   "PIPELINE" — full pipeline orchestrator run
                   "RETRIEVAL" — search/retrieval orchestrator run
                   "Ingestion" — parsing + chunking stage
                   "Indexing" — BM25 + embedding stage
                   Omit for all sessions.
          level: Filter verbosity.
                   "verbose" — SUCCESS + WARNING + ERROR (default)
                   "quiet"   — SUCCESS + ERROR only
                   Omit for verbose.
          limit: Max log entries to return (1-1000, default 100).

        Notes:
          - logtype.INFO is never returned — it's noise for AI consumers.
          - Results are oldest-first (top to bottom chronological order).
          - Only the 4 production sessions exist. CLI one-shot sessions
            (parsing, chunking, etc.) are transient and not queryable.
        """
        async with darwin_bridge(ctx):
            try:
                _limit = limit or _DEFAULT_LIMIT
                if _limit < 1:
                    _limit = 1
                if _limit > _MAX_LIMIT:
                    _limit = _MAX_LIMIT

                if session is not None:
                    valid = [s.value for s in SessionName]
                    if session not in valid:
                        return {
                            "success": False,
                            "error": f"Invalid session: {session!r}. Valid: {valid}",
                        }

                if level == "quiet":
                    levels = _AGENT_LEVELS_QUIET
                else:
                    levels = _AGENT_LEVELS_VERBOSE

                logs = LoggerOrchestrator.get_logs(
                    session=session,
                    levels=",".join(lt.value for lt in levels),
                    limit=_limit,
                )

                logs.reverse()

                return {
                    "success": True,
                    "data": {
                        "logs": logs,
                        "total": len(logs),
                        "session": session,
                        "level": level or "verbose",
                    },
                }
            except Exception as e:
                darwin.error(f"get_logs failed: {type(e).__name__}: {e}")
                return {"success": False, "error": str(e)}
