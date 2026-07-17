"""
Log history query and management command handlers for darwin-admin.

Commands: query, sessions, session-info, stats, purge, purge-session,
          export, compact, set-retention
"""

from src.admin.lib import (
    LogAction,
    ns_to,
    LogsQueryArgs,
    LogsSessionsArgs,
    LogsSessionInfoArgs,
    LogsStatsArgs,
    LogsPurgeArgs,
    LogsPurgeSessionArgs,
    LogsExportArgs,
    LogsCompactArgs,
    LogsSetRetentionArgs,
)
from src.admin.renderer import _r


def cmd_logs_query(args: LogsQueryArgs) -> None:
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    logs = LoggerOrchestrator.get_logs(
        session=args.session, levels=args.levels, limit=args.limit,
    )
    if not logs:
        _r.warn("No logs found matching filters.")
        return
    _r.header(f"Logs ({len(logs)} entries)")
    for l in logs:
        icon = {"info": ">", "warning": "!", "error": "x", "success": "OK"}.get(
            l["log_type"], "?"
        )
        session_tag = f"[{l['session']}] " if l.get("session") else ""
        msg = l["message"][:120]
        _r._out(f"  {icon} {session_tag}{msg}")
    _r._out(f"\n  {_r._dim(f'{len(logs)} log(s)')}")


def cmd_logs_sessions(args: LogsSessionsArgs) -> None:
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    sessions = LoggerOrchestrator.get_sessions()
    if not sessions:
        _r.warn("No sessions found.")
        return
    rows = []
    for s in sessions:
        info = LoggerOrchestrator.get_session_info(s)
        rows.append(
            (
                s,
                info["total_logs"],
                info["errors"],
                info["warnings"],
                f"{info['duration_seconds']}s"
                if info.get("duration_seconds")
                else "running",
            )
        )
    _r.table(
        "Sessions", ["Session", "Logs", "Errors", "Warnings", "Duration"], rows
    )


def cmd_logs_session_info(args: LogsSessionInfoArgs) -> None:
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    info = LoggerOrchestrator.get_session_info(args.session_name)
    _r.panel(f"Session: {args.session_name}", str(info))


def cmd_logs_stats(args: LogsStatsArgs) -> None:
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    stats = LoggerOrchestrator.history_stats()
    _r.panel("Log History Stats", str(stats))


def cmd_logs_purge(args: LogsPurgeArgs) -> None:
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    count = LoggerOrchestrator.purge_logs(days=args.days)
    _r.ok(f"Purged {count} log(s) older than {args.days} day(s)")


def cmd_logs_purge_session(args: LogsPurgeSessionArgs) -> None:
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    count = LoggerOrchestrator.purge_session(args.session_name)
    _r.ok(f"Purged {count} log(s) for session '{args.session_name}'")


def cmd_logs_export(args: LogsExportArgs) -> None:
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    count = LoggerOrchestrator.export_logs(args.output, session=args.session)
    _r.ok(f"Exported {count} log(s) to {args.output}")


def cmd_logs_compact(args: LogsCompactArgs) -> None:
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    result = LoggerOrchestrator.compact()
    _r.panel("Compact Result", str(result))


def cmd_logs_set_retention(args: LogsSetRetentionArgs) -> None:
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    LoggerOrchestrator.set_retention(days=args.days)
    _r.ok(f"Retention set to {args.days} day(s)")


def add_args(sub) -> None:
    """Register logs subcommands."""
    p = sub.add_parser("logs", help="Log history query and management")
    sp = p.add_subparsers(dest="logs_action", required=True)

    q = sp.add_parser("query", help="Query logs with optional filters")
    q.add_argument("--session", "-s", help="Filter by session name")
    q.add_argument("--levels", "-l", help="Comma-separated: info,warning,error,success")
    q.add_argument("--limit", type=int, default=50, help="Max logs to return")
    q.set_defaults(func=lambda a: cmd_logs_query(ns_to(LogsQueryArgs, a)))

    ss = sp.add_parser("sessions", help="List all sessions")
    ss.set_defaults(func=lambda a: cmd_logs_sessions(ns_to(LogsSessionsArgs, a)))

    si = sp.add_parser("session-info", help="Show detailed session info")
    si.add_argument("session_name", help="Session name")
    si.set_defaults(func=lambda a: cmd_logs_session_info(ns_to(LogsSessionInfoArgs, a)))

    st = sp.add_parser("stats", help="Show history statistics")
    st.set_defaults(func=lambda a: cmd_logs_stats(ns_to(LogsStatsArgs, a)))

    pu = sp.add_parser("purge", help="Purge logs older than N days (admin)")
    pu.add_argument(
        "--days", type=int, default=14, help="Delete logs older than N days"
    )
    pu.set_defaults(func=lambda a: cmd_logs_purge(ns_to(LogsPurgeArgs, a)))

    ps = sp.add_parser("purge-session", help="Purge all logs for a session (admin)")
    ps.add_argument("session_name", help="Session to purge")
    ps.set_defaults(func=lambda a: cmd_logs_purge_session(ns_to(LogsPurgeSessionArgs, a)))

    ex = sp.add_parser("export", help="Export logs to JSONL file (admin)")
    ex.add_argument("--output", "-o", required=True, help="Output file path (.jsonl)")
    ex.add_argument("--session", "-s", help="Filter by session")
    ex.set_defaults(func=lambda a: cmd_logs_export(ns_to(LogsExportArgs, a)))

    co = sp.add_parser("compact", help="VACUUM SQLite database (admin)")
    co.set_defaults(func=lambda a: cmd_logs_compact(ns_to(LogsCompactArgs, a)))

    rt = sp.add_parser("set-retention", help="Change log retention in days (admin)")
    rt.add_argument("--days", type=int, default=14, help="Retention period in days")
    rt.set_defaults(func=lambda a: cmd_logs_set_retention(ns_to(LogsSetRetentionArgs, a)))
