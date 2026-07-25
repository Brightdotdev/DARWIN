"""
Store introspection and maintenance command handlers for darwin-admin.

Commands: status, files, file, audit, health, repair
"""

from src.admin.lib import (
    ns_to,
    StoreStatusArgs,
    StoreFilesArgs,
    StoreFileArgs,
    StoreAuditArgs,
    StoreHealthArgs,
    StoreRepairArgs,
)
from src.admin.renderer import _r


def cmd_store_status(args: StoreStatusArgs, _get_store) -> None:
    store = _get_store()
    s = store.status()
    _r.header("Store Status")
    _r.tree("Store", [
        ("Name", s.get("name", "?")),
        ("Path", s.get("path", "?")),
        ("Sources", s.get("source_count", 0)),
        ("Pipeline stages", ", ".join(s.get("pipeline_stages", []))),
        ("Searchable files", s.get("searchable_count", 0)),
    ])


def cmd_store_files(args: StoreFilesArgs, _get_store) -> None:
    store = _get_store()
    report = store.get_pipeline_report()
    if isinstance(report, dict):
        report = [report]
    if not report:
        _r.warn("No files in store.")
        return

    def _truncate(text: str, max_len: int = 40) -> str:
        if not text or len(text) <= max_len:
            return text
        return text[:max_len - 1] + "\u2026"

    rows = []
    for r in report:
        search_ready = r.get("core_complete", False) and not r.get("missing")
        status = _r._green("ready") if search_ready else _r._dim("pending")

        completed = r.get("completed", [])
        missing = r.get("missing", [])
        stage_summary = f"{len(completed)}/{len(completed)+len(missing)} done"

        rows.append((
            _truncate(r.get("filename", "?"), 40),
            stage_summary,
            status,
            f"{r.get('progress_percent', 0)}%",
            str(r.get("chunk_count", 0)),
            _truncate(r.get("embedding_model", "N/A"), 20),
        ))

    _r.table(
        "Files in pipeline",
        ["File", "Stages", "Search Ready", "Progress", "Chunks", "Model"],
        rows,
    )


def cmd_store_file(args: StoreFileArgs, _get_store) -> None:
    store = _get_store()
    s = store.file_status(args.filename)
    if isinstance(s, dict):
        _r.tree(f"File: {args.filename}", [(k, v) for k, v in s.items()])
    else:
        _r.panel(f"File: {args.filename}", str(s))


def cmd_store_audit(args: StoreAuditArgs, _get_store) -> None:
    store = _get_store()
    a = store.full_audit()
    if isinstance(a, dict):
        _r.tree("Full Audit", [(k, v) for k, v in a.items()])
    else:
        _r.panel("Full Audit", str(a))


def cmd_store_health(args: StoreHealthArgs, _get_store) -> None:
    store = _get_store()
    if args.model:
        h = store.get_embedding_health(args.model)
        _r.panel(f"Embedding Health: {args.model}", str(h))
    else:
        h = store.full_registry_health_check()
        _r.panel("Registry Health", str(h))


def cmd_store_repair(args: StoreRepairArgs, _get_store) -> None:
    store = _get_store()
    r = store.repair(dry_run=args.dry_run)
    _r.panel("Repair Report", str(r))


def add_args(sub, _get_store) -> None:
    """Register store subcommands."""
    p = sub.add_parser("store", help="Store introspection and maintenance")
    sp = p.add_subparsers(dest="store_action", required=True)

    s = sp.add_parser("status", help="Store overview")
    s.set_defaults(func=lambda a: cmd_store_status(ns_to(StoreStatusArgs, a), _get_store))

    f = sp.add_parser("files", help="List all files with pipeline status")
    f.set_defaults(func=lambda a: cmd_store_files(ns_to(StoreFilesArgs, a), _get_store))

    fi = sp.add_parser("file", help="Detailed status for one file")
    fi.add_argument("filename", help="File display name")
    fi.set_defaults(func=lambda a: cmd_store_file(ns_to(StoreFileArgs, a), _get_store))

    a = sp.add_parser("audit", help="Full store audit")
    a.set_defaults(func=lambda a: cmd_store_audit(ns_to(StoreAuditArgs, a), _get_store))

    h = sp.add_parser("health", help="Health check / registry health")
    h.add_argument("--model", "-m", help="Embedding model ID for health check")
    h.set_defaults(func=lambda a: cmd_store_health(ns_to(StoreHealthArgs, a), _get_store))

    r = sp.add_parser("repair", help="Repair store integrity")
    r.add_argument("--dry-run", action="store_true", help="Preview without changes")
    r.set_defaults(func=lambda a: cmd_store_repair(ns_to(StoreRepairArgs, a), _get_store))
