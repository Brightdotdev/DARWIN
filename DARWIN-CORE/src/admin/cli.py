"""
darwin admin — Full orchestrator control for power users.
All knobs, all dials. Wraps the four orchestrators.

Usage:
    python -m src.admin setup --interactive
    python -m src.admin models status
    python -m src.admin store files
    python -m src.admin pipeline run --target ./docs
"""

import argparse
import sys
import traceback

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# ── Lazy imports (orchestrators init on first use) ──

_store = None
_retrieval = None
_pipeline = None
_models = None

console = Console()


def _get_store(store_ref: str | None = None):
    global _store
    if _store is None or store_ref is not None:
        from src.core.Storage.DarwinStore import DarwinStore
        from src.core.Orchestrators.lib import resolve_store

        resolved = resolve_store(store_ref)
        resolved.bootstrap()
        if store_ref is None:
            _store = resolved
        return resolved
    return _store


def _get_pipeline(store_ref: str | None = None):
    global _pipeline
    if _pipeline is None or store_ref is not None:
        from src.core.Orchestrators.PipelineOrchestrator import PipelineOrchestrator

        _pipeline = PipelineOrchestrator(store=_get_store(store_ref))
    return _pipeline


def _get_models():
    global _models
    if _models is None:
        from src.core.Orchestrators.ModelOrchestrator import modelsOrchestrator

        _models = modelsOrchestrator
        _models.status()
    return _models


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _print_table(title, columns, rows):
    t = Table(title=title, box=box.ROUNDED)
    for c in columns:
        t.add_column(c)
    for r in rows:
        t.add_row(*[str(v) for v in r])
    console.print(t)


def _ok(msg):
    console.print(f"[green]✓[/green] {msg}")


def _warn(msg):
    console.print(f"[yellow]⚠[/yellow] {msg}")


def _err(msg):
    console.print(f"[red]✗[/red] {msg}")


def _panel(title, content):
    console.print(Panel(content, title=title, border_style="blue"))


# ══════════════════════════════════════════════════════════════
# SETUP / ONBOARDING
# ══════════════════════════════════════════════════════════════


def _interactive_select(prompt, options, default=None):
    """Numbered interactive selector (no extra deps)."""
    console.print(f"\n[bold]{prompt}[/bold]")
    for i, (key, desc) in enumerate(options, 1):
        console.print(f"  {i}. {key}  [dim]{desc}[/dim]")
    while True:
        raw = input(
            f"Enter number [1-{len(options)}]"
            + (f" (default: 1)" if default else "")
            + ": "
        ).strip()
        if not raw and default is not None:
            return options[default - 1][0]
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        except ValueError:
            pass
        console.print("[red]Invalid selection.[/red]")


def cmd_setup_interactive(args):
    """Full interactive onboarding with numbered menus."""
    models = _get_models()
    console.print(Panel.fit("[bold blue]Darwin Interactive Setup[/bold blue]"))

    # 1. Detect system
    console.print("\n[bold]Step 1:[/bold] Detecting system capabilities...")
    system = models.detect_system()
    _print_table(
        "System",
        ["Property", "Value"],
        [
            ("RAM", f"{system.ram_gb} GB"),
            ("Disk Free", f"{system.disk_free_gb} GB"),
            ("CPU Cores", str(system.cpu_cores)),
            ("GPU", system.gpu_name if system.has_gpu else "None"),
            ("Tier", system.tier),
        ],
    )

    # 2. Show recommendations
    recs = models.get_recommendations(system)
    console.print(f"\n[bold]Recommended models for your system:[/bold]")
    choices = {}
    cat_labels = [
        ("embedding", "Vector embeddings for semantic search (required)"),
        ("nlp", "Natural language processing / NER / summaries"),
        ("ocr", "Optical character recognition for scanned PDFs"),
        ("reranker", "Re-rank search results for precision"),
        ("vision", "Image analysis and captioning"),
    ]
    for cat, desc in cat_labels:
        recommended = getattr(recs, cat, None)
        if recommended:
            console.print(f"  [green]{cat}[/green]: {recommended}  [dim]— {desc}[/dim]")
            choices[cat] = recommended
        else:
            console.print(f"  [dim]{cat}: none available[/dim]")

    # 3. Confirm which to install
    console.print(
        "\n[bold]Step 2:[/bold] Select models to install (comma-separated, or 'all'):"
    )
    for cat, mid in choices.items():
        console.print(f"  [cyan]{cat}[/cyan]: {mid}")
    raw = input("\nModels to install [all]: ").strip().lower()
    if raw == "" or raw == "all":
        selected = choices
    else:
        selected_cats = [c.strip() for c in raw.split(",")]
        selected = {c: choices[c] for c in selected_cats if c in choices}

    if not selected:
        _warn("No models selected. Skipping installation.")
        return

    # 4. API keys for provider-based models
    all_specs = models.list_registered()
    api_models = [s for s in all_specs if s.get("requires_provider")]
    needed_providers = set()
    for cat, mid in selected.items():
        for s in all_specs:
            if s["model_id"] == mid and s.get("requires_provider"):
                needed_providers.add(s["requires_provider"])

    if needed_providers:
        console.print(
            f"\n[bold]Step 3:[/bold] API key(s) needed for: {', '.join(needed_providers)}"
        )
        for prov in needed_providers:
            key = input(f"  Enter {prov} API key (or leave blank to skip): ").strip()
            if key:
                models.set_key(prov, key)
                _ok(f"{prov} key set")
            else:
                _warn(f"Skipped {prov} key")

    # 5. Install
    console.print(f"\n[bold]Step 4:[/bold] Installing {len(selected)} model(s)...")
    from src.core.Orchestrators.ModelOrchestrator import ModelCategory

    cat_map = {c.value.lower(): c for c in ModelCategory}
    for cat_str, mid in selected.items():
        cat_enum = cat_map.get(cat_str)
        if cat_enum:
            console.print(f"  Installing {mid} for {cat_str}...")
            result = models.install(mid)
            if result.success:
                _ok(f"{mid} installed")
            else:
                _err(f"{mid}: {result.error}")

    console.print("\n[bold green]Setup complete![/bold green]")
    console.print("Start the MCP server:  [cyan]python -m src.mcp[/cyan]")


def cmd_setup_profile(args):
    """Non-interactive profile-based setup."""
    models = _get_models()
    results = models.setup_profile(
        profile=args.profile, auto_download=not args.no_download
    )
    for cat, res in results.items():
        if res.success:
            _ok(f"{cat.value}: {res.model_id}")
        else:
            _err(f"{cat.value}: {res.error or 'failed'}")


def cmd_setup_onboard(args):
    """Non-interactive onboarding with preference dict."""
    models = _get_models()
    if args.preferences:
        prefs = dict(kv.split("=") for kv in args.preferences)
    else:
        prefs = None
    report = models.onboard(preferences=prefs, auto_download=not args.no_download)
    summary = report.get("summary", {})
    for cat in summary.get("ready", []):
        _ok(f"{cat} ready")
    for cat, err in summary.get("failed", {}).items():
        _err(f"{cat}: {err}")


# ══════════════════════════════════════════════════════════════
# MODEL COMMANDS
# ══════════════════════════════════════════════════════════════


def cmd_models_status(args):
    models = _get_models()
    status = models.status()
    console.print(Panel.fit(f"[bold]Model Status[/bold]  —  {status.summary}"))
    rows = []
    for name, cs in status.categories.items():
        rows.append(
            (
                name,
                cs.active_model_id or "[dim]none[/dim]",
                "[green]ready[/green]" if cs.available else "[red]missing[/red]",
                cs.error or "",
            )
        )
    _print_table("Categories", ["Category", "Active Model", "Status", "Error"], rows)


def cmd_models_list(args):
    models = _get_models()
    specs = models.list_registered()
    if args.category:
        specs = [s for s in specs if s["category"] == args.category]
    if args.provider:
        specs = [s for s in specs if s.get("requires_provider") == args.provider]

    rows = []
    for s in specs:
        rows.append(
            (
                s["model_id"],
                s["category"],
                s["tier"].lower(),
                "[green]local[/green]" if s["is_local"] else "[blue]api[/blue]",
                s.get("requires_provider") or "[dim]—[/dim]",
                f"{s['disk_mb']}MB" if s["disk_mb"] else "[dim]—[/dim]",
            )
        )
    _print_table(
        "Registered Models",
        ["Model ID", "Category", "Tier", "Type", "Provider", "Size"],
        rows,
    )


def cmd_models_install(args):
    models = _get_models()
    for mid in args.model_ids:
        console.print(f"Installing {mid}...")
        result = models.install(
            mid, auto_download=not args.no_download, force=args.force
        )
        if result.success:
            _ok(f"{mid} installed")
        else:
            _err(f"{mid}: {result.error}")


def cmd_models_uninstall(args):
    models = _get_models()
    for mid in args.model_ids:
        result = models.uninstall(mid, delete_files=not args.keep_files)
        _ok(
            f"{mid}: {result.message}"
            if result.success
            else _err(f"{mid}: {result.error}")
        )


def cmd_models_switch(args):
    models = _get_models()
    from src.core.Orchestrators.ModelOrchestrator import ModelCategory

    cat = ModelCategory(args.category)
    result = models.switch(cat, args.model_id, auto_download=not args.no_download)
    if result.success:
        _ok(f"Switched {args.category} → {args.model_id}")
    else:
        _err(f"Switch failed: {result.error}")


def cmd_models_keys(args):
    models = _get_models()
    if args.action == "set":
        ok = models.set_key(args.provider, args.key)
        _ok(f"{args.provider} key set" if ok else _err("Failed to set key"))
    elif args.action == "status":
        statuses = models.list_key_status()
        rows = [
            (p, "[green]set[/green]" if s else "[red]missing[/red]")
            for p, s in statuses.items()
        ]
        _print_table("API Keys", ["Provider", "Status"], rows)
    elif args.action == "remove":
        ok = models.remove_key(args.provider)
        _ok(f"{args.provider} key removed" if ok else _err("Failed to remove key"))


# ══════════════════════════════════════════════════════════════
# STORE COMMANDS
# ══════════════════════════════════════════════════════════════


def cmd_store_status(args):
    store = _get_store()
    s = store.status()
    from rich.tree import Tree

    t = Tree("[bold]Store Status[/bold]")
    t.add(f"Name: {s.get('name', '?')}")
    t.add(f"Path: {s.get('path', '?')}")
    t.add(f"Sources: {s.get('source_count', 0)}")
    t.add(f"Pipeline stages: {', '.join(s.get('pipeline_stages', []))}")
    t.add(f"Searchable files: {s.get('searchable_count', 0)}")
    console.print(t)


def cmd_store_files(args):
    store = _get_store()
    report = store.get_pipeline_report()
    if isinstance(report, dict):
        report = [report]
    if not report:
        _warn("No files in store.")
        return
    rows = []
    for r in report:
        rows.append(
            (
                r.get("filename", r.get("display_name", "?")),
                r.get("stage", r.get("pipeline_stage", "?")),
                "[green]ready[/green]"
                if r.get("search_ready")
                else "[dim]pending[/dim]",
                r.get("file_size", ""),
            )
        )
    _print_table("Files", ["File", "Stage", "Search Ready", "Size"], rows)


def cmd_store_file(args):
    store = _get_store()
    s = store.file_status(args.filename)
    _panel(f"File: {args.filename}", str(s))


def cmd_store_audit(args):
    store = _get_store()
    a = store.full_audit()
    _panel("Full Audit", str(a))


def cmd_store_health(args):
    store = _get_store()
    if args.model:
        h = store.get_embedding_health(args.model)
        _panel(f"Embedding Health: {args.model}", str(h))
    else:
        h = store.full_registry_health_check()
        _panel("Registry Health", str(h))


def cmd_store_repair(args):
    store = _get_store()
    r = store.repair(dry_run=args.dry_run)
    _panel("Repair Report", str(r))


# ══════════════════════════════════════════════════════════════
# PIPELINE COMMANDS
# ══════════════════════════════════════════════════════════════


def _run_pipeline_stages(
    target,
    files,
    stages,
    ocr,
    recursive,
    precise,
    threshold,
    idf_smoothing,
    only_kw,
    only_semantic,
    rerun,
    store_ref=None,
):
    pipe = _get_pipeline(store_ref)
    from src.core.Orchestrators.lib import PipelineParameters as PipelineInput
    from src.core.Logger.DarwinLogger import darwin
    darwin.configure()

    params = PipelineInput(
        target=target or "",
        store=store_ref,
        stages=stages,
        files=[files] if files else None,
        ocr=ocr,
        recursive=recursive,
        precise=precise,
        rerun=rerun,
        threshold=threshold or 0.5,
        idf_smoothing=idf_smoothing or 1.5,
        only_kw=only_kw,
        only_semantic=only_semantic,
    )
    console.print(f"Running pipeline stages: {stages}")
    result = pipe.build_pipeline(params)
    if result.success:
        _ok(f"{', '.join(s.value for s in result.stages_completed)} complete")
    if result.stages_failed:
        for stage, err in result.stages_failed.items():
            _err(f"{stage}: {err}")
    console.print(result.summary)


def cmd_pipeline_run(args):
    _run_pipeline_stages(
        args.target,
        args.files,
        ["ingestion", "indexing"],
        args.ocr,
        args.recursive,
        args.precise,
        args.threshold,
        args.idf_smoothing,
        args.only_kw,
        args.only_semantic,
        args.rerun,
        store_ref=getattr(args, "store", None),
    )


def cmd_pipeline_ingest(args):
    _run_pipeline_stages(
        args.target,
        args.files,
        ["ingestion"],
        args.ocr,
        args.recursive,
        args.precise,
        None,
        None,
        False,
        False,
        args.rerun,
        store_ref=getattr(args, "store", None),
    )


def cmd_pipeline_index(args):
    _run_pipeline_stages(
        None,
        args.files,
        ["indexing"],
        False,
        False,
        False,
        args.threshold,
        args.idf_smoothing,
        args.only_kw,
        args.only_semantic,
        args.rerun,
    )


def cmd_pipeline_purge(args):
    pipe = _get_pipeline()
    pipe.purge(args.stages, args.filenames)
    _ok(f"Purged {', '.join(args.stages)} for {len(args.filenames)} file(s)")


# ══════════════════════════════════════════════════════════════
# LOGS COMMANDS
# ══════════════════════════════════════════════════════════════


def cmd_logs_query(args):
    """Query logs with optional filters."""
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    logs = LoggerOrchestrator.get_logs(
        session=args.session,
        levels=args.levels,
        limit=args.limit,
    )
    if not logs:
        _warn("No logs found matching filters.")
        return
    for l in logs:
        icon = {"info": "ℹ", "warning": "⚠", "error": "✗", "success": "✓"}.get(
            l["log_type"], "?"
        )
        ts = l.get("datetime", "").split(".")[0] if l.get("datetime") else ""
        session_tag = f"[dim][{l['session']}][/dim] " if l.get("session") else ""
        console.print(f"  {icon} {session_tag}[bold]{l['message'][:120]}[/bold]")
    console.print(f"\n[dim]{len(logs)} log(s)[/dim]")


def cmd_logs_sessions(args):
    """List all sessions."""
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    sessions = LoggerOrchestrator.get_sessions()
    if not sessions:
        _warn("No sessions found.")
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
    _print_table(
        "Sessions", ["Session", "Logs", "Errors", "Warnings", "Duration"], rows
    )


def cmd_logs_session_info(args):
    """Show detailed info for a session."""
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    info = LoggerOrchestrator.get_session_info(args.session_name)
    _panel(f"Session: {args.session_name}", str(info))


def cmd_logs_stats(args):
    """Show history statistics."""
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    stats = LoggerOrchestrator.history_stats()
    _panel("Log History Stats", str(stats))


def cmd_logs_purge(args):
    """Purge logs older than N days."""
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    count = LoggerOrchestrator.purge_logs(days=args.days)
    _ok(f"Purged {count} log(s) older than {args.days} day(s)")


def cmd_logs_purge_session(args):
    """Purge all logs for a session."""
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    count = LoggerOrchestrator.purge_session(args.session_name)
    _ok(f"Purged {count} log(s) for session '{args.session_name}'")


def cmd_logs_export(args):
    """Export logs to JSONL file."""
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    count = LoggerOrchestrator.export_logs(args.output, session=args.session)
    _ok(f"Exported {count} log(s) to {args.output}")


def cmd_logs_compact(args):
    """VACUUM the SQLite database."""
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    result = LoggerOrchestrator.compact()
    _panel("Compact Result", str(result))


def cmd_logs_set_retention(args):
    """Change log retention period."""
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    LoggerOrchestrator.set_retention(days=args.days)
    _ok(f"Retention set to {args.days} day(s)")


# ══════════════════════════════════════════════════════════════
# SYSTEM COMMANDS
# ══════════════════════════════════════════════════════════════


def cmd_system_info(args):
    models = _get_models()
    system = models.detect_system()
    rows = [
        ("RAM", f"{system.ram_gb} GB"),
        ("Disk Free", f"{system.disk_free_gb} GB"),
        ("CPU Cores", str(system.cpu_cores)),
        ("GPU", system.gpu_name if system.has_gpu else "None"),
        ("Tier", system.tier),
    ]
    _print_table("System Info", ["Property", "Value"], rows)

    recs = models.get_recommendations(system)
    _print_table(
        "Recommended Models",
        ["Category", "Model ID"],
        [
            ("embedding", recs.embedding or "[dim]none[/dim]"),
            ("nlp", recs.nlp or "[dim]none[/dim]"),
            ("ocr", recs.ocr or "[dim]none[/dim]"),
            ("reranker", recs.reranker or "[dim]none[/dim]"),
            ("vision", recs.vision or "[dim]none[/dim]"),
        ],
    )


def cmd_status(args):
    """Comprehensive overview: store files, installed models, preferences."""

    # ── Store files ──
    try:
        store = _get_store()
        report = store.get_pipeline_report()
        if isinstance(report, dict):
            report = [report]
        file_count = len(report) if report else 0
        ready_count = sum(
            1 for r in (report or []) if r.get("search_ready")
        )
        source_count = store.status().get("source_count", 0)
    except Exception:
        file_count = 0
        ready_count = 0
        source_count = 0

    # ── Models ──
    try:
        models = _get_models()
        model_status = models.status()
        installed = {}
        for name, cs in model_status.categories.items():
            if cs.active_model_id:
                installed[name] = cs.active_model_id
        installed_count = len(installed)
    except Exception:
        installed_count = 0
        installed = {}

    # ── Preferences (from config) ──
    try:
        from src.Setup import DarwinSetup

        setup = DarwinSetup()
        config = setup.load_config() or {}
        prefs = config.get("models", {})
    except Exception:
        prefs = {}

    # ── Display ──
    console.print()
    _panel(
        "Darwin Status Overview",
        f"[bold]Store:[/bold] {source_count} sources, {file_count} files ({ready_count} search-ready)\n"
        f"[bold]Models:[/bold] {installed_count} installed\n"
        f"[bold]Preferences:[/bold] {len(prefs)} configured",
    )

    if installed:
        _print_table(
            "Installed Models",
            ["Category", "Model ID"],
            [(cat, mid) for cat, mid in installed.items()],
        )

    if prefs:
        _print_table(
            "Model Preferences (config)",
            ["Category", "Model ID"],
            [(cat, mid) for cat, mid in prefs.items()],
        )

    if file_count:
        _print_table(
            "Store File Summary",
            ["Status", "Count"],
            [
                ("Total", str(file_count)),
                ("Search Ready", str(ready_count)),
                ("Pending", str(file_count - ready_count)),
            ],
        )


# ══════════════════════════════════════════════════════════════
# CLI PARSER
# ══════════════════════════════════════════════════════════════


def _add_model_args(sub):
    p = sub.add_parser("models", help="Model lifecycle management")
    sp = p.add_subparsers(dest="model_action", required=True)

    s = sp.add_parser("status", help="Show model status across categories")
    s.set_defaults(func=cmd_models_status)

    l = sp.add_parser("list", help="List registered models")
    l.add_argument("--category", "-c", help="Filter by category")
    l.add_argument("--provider", "-p", help="Filter by provider")
    l.set_defaults(func=cmd_models_list)

    i = sp.add_parser("install", help="Install one or more models")
    i.add_argument("model_ids", nargs="+", help="Model ID(s) to install")
    i.add_argument("--no-download", action="store_true", help="Skip auto-download")
    i.add_argument("--force", action="store_true", help="Force re-download")
    i.set_defaults(func=cmd_models_install)

    u = sp.add_parser("uninstall", help="Uninstall models")
    u.add_argument("model_ids", nargs="+", help="Model ID(s) to remove")
    u.add_argument("--keep-files", action="store_true", help="Keep files on disk")
    u.set_defaults(func=cmd_models_uninstall)

    sw = sp.add_parser("switch", help="Switch active model for a category")
    sw.add_argument("category", help="Category (embedding, nlp, ocr, reranker, vision)")
    sw.add_argument("model_id", help="Target model ID")
    sw.add_argument(
        "--no-download", action="store_true", help="Skip auto-download if missing"
    )
    sw.set_defaults(func=cmd_models_switch)

    k = sp.add_parser("keys", help="API key management")
    k.add_argument("action", choices=["set", "status", "remove"], help="Key operation")
    k.add_argument(
        "--provider", "-p", help="Provider name (openai, gemini, groq, etc.)"
    )
    k.add_argument("--key", "-k", help="API key value (required for 'set')")
    k.set_defaults(func=cmd_models_keys)


def _add_store_args(sub):
    p = sub.add_parser("store", help="Store introspection and maintenance")
    sp = p.add_subparsers(dest="store_action", required=True)

    s = sp.add_parser("status", help="Store overview")
    s.set_defaults(func=cmd_store_status)

    f = sp.add_parser("files", help="List all files with pipeline status")
    f.set_defaults(func=cmd_store_files)

    fi = sp.add_parser("file", help="Detailed status for one file")
    fi.add_argument("filename", help="File display name")
    fi.set_defaults(func=cmd_store_file)

    a = sp.add_parser("audit", help="Full store audit")
    a.set_defaults(func=cmd_store_audit)

    h = sp.add_parser("health", help="Health check / registry health")
    h.add_argument("--model", "-m", help="Embedding model ID for health check")
    h.set_defaults(func=cmd_store_health)

    r = sp.add_parser("repair", help="Repair store integrity")
    r.add_argument("--dry-run", action="store_true", help="Preview without changes")
    r.set_defaults(func=cmd_store_repair)


def _add_pipeline_args(sub):
    p = sub.add_parser("pipeline", help="Pipeline execution and maintenance")
    sp = p.add_subparsers(dest="pipeline_action", required=True)

    r = sp.add_parser("run", help="Full ingest + index pipeline")
    r.add_argument("target", help="Path or static URL to ingest")
    r.add_argument("--store", help="Store ID, name, or path (default: auto)")
    r.add_argument("--files", "-f", help="Specific files (comma-separated)")
    r.add_argument("--ocr", action="store_true", help="Enable OCR for scanned docs")
    r.add_argument("--recursive", action="store_true", help="Recursive directory scan")
    r.add_argument("--precise", action="store_true", help="Precision parsing mode")
    r.add_argument(
        "--threshold", type=float, default=0.5, help="Indexing threshold (default: 0.5)"
    )
    r.add_argument(
        "--idf-smoothing",
        type=float,
        default=1.5,
        help="BM25 IDF smoothing (default: 1.5)",
    )
    r.add_argument("--only-kw", action="store_true", help="Keyword-only indexing")
    r.add_argument(
        "--only-semantic", action="store_true", help="Semantic-only indexing"
    )
    r.add_argument("--rerun", action="store_true", help="Re-process even if done")
    r.set_defaults(func=cmd_pipeline_run)

    i = sp.add_parser("ingest", help="Ingestion only (parse + chunk)")
    i.add_argument("target", help="Path or static URL to ingest")
    i.add_argument("--store", help="Store ID, name, or path (default: auto)")
    i.add_argument("--files", "-f", help="Specific files")
    i.add_argument("--ocr", action="store_true")
    i.add_argument("--recursive", action="store_true")
    i.add_argument("--precise", action="store_true")
    i.add_argument("--rerun", action="store_true")
    i.set_defaults(func=cmd_pipeline_ingest)

    ix = sp.add_parser("index", help="Indexing only (BM25 + embeddings)")
    ix.add_argument("--files", "-f", help="Specific files to index")
    ix.add_argument("--threshold", type=float, default=0.5)
    ix.add_argument("--idf-smoothing", type=float, default=1.5)
    ix.add_argument("--only-kw", action="store_true")
    ix.add_argument("--only-semantic", action="store_true")
    ix.add_argument("--rerun", action="store_true")
    ix.set_defaults(func=cmd_pipeline_index)

    pu = sp.add_parser("purge", help="Purge pipeline artifacts for files")
    pu.add_argument("filenames", nargs="+", help="Files to purge")
    pu.add_argument(
        "--stages",
        nargs="+",
        default=["ingestion", "indexing"],
        choices=["ingestion", "indexing"],
        help="Stages to purge",
    )
    pu.set_defaults(func=cmd_pipeline_purge)


def _add_setup_args(sub):
    p = sub.add_parser("setup", help="Initial setup and onboarding")
    sp = p.add_subparsers(dest="setup_action", required=True)

    i = sp.add_parser("interactive", help="Interactive setup with guided menus")
    i.set_defaults(func=cmd_setup_interactive)

    prof = sp.add_parser("profile", help="Non-interactive profile-based setup")
    prof.add_argument(
        "--profile",
        default="required",
        choices=["required", "recommended"],
        help="Model profile to install",
    )
    prof.add_argument("--no-download", action="store_true", help="Skip auto-download")
    prof.set_defaults(func=cmd_setup_profile)

    o = sp.add_parser("onboard", help="Non-interactive onboard with preferences")
    o.add_argument(
        "--preferences",
        "-p",
        nargs="+",
        metavar="CAT=MODEL_ID",
        help="e.g. embedding=intfloat/multilingual-e5-small-onnx",
    )
    o.add_argument("--no-download", action="store_true")
    o.set_defaults(func=cmd_setup_onboard)


def _add_logs_args(sub):
    p = sub.add_parser("logs", help="Log history query and management")
    sp = p.add_subparsers(dest="logs_action", required=True)

    q = sp.add_parser("query", help="Query logs with optional filters")
    q.add_argument("--session", "-s", help="Filter by session name")
    q.add_argument("--levels", "-l", help="Comma-separated: info,warning,error,success")
    q.add_argument("--limit", type=int, default=50, help="Max logs to return")
    q.set_defaults(func=cmd_logs_query)

    ss = sp.add_parser("sessions", help="List all sessions")
    ss.set_defaults(func=cmd_logs_sessions)

    si = sp.add_parser("session-info", help="Show detailed session info")
    si.add_argument("session_name", help="Session name")
    si.set_defaults(func=cmd_logs_session_info)

    st = sp.add_parser("stats", help="Show history statistics")
    st.set_defaults(func=cmd_logs_stats)

    pu = sp.add_parser("purge", help="Purge logs older than N days (admin)")
    pu.add_argument(
        "--days", type=int, default=14, help="Delete logs older than N days"
    )
    pu.set_defaults(func=cmd_logs_purge)

    ps = sp.add_parser("purge-session", help="Purge all logs for a session (admin)")
    ps.add_argument("session_name", help="Session to purge")
    ps.set_defaults(func=cmd_logs_purge_session)

    ex = sp.add_parser("export", help="Export logs to JSONL file (admin)")
    ex.add_argument("--output", "-o", required=True, help="Output file path (.jsonl)")
    ex.add_argument("--session", "-s", help="Filter by session")
    ex.set_defaults(func=cmd_logs_export)

    co = sp.add_parser("compact", help="VACUUM SQLite database (admin)")
    co.set_defaults(func=cmd_logs_compact)

    rt = sp.add_parser("set-retention", help="Change log retention in days (admin)")
    rt.add_argument("--days", type=int, default=14, help="Retention period in days")
    rt.set_defaults(func=cmd_logs_set_retention)


def _add_system_args(sub):
    p = sub.add_parser("system", help="System info and diagnostics")
    p.add_argument(
        "action", nargs="?", default="info", choices=["info"], help="Show system info"
    )
    p.set_defaults(func=cmd_system_info)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        prog="darwin-admin",
        description="Darwin Admin — Full orchestrator control for power users",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python -m src.admin setup interactive
  python -m src.admin status
  python -m src.admin models status
  python -m src.admin models install intfloat/multilingual-e5-small-onnx
  python -m src.admin models keys set openai sk-...
  python -m src.admin store files
  python -m src.admin store audit
  python -m src.admin pipeline run ./docs --ocr
  python -m src.admin pipeline purge myfile.pdf --stages ingestion
  python -m src.admin system info
    """,
    )

    sub = parser.add_subparsers(dest="command", required=True)
    _add_setup_args(sub)
    _add_model_args(sub)
    _add_store_args(sub)
    _add_pipeline_args(sub)
    _add_logs_args(sub)
    _add_system_args(sub)

    # Top-level status
    sp = sub.add_parser("status", help="Overview: store files, installed models, preferences")
    sp.set_defaults(func=cmd_status)

    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as e:
        _err(f"{type(e).__name__}: {e}")
        if getattr(args, "verbose", False):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
