"""
System info and status overview command handlers for darwin-admin.

Commands: system info, status (top-level)
"""

from src.admin.renderer import _r


def cmd_system_info(args, _get_models):
    models = _get_models(init_models=False)
    system = models.detect_system()
    _r.header("System Info")
    _r.table(
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

    recs = models.get_recommendations(system)
    _r.table(
        "Recommended Models",
        ["Category", "Model ID"],
        [
            ("embedding", recs.embedding or _r._dim("none")),
            ("nlp", recs.nlp or _r._dim("none")),
            ("ocr", recs.ocr or _r._dim("none")),
            ("reranker", recs.reranker or _r._dim("none")),
            ("vision", recs.vision or _r._dim("none")),
        ],
    )


def cmd_status(args):
    """Comprehensive overview: store files, installed models, preferences."""
    from src.core.Orchestrators.StoreOrchestrator import StoreOrchestrator
    from src.core.Orchestrators.lib import resolve_store

    if args.store:
        ds = resolve_store(args.store)
        orch = StoreOrchestrator(ds.darwin_dir, ds.store_name)
    else:
        orch = StoreOrchestrator.get_default() or StoreOrchestrator()

    init_models = getattr(args, "initialize", False)
    orch.bootstrap(initialize_models=init_models)


def add_args(sub, _get_models):
    """Register system subcommands."""
    p = sub.add_parser("system", help="System info and diagnostics")
    p.add_argument(
        "action", nargs="?", default="info", choices=["info"], help="Show system info"
    )
    p.set_defaults(func=lambda a: cmd_system_info(a, _get_models))


def add_status_args(sub):
    """Register the top-level 'status' command."""
    sp = sub.add_parser("status", help="Overview: store files, installed models, preferences")
    sp.add_argument("--store", help="Store ID, name, or path")
    sp.add_argument("--initialize", action="store_true", help="Also initialize models (slow)")
    sp.set_defaults(func=cmd_status)
