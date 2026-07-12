"""
Model lifecycle command handlers for darwin-admin.

Commands: status, list, install, uninstall, switch, keys
"""

from src.admin.renderer import _r


def cmd_models_status(args, _get_models):
    models = _get_models(init_models=True)
    status = models.status()
    _r.header(f"Model Status — {status.summary}")
    rows = []
    for name, cs in status.categories.items():
        rows.append(
            (
                name,
                cs.active_model_id or _r._dim("none"),
                _r._green("ready") if cs.available else _r._red("missing"),
                cs.error or "",
            )
        )
    _r.table("Categories", ["Category", "Active Model", "Status", "Error"], rows)


def cmd_models_list(args, _get_models):
    models = _get_models(init_models=False)
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
                _r._green("local") if s["is_local"] else _r._blue("api"),
                s.get("requires_provider") or _r._dim("—"),
                f"{s['disk_mb']}MB" if s["disk_mb"] else _r._dim("—"),
            )
        )
    _r.table(
        "Registered Models",
        ["Model ID", "Category", "Tier", "Type", "Provider", "Size"],
        rows,
    )


def cmd_models_install(args, _get_models):
    models = _get_models(init_models=False)
    _r.header(f"Installing {len(args.model_ids)} model(s)")
    for mid in args.model_ids:
        _r.info(f"Installing {mid}...")
        result = models.install(
            mid, auto_download=not args.no_download, force=args.force
        )
        if result.success:
            _r.ok(f"{mid} installed")
        else:
            _r.err(f"{mid}: {result.error}")


def cmd_models_uninstall(args, _get_models):
    models = _get_models(init_models=False)
    _r.header(f"Uninstalling {len(args.model_ids)} model(s)")
    for mid in args.model_ids:
        result = models.uninstall(mid, delete_files=not args.keep_files)
        if result.success:
            _r.ok(f"{mid}: {result.message}")
        else:
            _r.err(f"{mid}: {result.error}")


def cmd_models_switch(args, _get_models):
    models = _get_models(init_models=False)
    from src.core.Models.Lib.ModelSpec import ModelCategory

    cat = ModelCategory(args.category)
    _r.info(f"Switching {args.category} to {args.model_id}...")
    result = models.switch(cat, args.model_id, auto_download=not args.no_download)
    if result.success:
        _r.ok(f"Switched {args.category} → {args.model_id}")
    else:
        _r.err(f"Switch failed: {result.error}")


def cmd_models_keys(args, _get_models):
    models = _get_models(init_models=False)
    if args.action == "set":
        ok = models.set_key(args.provider, args.key)
        if ok:
            _r.ok(f"{args.provider} key set")
        else:
            _r.err("Failed to set key")
    elif args.action == "status":
        statuses = models.list_key_status()
        rows = [
            (p, _r._green("set") if s else _r._red("missing"))
            for p, s in statuses.items()
        ]
        _r.table("API Keys", ["Provider", "Status"], rows)
    elif args.action == "remove":
        ok = models.remove_key(args.provider)
        if ok:
            _r.ok(f"{args.provider} key removed")
        else:
            _r.err("Failed to remove key")


def add_args(sub, _get_models):
    """Register model subcommands and return a lazy-bound handler map."""
    p = sub.add_parser("models", help="Model lifecycle management")
    sp = p.add_subparsers(dest="model_action", required=True)

    s = sp.add_parser("status", help="Show model status across categories")
    s.set_defaults(func=lambda a: cmd_models_status(a, _get_models))

    l = sp.add_parser("list", help="List registered models")
    l.add_argument("--category", "-c", help="Filter by category")
    l.add_argument("--provider", "-p", help="Filter by provider")
    l.set_defaults(func=lambda a: cmd_models_list(a, _get_models))

    i = sp.add_parser("install", help="Install one or more models")
    i.add_argument("model_ids", nargs="+", help="Model ID(s) to install")
    i.add_argument("--no-download", action="store_true", help="Skip auto-download")
    i.add_argument("--force", action="store_true", help="Force re-download")
    i.set_defaults(func=lambda a: cmd_models_install(a, _get_models))

    u = sp.add_parser("uninstall", help="Uninstall models")
    u.add_argument("model_ids", nargs="+", help="Model ID(s) to remove")
    u.add_argument("--keep-files", action="store_true", help="Keep files on disk")
    u.set_defaults(func=lambda a: cmd_models_uninstall(a, _get_models))

    sw = sp.add_parser("switch", help="Switch active model for a category")
    sw.add_argument("category", help="Category (embedding, nlp, ocr, reranker, vision)")
    sw.add_argument("model_id", help="Target model ID")
    sw.add_argument(
        "--no-download", action="store_true", help="Skip auto-download if missing"
    )
    sw.set_defaults(func=lambda a: cmd_models_switch(a, _get_models))

    k = sp.add_parser("keys", help="API key management")
    k.add_argument("action", choices=["set", "status", "remove"], help="Key operation")
    k.add_argument(
        "--provider", "-p", help="Provider name (openai, gemini, groq, etc.)"
    )
    k.add_argument("--key", "-k", help="API key value (required for 'set')")
    k.set_defaults(func=lambda a: cmd_models_keys(a, _get_models))
