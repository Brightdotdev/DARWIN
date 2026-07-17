"""
Model lifecycle command handlers for darwin-admin.

Commands: status, list, install, discard, switch, keys
"""

from typing import List

from src.admin.lib import (
    KeyAction,
    ModelCategory,
    ns_to,
    ModelsStatusArgs,
    ModelsListArgs,
    ModelsInstallArgs,
    ModelsDiscardArgs,
    ModelsSwitchArgs,
    ModelsKeysArgs,
)
from src.admin.renderer import _r


def _get_installed_in_category(models, cat_enum: ModelCategory) -> List[str]:
    """Return list of model_ids that are installed (downloaded or API-keyed) in a category."""
    from src.lib.KeyManager import is_key_set, load_keys_from_keychain

    specs = models.list_registered(category=cat_enum)
    installed: list[str] = []
    load_keys_from_keychain(log=False)
    for s in specs:
        if s["is_api"]:
            if s.get("requires_provider") and is_key_set(s["requires_provider"]):
                installed.append(s["model_id"])
        else:
            if models.is_downloaded(cat_enum, s["model_id"]):
                installed.append(s["model_id"])
    return installed


def cmd_models_status(args: ModelsStatusArgs, _get_models) -> None:
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


def cmd_models_list(args: ModelsListArgs, _get_models) -> None:
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


def cmd_models_install(args: ModelsInstallArgs, _get_models) -> None:
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


def cmd_models_discard(args: ModelsDiscardArgs, _get_models) -> None:
    """Discard models per category with --all/--active/--inactive selection."""
    from src.core.Models.ModelConfig import get_active_model

    models = _get_models(init_models=False)
    delete = args.delete
    categories = args.category

    if not categories:
        _r.err("No categories specified. Use --category <name> [--category <name> ...]")
        return

    # Validate categories
    valid_cats = {c.value.lower() for c in ModelCategory}
    for cat in categories:
        if cat.lower() not in valid_cats:
            _r.err(f"Unknown category: {cat} (valid: {', '.join(sorted(valid_cats))})")
            return

    # Determine scope flags
    scope_all = args.discard_all
    scope_active = args.active
    scope_inactive = args.inactive

    scope_flags = sum([scope_all, scope_active, scope_inactive])
    if scope_flags > 1:
        _r.err("Use only one of --all, --active, or --inactive")
        return

    total_discarded = 0

    for cat_name in categories:
        cat_enum = ModelCategory(cat_name.lower())
        installed = _get_installed_in_category(models, cat_enum)

        if not installed:
            _r._out(f"\n  {_r._dim(cat_name + ': no installed models, skipping')}")
            continue

        active_id = get_active_model(cat_name.lower())

        to_discard: list[str] = []

        if scope_all:
            to_discard = list(installed)
        elif scope_active:
            if active_id and active_id in installed:
                to_discard = [active_id]
            else:
                _r._out(f"\n  {_r._dim(cat_name + ': no active model to discard')}")
                continue
        elif scope_inactive:
            to_discard = [m for m in installed if m != active_id]
            if not to_discard:
                _r._out(f"\n  {_r._dim(cat_name + ': no inactive models to discard')}")
                continue
        else:
            # Interactive: no scope flag — let user pick
            if len(installed) == 1:
                to_discard = list(installed)
            else:
                options = [("all", f"Discard ALL {len(installed)} models in {cat_name}")]
                for mid in installed:
                    tag = " (active)" if mid == active_id else ""
                    options.append((mid, mid + tag))
                options.append(("skip", f"Skip {cat_name}"))

                choice = _r.interactive_select(
                    f"Discard models from {cat_name}?",
                    options,
                    default=len(options),
                )

                if choice == "skip":
                    _r._out(f"  {_r._dim(cat_name + ': skipped')}")
                    continue
                elif choice == "all":
                    to_discard = list(installed)
                else:
                    to_discard = [choice]

        _r.section(f"Discarding {len(to_discard)} model(s) from {cat_name}")
        for mid in to_discard:
            _r.info(f"Discarding {mid}...")
            result = models.uninstall(mid, delete_files=delete)
            if result.success:
                _r.ok(f"{mid} discarded" + (" (files deleted)" if delete else ""))
                total_discarded += 1
            else:
                _r.err(f"{mid}: {result.error}")

    _r._out()
    _r.ok(f"Total discarded: {total_discarded} model(s)")


def cmd_models_switch(args: ModelsSwitchArgs, _get_models) -> None:
    """Switch active model. Fully interactive: no args → category → model."""
    from src.core.Models.ModelConfig import get_active_model
    from src.core.Models.Lib.ModelRegistry import find_spec

    models = _get_models(init_models=False)

    # ── Step 1: resolve category ──────────────────────────────────────
    cat_name = args.category
    if not cat_name:
        options = []
        for cat_enum in ModelCategory:
            cname = cat_enum.value
            installed = _get_installed_in_category(models, cat_enum)
            count = len(installed)
            active = get_active_model(cname)
            status = _r._green(active) if active else _r._dim("no active model")
            label = f"{cname}  ({count} installed)  {status}"
            options.append((cname, label))

        options.append(("cancel", "Cancel"))
        cat_name = _r.interactive_select("Which category?", options, default=len(options))
        if cat_name == "cancel":
            _r._out(f"  {_r._dim('switch cancelled')}")
            return

    # Validate category
    try:
        cat = ModelCategory(cat_name)
    except ValueError:
        _r.err(f"Unknown category: '{cat_name}'. Valid: {[c.value for c in ModelCategory]}")
        return

    # ── Step 2: resolve model_id ──────────────────────────────────────
    target_id = args.model_id

    if not target_id:
        installed = _get_installed_in_category(models, cat)

        if not installed:
            _r.warn(f"No installed models in {cat_name}. Install one first.")
            return

        active_id = get_active_model(cat_name.lower())

        options = []
        for mid in installed:
            tag = " (active)" if mid == active_id else ""
            spec = find_spec(mid)
            type_tag = "API" if spec and spec.is_api else "local"
            provider_tag = f" [{spec.requires_provider}]" if spec and spec.requires_provider else ""
            options.append((mid, f"{mid}{tag}  {_r._dim(f'[{type_tag}]{provider_tag}')}"))

        if len(options) == 1:
            target_id = installed[0]
            _r.info(f"Only one model installed: {target_id}")
            try:
                confirm = input(f"  Switch {cat_name} to {target_id}? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                _r._out()
                return
            if confirm == "n":
                _r._out(f"  {_r._dim('switch cancelled')}")
                return
        else:
            options.append(("cancel", "Cancel — don't switch"))
            choice = _r.interactive_select(
                f"Switch {cat_name} to which model?",
                options,
                default=len(options),
            )
            if choice == "cancel":
                _r._out(f"  {_r._dim('switch cancelled')}")
                return
            target_id = choice

    # ── Step 3: resolve old provider before switching ─────────────────
    old_active_id = get_active_model(cat_name.lower())
    old_provider = None
    if old_active_id:
        old_spec = find_spec(old_active_id)
        if old_spec:
            old_provider = old_spec.requires_provider

    # ── Step 4: resolve new provider ──────────────────────────────────
    new_spec = find_spec(target_id)
    new_provider = new_spec.requires_provider if new_spec else None

    # ── Step 5: perform the switch ────────────────────────────────────
    _r.info(f"Switching {cat_name} to {target_id}...")
    result = models.switch(cat, target_id, auto_download=not args.no_download)
    if not result.success:
        _r.err(f"Switch failed: {result.error}")
        return

    _r.ok(f"Switched {cat_name} → {target_id}")

    # ── Step 6: provider comparison + key management ──────────────────
    if old_provider and new_provider and old_provider == new_provider:
        _r.info(f"Same provider ({old_provider}) — key reused, no action needed")
        return

    if old_provider and old_provider != new_provider:
        _r._out()
        _r._out(f"  Old provider: {_r._yellow(old_provider)}")
        _r._out(f"  New provider: {_r._yellow(new_provider or 'none (local)')}")

        if new_provider:
            from src.lib.KeyManager import is_key_set, load_keys_from_keychain
            load_keys_from_keychain(log=False)
            if is_key_set(new_provider):
                _r.info(f"Key for {new_provider} already set — ready to use")
            else:
                _r.warn(f"No key set for {new_provider} — you'll need to add one:")
                _r._out(f"    darwin-admin models keys set {new_provider} <your-key>")

        try:
            raw = input(
                f"  Discard old key for {old_provider}? [y/N] (default: N): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            _r._out()
            return

        if raw == "y":
            ok = models.remove_key(old_provider)
            if ok:
                _r.ok(f"Key for {old_provider} discarded")
            else:
                _r.err(f"Failed to discard key for {old_provider}")
        else:
            _r.info(f"Key for {old_provider} kept (remove later with: darwin-admin models keys remove {old_provider})")


def cmd_models_keys(args: ModelsKeysArgs, _get_models) -> None:
    models = _get_models(init_models=False)
    if args.action in (KeyAction.SET.value, KeyAction.REMOVE.value) and not args.provider:
        _r.err(f"--provider is required for '{args.action}'")
        return
    if args.action == KeyAction.SET.value and not args.key:
        _r.err("--key is required for 'set'")
        return
    if args.action == KeyAction.SET.value:
        ok = models.set_key(args.provider, args.key)
        if ok:
            _r.ok(f"{args.provider} key set")
        else:
            _r.err("Failed to set key")
    elif args.action == KeyAction.STATUS.value:
        statuses = models.list_key_status()
        rows = [
            (p, _r._green("set") if s else _r._red("missing"))
            for p, s in statuses.items()
        ]
        _r.table("API Keys", ["Provider", "Status"], rows)
    elif args.action == KeyAction.REMOVE.value:
        ok = models.remove_key(args.provider)
        if ok:
            _r.ok(f"{args.provider} key removed")
        else:
            _r.err("Failed to remove key")


def add_args(sub, _get_models) -> None:
    """Register model subcommands."""
    p = sub.add_parser("models", help="Model lifecycle management")
    sp = p.add_subparsers(dest="model_action", required=True)

    s = sp.add_parser("status", help="Show model status across categories")
    s.set_defaults(func=lambda a: cmd_models_status(ns_to(ModelsStatusArgs, a), _get_models))

    l = sp.add_parser("list", help="List registered models")
    l.add_argument("--category", "-c", help="Filter by category")
    l.add_argument("--provider", "-p", help="Filter by provider")
    l.set_defaults(func=lambda a: cmd_models_list(ns_to(ModelsListArgs, a), _get_models))

    i = sp.add_parser("install", help="Install one or more models")
    i.add_argument("model_ids", nargs="+", help="Model ID(s) to install")
    i.add_argument("--no-download", action="store_true", help="Skip auto-download")
    i.add_argument("--force", action="store_true", help="Force re-download")
    i.set_defaults(func=lambda a: cmd_models_install(ns_to(ModelsInstallArgs, a), _get_models))

    d = sp.add_parser("discard", help="Discard models per category")
    d.add_argument(
        "--category", "-c", nargs="+", required=True, metavar="CAT",
        help="Category name(s): embedding, reranker, nlp, ocr, vision",
    )
    d.add_argument("--all", action="store_true", dest="discard_all",
                   help="Discard all installed models in each category")
    d.add_argument("--active", action="store_true",
                   help="Discard only the active model in each category")
    d.add_argument("--inactive", action="store_true",
                   help="Discard only inactive models in each category")
    d.add_argument("--delete", action="store_true",
                   help="Delete model files from disk")
    d.set_defaults(func=lambda a: cmd_models_discard(ns_to(ModelsDiscardArgs, a), _get_models))

    sw = sp.add_parser("switch", help="Switch active model for a category")
    sw.add_argument("category", nargs="?", default=None, help="Category (interactive if omitted)")
    sw.add_argument("model_id", nargs="?", default=None, help="Target model ID (interactive if omitted)")
    sw.add_argument(
        "--no-download", action="store_true", help="Skip auto-download if missing"
    )
    sw.set_defaults(func=lambda a: cmd_models_switch(ns_to(ModelsSwitchArgs, a), _get_models))

    k = sp.add_parser("keys", help="API key management")
    k.add_argument("action", choices=[a.value for a in KeyAction], help="Key operation")
    k.add_argument(
        "--provider", "-p", help="Provider name (openai, gemini, groq, etc.)"
    )
    k.add_argument("--key", "-k", help="API key value (required for 'set')")
    k.set_defaults(func=lambda a: cmd_models_keys(ns_to(ModelsKeysArgs, a), _get_models))
