"""
Interactive setup and onboarding command handlers for darwin-admin.

Commands: interactive, profile, onboard
"""

import json as _json

from src.admin.renderer import _r


# ══════════════════════════════════════════════════════════════
# Interactive helpers
# ══════════════════════════════════════════════════════════════


def _interactive_select(prompt, options, default=None):
    """Numbered interactive selector."""
    _r.section(prompt)
    for i, (key, desc) in enumerate(options, 1):
        _r._out(f"  [{i}] {key}  {_r._dim(desc)}")
    while True:
        try:
            raw = input(
                f"  Enter [1-{len(options)}]"
                + (f" (default: {default})" if default else "")
                + ": "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            _r._out()
            raise KeyboardInterrupt
        if not raw and default is not None:
            return options[default - 1][0]
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        except ValueError:
            pass
        _r.err("Invalid selection.")


def _setup_status():
    """Check all three config files via AppPaths.

    Returns (status, content) where:
      status  = {"darwin_json": bool, "logging_json": bool, "models_json": bool}
      content = loaded JSON content for each file that exists, None otherwise
    """
    from src.lib.AppPaths import default_config_dir, default_models_config_dir

    darwin_path = default_config_dir() / "darwin.json"
    logging_path = default_config_dir() / "logging.json"
    models_path = default_models_config_dir() / "models.json"

    status = {
        "darwin_json": darwin_path.exists(),
        "logging_json": logging_path.exists(),
        "models_json": models_path.exists(),
    }

    content = {}

    if status["darwin_json"]:
        try:
            content["darwin"] = _json.loads(darwin_path.read_text(encoding="utf-8"))
        except Exception:
            content["darwin"] = None

    if status["logging_json"]:
        try:
            content["logging"] = _json.loads(logging_path.read_text(encoding="utf-8"))
        except Exception:
            content["logging"] = None

    if status["models_json"]:
        try:
            from src.core.Models.ModelConfig import load_active_models
            content["models"] = load_active_models()
        except Exception:
            content["models"] = None

    return status, content


def _show_setup_status(status, content):
    """Display a summary table of what is configured."""
    _r.section("Current configuration")

    def _tag(ok):
        return _r._green("yes") if ok else _r._dim("no")

    _r._out(f"  darwin.json   (setup config)     : {_tag(status['darwin_json'])}")
    _r._out(f"  logging.json  (logging settings) : {_tag(status['logging_json'])}")
    _r._out(f"  models.json   (active models)    : {_tag(status['models_json'])}")

    models_cfg = content.get("models")
    if models_cfg:
        active = [f"{cat}={mid}" for cat, mid in models_cfg.items() if mid]
        if active:
            _r._out(f"  {_r._dim('Active: ' + ', '.join(active))}")

    darwin_cfg = content.get("darwin")
    if darwin_cfg:
        ts = darwin_cfg.get("setup_completed", "unknown")
        _r._out(f"  {_r._dim('Last setup: ' + str(ts))}")

    _r._out()


def _setup_detect_and_recommend(_get_models):
    """Detect system and show model recommendations. Returns (models, system, recs)."""
    _r.section("Step 1: Detecting system capabilities")
    models = _get_models(init_models=False)
    system = models.detect_system()
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
    return models, system, recs


def _setup_category(models, cat, desc, recommended, *, required=False):
    """Walk user through one category: list models, let them pick.

    Returns (model_id, api_key_needed) or (None, None) if skipped.
    """
    from src.core.Models.Lib.ModelSpec import ModelCategory

    cat_enum = ModelCategory(cat)
    all_models = models.list_registered(category=cat_enum)

    if not all_models:
        _r._out(f"  {_r._dim(cat + ': no models registered')}")
        return None, None

    _r.section(f"Category: {cat} {_r._dim('(required)') if required else ''}")
    _r._out(f"  {_r._dim(desc)}")

    if required:
        _r._out()
        for i, m in enumerate(all_models, 1):
            tag = _r._green(" (recommended)") if m["model_id"] == recommended else ""
            type_tag = "API" if m["is_api"] else "local"
            provider = f" — needs {m['requires_provider']} key" if m.get("requires_provider") else ""
            _r._out(f"  [{i}] {m['model_id']}  {_r._dim(f'[{type_tag}]{provider}')}{tag}")

        try:
            raw = input(f"  Pick [1-{len(all_models)}] (default: recommended): ").strip()
        except (EOFError, KeyboardInterrupt):
            _r._out()
            raise KeyboardInterrupt

        if not raw:
            idx = next((i for i, m in enumerate(all_models, 1) if m["model_id"] == recommended), 1)
        else:
            try:
                idx = int(raw)
                if not (1 <= idx <= len(all_models)):
                    idx = 1
            except ValueError:
                idx = 1

        picked = all_models[idx - 1]
        api_key = picked.get("requires_provider")
        return picked["model_id"], api_key

    # Optional category
    rec_spec = next((m for m in all_models if m["model_id"] == recommended), None)

    if rec_spec:
        type_tag = "API (requires key)" if rec_spec["is_api"] else "local (auto-download)"
        _r._out(f"  Recommended: {_r._green(recommended)}  {_r._dim(type_tag)}")
        if rec_spec.get("notes"):
            _r._out(f"  {_r._dim(rec_spec['notes'])}")
    else:
        _r._out(f"  {_r._dim('No recommended model for this category.')}")

    _r._out()
    options = [("skip", "Skip this category")]
    if rec_spec:
        options.insert(0, ("recommended", f"Use recommended: {recommended}"))
    options.append(("browse", "Browse all options in this category"))

    choice = _interactive_select(
        f"  {cat}?",
        options,
        default=1 if rec_spec else 2,
    )

    if choice == "skip":
        _r._out(f"  {_r._dim(cat + ': skipped')}")
        return None, None

    if choice == "recommended" and rec_spec:
        api_key = rec_spec.get("requires_provider")
        return rec_spec["model_id"], api_key

    # Browse all options
    _r._out()
    for i, m in enumerate(all_models, 1):
        type_tag = "API" if m["is_api"] else "local"
        provider = f" — needs {m['requires_provider']} key" if m.get("requires_provider") else ""
        _r._out(f"  [{i}] {m['model_id']}  {_r._dim(f'[{type_tag}]{provider}')}")

    try:
        raw = input(f"  Pick [1-{len(all_models)}] or 's' to skip: ").strip()
    except (EOFError, KeyboardInterrupt):
        _r._out()
        raise KeyboardInterrupt

    if raw.lower() == "s" or not raw:
        _r._out(f"  {_r._dim(cat + ': skipped')}")
        return None, None

    try:
        idx = int(raw)
        if not (1 <= idx <= len(all_models)):
            _r._out(f"  {_r._dim(cat + ': skipped')}")
            return None, None
    except ValueError:
        _r._out(f"  {_r._dim(cat + ': skipped')}")
        return None, None

    picked = all_models[idx - 1]
    api_key = picked.get("requires_provider")
    return picked["model_id"], api_key


def _setup_per_category(models, recs):
    """Walk user through each category one by one.

    Returns {category: model_id} (only categories the user picked).
    """
    cat_labels = [
        ("embedding", "Vector embeddings for semantic search", True),
        ("reranker", "Re-rank search results for higher precision", False),
        ("nlp", "Named entity recognition, summarization, NLP", False),
        ("ocr", "Optical character recognition for scanned documents", False),
        ("vision", "Image analysis and captioning (API-only for now)", False),
    ]

    _r.header("Step 2: Configure models per category")
    selected = {}

    for cat, desc, required in cat_labels:
        recommended = getattr(recs, cat, None)
        if not recommended and not required:
            continue

        model_id, api_key_needed = _setup_category(
            models, cat, desc, recommended, required=required,
        )

        if model_id:
            selected[cat] = (model_id, api_key_needed)

    return selected


def _setup_api_keys_for_selected(models, selected):
    """Prompt for API keys needed by selected API-based models."""
    providers_needed = set()
    for cat, (mid, api_key) in selected.items():
        if api_key:
            providers_needed.add(api_key)

    if not providers_needed:
        return

    _r.section(f"Step 3: API key(s) needed for: {', '.join(sorted(providers_needed))}")
    for prov in sorted(providers_needed):
        try:
            key = input(f"  Enter {prov} API key (or leave blank to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            _r._out()
            raise KeyboardInterrupt
        if key:
            models.set_key(prov, key)
            _r.ok(f"{prov} key set")
        else:
            _r.warn(f"Skipped {prov} key")


def _setup_install_selected(models, selected):
    """Install only local models. API models just need a key (already handled)."""
    all_specs = models.list_registered()
    spec_map = {s["model_id"]: s for s in all_specs}

    to_install = []
    api_noted = []
    for cat, (mid, _) in selected.items():
        spec = spec_map.get(mid)
        if spec and spec.get("is_api"):
            api_noted.append((cat, mid))
        else:
            to_install.append((cat, mid))

    if api_noted:
        _r.section("API models (key required, no download needed)")
        for cat, mid in api_noted:
            _r._out(f"  {cat}: {_r._green(mid)}  {_r._dim('— will use API key')}")

    if to_install:
        _r.section(f"Step 4: Installing {len(to_install)} local model(s)")
        for cat, mid in to_install:
            _r.info(f"Installing {mid} for {cat}...")
            result = models.install(mid, auto_download=True)
            if result.success:
                _r.ok(f"{mid} installed")
            else:
                _r.err(f"{mid}: {result.error}")
    elif not api_noted:
        _r.info("No models to install.")


def _setup_logging_interactive():
    """Interactively configure logging.json.

    Modes:
      terminal — Console logs only (default)
      sse      — Stream logs via SSE (for MCP / remote viewing)
      both     — Console + SSE simultaneously
      none     — Silent (no log output)

    The persist flag (save to SQLite history) is asked separately.
    If SSE/BOTH is selected, a working port is found and saved to logging.json.
    MCP reads this port at startup.
    """
    from src.lib.AppPaths import default_config_dir

    _r.section("Step 5: Logging")
    _r._out("  [1] terminal   — Logs to console only")
    _r._out("  [2] sse        — Stream logs via SSE (for MCP / remote)")
    _r._out("  [3] both       — Console + SSE simultaneously")
    _r._out("  [4] none       — Silent (no log output)")
    try:
        raw = input("  Select mode [1-4] (default: 1): ").strip()
    except (EOFError, KeyboardInterrupt):
        _r._out()
        raise KeyboardInterrupt
    mode_map = {"1": "terminal", "2": "sse", "3": "both", "4": "none"}
    mode = mode_map.get(raw, "terminal")

    _r._out()
    try:
        persist_raw = input("  Save logs to history? [y/n] (default: y): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        _r._out()
        raise KeyboardInterrupt
    persist = persist_raw != "n"

    _r._out()
    ret_choice = _interactive_select(
        "Log retention",
        [
            ("3", "3 days"),
            ("7", "7 days"),
            ("14", "14 days"),
            ("30", "30 days"),
            ("0", "Never — delete logs immediately"),
        ],
        default=3,
    )
    retention = int(ret_choice)

    sse_port = 8765
    sse_host = "127.0.0.1"
    if mode in ("sse", "both"):
        from src.lib.NetUtils import is_port_available, find_available_port

        if is_port_available(sse_host, sse_port):
            _r.ok(f"Port {sse_port} available")
        else:
            _r.warn(f"Port {sse_port} in use — scanning...")
            found = find_available_port(sse_host, sse_port + 1, 10)
            if found:
                sse_port = found
                _r.ok(f"Using port {sse_port}")
            else:
                _r.err("No available port found — SSE disabled")
                mode = "terminal"

    logging_cfg = {
        "mode": mode,
        "persist": persist,
        "retention_days": retention,
        "max_memory": 10000,
        "sse_port": sse_port,
        "sse_host": sse_host,
        "show_banner": True,
        "log_filter": "all",
    }

    path = default_config_dir() / "logging.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(logging_cfg, indent=2), encoding="utf-8")

    port_note = f", port {sse_port}" if mode in ("sse", "both") else ""
    _r.ok(f"logging.json written ({mode}, persist={persist}, {retention}d retention{port_note})")


def _cmd_change_settings(_get_models):
    """Sub-menu for changing specific config areas."""
    _r.header("Change Settings")
    choice = _interactive_select(
        "What would you like to change?",
        [
            ("models", "Re-select and reinstall models"),
            ("logging", "Change logging mode and retention"),
            ("api-keys", "Add or change API keys"),
            ("all", "Full re-run of entire setup"),
        ],
        default=4,
    )

    if choice == "models":
        models, system, recs = _setup_detect_and_recommend(_get_models)
        selected = _setup_per_category(models, recs)
        if selected:
            _setup_api_keys_for_selected(models, selected)
            _setup_install_selected(models, selected)

    elif choice == "logging":
        _setup_logging_interactive()

    elif choice == "api-keys":
        models = _get_models(init_models=False)
        all_specs = models.list_registered()
        providers = set()
        for s in all_specs:
            if s.get("requires_provider"):
                providers.add(s["requires_provider"])
        if not providers:
            _r.info("No provider-based models registered.")
            return
        _r.section("API Keys")
        for prov in sorted(providers):
            try:
                key = input(f"  {prov} API key (blank to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                _r._out()
                raise KeyboardInterrupt
            if key:
                models.set_key(prov, key)
                _r.ok(f"{prov} key set")

    elif choice == "all":
        _setup_full_interactive(_get_models)


def _setup_full_interactive(_get_models):
    """The full interactive setup flow."""
    _r.header("Darwin Interactive Setup")

    models, system, recs = _setup_detect_and_recommend(_get_models)

    selected = _setup_per_category(models, recs)
    if not selected:
        _r.info("No models selected.")
        _r.next_steps([
            "darwin-admin setup interactive   # Run setup again later",
            "darwin-admin models install <id>  # Install a specific model",
        ])
        return

    _setup_api_keys_for_selected(models, selected)
    _setup_install_selected(models, selected)
    _setup_logging_interactive()

    _r._out()
    _r.ok("Setup complete")
    _r.next_steps([
        "darwin mcp               # Start the MCP server",
        "darwin status             # Check system status",
        "darwin-admin models status  # Check model status",
    ])


# ══════════════════════════════════════════════════════════════
# Command handlers
# ══════════════════════════════════════════════════════════════


def cmd_setup_interactive(args, _get_models):
    """Config-aware interactive onboarding."""
    status, content = _setup_status()
    count = sum(status.values())

    if count == 3:
        _r.header("Darwin Setup")
        _show_setup_status(status, content)
        choice = _interactive_select(
            "Everything looks configured. What would you like to do?",
            [
                ("change", "Change a specific setting (models, logging, API keys)"),
                ("re-run", "Full re-run (reconfigure everything)"),
                ("exit", "Exit — current config is fine"),
            ],
            default=3,
        )
        if choice == "exit":
            _r.ok("Keeping current configuration.")
            return
        elif choice == "change":
            _cmd_change_settings(_get_models)
        else:
            _setup_full_interactive(_get_models)

    elif count > 0:
        _r.header("Darwin Setup")
        missing = [k.replace("_json", ".json") for k, v in status.items() if not v]
        _show_setup_status(status, content)
        _r.info(f"Missing: {', '.join(missing)}")
        choice = _interactive_select(
            "How would you like to proceed?",
            [
                ("complete", "Complete the missing pieces"),
                ("re-run", "Full re-run (reconfigure everything)"),
                ("exit", "Exit — do nothing"),
            ],
            default=1,
        )
        if choice == "exit":
            _r.ok("Keeping current configuration.")
            return
        elif choice == "complete":
            _cmd_change_settings(_get_models)
        else:
            _setup_full_interactive(_get_models)

    else:
        _setup_full_interactive(_get_models)


def cmd_setup_profile(args, _get_models):
    """Non-interactive profile-based setup."""
    models = _get_models(init_models=False)
    results = models.setup_profile(
        profile=args.profile, auto_download=not args.no_download
    )
    _r.header(f"Setup Profile: {args.profile}")
    for cat, res in results.items():
        if res.success:
            _r.ok(f"{cat.value}: {res.model_id}")
        else:
            _r.err(f"{cat.value}: {res.error or 'failed'}")


def cmd_setup_onboard(args, _get_models):
    """Non-interactive onboarding with preference dict."""
    models = _get_models(init_models=False)
    if args.preferences:
        prefs = dict(kv.split("=", 1) for kv in args.preferences)
    else:
        prefs = None
    report = models.onboard(preferences=prefs, auto_download=not args.no_download)
    summary = report.get("summary", {})
    _r.header("Onboard Report")
    for cat in summary.get("ready", []):
        _r.ok(f"{cat} ready")
    for cat, err in summary.get("failed", {}).items():
        _r.err(f"{cat}: {err}")


def add_args(sub, _get_models):
    """Register setup subcommands."""
    p = sub.add_parser("setup", help="Initial setup and onboarding")
    sp = p.add_subparsers(dest="setup_action", required=True)

    i = sp.add_parser("interactive", help="Interactive setup with guided menus")
    i.set_defaults(func=lambda a: cmd_setup_interactive(a, _get_models))

    prof = sp.add_parser("profile", help="Non-interactive profile-based setup")
    prof.add_argument(
        "--profile", default="required",
        choices=["required", "recommended"], help="Model profile to install",
    )
    prof.add_argument("--no-download", action="store_true", help="Skip auto-download")
    prof.set_defaults(func=lambda a: cmd_setup_profile(a, _get_models))

    o = sp.add_parser("onboard", help="Non-interactive onboard with preferences")
    o.add_argument(
        "--preferences", "-p", nargs="+", metavar="CAT=MODEL_ID",
        help="e.g. embedding=intfloat/multilingual-e5-small-onnx",
    )
    o.add_argument("--no-download", action="store_true")
    o.set_defaults(func=lambda a: cmd_setup_onboard(a, _get_models))
