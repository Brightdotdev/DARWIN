"""
Interactive setup and onboarding command handlers for darwin-admin.

Commands: interactive, profile, onboard, preset, validate, doctor, reset, logging
"""

import json as _json
import shutil
import time

from src.core.Models.Lib.ModelSpec import ModelCategory
from src.core.Logger.lib import OutputMode
from src.admin.lib import (
    SetupAction,
    SetupPreset,
    SetupStatus,
    DiskImpact,
    LoggingConfig,
    DoctorCheck,
    CATEGORY_SETUP_ORDER,
    DEFAULT_SSE_PORT,
    DEFAULT_SSE_HOST,
    DEFAULT_RETENTION_DAYS,
    atomic_write_json,
    ns_to,
    SetupInteractiveArgs,
    SetupPresetArgs,
    SetupProfileArgs,
    SetupOnboardArgs,
    SetupValidateArgs,
    SetupResetArgs,
    SetupLoggingArgs,
)
from src.admin.renderer import _r


# ══════════════════════════════════════════════════════════════
# Config status helpers
# ══════════════════════════════════════════════════════════════


def _setup_status() -> SetupStatus:
    """Check which config files exist."""
    from src.lib.AppPaths import default_logging_config, default_models_config_dir, default_darwin_config

    return SetupStatus(
        darwin_json=default_darwin_config().exists(),
        logging_json=default_logging_config().exists(),
        models_json=(default_models_config_dir() / "models.json").exists(),
    )


def _load_config_content(status: SetupStatus) -> dict:
    """Load parsed JSON content for each config file that exists."""
    from src.lib.AppPaths import default_logging_config, default_darwin_config

    content: dict = {}

    if status.darwin_json:
        try:
            content["darwin"] = _json.loads(
                default_darwin_config().read_text(encoding="utf-8")
            )
        except Exception:
            content["darwin"] = None

    if status.logging_json:
        try:
            content["logging"] = _json.loads(
                default_logging_config().read_text(encoding="utf-8")
            )
        except Exception:
            content["logging"] = None

    if status.models_json:
        try:
            from src.core.Models.ModelConfig import load_active_models
            content["models"] = load_active_models()
        except Exception:
            content["models"] = None

    return content


def _show_setup_status(status: SetupStatus, content: dict) -> None:
    """Display a summary table of what is configured."""
    _r.section("Current configuration")

    def _tag(ok: bool) -> str:
        return _r._green("yes") if ok else _r._dim("no")

    _r._out(f"  darwin.json   (setup config)     : {_tag(status.darwin_json)}")
    _r._out(f"  logging.json  (logging settings) : {_tag(status.logging_json)}")
    _r._out(f"  models.json   (active models)    : {_tag(status.models_json)}")

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


# ══════════════════════════════════════════════════════════════
# Disk impact
# ══════════════════════════════════════════════════════════════


def calculate_disk_impact(preferences: dict[str, str]) -> DiskImpact:
    """Calculate disk space impact of model installation.

    Args:
        preferences: {category_name: model_id}

    Returns:
        DiskImpact dataclass
    """
    from src.core.Models.Lib.ModelRegistry import find_spec
    from src.lib.AppPaths import set_darwin_base_dir

    required_mb = 0
    for model_id in preferences.values():
        spec = find_spec(model_id)
        if spec and spec.is_local and spec.disk_mb:
            required_mb += spec.disk_mb

    base = set_darwin_base_dir()
    try:
        usage = shutil.disk_usage(str(base))
        available_mb = usage.free // (1024 * 1024)
    except Exception:
        available_mb = None

    if available_mb is not None:
        after_mb = available_mb - required_mb
        warning = None
        if after_mb < 500:
            warning = f"Low disk space ({after_mb}MB remaining)"
        elif after_mb < 1024:
            warning = f"Disk space getting tight ({after_mb}MB remaining)"
    else:
        after_mb = None
        warning = "Could not check disk space"

    return DiskImpact(
        required_mb=required_mb,
        available_mb=available_mb,
        after_mb=after_mb,
        warning=warning,
    )


# ══════════════════════════════════════════════════════════════
# Setup flow helpers
# ══════════════════════════════════════════════════════════════


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


def _setup_category(models, cat: str, desc: str, recommended: str,
                    *, required: bool = False) -> tuple[str | None, str | None]:
    """Walk user through one category: list models, let them pick.

    Returns (model_id, api_key_needed) or (None, None) if skipped.
    """
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
        return picked["model_id"], picked.get("requires_provider")

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

    choice = _r.interactive_select(
        f"  {cat}?",
        options,
        default=1 if rec_spec else 2,
    )

    if choice == "skip":
        _r._out(f"  {_r._dim(cat + ': skipped')}")
        return None, None

    if choice == "recommended" and rec_spec:
        return rec_spec["model_id"], rec_spec.get("requires_provider")

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
    return picked["model_id"], picked.get("requires_provider")


def _setup_per_category(models, recs) -> dict[str, tuple[str, str | None]]:
    """Walk user through each category one by one.

    Returns {category: (model_id, api_key_provider)} for categories the user picked.
    """
    _r.header("Step 2: Configure models per category")
    selected: dict[str, tuple[str, str | None]] = {}

    for label in CATEGORY_SETUP_ORDER:
        recommended = getattr(recs, label.category, None)
        if not recommended and not label.required:
            continue

        model_id, api_key_needed = _setup_category(
            models, label.category, label.description, recommended,
            required=label.required,
        )

        if model_id:
            selected[label.category] = (model_id, api_key_needed)

    return selected


def _setup_api_keys_for_selected(models, selected: dict) -> None:
    """Prompt for API keys needed by selected API-based models."""
    providers_needed: set[str] = set()
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


def _setup_install_selected(models, selected: dict) -> None:
    """Install only local models. API models just need a key (already handled)."""
    all_specs = models.list_registered()
    spec_map = {s["model_id"]: s for s in all_specs}

    to_install: list[tuple[str, str]] = []
    api_noted: list[tuple[str, str]] = []
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


def _resolve_sse_port(mode: str) -> tuple[str, int, str]:
    """If SSE/BOTH mode, find an available port. Returns (mode, port, host)."""
    port = DEFAULT_SSE_PORT
    host = DEFAULT_SSE_HOST

    if mode in (OutputMode.SSE.value, OutputMode.BOTH.value):
        from src.lib.NetUtils import is_port_available, find_available_port

        if is_port_available(host, port):
            _r.ok(f"Port {port} available")
        else:
            _r.warn(f"Port {port} in use — scanning...")
            found = find_available_port(host, port + 1, 10)
            if found:
                port = found
                _r.ok(f"Using port {port}")
            else:
                _r.err("No available port found — SSE disabled")
                mode = OutputMode.TERMINAL.value

    return mode, port, host


def _setup_logging_interactive(yes: bool = False) -> None:
    """Interactively configure logging.json."""
    from src.lib.AppPaths import default_logging_config

    _r.section("Step 5: Logging")

    if yes:
        mode = OutputMode.TERMINAL.value
        persist = True
        retention = DEFAULT_RETENTION_DAYS
        log_filter = "all"
        _r._out(f"  Mode:      {_r._dim(mode)} (default)")
        _r._out(f"  Persist:   {_r._dim('yes')} (default)")
        _r._out(f"  Retention: {_r._dim(f'{DEFAULT_RETENTION_DAYS} days')} (default)")
    else:
        mode = _r.interactive_select(
            "Output mode",
            [
                (OutputMode.TERMINAL.value, "Console logs only"),
                (OutputMode.SSE.value, "Stream logs via SSE (for MCP / remote)"),
                (OutputMode.BOTH.value, "Console + SSE simultaneously"),
                (OutputMode.NONE.value, "Silent (no log output)"),
            ],
            default=1,
        )

        _r._out()
        try:
            persist_raw = input("  Save logs to history? [y/n] (default: y): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            _r._out()
            raise KeyboardInterrupt
        persist = persist_raw != "n"

        log_filter = _r.interactive_select(
            "Log verbosity",
            [
                ("all", "Everything (default)"),
                ("quiet", "Errors + warnings + success only"),
                ("silent", "Errors only"),
            ],
            default=1,
        )

        _r._out()
        ret_choice = _r.interactive_select(
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

    mode, sse_port, sse_host = _resolve_sse_port(mode)

    cfg = LoggingConfig(
        mode=mode,
        persist=persist,
        retention_days=retention,
        sse_port=sse_port,
        sse_host=sse_host,
        log_filter=log_filter if not yes else "all",
    )

    atomic_write_json(default_logging_config(), cfg.to_dict())

    port_note = f", port {sse_port}" if mode in (OutputMode.SSE.value, OutputMode.BOTH.value) else ""
    _r.ok(f"logging.json written ({mode}, persist={persist}, {retention}d retention{port_note})")


def _setup_logging_defaults(retention_days: int = DEFAULT_RETENTION_DAYS) -> None:
    """Write recommended logging.json without prompts (for presets)."""
    from src.lib.AppPaths import default_logging_config
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

    cfg = LoggingConfig(retention_days=retention_days)
    atomic_write_json(default_logging_config(), cfg.to_dict())

    LoggerOrchestrator.configure(
        mode=OutputMode.TERMINAL.value,
        persist=True,
        log_filter="all",
        persist_config=True,
        override=True,
    )

    _r.ok(f"Logging configured (terminal, persist=true, {retention_days}d retention)")


def _cmd_change_settings(_get_models, yes: bool = False) -> None:
    """Sub-menu for changing specific config areas."""
    _r.header("Change Settings")
    choice = _r.interactive_select(
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
        _setup_logging_interactive(yes=yes)

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


def _setup_full_interactive(_get_models, yes: bool = False) -> None:
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
    _setup_logging_interactive(yes=yes)

    # Disk impact summary
    prefs = {cat: mid for cat, (mid, _) in selected.items()}
    if prefs:
        impact = calculate_disk_impact(prefs)
        _r.section("Disk Impact")
        _r._out(f"    Required:  {impact.required_mb}MB")
        if impact.available_mb is not None:
            _r._out(f"    Available: {impact.available_mb}MB")
            _r._out(f"    After:     {impact.after_mb}MB")
        else:
            _r._out(f"    Available: {_r._dim('unknown')}")
        if impact.warning:
            _r.warn(impact.warning)

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


def cmd_setup_interactive(args: SetupInteractiveArgs, _get_models) -> None:
    """Config-aware interactive onboarding."""
    status = _setup_status()
    content = _load_config_content(status)

    if status.all_configured:
        _r.header("Darwin Setup")
        _show_setup_status(status, content)
        choice = _r.interactive_select(
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
            _cmd_change_settings(_get_models, yes=args.yes)
        else:
            _setup_full_interactive(_get_models, yes=args.yes)

    elif status.any_configured:
        _r.header("Darwin Setup")
        _show_setup_status(status, content)
        _r.info(f"Missing: {', '.join(status.missing_files)}")
        choice = _r.interactive_select(
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
            _cmd_change_settings(_get_models, yes=args.yes)
        else:
            _setup_full_interactive(_get_models, yes=args.yes)

    else:
        _setup_full_interactive(_get_models, yes=args.yes)


def cmd_setup_profile(args: SetupProfileArgs, _get_models) -> None:
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


def cmd_setup_onboard(args: SetupOnboardArgs, _get_models) -> None:
    """Non-interactive onboarding with preference dict."""
    models = _get_models(init_models=False)
    if args.preferences:
        prefs = {}
        for kv in args.preferences:
            if "=" not in kv:
                _r.err(f"Invalid preference format: '{kv}' (expected CATEGORY=MODEL_ID)")
                return
            cat, model_id = kv.split("=", 1)
            if not cat or not model_id:
                _r.err(f"Invalid preference: '{kv}' (both category and model required)")
                return
            prefs[cat] = model_id
        valid_cats = {c.value for c in ModelCategory}
        for cat in prefs:
            if cat.lower() not in valid_cats:
                _r.err(f"Unknown category: '{cat}' (valid: {', '.join(sorted(valid_cats))})")
                return
    else:
        prefs = None
    report = models.onboard(preferences=prefs, auto_download=not args.no_download)
    summary = report.get("summary", {})
    _r.header("Onboard Report")
    for cat in summary.get("ready", []):
        _r.ok(f"{cat} ready")
    for cat, err in summary.get("failed", {}).items():
        _r.err(f"{cat}: {err}")


def cmd_setup_validate(args: SetupValidateArgs) -> None:
    """Validate the current Darwin configuration."""
    from src.lib.AppPaths import default_logging_config, default_models_config_dir, default_darwin_config

    darwin_path = default_darwin_config()
    logging_path = default_logging_config()
    models_path = default_models_config_dir() / "models.json"

    _r.header("Validate Configuration")
    ok: list[str] = []
    issues: list[str] = []

    # Config files
    for label, path in [("darwin.json", darwin_path), ("logging.json", logging_path), ("models.json", models_path)]:
        if path.exists():
            ok.append(f"{label} exists")
            try:
                _json.loads(path.read_text(encoding="utf-8"))
                ok.append(f"{label} is valid JSON")
            except Exception as e:
                issues.append(f"{label} is corrupted: {e}")
        else:
            issues.append(f"{label} missing")

    # Embedding model
    if models_path.exists():
        try:
            from src.core.Models.ModelConfig import load_active_models
            active = load_active_models()
            emb = active.get("embedding")
            if emb:
                from src.core.Models.Lib.ModelRegistry import find_spec
                spec = find_spec(emb)
                if spec:
                    ok.append(f"Embedding model registered: {emb}")
                else:
                    issues.append(f"Embedding model unknown: {emb}")
            else:
                issues.append("No embedding model configured")
        except Exception as e:
            issues.append(f"Could not read models config: {e}")

    for item in ok:
        _r.ok(item)
    for item in issues:
        _r.err(item)

    _r._out()
    if not issues:
        _r.ok("Configuration is valid")
    else:
        _r.err(f"{len(issues)} issue(s) found")


def cmd_setup_doctor(_get_models) -> None:
    """Diagnose common Darwin issues."""
    from src.lib.AppPaths import default_logging_config, default_models_config_dir, set_darwin_base_dir, default_darwin_config

    _r.header("Darwin Doctor")
    checks: list[DoctorCheck] = []

    # Config files
    darwin_path = default_darwin_config()
    logging_path = default_logging_config()
    models_path = default_models_config_dir() / "models.json"

    checks.append(DoctorCheck("Config file exists", darwin_path.exists(), "Run: darwin-admin setup interactive"))
    checks.append(DoctorCheck("Logging config exists", logging_path.exists(), "Run: darwin-admin setup interactive"))
    checks.append(DoctorCheck("Models config exists", models_path.exists(), "Run: darwin-admin setup interactive"))

    # Disk space
    try:
        usage = shutil.disk_usage(str(set_darwin_base_dir()))
        free_mb = usage.free // (1024 * 1024)
        checks.append(DoctorCheck(f"Disk space ({free_mb}MB free)", free_mb > 500, "Free up disk space"))
    except Exception:
        checks.append(DoctorCheck("Disk space check", False, "Could not check disk space"))

    # Models
    if models_path.exists():
        try:
            from src.core.Models.ModelConfig import load_active_models
            active = load_active_models()
            emb = active.get("embedding")
            if emb:
                from src.core.Models.Lib.ModelRegistry import find_spec
                spec = find_spec(emb)
                checks.append(DoctorCheck(f"Embedding model '{emb}' registered", spec is not None, "Re-run setup"))
                if spec and spec.is_local:
                    models = _get_models(init_models=False)
                    downloaded = models.is_downloaded(spec.category, spec.model_id)
                    checks.append(DoctorCheck("Embedding model downloaded", downloaded, f"Run: darwin-admin models install {emb}"))
            else:
                checks.append(DoctorCheck("Embedding model configured", False, "Run: darwin-admin setup interactive"))
        except Exception:
            checks.append(DoctorCheck("Model check", False, "Could not read model config"))

    for check in checks:
        if check.passed:
            _r.ok(check.name)
        else:
            _r.err(f"{check.name} — {check.fix}")

    _r._out()
    failed = sum(1 for c in checks if not c.passed)
    if failed == 0:
        _r.ok("All checks passed")
    else:
        _r.warn(f"{failed} issue(s) found")


def cmd_setup_reset(args: SetupResetArgs) -> None:
    """Reset Darwin configuration. Optionally keep downloaded models."""
    from src.lib.AppPaths import default_logging_config, default_models_config_dir, default_darwin_config

    if not args.yes:
        action = "keep models" if args.keep_models else "remove everything"
        try:
            confirm = input(f"  Reset Darwin? This will {action}. [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            _r._out()
            return
        if confirm != "y":
            _r._out("Cancelled.")
            return

    _r.header("Reset Darwin Configuration")

    # Backup darwin.json
    darwin_path = default_darwin_config()
    if darwin_path.exists():
        backup = darwin_path.with_suffix(f".backup.{int(time.time())}.json")
        shutil.copy2(darwin_path, backup)
        _r.ok(f"Config backed up to {backup}")
        darwin_path.unlink()

    # Remove logging.json
    logging_path = default_logging_config()
    if logging_path.exists():
        logging_path.unlink()

    # Remove models.json (unless keeping configs)
    models_path = default_models_config_dir() / "models.json"
    if not args.keep_models and models_path.exists():
        models_path.unlink()

    # Optionally remove downloaded models
    if not args.keep_models:
        models_dir = default_models_config_dir().parent / "models"
        if models_dir.exists():
            shutil.rmtree(models_dir)
            _r.ok(f"Models directory removed: {models_dir}")

    _r.ok("Configuration reset")
    _r.next_steps([
        "darwin-admin setup interactive   # Re-run setup",
    ])


def cmd_setup_logging(_args: SetupLoggingArgs, _get_models) -> None:
    """Interactively reconfigure logging without re-running full setup."""
    from src.lib.AppPaths import default_logging_config

    _r.header("Reconfigure Logging")

    # Show current config
    cfg_path = default_logging_config()
    if cfg_path.exists():
        try:
            current = _json.loads(cfg_path.read_text(encoding="utf-8"))
            _r.section("Current")
            _r._out(f"    Mode:        {current.get('mode', 'terminal')}")
            _r._out(f"    Persist:     {current.get('persist', True)}")
            _r._out(f"    Filter:      {current.get('log_filter', 'all')}")
            _r._out(f"    Retention:   {current.get('retention_days', '?')} days")
            _r._out()
        except Exception:
            pass

    mode = _r.interactive_select(
        "Output mode",
        [
            (OutputMode.TERMINAL.value, "Console logs only (recommended)"),
            (OutputMode.SSE.value, "Stream logs via SSE (for MCP / remote)"),
            (OutputMode.BOTH.value, "Console + SSE simultaneously"),
            (OutputMode.NONE.value, "Silent (no log output)"),
        ],
        default=1,
    )

    # Persist
    _r._out()
    try:
        persist_raw = input("  Save logs to history? [y/n] (default: y): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        _r._out()
        raise KeyboardInterrupt
    persist = persist_raw != "n"

    log_filter = _r.interactive_select(
        "Log verbosity",
        [
            ("all", "Everything (default)"),
            ("quiet", "Errors + warnings + success only"),
            ("silent", "Errors only"),
        ],
        default=1,
    )

    ret_choice = _r.interactive_select(
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

    mode, sse_port, sse_host = _resolve_sse_port(mode)

    cfg = LoggingConfig(
        mode=mode,
        persist=persist,
        retention_days=retention,
        sse_port=sse_port,
        sse_host=sse_host,
        log_filter=log_filter,
    )

    atomic_write_json(cfg_path, cfg.to_dict())

    # Apply immediately
    from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator
    LoggerOrchestrator.configure(
        mode=mode,
        persist=persist,
        log_filter=log_filter,
        persist_config=True,
        override=True,
    )

    port_note = f", port {sse_port}" if mode in (OutputMode.SSE.value, OutputMode.BOTH.value) else ""
    _r.ok(f"Logging updated ({mode}, persist={persist}, filter={log_filter}, {retention}d retention{port_note})")


def _setup_preset(args: SetupPresetArgs, _get_models) -> None:
    """One-shot preset-based setup — resolve, download, configure."""
    models = _get_models(init_models=False)
    preset = args.preset

    _r.header(f"Setup: {preset}")

    # Show system info
    system = models.detect_system()
    _r.section("System")
    _r._out(f"    RAM: {system.ram_gb}GB | Disk: {system.disk_free_gb}GB | "
            f"CPU: {system.cpu_cores} cores | GPU: {system.gpu_name or 'None'}")
    _r._out(f"    Tier: {system.tier}")

    # Resolve and install via orchestrator
    results = models.setup_profile(
        profile=preset,
        auto_download=not args.no_download,
    )

    if not results:
        _r.err("Profile resolved to no models")
        return

    _r.section("Models")
    all_ok = True
    for cat, res in results.items():
        if res.success:
            _r.ok(f"{cat.value}: {res.model_id}")
        else:
            _r.err(f"{cat.value}: {res.error or 'failed'}")
            all_ok = False

    _setup_logging_defaults()

    if all_ok:
        _r.ok("Setup complete")
    else:
        _r.warn("Setup completed with some failures")

    _r.next_steps([
        "darwin mcp               # Start the MCP server",
        "darwin-admin status       # Check system status",
        "darwin-admin setup logging  # Reconfigure logging",
    ])


def _setup_dispatch(args, _get_models) -> None:
    """Dispatch setup — subcommand, preset, or default to interactive."""
    action = getattr(args, "action", None)

    if action in (SetupPreset.REQUIRED.value, SetupPreset.RECOMMENDED.value):
        _r.err(f"Did you mean: darwin-admin setup --preset {action}?")
        return

    dispatch_map = {
        SetupAction.VALIDATE.value: lambda: cmd_setup_validate(ns_to(SetupValidateArgs, args)),
        SetupAction.DOCTOR.value: lambda: cmd_setup_doctor(_get_models),
        SetupAction.RESET.value: lambda: cmd_setup_reset(ns_to(SetupResetArgs, args)),
        SetupAction.PROFILE.value: lambda: cmd_setup_profile(ns_to(SetupProfileArgs, args), _get_models),
        SetupAction.ONBOARD.value: lambda: cmd_setup_onboard(ns_to(SetupOnboardArgs, args), _get_models),
        SetupAction.LOGGING.value: lambda: cmd_setup_logging(ns_to(SetupLoggingArgs, args), _get_models),
    }

    handler = dispatch_map.get(action)
    if handler:
        handler()
    elif getattr(args, "preset", None):
        _setup_preset(ns_to(SetupPresetArgs, args), _get_models)
    else:
        cmd_setup_interactive(ns_to(SetupInteractiveArgs, args), _get_models)


def add_args(sub, _get_models) -> None:
    """Register setup subcommands."""
    p = sub.add_parser("setup", help="Setup and onboarding")
    p.add_argument(
        "action", nargs="?", default=None,
        choices=[a.value for a in SetupAction],
        help="Subcommand (default: interactive)",
    )
    p.add_argument(
        "--preset", choices=[p.value for p in SetupPreset],
        help="One-shot setup with a preset (downloads models)",
    )
    p.add_argument(
        "--no-download", action="store_true",
        help="Skip model downloads (assumes models are cached locally)",
    )
    p.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmations",
    )
    p.add_argument(
        "--profile", default="required",
        choices=[p.value for p in SetupPreset],
        help="Profile for 'profile' action (default: required)",
    )
    p.add_argument(
        "--preferences", "-p", nargs="+", metavar="CAT=MODEL_ID",
        help="Preferences for 'onboard' action",
    )
    p.add_argument(
        "--keep-models", action="store_true",
        help="Keep downloaded models on reset",
    )
    p.set_defaults(func=lambda a: _setup_dispatch(a, _get_models))
