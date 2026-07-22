"""
Darwin uninstall logic.

Safely discovers all stores, removes darwin artifacts from each,
cleans up the main data directory, and optionally uninstalls the pip package.

Two modes:
  --clean    Remove all darwin data (stores, indexes, configs, logs)
   --package  Also pip-uninstall darwin-rag after cleanup
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List,  Tuple

from src.admin.lib import UninstallArgs, ns_to
from src.admin.renderer import _r


# ══════════════════════════════════════════════════════════════
# Discovery
# ══════════════════════════════════════════════════════════════


def _discover_stores() -> List[dict]:
    """Load all registered stores from the persistent StoreRegistry."""
    try:
        from src.core.Storage.StoreRegistry import StoreRegistry
        reg = StoreRegistry()
        return reg.list_stores()
    except Exception:
        return []


def _get_base_dir() -> Path:
    """Get the platform-specific darwin base directory."""
    from src.lib.AppPaths import set_darwin_base_dir
    return set_darwin_base_dir()


def _scan_extra_store(dirty_path: Path) -> Tuple[bool, List[Path], str]:
    """Check if a path contains darwin artifacts.

    Returns (is_extra_store, artifacts_to_remove, description).
    An 'extra store' is a user directory that has darwin indexes inside it.
    """
    darwin_dir = dirty_path
    indexes_dir = darwin_dir / "indexes"
    registry_db = indexes_dir / "registry.db" if indexes_dir.exists() else None
    store_id_file = indexes_dir / ".darwin_store_id" if indexes_dir.exists() else None

    artifacts = []
    desc_parts = []

    # Check for indexes/ directory with content
    if indexes_dir.exists() and indexes_dir.is_dir():
        # Check for hash directories (pipeline artifacts)
        hash_dirs = [d for d in indexes_dir.iterdir()
                     if d.is_dir() and len(d.name) == 64]  # SHA-256 = 64 hex chars
        if hash_dirs or registry_db or store_id_file:
            artifacts.append(indexes_dir)
            desc_parts.append(f"indexes/ ({len(hash_dirs)} file hashes)")

    if registry_db and registry_db.exists():
        # Already captured as part of indexes/, but note it
        desc_parts.append(f"registry.db ({registry_db.stat().st_size // 1024}KB)")

    if store_id_file and store_id_file.exists():
        desc_parts.append(".darwin_store_id")

    # Also check for checkpoints/ and conversations/ dirs
    for subdir_name in ("checkpoints", "conversations"):
        subdir = darwin_dir / subdir_name
        if subdir.exists() and subdir.is_dir():
            artifacts.append(subdir)
            desc_parts.append(f"{subdir_name}/")

    is_store = len(artifacts) > 0
    return is_store, artifacts, "; ".join(desc_parts)


def _scan_base_dir(base: Path) -> List[Path]:
    """List all items in the base directory that darwin owns."""
    if not base.exists():
        return []

    owned = []
    for item in base.iterdir():
        owned.append(item)
    return owned


# ══════════════════════════════════════════════════════════════
# Cleanup
# ══════════════════════════════════════════════════════════════


def _remove_artifacts(artifacts: List[Path], dry_run: bool = False) -> List[str]:
    """Remove a list of files/directories. Returns log of actions."""
    log = []
    for path in artifacts:
        if not path.exists():
            continue
        if dry_run:
            log.append(f"[dry-run] would remove: {path}")
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
                log.append(f"removed directory: {path}")
            else:
                path.unlink()
                log.append(f"removed file: {path}")
        except Exception as e:
            log.append(f"FAILED to remove {path}: {e}")
    return log


def _remove_global_registry(dry_run: bool = False) -> List[str]:
    """Remove ~/.darwin/ (the global store registry)."""
    darwin_home = Path.home() / ".darwin"
    if not darwin_home.exists():
        return ["~/.darwin/ not found (nothing to remove)"]

    # Safety check: validate it looks like a darwin registry
    expected = {"stores_registry.db", "config"}
    actual = {d.name for d in darwin_home.iterdir()} if darwin_home.is_dir() else set()
    if not (expected & actual):
        return [
            f"WARNING: ~/.darwin/ doesn't look like a darwin registry "
            f"(contents: {actual}). Skipping."
        ]

    if dry_run:
        return [f"[dry-run] would remove {darwin_home}"]

    try:
        shutil.rmtree(darwin_home)
        return [f"removed {darwin_home}"]
    except Exception as e:
        return [f"FAILED to remove {darwin_home}: {e}"]


def _remove_base_dir(base: Path, dry_run: bool = False) -> List[str]:
    """Safely remove the main darwin base directory."""
    if not base.exists():
        return [f"{base} not found (nothing to remove)"]

    # Safety check: only remove if it looks like a darwin dir
    expected_subdirs = {"config", "models", "logs", "indexes"}
    actual_subdirs = {d.name for d in base.iterdir() if d.is_dir()}
    darwin_files = {f.name for f in base.iterdir() if f.is_file()}

    # Must have at least some darwin indicators
    has_darwin_content = (
        expected_subdirs & actual_subdirs
        or any(f.endswith(".json") for f in darwin_files)
    )

    if not has_darwin_content:
        return [
            f"WARNING: {base} doesn't look like a darwin directory "
            f"(contents: {actual_subdirs | darwin_files}). Skipping."
        ]

    if dry_run:
        items = list(base.iterdir())
        return [f"[dry-run] would remove {base}/ ({len(items)} items)"]

    try:
        shutil.rmtree(base)
        return [f"removed {base}"]
    except Exception as e:
        return [f"FAILED to remove {base}: {e}"]


def _pip_uninstall() -> bool:
    """Run pip uninstall darwin-rag --yes."""
    _r.info("Running: pip uninstall darwin-rag --yes")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "darwin-rag", "--yes"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            _r.ok("darwin-rag uninstalled")
            return True
        else:
            _r.err(f"pip uninstall failed: {result.stderr.strip()}")
            return False
    except Exception as e:
        _r.err(f"pip uninstall error: {e}")
        return False


# ══════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════


def cmd_uninstall(args: UninstallArgs):
    """Full darwin uninstall: clean data, optionally pip uninstall."""
    dry_run = args.dry_run
    skip_package = args.keep_package

    _r.header("Darwin Uninstall")
    if dry_run:
        _r.warn("DRY RUN — no changes will be made")

    all_logs = []

    # ── Step 1: Discover extra stores ──
    _r.section("Step 1: Discovering stores")
    stores = _discover_stores()
    base = _get_base_dir()

    extra_stores = []
    main_store = None

    for store_info in stores:
        darwin_dir = Path(store_info["darwin_dir"])
        is_default = store_info.get("is_default", False)
        name = store_info.get("name", store_info.get("slug", "?"))

        if darwin_dir == base or is_default:
            main_store = store_info
            _r._out(f"  main store: {darwin_dir}  {_r._dim(f'(name={name})')}")
        else:
            is_extra, artifacts, desc = _scan_extra_store(darwin_dir)
            if is_extra:
                extra_stores.append((darwin_dir, artifacts, desc, name))
                _r._out(f"  extra store: {darwin_dir}  {_r._dim(f'{name}: {desc}')}")

    if not extra_stores and main_store is None:
        _r._out(f"  {_r._dim('no registered stores found')}")

    # Also scan the base directory
    base_items = _scan_base_dir(base) if base.exists() else []
    if base_items:
        _r._out(f"  base dir: {base}  {_r._dim(f'{len(base_items)} items')}")

    # ── Step 2: Clean extra stores (remove artifacts, keep user files) ──
    if extra_stores:
        _r.section("Step 2: Cleaning extra stores")
        for darwin_dir, artifacts, desc, name in extra_stores:
            _r.info(f"Cleaning {name} ({darwin_dir})...")
            logs = _remove_artifacts(artifacts, dry_run=dry_run)
            all_logs.extend(logs)
            for log in logs:
                _r._out(f"    {log}")

    # ── Step 3: Remove global store registry ──
    _r.section("Step 3: Removing global store registry")
    logs = _remove_global_registry(dry_run=dry_run)
    all_logs.extend(logs)
    for log in logs:
        _r._out(f"  {log}")

    # ── Step 4: Remove main base directory ──
    _r.section("Step 4: Removing main data directory")
    logs = _remove_base_dir(base, dry_run=dry_run)
    all_logs.extend(logs)
    for log in logs:
        _r._out(f"  {log}")

    # ── Step 5: Pip uninstall ──
    if not skip_package:
        _r.section("Step 5: Uninstalling darwin-rag package")
        if dry_run:
            _r._out("  [dry-run] would run: pip uninstall darwin-rag --yes")
        else:
            _pip_uninstall()

    # ── Summary ──
    _r._out()
    _r.header("Summary")
    errors = [l for l in all_logs if "FAILED" in l]
    if errors:
        for e in errors:
            _r.err(e)
    else:
        _r.ok("Darwin uninstalled successfully")

    if dry_run:
        _r._out()
        _r.info("This was a dry run. Run without --dry-run to apply changes.")


def add_args(sub):
    """Register the uninstall subcommand."""
    p = sub.add_parser("uninstall", help="Remove all darwin data and optionally the package")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be removed without making changes",
    )
    p.add_argument(
        "--keep-package", action="store_true",
        help="Only clean data, don't pip-uninstall darwin-rag",
    )
    p.set_defaults(func=lambda a: cmd_uninstall(ns_to(UninstallArgs, a)))
