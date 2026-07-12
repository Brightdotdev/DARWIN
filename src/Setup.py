"""
Darwin Setup — Onboarding and configuration for the Darwin RAG MCP server.

Command Hierarchy
─────────────────
  darwin setup                    -> Guided interactive onboarding
  darwin setup preset <name>      -> One-shot preset (non-interactive)
  darwin setup models             -> Configure / reconfigure models
  darwin setup embeddings         -> Configure embedding model only
  darwin setup logs               -> Configure logging
  darwin setup storage            -> Configure storage directory
  darwin setup paths              -> Show all derived paths
  darwin setup validate           -> Validate current configuration
  darwin setup status             -> Show current state, disk usage, active models
  darwin setup reset              -> Clear all config and start fresh
  darwin setup doctor             -> Diagnose common issues

Architecture
────────────
  DarwinSetup        -> Pure-Python programmatic API. Zero CLI dependencies.
  SetupRenderer      -> Human-facing output. JSON mode bypasses entirely.
  SetupFlow          -> Explicit state machine for the setup run.
  Click commands     -> Thin wrappers that call DarwinSetup methods.
"""

from __future__ import annotations

import json
import os
import sys
import time
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# ─────────────────────────────────────────────────────────────────────────
# Darwin imports
# ─────────────────────────────────────────────────────────────────────────
from src.lib.AppPaths import (
    set_darwin_base_dir,
    default_config_dir,
    default_models_dir,
    default_logs_dir,
    is_running_in_docker,
    ensure_directories,
)
from src.core.Orchestrators.ModelOrchestrator import (
    ModelsOrchestrator,
    modelsOrchestrator,
    OperationResult,
)
from src.core.Models.Lib.ModelRegistry import (
    find_spec,
    get_specs_by_category,
)
from src.core.Models.Lib.ModelSpec import ModelCategory, ModelTier
from src.core.Logger.DarwinLogger import darwin, LogFilter, OutputMode, LogType
from src.core.Orchestrators.LoggerOrchestrator import LoggerOrchestrator

# ─────────────────────────────────────────────────────────────────────────
# Lazy Click import
# ─────────────────────────────────────────────────────────────────────────
_click = None


def _get_click():
    global _click
    if _click is None:
        try:
            import click
            _click = click
        except ImportError as exc:
            # Fallback: print to stderr directly since logger may not be configured
            print("Click is required. Install: pip install click", file=sys.stderr)
            return None
    return _click


# ═══════════════════════════════════════════════════════════════════════
# Enums & Constants
# ═══════════════════════════════════════════════════════════════════════


class SetupPhase(Enum):
    """Ordered phases of the setup process."""
    VALIDATE = ("Validate environment", 1)
    RESOLVE = ("Resolve models", 2)
    INSTALL = ("Download & install", 3)
    CONFIGURE = ("Configure logging", 4)
    SAVE = ("Save configuration", 5)
    VERIFY = ("Verify setup", 6)

    def __init__(self, label: str, order: int):
        self.label = label
        self.order = order


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SetupConfig:
    """Configuration bundle for Darwin onboarding."""
    base_dir: Optional[Path] = None
    ignore_env_checks: bool = False
    preset: Optional[str] = None
    embedding_model: Optional[str] = None
    selected_models: Dict[str, List[str]] = field(default_factory=dict)
    api_keys: Dict[str, str] = field(default_factory=dict)
    log_level: str = "normal"
    log_retention: Optional[int] = 14
    log_persist: bool = True
    non_interactive: bool = False
    force: bool = False
    timeout: int = 300
    dry_run: bool = False
    json_output: bool = False
    plain: bool = False  # --plain / --no-color


@dataclass(slots=True)
class SetupResult:
    """Result of a setup run."""
    success: bool = False
    config_path: Optional[Path] = None
    installed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "config_path": str(self.config_path) if self.config_path else None,
            "installed": self.installed,
            "failed": self.failed,
            "skipped": self.skipped,
            "messages": self.messages,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass(frozen=True, slots=True)
class DiskImpact:
    """Disk space analysis for model installation."""
    required_mb: int
    available_mb: int
    after_mb: int
    warning: Optional[str] = None

    @property
    def is_safe(self) -> bool:
        return self.after_mb > 500  # Keep 500MB buffer


@dataclass(frozen=True, slots=True)
class PhaseResult:
    """Result of a single setup phase."""
    phase: SetupPhase
    success: bool
    elapsed: float
    details: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# SetupRenderer — Human-facing output
# ═══════════════════════════════════════════════════════════════════════


class SetupRenderer:
    """
    Handles all human-facing output for Darwin Setup.
    JSON mode bypasses this entirely.

    CRITICAL: This renderer writes DIRECTLY to the terminal via click.echo().
    It does NOT route through DarwinLogger. Mixing click.style() with
    DarwinLogger's TerminalUI causes double-ANSI and literal escape sequences
    on Windows (OSError 9 / Bad file descriptor territory).
    """

    def __init__(self, json_output: bool = False, plain: bool = False):
        self.json_output = json_output
        self.plain = plain
        self._click = _get_click()
        self._supports_color = self._detect_color()
        self._supports_overwrite = self._detect_overwrite()
        self._current_phase: Optional[SetupPhase] = None
        self._phase_start: float = 0.0

    # ── Terminal capability detection ────────────────────────────────

    def _detect_color(self) -> bool:
        if self.plain:
            return False
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("FORCE_COLOR"):
            return True
        return sys.stdout.isatty()

    def _detect_overwrite(self) -> bool:
        if self.plain:
            return False
        return sys.stdout.isatty() and not os.environ.get("TERM", "").startswith("dumb")

    # ── Core output primitive ──────────────────────────────────────
    # Everything goes through here. No DarwinLogger. No double-wrapping.

    def _out(self, text: str = "", err: bool = False) -> None:
        """Write a line directly to stdout/stderr. Bypasses logger entirely."""
        if self.json_output:
            return
        # Use click.echo if available (handles unicode on Windows better)
        if self._click:
            self._click.echo(text, err=err)
        else:
            stream = sys.stderr if err else sys.stdout
            print(text, file=stream)

    # ── Style helpers ────────────────────────────────────────────────

    def _style(self, text: str, fg: Optional[str] = None, bold: bool = False, dim: bool = False) -> str:
        if not self._supports_color or not self._click:
            return text
        try:
            return self._click.style(text, fg=fg, bold=bold, dim=dim)
        except Exception:
            return text

    def _dim(self, text: str) -> str:
        return self._style(text, dim=True)

    def _green(self, text: str) -> str:
        return self._style(text, fg="green")

    def _red(self, text: str) -> str:
        return self._style(text, fg="red")

    def _yellow(self, text: str) -> str:
        return self._style(text, fg="yellow")

    def _bright(self, text: str) -> str:
        return self._style(text, fg="bright_white", bold=True)

    def _cyan(self, text: str) -> str:
        return self._style(text, fg="cyan")

    # ── Section formatting ───────────────────────────────────────────

    def _divider(self, width: int = 60) -> str:
        return "─" * width

    def _header(self, title: str) -> None:
        if self.json_output:
            return
        self._out(self._divider())
        self._out(f"  {self._bright(title)}")
        self._out(self._divider())

    def _subheader(self, title: str) -> None:
        if self.json_output:
            return
        self._out(f"{self._bright(title)}")

    # ── Phase tracking ───────────────────────────────────────────────

    @contextmanager
    def phase(self, phase: SetupPhase, current: int, total: int):
        """Context manager for phase-aware progress reporting."""
        if self.json_output:
            yield
            return

        self._current_phase = phase
        self._phase_start = time.monotonic()

        status = f"[{current}/{total}]"
        self._out(f"{self._cyan(status)} {phase.label} ...")

        try:
            yield
            elapsed = time.monotonic() - self._phase_start
            self._out(f"  {self._green('✓')} {phase.label} — {elapsed:.1f}s")
        except Exception as exc:
            elapsed = time.monotonic() - self._phase_start
            self._out(f"  {self._red('✗')} {phase.label} — {elapsed:.1f}s", err=True)
            raise

    # ── Model tables ─────────────────────────────────────────────────

    def model_table(self, category: ModelCategory, models: List[Any], required: bool = False, default_idx: int = 0) -> None:
        """Print a model selection table."""
        if self.json_output:
            return

        tag = "Required" if required else "Optional"
        self._out(f"{self._bright(category.value.title())} — {self._dim(tag)}")

        for i, spec in enumerate(models, 1):
            kind = "Local" if spec.is_local else "API"
            provider = f" · {spec.requires_provider}" if spec.requires_provider else ""
            size = ""
            if spec.is_local:
                size = f" · {spec.ram_gb}GB RAM" if spec.ram_gb else ""
                size += f" · {spec.disk_mb}MB" if spec.disk_mb else ""
            notes = f"  — {spec.notes}" if spec.notes else ""
            marker = self._green("→") if i == default_idx else " "
            self._out(f"  {marker} [{i}] {spec.model_id}")
            self._out(f"      {self._dim(f'{kind}{provider}{size}{notes}')}")

        if not required:
            self._out(f"      {self._dim('[0] Skip this category')}")

    # ── Disk impact ──────────────────────────────────────────────────

    def disk_impact(self, impact: DiskImpact) -> None:
        """Show disk space analysis."""
        if self.json_output:
            return

        required_gb = impact.required_mb / 1024
        available_gb = impact.available_mb / 1024
        after_gb = impact.after_mb / 1024

        self._out(f"{self._bright('Storage Impact')}")
        self._out(f"    Required:  {required_gb:.1f}GB")
        self._out(f"    Available: {available_gb:.1f}GB")
        self._out(f"    After:     {after_gb:.1f}GB")

        if impact.warning:
            self._out(f"    {self._yellow('⚠ ' + impact.warning)}")

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self, prefs: Dict[str, str], base_dir: Optional[Path], disk_impact: Optional[DiskImpact], log_level: str, retention: Optional[int]) -> None:
        """Show pre-installation summary."""
        if self.json_output:
            return

        self._header("Setup Summary")
        self._out(f"  Storage:     {base_dir or 'Default'}")
        env = detect_environment()
        self._out(f"  Environment: {'Docker' if env.get('docker') else 'Standard'}")

        self._out(f"{self._bright('Models')}")
        for cat, model_id in prefs.items():
            label = model_id or "(skipped)"
            spec = find_spec(model_id) if model_id else None
            kind = f" [{self._dim('Local' if spec and spec.is_local else 'API')}]" if spec else ""
            self._out(f"    {cat.title()}: {label}{kind}")

        if disk_impact:
            self._out(f"{self._bright('Disk')}")
            self._out(f"    After install: {disk_impact.after_mb / 1024:.1f}GB free")

        self._out(f"{self._bright('Logging')}")
        self._out(f"    Level:     {log_level}")
        self._out(f"    Retention: {retention or 'forever'} days")

    # ── Confirmation prompts ─────────────────────────────────────────

    def confirm_install(self, existing: bool = False) -> bool:
        """Context-aware installation confirmation."""
        if self.json_output or not self._click:
            return True

        c = self._click
        if existing:
            return c.confirm(
                self._style("  Update existing configuration?", fg="bright_white"),
                default=True,
            )
        return c.confirm(
            self._style("  Proceed with installation?", fg="bright_white"),
            default=False,  # Safe default for destructive ops
        )

    def confirm_preset(self, preset: str, total_mb: int) -> bool:
        """Preset-specific confirmation."""
        if self.json_output or not self._click:
            return True

        c = self._click
        if total_mb:
            self._out(f"  Total local download: ~{total_mb}MB")
        return c.confirm(
            self._style("  Proceed with download and installation?", fg="bright_white"),
            default=False,
        )

    # ── Structured failures ──────────────────────────────────────────

    def failure(self, error: str, reason: Optional[str] = None, fix: Optional[str] = None, skip: Optional[str] = None) -> None:
        """Print an actionable failure report."""
        if self.json_output:
            return

        self._out(f"{self._red('[ERR]')} {error}", err=True)
        if reason:
            self._out(f"        Reason:  {reason}", err=True)
        if fix:
            self._out(f"        Fix:     {self._cyan(fix)}", err=True)
        if skip:
            self._out(f"        Skip:    {self._dim(skip)}", err=True)

    # ── Results ──────────────────────────────────────────────────────

    def show_result(self, result: SetupResult) -> None:
        """Display setup result with next steps."""
        if self.json_output:
            return

        if result.success:
            self._out(f"\n  {self._green('Setup Complete')}")
            self._out(f"  Config saved to {result.config_path}")
            self._next_steps()
        else:
            self._out(f"\n  {self._red('Setup Failed')}", err=True)
            for err in result.errors:
                self._out(f"    • {err}", err=True)

    def _next_steps(self) -> None:
        """Post-setup guidance."""
        self._out(f"\n  {self._bright('Next steps')}")
        self._out(f"    darwin status          # See what's active")
        self._out(f"    darwin ingest --help   # Start adding documents")
        self._out(f"    darwin serve           # Start the MCP server")
        self._out(f"\n  {self._dim('Docs: https://docs.darwin.dev/quickstart')}")

    # ── Dry-run preview ──────────────────────────────────────────────

    def dry_run_preview(self, actions: List[Tuple[str, str, Optional[int], str]]) -> None:
        """Show dry-run action table."""
        if self.json_output:
            return

        self._header("Dry Run — No changes will be made")
        self._out(f"  {'Action':12} {'Model':30} {'Size':10} {'Status'}")
        self._out(f"  {'─' * 11} {'─' * 29} {'─' * 9} {'─' * 20}")
        for action, model, size, status in actions:
            size_str = f"{size}MB" if size else "—"
            self._out(f"  {action:12} {model:30} {size_str:10} {self._dim(status)}")

    # ── Validation results ───────────────────────────────────────────

    def validation(self, result: Dict[str, Any]) -> None:
        """Display validation results."""
        if self.json_output:
            return

        self._header("Validation Results")
        for item in result.get("ok", []):
            self._out(f"  {self._green('✓')} {item}")
        for item in result.get("warnings", []):
            self._out(f"  {self._yellow('!')} {item}")
        for item in result.get("issues", []):
            self._out(f"  {self._red('✗')} {item}", err=True)

        status = "Valid" if result["valid"] else "Invalid"
        fg = "green" if result["valid"] else "red"
        self._out(f"\n  {self._style(f'Status: {status}', fg=fg, bold=True)}")

    # ── Existing config prompt ───────────────────────────────────────

    def existing_config_prompt(self, existing: Dict[str, Any]) -> Optional[str]:
        """Show existing config and ask what to do. Returns choice: keep, update, full."""
        if self.json_output or not self._click:
            return "full"

        c = self._click
        self._out(f"\n  {self._bright('Existing Configuration Detected')}")
        self._out(f"    Setup: {existing.get('setup_completed', 'unknown')}")

        models = existing.get("models", {})
        if models:
            self._out(f"\n    {self._bright('Active Models')}")
            for cat, model_id in models.items():
                self._out(f"      {cat.title()}: {model_id}")

        self._out(f"\n    [1] Keep existing config")
        self._out(f"    [2] Update models only")
        self._out(f"    [3] Full reconfiguration")

        choice = c.prompt(
            "  Choice",
            type=c.Choice(["1", "2", "3"]),
            default="1",
        )
        return {"1": "keep", "2": "update", "3": "full"}[choice]

    # ── Status display ───────────────────────────────────────────────

    def status(self, config: Dict[str, Any], orch: ModelsOrchestrator) -> None:
        """Show current Darwin status."""
        if self.json_output:
            return

        self._header("Darwin Status")

        base = config.get("base_dir", str(set_darwin_base_dir()))
        self._out(f"  Base dir:    {base}")
        self._out(f"  Config:      {default_config_dir() / 'darwin.json'}")

        models = config.get("models", {})
        if models:
            self._out(f"\n  {self._bright('Models')}")
            for cat, model_id in models.items():
                spec = find_spec(model_id)
                status = "✓" if spec and (not spec.is_local or orch.is_downloaded(spec.category, spec.model_id)) else "✗"
                color = self._green if status == "✓" else self._red
                self._out(f"    {color(status)} {cat.title()}: {model_id}")

        log = config.get("logging", {})
        if log:
            self._out(f"\n  {self._bright('Logging')}")
            self._out(f"    Level:     {log.get('level', 'normal')}")
            self._out(f"    Retention: {log.get('retention_days', 14)} days")

    # ── Doctor results ───────────────────────────────────────────────

    def doctor(self, checks: List[Tuple[str, bool, Optional[str]]]) -> None:
        """Display doctor check results."""
        if self.json_output:
            return

        self._header("Darwin Doctor")
        for name, ok, fix in checks:
            if ok:
                self._out(f"  {self._green('✓')} {name}")
            else:
                self._out(f"  {self._red('✗')} {name}", err=True)
                if fix:
                    self._out(f"    {self._cyan('→')} {fix}", err=True)


# ═══════════════════════════════════════════════════════════════════════
# Disk impact calculator
# ═══════════════════════════════════════════════════════════════════════


def calculate_disk_impact(preferences: Dict[ModelCategory, str]) -> DiskImpact:
    """Calculate disk space impact of model installation."""
    required_mb = 0
    for model_id in preferences.values():
        spec = find_spec(model_id)
        if spec and spec.is_local and spec.disk_mb:
            required_mb += spec.disk_mb

    base = set_darwin_base_dir()
    try:
        stat = os.statvfs(base)
        available_mb = (stat.f_bavail * stat.f_frsize) // (1024 * 1024)
    except Exception:
        available_mb = 0

    after_mb = available_mb - required_mb
    warning = None
    if after_mb < 500:
        warning = f"Low disk space ({after_mb}MB remaining)"
    elif after_mb < 1024:
        warning = f"Disk space getting tight ({after_mb}MB remaining)"

    return DiskImpact(
        required_mb=required_mb,
        available_mb=available_mb,
        after_mb=after_mb,
        warning=warning,
    )


# ═══════════════════════════════════════════════════════════════════════
# Environment detection
# ═══════════════════════════════════════════════════════════════════════


def detect_environment() -> Dict[str, Any]:
    """Detect runtime environment characteristics."""
    env = {
        "docker": is_running_in_docker(),
        "ci": any(
            os.environ.get(k)
            for k in [
                "CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL",
                "TRAVIS", "CIRCLECI", "BUILDKITE", "DRONE", "APPVEYOR",
            ]
        ),
        "ssh": any(
            os.environ.get(k) for k in ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"]
        ),
        "kubernetes": "KUBERNETES_SERVICE_HOST" in os.environ,
        "platform": sys.platform,
    }
    env["ephemeral"] = env["docker"] or env["ci"] or env["kubernetes"]
    env["headless"] = env["ci"] or env["ssh"] or not sys.stdin.isatty()
    return env


# ═══════════════════════════════════════════════════════════════════════
# DarwinSetup — Programmatic API (shared core)
# ═══════════════════════════════════════════════════════════════════════


class DarwinSetup:
    """Programmatic onboarding API. No CLI dependencies."""

    CONFIG_FILE = "darwin.json"
    CONFIG_VERSION = "1.0"
    PRESETS = ["minimal", "recommended", "complete", "local-only", "api-only", "server"]

    PRESET_DESCRIPTIONS = {
        "minimal": "Embedding only. Fastest local setup.",
        "recommended": "Embedding + Reranker. No spaCy. Best for most users.",
        "complete": "Everything: embedding + reranker + NLP + OCR.",
        "local-only": "All local models. No API keys required.",
        "api-only": "API models only. Fastest setup, requires keys.",
        "server": "Light local embedding + API vision. Server-optimized.",
    }

    def __init__(self, orchestrator: Optional[ModelsOrchestrator] = None):
        self.orch = orchestrator or modelsOrchestrator
        self._config: Dict[str, Any] = {}
        self.result = SetupResult()

    # ── Config I/O (atomic) ──────────────────────────────────────────

    def _config_path(self) -> Path:
        return default_config_dir() / self.CONFIG_FILE

    def load_config(self) -> Optional[Dict[str, Any]]:
        path = self._config_path()
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                self.result.warnings.append(f"Config file corrupted: {exc}")
            except Exception as exc:
                self.result.warnings.append(f"Could not read config: {exc}")
        return None

    def save_config(self, config: Dict[str, Any]) -> Path:
        """Atomically write config to disk."""
        path = self._config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        config["version"] = self.CONFIG_VERSION
        config["setup_completed"] = datetime.now().isoformat()

        temp_path = path.with_suffix(".tmp")
        try:
            temp_path.write_text(
                json.dumps(config, indent=2, default=str),
                encoding="utf-8",
            )
            temp_path.chmod(0o600)
            # Windows: os.fsync on read handle fails with OSError 9.
            # NTFS rename is atomic enough; skip fsync on Windows.
            if sys.platform != "win32":
                with temp_path.open("r+b") as f:
                    os.fsync(f.fileno())
            os.replace(temp_path, path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
        return path

    def is_configured(self) -> bool:
        return self._config_path().exists()

    # ── Environment & Storage ────────────────────────────────────────

    def validate_storage(
        self, base_dir: Optional[Path] = None, ignore: bool = False
    ) -> bool:
        if ignore:
            if base_dir:
                set_darwin_base_dir(base_dir)
            return True

        env = detect_environment()
        if env["ephemeral"] and not base_dir:
            self.result.errors.append(
                "Ephemeral environment detected. A persistent --base-dir is required. "
                "Use --yolo to skip this check."
            )
            self.result.success = False
            return False

        if base_dir:
            set_darwin_base_dir(base_dir)

        if not ensure_directories():
            self.result.errors.append(
                "Failed to create or write to Darwin directories."
            )
            self.result.success = False
            return False
        return True

    def set_storage(self, base_dir: Optional[Path] = None) -> Path:
        """Set or update the storage directory."""
        if base_dir:
            set_darwin_base_dir(base_dir)
        ensure_directories()
        return set_darwin_base_dir()

    # ── Preset resolution ─────────────────────────────────────────────

    def resolve_preset(self, preset: str) -> Dict[ModelCategory, str]:
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Choose from: {self.PRESETS}")

        system = self.orch.detect_system()
        recs = self.orch.get_recommendations(system)
        prefs: Dict[ModelCategory, str] = {}

        if preset == "minimal":
            if recs.embedding:
                prefs[ModelCategory.EMBEDDING] = recs.embedding

        elif preset == "recommended":
            for attr, cat in [
                ("embedding", ModelCategory.EMBEDDING),
                ("reranker", ModelCategory.RERANKER),
            ]:
                model_id = getattr(recs, attr)
                if model_id:
                    prefs[cat] = model_id

        elif preset == "complete":
            for attr, cat in [
                ("embedding", ModelCategory.EMBEDDING),
                ("reranker", ModelCategory.RERANKER),
                ("nlp", ModelCategory.NLP),
                ("ocr", ModelCategory.OCR),
            ]:
                model_id = getattr(recs, attr)
                if model_id:
                    prefs[cat] = model_id

        elif preset == "local-only":
            for cat in ModelCategory:
                specs = get_specs_by_category(cat)
                local = [s for s in specs if s.is_local]
                if local:
                    best = min(local, key=lambda s: (s.ram_gb or 999, s.disk_mb or 999))
                    prefs[cat] = best.model_id

        elif preset == "api-only":
            for cat in [ModelCategory.EMBEDDING, ModelCategory.VISION]:
                specs = get_specs_by_category(cat)
                api = [s for s in specs if s.is_api]
                if api:
                    best = min(api, key=lambda s: s.tier.value)
                    prefs[cat] = best.model_id

        elif preset == "server":
            emb_specs = get_specs_by_category(ModelCategory.EMBEDDING)
            local_emb = [s for s in emb_specs if s.is_local]
            if local_emb:
                prefs[ModelCategory.EMBEDDING] = min(
                    local_emb, key=lambda s: s.ram_gb or 999
                ).model_id
            vis_specs = get_specs_by_category(ModelCategory.VISION)
            cheap_vis = [
                s for s in vis_specs if s.is_api and s.tier == ModelTier.DOLPHIN
            ]
            if cheap_vis:
                prefs[ModelCategory.VISION] = cheap_vis[0].model_id

        return prefs

    # ── Model installation ───────────────────────────────────────────

    def install_model(
        self, model_id: str, force: bool = False, timeout: int = 300
    ) -> OperationResult:
        spec = find_spec(model_id)
        if not spec:
            return OperationResult(
                success=False,
                model_id=model_id,
                error=f"Model '{model_id}' not found in registry",
            )

        # Retry with exponential backoff
        for attempt in range(1, 4):
            result = self.orch.install(
                model_id,
                auto_download=True,
                force=force,
                set_active=True,
            )
            if result.success:
                return result
            if attempt < 3:
                delay = 2 ** attempt  # 2s, 4s
                time.sleep(delay)
        return result

    def install_models(
        self,
        preferences: Dict[ModelCategory, str],
        force: bool = False,
        timeout: int = 300,
    ) -> None:
        for cat, model_id in preferences.items():
            if not model_id:
                self.result.skipped.append(cat.value)
                continue
            op = self.install_model(model_id, force=force, timeout=timeout)
            if op.success:
                self.result.installed.append(model_id)
                self.result.messages.append(f"Installed {model_id}")
            else:
                self.result.failed.append(model_id)
                self.result.errors.append(f"Failed to install {model_id}: {op.error}")

    # ── Logging configuration ────────────────────────────────────────

    def configure_logging(
        self, level: str = "normal", retention: Optional[int] = 14, persist: bool = True
    ) -> None:
        filter_map = {
            "normal": LogFilter.ALL,
            "quiet": LogFilter.QUIET,
            "silent": LogFilter.SILENT,
        }
        log_filter = filter_map.get(level, LogFilter.ALL)
        mode = (
            OutputMode.TERMINAL
            if not detect_environment()["headless"]
            else OutputMode.NONE
        )

        LoggerOrchestrator.configure(
            mode=mode.value,
            persist=persist,
            show_banner=False,
            log_filter=log_filter.value,
            persist_config=True,
            override=True,
        )

        self.result.config["logging"] = {
            "mode": mode.value,
            "level": level,
            "retention_days": retention,
            "persist": persist,
        }

    # ── Validation ───────────────────────────────────────────────────

    def validate_config(
        self, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate a configuration dict (or current saved config)."""
        if config is None:
            config = self.load_config() or {}

        issues: List[str] = []
        warnings: List[str] = []
        ok: List[str] = []

        # Storage
        base = Path(config.get("base_dir", set_darwin_base_dir()))
        if base.exists() and os.access(base, os.W_OK):
            ok.append(f"Storage writable: {base}")
        else:
            issues.append(f"Storage not writable: {base}")

        # Models
        models = config.get("models", {})
        if not models.get("embedding"):
            issues.append("No embedding model configured")
        else:
            spec = find_spec(models["embedding"])
            if spec:
                ok.append(f"Embedding model registered: {spec.model_id}")
                if spec.is_local and not self.orch.is_downloaded(
                    spec.category, spec.model_id
                ):
                    issues.append(f"Embedding model not downloaded: {spec.model_id}")
            else:
                issues.append(f"Unknown embedding model: {models['embedding']}")

        # Logging
        log = config.get("logging", {})
        if log.get("level"):
            ok.append(f"Logging level: {log['level']}")
        else:
            warnings.append("Logging not configured")

        return {
            "valid": len(issues) == 0,
            "ok": ok,
            "warnings": warnings,
            "issues": issues,
        }

    # ── Doctor ───────────────────────────────────────────────────────

    def doctor(self) -> List[Tuple[str, bool, Optional[str]]]:
        """Run diagnostic checks and return (name, ok, fix_message)."""
        checks: List[Tuple[str, bool, Optional[str]]] = []

        # Storage
        base = set_darwin_base_dir()
        checks.append((
            f"Storage directory writable ({base})",
            base.exists() and os.access(base, os.W_OK),
            f"mkdir -p {base} && chmod 755 {base}"
        ))

        # Config
        config = self.load_config()
        checks.append((
            "Configuration file exists",
            config is not None,
            "Run `darwin setup` to create config"
        ))

        # Embedding model
        if config:
            emb = config.get("models", {}).get("embedding")
            if emb:
                spec = find_spec(emb)
                checks.append((
                    f"Embedding model '{emb}' registered",
                    spec is not None,
                    f"Run `darwin setup embeddings <model>`"
                ))
                if spec and spec.is_local:
                    checks.append((
                        f"Embedding model '{emb}' downloaded",
                        self.orch.is_downloaded(spec.category, spec.model_id),
                        f"Run `darwin setup models {emb}`"
                    ))

        # Disk space
        try:
            stat = os.statvfs(base)
            free_mb = (stat.f_bavail * stat.f_frsize) // (1024 * 1024)
            checks.append((
                f"Disk space ({free_mb}MB free)",
                free_mb > 500,
                "Free up disk space"
            ))
        except Exception:
            checks.append(("Disk space check", False, "Could not check disk space"))

        return checks

    # ── Status ───────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get current Darwin status for display."""
        config = self.load_config() or {}
        return {
            "config": config,
            "base_dir": str(set_darwin_base_dir()),
            "config_path": str(self._config_path()),
            "models_dir": str(default_models_dir()),
            "logs_dir": str(default_logs_dir()),
        }

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self, keep_models: bool = False) -> SetupResult:
        """Reset Darwin configuration. Optionally keep downloaded models."""
        self.result = SetupResult()
        config_path = self._config_path()

        if config_path.exists():
            backup_path = config_path.with_suffix(f".backup.{int(time.time())}.json")
            shutil.copy2(config_path, backup_path)
            self.result.messages.append(f"Config backed up to {backup_path}")
            config_path.unlink()

        if not keep_models:
            models_dir = default_models_dir()
            if models_dir.exists():
                shutil.rmtree(models_dir)
                self.result.messages.append(f"Models directory removed: {models_dir}")

        self.result.success = True
        self.result.messages.append("Darwin configuration reset")
        return self.result

    # ── Core run logic (state machine) ───────────────────────────────

    def run(self, cfg: SetupConfig, renderer: Optional[SetupRenderer] = None) -> SetupResult:
        """Execute full onboarding based on a SetupConfig."""
        self.result = SetupResult()
        self.result.config = asdict(cfg)
        renderer = renderer or SetupRenderer(json_output=cfg.json_output, plain=cfg.plain)

        phases = [SetupPhase.VALIDATE, SetupPhase.RESOLVE, SetupPhase.INSTALL, SetupPhase.CONFIGURE, SetupPhase.SAVE]
        phase_results: List[PhaseResult] = []

        # Phase 1: Validate
        with renderer.phase(SetupPhase.VALIDATE, 1, len(phases)):
            if not self.validate_storage(cfg.base_dir, cfg.ignore_env_checks):
                return self.result
            phase_results.append(PhaseResult(SetupPhase.VALIDATE, True, 0.0))

        existing = self.load_config()
        preferences: Dict[ModelCategory, str] = {}

        # Phase 2: Resolve
        with renderer.phase(SetupPhase.RESOLVE, 2, len(phases)):
            if cfg.preset:
                try:
                    preferences = self.resolve_preset(cfg.preset)
                    self.result.messages.append(f"Applied preset: {cfg.preset}")
                except ValueError as e:
                    self.result.errors.append(str(e))
                    return self.result
            else:
                if cfg.embedding_model:
                    preferences[ModelCategory.EMBEDDING] = cfg.embedding_model
                for cat_str, model_ids in cfg.selected_models.items():
                    try:
                        cat = ModelCategory[cat_str.upper()]
                        if model_ids:
                            preferences[cat] = model_ids[0]
                    except KeyError:
                        self.result.warnings.append(f"Unknown category: {cat_str}")

            if not preferences.get(ModelCategory.EMBEDDING):
                self.result.errors.append("An embedding model is required.")
                return self.result

            for cat, model_id in preferences.items():
                spec = find_spec(model_id)
                if not spec:
                    self.result.errors.append(f"Unknown model: {model_id}")
                    return self.result
                if spec.category != cat:
                    self.result.warnings.append(
                        f"Model {model_id} registered under '{spec.category.value}', not '{cat.value}'"
                    )

            phase_results.append(PhaseResult(SetupPhase.RESOLVE, True, 0.0))

        # Phase 3: Install
        with renderer.phase(SetupPhase.INSTALL, 3, len(phases)):
            if cfg.dry_run:
                actions = []
                for cat, model_id in preferences.items():
                    spec = find_spec(model_id)
                    size = spec.disk_mb if spec and spec.is_local else None
                    status = "will download" if spec and spec.is_local else "will configure"
                    actions.append(("INSTALL", model_id, size, status))
                renderer.dry_run_preview(actions)
                self.result.messages.append("Dry run — no changes made.")
                self.result.success = True
                self.result.config["models"] = {k.value: v for k, v in preferences.items()}
                return self.result

            self.install_models(preferences, force=cfg.force, timeout=cfg.timeout)
            phase_results.append(PhaseResult(SetupPhase.INSTALL, len(self.result.failed) == 0, 0.0))

        # Phase 4: Configure logging
        with renderer.phase(SetupPhase.CONFIGURE, 4, len(phases)):
            self.configure_logging(
                level=cfg.log_level, retention=cfg.log_retention, persist=cfg.log_persist
            )
            phase_results.append(PhaseResult(SetupPhase.CONFIGURE, True, 0.0))

        # Phase 5: Save
        with renderer.phase(SetupPhase.SAVE, 5, len(phases)):
            config_to_save = {
                "base_dir": str(set_darwin_base_dir()),
                "environment": detect_environment(),
                "models": {k.value: v for k, v in preferences.items()},
                "api_keys": {k: "[REDACTED]" for k in cfg.api_keys},
                "logging": self.result.config.get("logging", {}),
            }
            if existing:
                config_to_save["previous_setup"] = existing.get("setup_completed")

            path = self.save_config(config_to_save)
            self.result.config_path = path
            self.result.config = config_to_save
            self.result.success = (
                len(self.result.failed) == 0 and len(self.result.errors) == 0
            )
            phase_results.append(PhaseResult(SetupPhase.SAVE, True, 0.0))

        renderer.show_result(self.result)
        return self.result

    # ── Granular helpers (used by subcommands) ───────────────────────

    def configure_models(
        self, model_ids: List[str], force: bool = False
    ) -> SetupResult:
        """Install/reconfigure specific models without touching other config."""
        self.result = SetupResult()
        preferences: Dict[ModelCategory, str] = {}

        for mid in model_ids:
            spec = find_spec(mid)
            if not spec:
                self.result.errors.append(f"Unknown model: {mid}")
                continue
            preferences[spec.category] = mid

        if not preferences:
            self.result.errors.append("No valid models specified.")
            return self.result

        self.install_models(preferences, force=force)

        # Merge into existing config
        existing = self.load_config() or {}
        existing.setdefault("models", {}).update(
            {k.value: v for k, v in preferences.items()}
        )
        path = self.save_config(existing)
        self.result.config_path = path
        self.result.success = len(self.result.failed) == 0
        return self.result

    def configure_embedding(self, model_id: str, force: bool = False) -> SetupResult:
        """Set or change the embedding model only."""
        return self.configure_models([model_id], force=force)

    def configure_logs(
        self,
        level: Optional[str] = None,
        retention: Optional[int] = None,
        persist: Optional[bool] = None,
    ) -> SetupResult:
        """Update logging configuration without touching models."""
        self.result = SetupResult()
        existing = self.load_config() or {}

        current_log = existing.get("logging", {})
        new_level = level or current_log.get("level", "normal")
        new_retention = (
            retention
            if retention is not None
            else current_log.get("retention_days", 14)
        )
        new_persist = (
            persist if persist is not None else current_log.get("persist", True)
        )

        self.configure_logging(
            level=new_level, retention=new_retention, persist=new_persist
        )

        existing["logging"] = self.result.config["logging"]
        path = self.save_config(existing)
        self.result.config_path = path
        self.result.success = True
        self.result.messages.append(f"Logging configured: {new_level}")
        return self.result


# ═══════════════════════════════════════════════════════════════════════
# Click CLI — Command Group
# ═══════════════════════════════════════════════════════════════════════


def _build_cli():
    click = _get_click()

    @click.group(name="setup")
    def setup_group():
        """Configure Darwin — models, storage, logging, and more."""
        pass

    # ── darwin setup (guided onboarding) ─────────────────────────────
    @setup_group.command(name="run")
    @click.option("--base-dir", type=click.Path(path_type=Path), help="Darwin base directory")
    @click.option("--ignore-env", "--yolo", is_flag=True, help="Skip ephemeral-environment checks")
    @click.option("--preset", type=click.Choice(DarwinSetup.PRESETS), help="Use a preset")
    @click.option("--embedding", help="Embedding model ID")
    @click.option("--model", "models", multiple=True, help="Additional model IDs as category:model_id")
    @click.option("--api-key", "api_keys", multiple=True, help="API keys as provider:key")
    @click.option("--log-level", type=click.Choice(["info", "normal", "quiet", "silent"]), default="normal")
    @click.option("--log-retention", type=click.Choice(["forever", "7", "14", "30", "90", "none"]), default="14")
    @click.option("--no-logs", is_flag=True, help="Disable log persistence")
    @click.option("--non-interactive", "-y", is_flag=True, help="Skip all prompts")
    @click.option("--force", "-f", is_flag=True, help="Force re-download")
    @click.option("--timeout", default=300, help="Download timeout")
    @click.option("--dry-run", is_flag=True, help="Preview without installing")
    @click.option("--json", "json_output", is_flag=True, help="Output as JSON")
    @click.option("--plain", is_flag=True, help="Disable colors and animations")
    @click.pass_context
    def setup_run(ctx, base_dir, ignore_env, preset, embedding, models, api_keys,
                  log_level, log_retention, no_logs, non_interactive, force, timeout,
                  dry_run, json_output, plain):
        """Launch the full guided onboarding experience."""
        c = click
        renderer = SetupRenderer(json_output=json_output, plain=plain)

        if not non_interactive and not json_output:
            darwin.configure(show_banner=True, persist=False, persist_config=False)

        env = detect_environment()
        if env["ephemeral"] and not base_dir and not ignore_env:
            renderer.failure(
                "Ephemeral environment detected",
                reason="No persistent --base-dir specified",
                fix="Add --base-dir /persistent/path or use --yolo to skip",
            )
            ctx.exit(1)

        setup = DarwinSetup()
        existing = setup.load_config()

        # Existing config handling
        if existing and not non_interactive and not preset and not embedding and not models:
            choice = renderer.existing_config_prompt(existing)
            if choice == "keep":
                renderer._out("Keeping existing configuration.")
                ctx.exit(0)
            elif choice == "update":
                non_interactive = True  # Skip to model selection

        # Interactive model selection
        selected_models: Dict[str, List[str]] = {}
        resolved_api_keys: Dict[str, str] = {}

        if not non_interactive and not preset:
            renderer._out("\nModel Selection")
            darwin.debug("Choose one model per category. Embedding is required.")

            for cat in ModelCategory:
                specs = get_specs_by_category(cat)
                if not specs:
                    continue

                # Determine default index
                default_idx = 1
                if existing:
                    existing_id = existing.get("models", {}).get(cat.value)
                    if existing_id:
                        for i, s in enumerate(specs, 1):
                            if s.model_id == existing_id:
                                default_idx = i
                                break
                else:
                    if cat == ModelCategory.EMBEDDING:
                        for i, s in enumerate(specs, 1):
                            if s.is_local:
                                default_idx = i
                                break

                renderer.model_table(cat, specs, required=(cat == ModelCategory.EMBEDDING), default_idx=default_idx)

                prompt_text = "  Select" if cat == ModelCategory.EMBEDDING else "  Select (0 to skip)"
                default_val = default_idx if cat == ModelCategory.EMBEDDING else 0
                choice = c.prompt(prompt_text, type=int, default=default_val, show_default=True)

                if choice == 0 and cat != ModelCategory.EMBEDDING:
                    continue
                if 1 <= choice <= len(specs):
                    model_id = specs[choice - 1].model_id
                    selected_models[cat.value] = [model_id]
                    spec = specs[choice - 1]
                    if spec.requires_provider:
                        provider = spec.requires_provider
                        env_var = f"{provider.upper().replace('-', '_')}_API_KEY"
                        if os.environ.get(env_var):
                            renderer._out(f"{env_var} already set.")
                        else:
                            key = c.prompt(f"  Enter {provider} API key", hide_input=True, confirmation_prompt=True)
                            if key:
                                from src.lib.KeyManager import set_api_key
                                if set_api_key(provider, key, validate=True) is not None:
                                    resolved_api_keys[provider] = key
                                else:
                                    renderer._out(f"  Invalid {provider} key — you can set it later with 'darwin key set {provider}'")
                elif cat == ModelCategory.EMBEDDING:
                    renderer.failure("Embedding model is required")
                    ctx.exit(1)

            # Logging interactive
            renderer._out("\nLogging Configuration")
            log_level = c.prompt(
                "  Log level",
                type=click.Choice(["normal", "quiet", "silent"]),
                default=log_level,
            )
            retention_choice = c.prompt(
                "  Log retention (days)",
                type=click.Choice(["forever", "7", "14", "30", "90", "none"]),
                default="14",
            )
            log_retention = retention_choice

        # Parse explicit flags
        if embedding:
            selected_models[ModelCategory.EMBEDDING.value] = [embedding]
        for m in models:
            if ":" in m:
                cat_str, model_id = m.split(":", 1)
                selected_models.setdefault(cat_str, []).append(model_id)
            else:
                spec = find_spec(m)
                if spec:
                    selected_models.setdefault(spec.category.value, []).append(m)
        for ak in api_keys:
            if ":" in ak:
                provider, key = ak.split(":", 1)
                from src.lib.KeyManager import set_api_key
                if set_api_key(provider, key, validate=True) is not None:
                    resolved_api_keys[provider] = key
                else:
                    renderer._out(f"  Invalid {provider} key — skipping")

        # Resolve retention
        if log_retention == "forever":
            retention_days = None
        elif log_retention == "none":
            retention_days = 0
            no_logs = True
        else:
            retention_days = int(log_retention)

        # Build preferences for summary
        prefs: Dict[str, str] = {}
        for cat_str, mids in selected_models.items():
            if mids:
                prefs[cat_str] = mids[0]
        if preset:
            try:
                preset_prefs = setup.resolve_preset(preset)
                prefs = {k.value: v for k, v in preset_prefs.items()}
            except ValueError:
                pass

        # Disk impact + summary
        if not json_output and not dry_run:
            disk_impact = None
            if preset:
                try:
                    preset_prefs = setup.resolve_preset(preset)
                    disk_impact = calculate_disk_impact(preset_prefs)
                except ValueError:
                    pass
            elif selected_models:
                try:
                    temp_prefs = {}
                    for cat_str, mids in selected_models.items():
                        if mids:
                            temp_prefs[ModelCategory[cat_str.upper()]] = mids[0]
                    disk_impact = calculate_disk_impact(temp_prefs)
                except (KeyError, ValueError):
                    pass
            renderer.summary(prefs, base_dir, disk_impact, log_level, retention_days)

        # Preset preview + confirmation
        if preset and not non_interactive and not json_output and not dry_run:
            try:
                preset_prefs = setup.resolve_preset(preset)
                renderer._out(f"\nPreset: {preset}")
                renderer._out(DarwinSetup.PRESET_DESCRIPTIONS.get(preset, ""))
                total_size_mb = 0
                for cat, model_id in preset_prefs.items():
                    spec = find_spec(model_id)
                    if not spec:
                        continue
                    size_str = f" ({spec.disk_mb}MB)" if spec.disk_mb else ""
                    kind = "Local" if spec.is_local else "API"
                    renderer._out(f"  -> {cat.value}: {model_id} [{kind}]{size_str}")
                    if spec.disk_mb:
                        total_size_mb += spec.disk_mb

                if not renderer.confirm_preset(preset, total_size_mb):
                    renderer._out("Setup cancelled.")
                    ctx.exit(0)
            except ValueError:
                pass

        # Non-preset interactive confirmation
        if not non_interactive and not json_output and not dry_run and not preset:
            if not renderer.confirm_install(existing=bool(existing)):
                renderer._out("Setup cancelled.")
                ctx.exit(0)

        cfg = SetupConfig(
            base_dir=base_dir,
            ignore_env_checks=ignore_env,
            preset=preset,
            embedding_model=embedding,
            selected_models=selected_models,
            api_keys=resolved_api_keys,
            log_level=log_level,
            log_retention=retention_days,
            log_persist=not no_logs,
            non_interactive=non_interactive,
            force=force,
            timeout=timeout,
            dry_run=dry_run,
            json_output=json_output,
            plain=plain,
        )

        result = setup.run(cfg, renderer=renderer)

        if json_output:
            c.echo(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            for msg in result.messages:
                renderer._out(f"  [OK] {msg}")
            for warn in result.warnings:
                renderer._out(f"  [!] {warn}")
            for err in result.errors:
                renderer._out(f"  [ERR] {err}", err=True)

        ctx.exit(0 if result.success else 1)

    # ── darwin setup preset <name> ───────────────────────────────────
    @setup_group.command(name="preset")
    @click.argument("name", type=click.Choice(DarwinSetup.PRESETS))
    @click.option("--base-dir", type=click.Path(path_type=Path))
    @click.option("--yolo", is_flag=True, help="Skip ephemeral checks")
    @click.option("--force", "-f", is_flag=True)
    @click.option("--dry-run", is_flag=True)
    @click.option("--json", "json_output", is_flag=True)
    @click.option("--plain", is_flag=True, help="Disable colors")
    @click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
    def setup_preset(name, base_dir, yolo, force, dry_run, json_output, plain, yes):
        """Apply a one-shot preset: minimal, recommended, complete, local-only, api-only, server."""
        c = click
        renderer = SetupRenderer(json_output=json_output, plain=plain)
        setup = DarwinSetup()

        if not setup.validate_storage(base_dir, yolo):
            renderer.failure("Storage validation failed")
            sys.exit(1)

        existing = setup.load_config()
        prefs = setup.resolve_preset(name)

        # Preview
        if not json_output:
            renderer._out(f"\nPreset: {name}")
            renderer._out(DarwinSetup.PRESET_DESCRIPTIONS.get(name, ""))
            renderer._out("\nModels to be configured:")
            total_size_mb = 0
            for cat, model_id in prefs.items():
                spec = find_spec(model_id)
                if not spec:
                    continue
                size_str = f" ({spec.disk_mb}MB)" if spec.disk_mb else ""
                kind = "Local" if spec.is_local else "API"
                renderer._out(f"  -> {cat.value}: {model_id} [{kind}]{size_str}")
                if spec.disk_mb:
                    total_size_mb += spec.disk_mb

            if total_size_mb:
                renderer._out(f"  Total local download: ~{total_size_mb}MB")

        # First-run confirmation only
        if not yes and not existing and not json_output and not dry_run:
            if not renderer.confirm_preset(name, total_size_mb):
                renderer._out("Cancelled.")
                sys.exit(0)

        cfg = SetupConfig(
            preset=name,
            base_dir=base_dir,
            ignore_env_checks=yolo,
            force=force,
            dry_run=dry_run,
            json_output=json_output,
            plain=plain,
            non_interactive=True,
        )
        result = setup.run(cfg, renderer=renderer)

        if json_output:
            c.echo(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            for msg in result.messages:
                renderer._out(f"  [OK] {msg}")
            for err in result.errors:
                renderer._out(f"  [ERR] {err}", err=True)
            if result.success:
                renderer._out(f"Saved to {result.config_path}")
        sys.exit(0 if result.success else 1)

    # ── darwin setup models ──────────────────────────────────────────
    @setup_group.command(name="models")
    @click.argument("model_ids", nargs=-1, required=True)
    @click.option("--force", "-f", is_flag=True)
    @click.option("--json", "json_output", is_flag=True)
    @click.option("--plain", is_flag=True)
    def setup_models(model_ids, force, json_output, plain):
        """Install or update specific models without changing other config."""
        c = click
        renderer = SetupRenderer(json_output=json_output, plain=plain)
        setup = DarwinSetup()
        result = setup.configure_models(list(model_ids), force=force)
        if json_output:
            c.echo(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            for msg in result.messages:
                c.echo(c.style(f"  [OK] {msg}", fg="green"))
            for err in result.errors:
                c.echo(c.style(f"  [ERR] {err}", fg="red"), err=True)
            if result.config_path:
                c.echo(c.style(f"  -> Config: {result.config_path}", dim=True))
        sys.exit(0 if result.success else 1)

    # ── darwin setup embeddings ──────────────────────────────────────
    @setup_group.command(name="embeddings")
    @click.argument("model_id")
    @click.option("--force", "-f", is_flag=True)
    @click.option("--json", "json_output", is_flag=True)
    @click.option("--plain", is_flag=True)
    def setup_embeddings(model_id, force, json_output, plain):
        """Set or change the embedding model."""
        c = click
        renderer = SetupRenderer(json_output=json_output, plain=plain)
        setup = DarwinSetup()
        result = setup.configure_embedding(model_id, force=force)
        if json_output:
            c.echo(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            for msg in result.messages:
                c.echo(c.style(f"  [OK] {msg}", fg="green"))
            for err in result.errors:
                c.echo(c.style(f"  [ERR] {err}", fg="red"), err=True)
        sys.exit(0 if result.success else 1)

    # ── darwin setup logs ────────────────────────────────────────────
    @setup_group.command(name="logs")
    @click.option("--level", type=click.Choice(["info", "normal", "quiet", "silent"]))
    @click.option("--retention", type=click.Choice(["forever", "7", "14", "30", "90", "none"]))
    @click.option("--persist/--no-persist", default=None)
    @click.option("--json", "json_output", is_flag=True)
    @click.option("--plain", is_flag=True)
    def setup_logs(level, retention, persist, json_output, plain):
        """Configure logging without touching models or storage."""
        c = click
        renderer = SetupRenderer(json_output=json_output, plain=plain)
        setup = DarwinSetup()

        retention_days = None
        if retention == "forever":
            retention_days = None
        elif retention == "none":
            retention_days = 0
        elif retention:
            retention_days = int(retention)

        result = setup.configure_logs(level=level, retention=retention_days, persist=persist)
        if json_output:
            c.echo(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            for msg in result.messages:
                c.echo(c.style(f"  [OK] {msg}", fg="green"))
        sys.exit(0 if result.success else 1)

    # ── darwin setup storage ─────────────────────────────────────────
    @setup_group.command(name="storage")
    @click.argument("path", type=click.Path(path_type=Path))
    @click.option("--json", "json_output", is_flag=True)
    @click.option("--plain", is_flag=True)
    def setup_storage(path, json_output, plain):
        """Set the Darwin base storage directory."""
        c = click
        setup = DarwinSetup()
        base = setup.set_storage(path)
        existing = setup.load_config() or {}
        existing["base_dir"] = str(base)
        setup.save_config(existing)
        if json_output:
            c.echo(json.dumps({"base_dir": str(base), "status": "ok"}))
        else:
            c.echo(c.style(f"  [OK] Storage set to: {base}", fg="green"))
        sys.exit(0)

    # ── darwin setup paths ───────────────────────────────────────────
    @setup_group.command(name="paths")
    @click.option("--json", "json_output", is_flag=True)
    @click.option("--plain", is_flag=True)
    def setup_paths(json_output, plain):
        """Show all derived Darwin paths."""
        c = click
        renderer = SetupRenderer(json_output=json_output, plain=plain)
        paths = {
            "base": str(set_darwin_base_dir()),
            "models": str(default_models_dir()),
            "config": str(default_config_dir()),
            "logs": str(default_logs_dir()),
        }
        if json_output:
            c.echo(json.dumps(paths, indent=2))
        else:
            renderer._out(renderer._divider())
            renderer._out(f"  {renderer._bright('Darwin Paths')}")
            renderer._out(renderer._divider())
            for name, p in paths.items():
                renderer._out(f"  {name:12} {p}")
        sys.exit(0)

    # ── darwin setup validate ────────────────────────────────────────
    @setup_group.command(name="validate")
    @click.option("--json", "json_output", is_flag=True)
    @click.option("--plain", is_flag=True)
    def setup_validate(json_output, plain):
        """Validate the current Darwin configuration."""
        c = click
        renderer = SetupRenderer(json_output=json_output, plain=plain)
        setup = DarwinSetup()
        result = setup.validate_config()
        if json_output:
            c.echo(json.dumps(result, indent=2))
        else:
            renderer.validation(result)
        sys.exit(0 if result["valid"] else 1)

    # ── darwin setup status ──────────────────────────────────────────
    @setup_group.command(name="status")
    @click.option("--json", "json_output", is_flag=True)
    @click.option("--plain", is_flag=True)
    def setup_status(json_output, plain):
        """Show current Darwin status."""
        c = click
        renderer = SetupRenderer(json_output=json_output, plain=plain)
        setup = DarwinSetup()
        status = setup.get_status()
        if json_output:
            c.echo(json.dumps(status, indent=2, default=str))
        else:
            renderer.status(status["config"], setup.orch)
        sys.exit(0)

    # ── darwin setup doctor ──────────────────────────────────────────
    @setup_group.command(name="doctor")
    @click.option("--json", "json_output", is_flag=True)
    @click.option("--plain", is_flag=True)
    def setup_doctor(json_output, plain):
        """Diagnose common Darwin issues."""
        c = click
        renderer = SetupRenderer(json_output=json_output, plain=plain)
        setup = DarwinSetup()
        checks = setup.doctor()
        if json_output:
            c.echo(json.dumps([{"name": n, "ok": o, "fix": f} for n, o, f in checks], indent=2))
        else:
            renderer.doctor(checks)
        all_ok = all(ok for _, ok, _ in checks)
        sys.exit(0 if all_ok else 1)

    # ── darwin setup reset ───────────────────────────────────────────
    @setup_group.command(name="reset")
    @click.option("--keep-models", is_flag=True, help="Keep downloaded models")
    @click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
    @click.option("--json", "json_output", is_flag=True)
    @click.option("--plain", is_flag=True)
    def setup_reset(keep_models, yes, json_output, plain):
        """Reset Darwin configuration. Optionally keep downloaded models."""
        c = click
        renderer = SetupRenderer(json_output=json_output, plain=plain)
        setup = DarwinSetup()

        if not yes and not json_output:
            action = "keep models" if keep_models else "remove everything"
            if not c.confirm(
                c.style(f"  Reset Darwin? This will {action}.", fg="bright_white"),
                default=False,
            ):
                renderer._out("Cancelled.")
                sys.exit(0)

        result = setup.reset(keep_models=keep_models)
        if json_output:
            c.echo(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            for msg in result.messages:
                renderer._out(f"  [OK] {msg}")
        sys.exit(0 if result.success else 1)

    return setup_group


# ═══════════════════════════════════════════════════════════════════════
# Module entry point
# ═══════════════════════════════════════════════════════════════════════


def main():
    click = _get_click()
    cli = click.group(invoke_without_command=True)(lambda: None)
    cli.add_command(_build_cli())
    cli()


if __name__ == "__main__":
    main()