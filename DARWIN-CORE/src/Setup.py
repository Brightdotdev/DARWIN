# src/core/Setup/DarwinSetup.py
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

Architecture
────────────
  DarwinSetup        -> Pure-Python programmatic API. Zero CLI dependencies.
  Click commands     -> Thin wrappers that call DarwinSetup methods.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

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
    MODEL_SPECS,
    find_spec,
    get_specs_by_category,
)
from src.core.Models.Lib.ModelSpec import ModelCategory, ModelTier
from src.core.Logger.DarwinLogger import darwin, LogFilter, OutputMode
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
            raise RuntimeError("Click is required. Install: pip install click") from exc
    return _click


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════


@dataclass
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


@dataclass
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


# ═══════════════════════════════════════════════════════════════════════
# UI Helpers
# ═══════════════════════════════════════════════════════════════════════


class _SetupUI:
    """Terminal styling helpers."""

    @staticmethod
    def banner():
        from src.core.Logger.lib import ASCII_BANNER
        import sys

        buf = sys.stdout.buffer
        for line in ASCII_BANNER:
            buf.write(f"  {line}\n".encode("utf-8"))
        buf.write(b"\n")
        buf.flush()

    @staticmethod
    def section(title: str):
        c = _get_click()
        c.echo()
        c.echo(c.style(f"  ## {title}", fg="bright_white", bold=True))
        c.echo(c.style(f"  {'--' * 26}", dim=True))

    @staticmethod
    def success(msg: str):
        c = _get_click()
        c.echo(c.style(f"  [OK] {msg}", fg="green"))

    @staticmethod
    def error(msg: str):
        c = _get_click()
        c.echo(c.style(f"  [ERR] {msg}", fg="red"), err=True)

    @staticmethod
    def warning(msg: str):
        c = _get_click()
        c.echo(c.style(f"  [!] {msg}", fg="yellow"))

    @staticmethod
    def info(msg: str):
        c = _get_click()
        c.echo(c.style(f"  -> {msg}", fg="blue"))

    @staticmethod
    def dim(msg: str):
        c = _get_click()
        c.echo(c.style(f"  {msg}", dim=True))

    @staticmethod
    def model_table(category: ModelCategory, models: List[Any], required: bool = False):
        c = _get_click()
        tag = "Required #" if required else "Optional o"
        c.echo(
            c.style(
                f"\n  {category.value.title()} Models  —  {tag}",
                fg="bright_white",
                bold=True,
            )
        )
        c.echo(c.style(f"  {'─' * 52}", dim=True))

        for i, spec in enumerate(models, 1):
            kind = "Local" if spec.is_local else "API"
            provider = f" · {spec.requires_provider}" if spec.requires_provider else ""
            size = ""
            if spec.is_local:
                size = f" · {spec.ram_gb}GB RAM" if spec.ram_gb else ""
                size += f" · {spec.disk_mb}MB" if spec.disk_mb else ""
            notes = f"  — {spec.notes}" if spec.notes else ""
            c.echo(f"  [{i}] {c.style(spec.model_id, fg='bright_white')}")
            c.echo(f"      {kind}{provider}{size}{c.style(notes, dim=True)}")

        if not required:
            c.echo(c.style(f"  [0] Skip this category", dim=True))

    @staticmethod
    def summary_table(config: Dict[str, Any], to_install: Dict[str, str]):
        c = _get_click()
        c.echo()
        c.echo(c.style("  == Setup Summary ==", fg="bright_white"))
        c.echo(f"     Storage:     {config.get('base_dir', 'Default')}")
        c.echo(f"     Environment: {config.get('environment', 'Standard')}")
        c.echo("")
        c.echo("     Models:")
        for cat, model_id in to_install.items():
            status = "[OK]" if model_id else " o "
            label = model_id or "(skipped)"
            c.echo(f"       {status} {cat.title()}: {label}")
        c.echo("")
        log = config.get("logging", {})
        level = log.get("level", "normal")
        retention = log.get("retention_days", 14)
        retention_str = "forever" if retention is None else f"{retention} days"
        c.echo(f"     Logging:     {level} / {retention_str}")
        c.echo()


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
                "CI",
                "GITHUB_ACTIONS",
                "GITLAB_CI",
                "JENKINS_URL",
                "TRAVIS",
                "CIRCLECI",
                "BUILDKITE",
                "DRONE",
                "APPVEYOR",
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

    # ── Config I/O ─────────────────────────────────────────────────────

    def _config_path(self) -> Path:
        return default_config_dir() / self.CONFIG_FILE

    def load_config(self) -> Optional[Dict[str, Any]]:
        path = self._config_path()
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return None

    def save_config(self, config: Dict[str, Any]) -> Path:
        path = self._config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        config["version"] = self.CONFIG_VERSION
        config["setup_completed"] = datetime.now().isoformat()
        path.write_text(json.dumps(config, indent=2, default=str))
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

        if spec.requires_provider:
            stored_keys = self.result.config.get("api_keys", {})
            if spec.requires_provider in stored_keys:
                self.orch.set_key(
                    spec.requires_provider,
                    stored_keys[spec.requires_provider],
                    validate=False,
                )

        for attempt in range(1, 3):
            result = self.orch.install(
                model_id,
                auto_download=True,
                force=force,
                set_active=True,
            )
            if result.success:
                return result
            if attempt < 2:
                time.sleep(2)
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
            "debug": LogFilter.ALL,
            "info": LogFilter.ALL,
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

    # ── Core run logic ───────────────────────────────────────────────

    def run(self, cfg: SetupConfig) -> SetupResult:
        """Execute full onboarding based on a SetupConfig."""
        self.result = SetupResult()
        self.result.config = asdict(cfg)

        if not self.validate_storage(cfg.base_dir, cfg.ignore_env_checks):
            return self.result

        existing = self.load_config()
        preferences: Dict[ModelCategory, str] = {}

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

        if (
            ModelCategory.EMBEDDING not in preferences
            or not preferences[ModelCategory.EMBEDDING]
        ):
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

        if cfg.dry_run:
            self.result.messages.append("Dry run — no changes made.")
            self.result.success = True
            self.result.config["models"] = {k.value: v for k, v in preferences.items()}
            return self.result

        self.install_models(preferences, force=cfg.force, timeout=cfg.timeout)
        self.configure_logging(
            level=cfg.log_level, retention=cfg.log_retention, persist=cfg.log_persist
        )

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
        """🚀  Configure Darwin — models, storage, logging, and more."""
        pass

    # ── darwin setup (guided onboarding) ─────────────────────────────
    @setup_group.command(name="run")
    @click.option(
        "--base-dir", type=click.Path(path_type=Path), help="Darwin base directory"
    )
    @click.option(
        "--ignore-env", "--yolo", is_flag=True, help="Skip ephemeral-environment checks"
    )
    @click.option(
        "--preset", type=click.Choice(DarwinSetup.PRESETS), help="Use a preset"
    )
    @click.option("--embedding", help="Embedding model ID")
    @click.option(
        "--model",
        "models",
        multiple=True,
        help="Additional model IDs as category:model_id",
    )
    @click.option(
        "--api-key", "api_keys", multiple=True, help="API keys as provider:key"
    )
    @click.option(
        "--log-level",
        type=click.Choice(["debug", "info", "normal", "quiet", "silent"]),
        default="normal",
    )
    @click.option(
        "--log-retention",
        type=click.Choice(["forever", "7", "14", "30", "90", "none"]),
        default="14",
    )
    @click.option("--no-logs", is_flag=True, help="Disable log persistence")
    @click.option("--non-interactive", "-y", is_flag=True, help="Skip all prompts")
    @click.option("--force", "-f", is_flag=True, help="Force re-download")
    @click.option("--timeout", default=300, help="Download timeout")
    @click.option("--dry-run", is_flag=True, help="Preview without installing")
    @click.option("--json", "json_output", is_flag=True, help="Output as JSON")
    @click.pass_context
    def setup_run(
        ctx,
        base_dir,
        ignore_env,
        preset,
        embedding,
        models,
        api_keys,
        log_level,
        log_retention,
        no_logs,
        non_interactive,
        force,
        timeout,
        dry_run,
        json_output,
    ):
        """Launch the full guided onboarding experience."""
        ui = _SetupUI()
        c = click

        if not non_interactive and not json_output:
            ui.banner()

        env = detect_environment()
        if env["ephemeral"] and not base_dir and not ignore_env:
            ui.error("Ephemeral environment. Use --base-dir or --yolo.")
            ctx.exit(1)

        setup = DarwinSetup()
        existing = setup.load_config()

        # Existing config handling
        if (
            existing
            and not non_interactive
            and not preset
            and not embedding
            and not models
        ):
            ui.section("Existing Installation Detected")
            ui.info(f"Previous setup: {existing.get('setup_completed', 'unknown')}")
            if not c.confirm(
                c.style("  Update existing configuration?", fg="bright_white"),
                default=True,
            ):
                ui.success("Keeping existing configuration.")
                ctx.exit(0)

        # Interactive model selection
        selected_models: Dict[str, List[str]] = {}
        resolved_api_keys: Dict[str, str] = {}

        if not non_interactive and not preset:
            ui.section("Model Selection")
            ui.dim("Choose one model per category. Embedding is required.")

            for cat in ModelCategory:
                specs = get_specs_by_category(cat)
                if not specs:
                    continue
                ui.model_table(cat, specs, required=(cat == ModelCategory.EMBEDDING))

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

                prompt_text = (
                    "  Select"
                    if cat == ModelCategory.EMBEDDING
                    else "  Select (0 to skip)"
                )
                default_val = default_idx if cat == ModelCategory.EMBEDDING else 0
                choice = c.prompt(
                    prompt_text, type=int, default=default_val, show_default=True
                )

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
                            ui.success(f"{env_var} already set.")
                        else:
                            key = c.prompt(
                                f"  Enter {provider} API key",
                                hide_input=True,
                                confirmation_prompt=True,
                            )
                            if key:
                                resolved_api_keys[provider] = key
                elif cat == ModelCategory.EMBEDDING:
                    ui.error("Embedding model is required.")
                    ctx.exit(1)

            # Logging interactive
            ui.section("Logging Configuration")
            log_level = c.prompt(
                "  Log level",
                type=click.Choice(["debug", "info", "normal", "quiet", "silent"]),
                default=log_level,
            )
            retention_choice = c.prompt(
                "  Log retention",
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
                resolved_api_keys[provider] = key

        # Resolve retention
        if log_retention == "forever":
            retention_days = None
        elif log_retention == "none":
            retention_days = 0
            no_logs = True
        else:
            retention_days = int(log_retention)

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
        )

        # Preset preview + first-run confirmation
        if preset and not non_interactive and not json_output and not dry_run:
            prefs = setup.resolve_preset(preset)
            darwin.info(f"Preset '{preset}' selected. Resolving models...")
            total_size_mb = 0
            for cat, model_id in prefs.items():
                spec = find_spec(model_id)
                if not spec:
                    continue
                size_str = ""
                if spec.disk_mb:
                    size_str = f" ({spec.disk_mb} MB)"
                    total_size_mb += spec.disk_mb
                kind = "Local" if spec.is_local else "API"
                darwin.info(f"  -> {cat.value}: {model_id} [{kind}]{size_str}")
            if total_size_mb:
                darwin.info(f"  Total local download size: ~{total_size_mb} MB")

            if not existing:
                c.echo()
                if not c.confirm(
                    c.style(
                        "  Proceed with download and installation?", fg="bright_white"
                    ),
                    default=True,
                ):
                    darwin.info("Setup cancelled.")
                    ctx.exit(0)

        # Summary for non-preset interactive
        if not non_interactive and not json_output and not dry_run and not preset:
            prefs = {}
            for cat_str, mids in selected_models.items():
                if mids:
                    prefs[cat_str] = mids[0]
            ui.summary_table(
                {
                    "base_dir": str(base_dir or "default"),
                    "environment": "Docker" if env.get("docker") else "Standard",
                    "logging": {"level": log_level, "retention_days": retention_days},
                },
                prefs,
            )
            if not c.confirm(
                c.style("  Proceed with installation?", fg="bright_white"), default=True
            ):
                ui.info("Setup cancelled.")
                ctx.exit(0)

        if not json_output and not dry_run:
            ui.section("Installing")

        result = setup.run(cfg)

        if json_output:
            c.echo(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            for msg in result.messages:
                ui.success(msg)
            for warn in result.warnings:
                ui.warning(warn)
            for err in result.errors:
                ui.error(err)
            if result.success:
                ui.section("Setup Complete")
                ui.success(f"Configuration saved to {result.config_path}")
                ui.dim("Run 'darwin status' to verify.")
            else:
                ui.section("Setup Failed")
                ctx.exit(1)
        ctx.exit(0 if result.success else 1)

    # ── darwin setup preset <name> ───────────────────────────────────
    @setup_group.command(name="preset")
    @click.argument("name", type=click.Choice(DarwinSetup.PRESETS))
    @click.option("--base-dir", type=click.Path(path_type=Path))
    @click.option("--yolo", is_flag=True, help="Skip ephemeral checks")
    @click.option("--force", "-f", is_flag=True)
    @click.option("--dry-run", is_flag=True)
    @click.option("--json", "json_output", is_flag=True)
    @click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
    def setup_preset(name, base_dir, yolo, force, dry_run, json_output, yes):
        """Apply a one-shot preset: minimal, recommended, complete, local-only, api-only, server."""
        ui = _SetupUI()
        c = click
        setup = DarwinSetup()

        if not setup.validate_storage(base_dir, yolo):
            ui.error("Storage validation failed.")
            sys.exit(1)

        existing = setup.load_config()
        prefs = setup.resolve_preset(name)

        # Log preview
        if not json_output:
            ui.section(f"Preset: {name}")
            ui.info(DarwinSetup.PRESET_DESCRIPTIONS.get(name, ""))
            darwin.info("Models to be configured:")
            total_size_mb = 0
            for cat, model_id in prefs.items():
                spec = find_spec(model_id)
                if not spec:
                    continue
                size_str = ""
                if spec.disk_mb:
                    size_str = f" ({spec.disk_mb} MB)"
                    total_size_mb += spec.disk_mb
                kind = "Local" if spec.is_local else "API"
                darwin.info(f"  -> {cat.value}: {model_id} [{kind}]{size_str}")
            if total_size_mb:
                darwin.info(f"  Total local download: ~{total_size_mb} MB")

        # First-run confirmation only
        if not yes and not existing and not json_output and not dry_run:
            c.echo()
            if not c.confirm(
                c.style("  Proceed with installation?", fg="bright_white"), default=True
            ):
                ui.info("Cancelled.")
                sys.exit(0)

        cfg = SetupConfig(
            preset=name,
            base_dir=base_dir,
            ignore_env_checks=yolo,
            force=force,
            dry_run=dry_run,
            json_output=json_output,
            non_interactive=True,
        )
        result = setup.run(cfg)

        if json_output:
            c.echo(json.dumps(result.to_dict(), indent=2, default=str))
        else:
            for msg in result.messages:
                ui.success(msg)
            for err in result.errors:
                ui.error(err)
            if result.success:
                ui.success(f"Saved to {result.config_path}")
        sys.exit(0 if result.success else 1)

    # ── darwin setup models ──────────────────────────────────────────
    @setup_group.command(name="models")
    @click.argument("model_ids", nargs=-1, required=True)
    @click.option("--force", "-f", is_flag=True)
    @click.option("--json", "json_output", is_flag=True)
    def setup_models(model_ids, force, json_output):
        """Install or update specific models without changing other config."""
        c = click
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
    def setup_embeddings(model_id, force, json_output):
        """Set or change the embedding model."""
        c = click
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
    @click.option(
        "--level", type=click.Choice(["debug", "info", "normal", "quiet", "silent"])
    )
    @click.option(
        "--retention", type=click.Choice(["forever", "7", "14", "30", "90", "none"])
    )
    @click.option("--persist/--no-persist", default=None)
    @click.option("--json", "json_output", is_flag=True)
    def setup_logs(level, retention, persist, json_output):
        """Configure logging without touching models or storage."""
        c = click
        setup = DarwinSetup()

        retention_days = None
        if retention == "forever":
            retention_days = None
        elif retention == "none":
            retention_days = 0
        elif retention:
            retention_days = int(retention)

        result = setup.configure_logs(
            level=level, retention=retention_days, persist=persist
        )
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
    def setup_storage(path, json_output):
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
    def setup_paths(json_output):
        """Show all derived Darwin paths."""
        c = click
        paths = {
            "base": str(set_darwin_base_dir()),
            "models": str(default_models_dir()),
            "config": str(default_config_dir()),
            "logs": str(default_logs_dir()),
        }
        if json_output:
            c.echo(json.dumps(paths, indent=2))
        else:
            c.echo(c.style("  Darwin Paths", fg="bright_white", bold=True))
            for name, p in paths.items():
                c.echo(f"  {name:12} {p}")
        sys.exit(0)

    # ── darwin setup validate ────────────────────────────────────────
    @setup_group.command(name="validate")
    @click.option("--json", "json_output", is_flag=True)
    def setup_validate(json_output):
        """Validate the current Darwin configuration."""
        c = click
        setup = DarwinSetup()
        result = setup.validate_config()
        if json_output:
            c.echo(json.dumps(result, indent=2))
        else:
            c.echo(c.style("  Validation Results", fg="bright_white", bold=True))
            for item in result.get("ok", []):
                c.echo(c.style(f"  [OK] {item}", fg="green"))
            for item in result.get("warnings", []):
                c.echo(c.style(f"  [!] {item}", fg="yellow"))
            for item in result.get("issues", []):
                c.echo(c.style(f"  [ERR] {item}", fg="red"), err=True)
            status = "Valid" if result["valid"] else "Invalid"
            c.echo(
                c.style(
                    f"\n  Status: {status}",
                    fg="green" if result["valid"] else "red",
                    bold=True,
                )
            )
        sys.exit(0 if result["valid"] else 1)

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
