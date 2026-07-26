"""
Shared types, constants, and data models for the admin module.

Every string that appears in more than one place gets a name here.
Every dict that gets passed between functions gets a TypedDict or dataclass.
No magic numbers, no magic strings.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import json as _json
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════

DEFAULT_SSE_PORT: int = 8765
DEFAULT_SSE_HOST: str = "127.0.0.1"
DEFAULT_RETENTION_DAYS: int = 14
DEFAULT_MAX_MEMORY: int = 10_000


# ══════════════════════════════════════════════════════════════
# Enums — admin-specific
# ══════════════════════════════════════════════════════════════


class SetupAction(str, Enum):
    """Subcommands under ``darwin-admin setup``."""

    INTERACTIVE = "interactive"
    PROFILE = "profile"
    ONBOARD = "onboard"
    VALIDATE = "validate"
    DOCTOR = "doctor"
    RESET = "reset"
    LOGGING = "logging"


class SetupPreset(str, Enum):
    """One-shot preset profiles."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"


class KeyAction(str, Enum):
    """Subcommands under ``darwin-admin models keys``."""

    SET = "set"
    STATUS = "status"
    REMOVE = "remove"


class StoreAction(str, Enum):
    """Subcommands under ``darwin-admin store``."""

    STATUS = "status"
    FILES = "files"
    FILE = "file"
    AUDIT = "audit"
    HEALTH = "health"
    REPAIR = "repair"


class PipelineAction(str, Enum):
    """Subcommands under ``darwin-admin pipeline``."""

    RUN = "run"
    INGEST = "ingest"
    INDEX = "index"
    PURGE = "purge"


class LogAction(str, Enum):
    """Subcommands under ``darwin-admin logs``."""

    QUERY = "query"
    SESSIONS = "sessions"
    SESSION_INFO = "session-info"
    STATS = "stats"
    PURGE = "purge"
    PURGE_SESSION = "purge-session"
    EXPORT = "export"
    COMPACT = "compact"
    SET_RETENTION = "set-retention"


class DiscardScope(str, Enum):
    """Scope flags for ``models discard``."""

    ALL = "all"
    ACTIVE = "active"
    INACTIVE = "inactive"


# ══════════════════════════════════════════════════════════════
# Config file keys
# ══════════════════════════════════════════════════════════════


class ConfigFile(str, Enum):
    """The three config files the admin module manages."""

    DARWIN = "darwin"
    LOGGING = "logging"
    MODELS = "models"

    @property
    def filename(self) -> str:
        return f"{self.value}.json"


# ══════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SetupStatus:
    """Which config files exist and their parsed content."""

    darwin_json: bool
    logging_json: bool
    models_json: bool

    @property
    def all_configured(self) -> bool:
        return self.darwin_json and self.logging_json and self.models_json

    @property
    def any_configured(self) -> bool:
        return self.darwin_json or self.logging_json or self.models_json

    @property
    def missing_files(self) -> List[str]:
        missing = []
        if not self.darwin_json:
            missing.append(ConfigFile.DARWIN.filename)
        if not self.logging_json:
            missing.append(ConfigFile.LOGGING.filename)
        if not self.models_json:
            missing.append(ConfigFile.MODELS.filename)
        return missing


@dataclass(frozen=True)
class DiskImpact:
    """Disk space impact of installing models."""

    required_mb: int
    available_mb: Optional[int]
    after_mb: Optional[int]
    warning: Optional[str]


@dataclass(frozen=True)
class ModelChoice:
    """A single model the user selected during setup."""

    model_id: str
    category: str
    api_key_provider: Optional[str] = None

    @property
    def needs_key(self) -> bool:
        return self.api_key_provider is not None


@dataclass(frozen=True)
class LoggingConfig:
    """Values that go into logging.json."""

    mode: str = "terminal"
    persist: bool = True
    retention_days: int = DEFAULT_RETENTION_DAYS
    max_memory: int = DEFAULT_MAX_MEMORY
    sse_port: int = DEFAULT_SSE_PORT
    sse_host: str = DEFAULT_SSE_HOST
    log_filter: str = "all"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "persist": self.persist,
            "retention_days": self.retention_days,
            "max_memory": self.max_memory,
            "sse_port": self.sse_port,
            "sse_host": self.sse_host,
            "log_filter": self.log_filter,
        }


@dataclass(frozen=True)
class DoctorCheck:
    """A single diagnostic check result."""

    name: str
    passed: bool
    fix: str


# ══════════════════════════════════════════════════════════════
# Category labels (used by setup flow)
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CategoryLabel:
    """Display metadata for a model category during interactive setup."""

    category: str
    description: str
    required: bool


# The order and descriptions shown during interactive setup
CATEGORY_SETUP_ORDER: List[CategoryLabel] = [
    CategoryLabel("embedding", "Vector embeddings for semantic search", True),
    CategoryLabel("reranker", "Re-rank search results for higher precision", False),
    CategoryLabel("ocr", "Optical character recognition for scanned documents", False),
    CategoryLabel("vision", "Image analysis and captioning (API-only for now)", False),
]


# ══════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically — temp file then os.replace().

    If interrupted mid-write, only the .tmp file exists.
    The original file is never half-written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            _json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def is_ci() -> bool:
    """True when running in CI/CD."""
    return any(
        os.environ.get(k)
        for k in ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL",
                   "TRAVIS", "CIRCLECI", "BUILDKITE", "DRONE"]
    )


def is_ssh() -> bool:
    """True when running over SSH."""
    return any(os.environ.get(k) for k in ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"])


def is_kubernetes() -> bool:
    """True when running in Kubernetes."""
    return "KUBERNETES_SERVICE_HOST" in os.environ


def is_headless() -> bool:
    """True when no interactive terminal is available."""
    return is_ci() or is_ssh() or not sys.stdin.isatty()


# ══════════════════════════════════════════════════════════════
# Arg dataclasses — typed wrappers for argparse.Namespace
# ══════════════════════════════════════════════════════════════
#
# Every command handler receives one of these instead of raw
# argparse.Namespace.  Fields use defaults so missing attrs
# don't crash — argparse always sets them, but the dataclass
# makes the contract explicit and type-checkable.
#


def ns_to(dc_class, ns: argparse.Namespace):
    """Convert an argparse Namespace to a dataclass instance.

    Only copies fields that exist on the dataclass — extra
    argparse attrs (like 'func', 'command') are ignored.
    """
    dc_fields = {f.name for f in fields(dc_class)}
    kwargs = {k: v for k, v in vars(ns).items() if k in dc_fields}
    return dc_class(**kwargs)


# ── Setup ──────────────────────────────────────────────────


@dataclass(frozen=True)
class SetupInteractiveArgs:
    """Args for ``darwin-admin setup`` (no subcommand / interactive)."""
    yes: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class SetupPresetArgs:
    """Args for ``darwin-admin setup --preset <name>``."""
    preset: str = ""
    no_download: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class SetupProfileArgs:
    """Args for ``darwin-admin setup profile``."""
    profile: str = "required"
    no_download: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class SetupOnboardArgs:
    """Args for ``darwin-admin setup onboard``."""
    preferences: Optional[List[str]] = None
    no_download: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class SetupValidateArgs:
    """Args for ``darwin-admin setup validate``."""
    verbose: bool = False


@dataclass(frozen=True)
class SetupDoctorArgs:
    """Args for ``darwin-admin setup doctor``."""
    verbose: bool = False


@dataclass(frozen=True)
class SetupResetArgs:
    """Args for ``darwin-admin setup reset``."""
    yes: bool = False
    keep_models: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class SetupLoggingArgs:
    """Args for ``darwin-admin setup logging``."""
    verbose: bool = False


# ── Models ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelsStatusArgs:
    """Args for ``darwin-admin models status``."""
    verbose: bool = False


@dataclass(frozen=True)
class ModelsListArgs:
    """Args for ``darwin-admin models list``."""
    category: Optional[str] = None
    provider: Optional[str] = None
    verbose: bool = False


@dataclass(frozen=True)
class ModelsInstallArgs:
    """Args for ``darwin-admin models install``."""
    model_ids: List[str] = field(default_factory=list)
    no_download: bool = False
    force: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class ModelsDiscardArgs:
    """Args for ``darwin-admin models discard``."""
    category: List[str] = field(default_factory=list)
    discard_all: bool = False
    active: bool = False
    inactive: bool = False
    delete: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class ModelsSwitchArgs:
    """Args for ``darwin-admin models switch``."""
    category: Optional[str] = None
    model_id: Optional[str] = None
    no_download: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class ModelsKeysArgs:
    """Args for ``darwin-admin models keys``."""
    action: str = ""
    provider: Optional[str] = None
    key: Optional[str] = None
    verbose: bool = False


# ── Store ──────────────────────────────────────────────────


@dataclass(frozen=True)
class StoreStatusArgs:
    """Args for ``darwin-admin store status``."""
    verbose: bool = False


@dataclass(frozen=True)
class StoreFilesArgs:
    """Args for ``darwin-admin store files``."""
    verbose: bool = False


@dataclass(frozen=True)
class StoreFileArgs:
    """Args for ``darwin-admin store file``."""
    filename: str = ""
    verbose: bool = False


@dataclass(frozen=True)
class StoreAuditArgs:
    """Args for ``darwin-admin store audit``."""
    verbose: bool = False


@dataclass(frozen=True)
class StoreHealthArgs:
    """Args for ``darwin-admin store health``."""
    model: Optional[str] = None
    verbose: bool = False


@dataclass(frozen=True)
class StoreRepairArgs:
    """Args for ``darwin-admin store repair``."""
    dry_run: bool = False
    verbose: bool = False


# ── Pipeline ───────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineRunArgs:
    """Args for ``darwin-admin pipeline run``."""
    target: Optional[str] = None
    store: Optional[str] = None
    files: Optional[List[str]] = None
    ocr: bool = False
    recursive: bool = False
    precise: bool = False
    threshold: Optional[float] = None
    idf_smoothing: Optional[float] = None
    only_kw: bool = False
    only_semantic: bool = False
    rerun: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class PipelineIngestArgs:
    """Args for ``darwin-admin pipeline ingest``."""
    target: Optional[str] = None
    store: Optional[str] = None
    files: Optional[List[str]] = None
    ocr: bool = False
    recursive: bool = False
    precise: bool = False
    rerun: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class PipelineIndexArgs:
    """Args for ``darwin-admin pipeline index``."""
    files: Optional[List[str]] = None
    store: Optional[str] = None
    threshold: Optional[float] = None
    idf_smoothing: Optional[float] = None
    only_kw: bool = False
    only_semantic: bool = False
    rerun: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class PipelinePurgeArgs:
    """Args for ``darwin-admin pipeline purge``."""
    filenames: List[str] = field(default_factory=list)
    store: Optional[str] = None
    stages: List[str] = field(default_factory=lambda: ["ingestion", "indexing"])
    verbose: bool = False


# ── Search ─────────────────────────────────────────────────


@dataclass(frozen=True)
class SearchArgs:
    """Args for ``darwin-admin search``."""
    query: str = ""
    method: Optional[str] = None
    top_k: Optional[int] = None
    max_passages: Optional[int] = None
    max_chars: Optional[int] = None
    min_passage_score: Optional[float] = None
    merge_strategy: Optional[str] = None
    files: Optional[List[str]] = None
    store: Optional[str] = None
    no_save: bool = False
    results_dir: Optional[str] = None
    rerank: bool = False
    structural_penalty_factor: Optional[float] = None
    diversity_rerank: Optional[bool] = None
    include_audit: Optional[bool] = None
    verbose: bool = False


# ── Logs ───────────────────────────────────────────────────


@dataclass(frozen=True)
class LogsQueryArgs:
    """Args for ``darwin-admin logs query``."""
    session: Optional[str] = None
    levels: Optional[str] = None
    limit: int = 50
    verbose: bool = False


@dataclass(frozen=True)
class LogsSessionsArgs:
    """Args for ``darwin-admin logs sessions``."""
    verbose: bool = False


@dataclass(frozen=True)
class LogsSessionInfoArgs:
    """Args for ``darwin-admin logs session-info``."""
    session_name: str = ""
    verbose: bool = False


@dataclass(frozen=True)
class LogsStatsArgs:
    """Args for ``darwin-admin logs stats``."""
    verbose: bool = False


@dataclass(frozen=True)
class LogsPurgeArgs:
    """Args for ``darwin-admin logs purge``."""
    days: int = 14
    verbose: bool = False


@dataclass(frozen=True)
class LogsPurgeSessionArgs:
    """Args for ``darwin-admin logs purge-session``."""
    session_name: str = ""
    verbose: bool = False


@dataclass(frozen=True)
class LogsExportArgs:
    """Args for ``darwin-admin logs export``."""
    output: str = ""
    session: Optional[str] = None
    verbose: bool = False


@dataclass(frozen=True)
class LogsCompactArgs:
    """Args for ``darwin-admin logs compact``."""
    verbose: bool = False


@dataclass(frozen=True)
class LogsSetRetentionArgs:
    """Args for ``darwin-admin logs set-retention``."""
    days: int = 14
    verbose: bool = False


# ── System ─────────────────────────────────────────────────


@dataclass(frozen=True)
class SystemInfoArgs:
    """Args for ``darwin-admin system info``."""
    verbose: bool = False


@dataclass(frozen=True)
class StatusArgs:
    """Args for ``darwin-admin status``."""
    store: Optional[str] = None
    initialize: bool = False
    verbose: bool = False


# ── Uninstall ──────────────────────────────────────────────


@dataclass(frozen=True)
class UninstallArgs:
    """Args for ``darwin-admin uninstall``."""
    dry_run: bool = False
    keep_package: bool = False
    verbose: bool = False
