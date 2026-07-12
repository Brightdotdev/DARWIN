# AppPaths.py
import logging
import platform
from pathlib import Path
from typing import Optional
import os

logger = logging.getLogger("darwin.app_paths")

_DARWIN_BASE: Optional[Path] = None

def set_darwin_base_dir(path: Optional[Path] = None) -> Path:
    """
    Set or get the Darwin base directory.

    Priority:
        1. Explicit `path` argument (programmatic override)
        2. Environment variable `DARWIN_BASE_DIR`
        3. Platform‑specific default (macOS, Windows, Linux)

    This is the **only** place where the base is determined.
    All sub‑directories (models, logs, etc.) are derived from this.
    """
    global _DARWIN_BASE

    if path is not None:
        _DARWIN_BASE = Path(path)
        return _DARWIN_BASE

    env_path = os.environ.get("DARWIN_BASE_DIR")
    if env_path:
        _DARWIN_BASE = Path(env_path)
        return _DARWIN_BASE

    # Platform defaults
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        _DARWIN_BASE = base / "Darwin"
    elif system == "Darwin":  # macOS
        _DARWIN_BASE = Path.home() / "Library" / "Application Support" / "Darwin"
    else:  # Linux and others
        xdg = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
        _DARWIN_BASE = Path(xdg) / "darwin"

    return _DARWIN_BASE





# Directory getters with environment variable overrides
def default_models_dir() -> Path:
    """Platform-appropriate models directory."""
    return set_darwin_base_dir() / "models"

def default_config_dir() -> Path:
    """Platform-appropriate config directory."""
    return set_darwin_base_dir() / "config"

def default_logs_dir() -> Path:
    """Platform-appropriate logs directory."""
    return set_darwin_base_dir() / "logs"




def default_models_config_dir() -> Path:
    """Platform-appropriate models config directory."""
    return default_models_dir() / "config"




def is_running_in_docker() -> bool:
    """Detect if we are inside a Docker container."""
    return Path("/.dockerenv").exists() or (
        Path("/proc/1/cgroup").exists() and "docker" in Path("/proc/1/cgroup").read_text()
    )

def ensure_directories() -> bool:
    """
    Ensure all Darwin directories exist and are writable.

    Returns:
        True if all directories are accessible and writable.
        False if any directory is missing or unwritable.

    In Docker, if the base directory is not writable, prints a strong warning.
    """
    dirs = [
        default_models_dir(),
        default_config_dir(),
        default_logs_dir(),
        default_models_config_dir(),
    ]

    base = set_darwin_base_dir()
    docker = is_running_in_docker()

    # Check base directory exists and is writable
    if not base.exists():
        if docker:
            logger.error(f"DARWIN_BASE_DIR='{base}' does not exist inside Docker.")
            logger.error("Please mount a persistent volume at that path.")
            return False
        else:
            # Try to create it
            try:
                base.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                logger.error(f"Cannot create DARWIN_BASE_DIR='{base}'. Permission denied.")
                return False

    # Check each subdirectory
    for d in dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            # Verify write access by creating a temporary file
            test_file = d / ".darwin_write_test"
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            logger.error(f"Directory '{d}' is not writable: {e}")
            if docker:
                logger.error("In Docker, ensure you have mounted a volume with write permissions.")
            return False

    # Warn if running in Docker and the base directory might be ephemeral
    if docker:
        # Check if base is on a tmpfs (simplistic: if it's under /tmp or /dev/shm)
        if any(base.is_relative_to(Path(p)) for p in ["/tmp", "/dev/shm", "/var/tmp"]):
            logger.warning("DARWIN_BASE_DIR points to a temporary location in Docker.")
            logger.warning("Data will be lost when the container stops.")
            logger.warning("Please use a persistent volume mount.")
        else:
            logger.info("Docker volume detected. Data will persist.")

    return True