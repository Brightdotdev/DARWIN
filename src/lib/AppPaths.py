import platform
from pathlib import Path
import os

def default_models_dir() -> Path:
    """Platform-appropriate global models directory."""
    system = platform.system()
    
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "Darwin" / "models"
    
    elif system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Darwin" / "models"
    
    else:  # Linux and everything else
        # XDG spec
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        return base / "darwin" / "models"
    


def ensure_models_dir(path: Path) -> Path:
    """Make sure the models directory exists, creating it if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path



def get_darwin_dir() -> Path:
    """Get the base Darwin directory (not models, just the parent)."""
    system = platform.system()
    
    if system == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Darwin"
    
    elif system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Darwin"
    
    else:  # Linux and everything else
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        return base / "darwin"