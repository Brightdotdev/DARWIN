import platform
from pathlib import Path
import os



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
    



def default_models_dir() -> Path:
    """Platform-appropriate global models directory."""
    return get_darwin_dir() / "models"
    


def default_plugins_dir() -> Path:
    """Platform-appropriate global plugins directory."""
    return get_darwin_dir() / "plugins"

def default_plugins_config_dir() -> Path:
    """Platform-appropriate global plugins directory."""
    return  default_plugins_dir() / "config"
    


def default_conversations_dir() -> Path:
    """Platform-appropriate global models directory."""
    return get_darwin_dir() / "conversations"
    


def default_models_config_dir() -> Path:
    """Platform-appropriate global models directory."""
    return  default_models_dir() / "config"
    

def conversation_memory_path() -> Path:
    """Platform-appropriate global models directory."""
    return  default_conversations_dir() / "conversations_memory"

def conversation_db_path() -> Path:
    """Platform-appropriate global models directory."""
    return  default_conversations_dir() / "darwin_conversations.db"


def default_plugins_dir() -> Path:
    """Platform-appropriate global plugins directory."""
    return get_darwin_dir() / "plugins"
