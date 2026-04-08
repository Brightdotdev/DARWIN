import os
import platform
from pathlib import Path
import subprocess

def open_in_explorer(path: str | Path):
    """
    Open a file or directory in the default file explorer.
    Works on Windows, macOS, and Linux.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    system = platform.system()

    if system == "Windows":
        os.startfile(path)  # native Windows call
    elif system == "Darwin":  # macOS
        subprocess.run(["open", str(path)])
    else:  # assume Linux/Unix
        subprocess.run(["xdg-open", str(path)])