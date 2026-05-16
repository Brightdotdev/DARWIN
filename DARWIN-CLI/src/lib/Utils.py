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



SUPPORTED = {".pdf", ".docx", ".pptx", ".txt", ".md"}

def scan_folder():
    files = []
    for f in Path.cwd().iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED:
            files.append(f)
    return files




def ensure_writable_directory(path_str: str) -> bool:
    try:
        path = Path(path_str)

        # Create directory if missing
        path.mkdir(parents=True, exist_ok=True)

        if not path.is_dir():
            return False

        return os.access(path, os.W_OK)

    except Exception:
        return False