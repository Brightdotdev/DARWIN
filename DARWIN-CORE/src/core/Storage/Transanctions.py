from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def write_atomic(path: Path, data: bytes) -> None:
    """Write bytes atomically via temp + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def delete_tree(path: Path) -> None:
    if path.exists() and path.is_dir():
        shutil.rmtree(path)