# src/core/Storage/DiskUsage.py
"""
Disk usage tree (JSON exportable) for the Darwin data directory.
Labels hashed directories with original filenames from the file registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

from src.core.Storage.StorePaths import StorePaths, ARTIFACT_MANIFEST
from src.core.Models.Lib.ModelRegistry import find_spec
from src.core.Models.ModelStore import ModelStore
from src.lib.AppPaths import default_models_dir

# ── size helpers ─────────────────────────────────────────────────────────

def dir_size(path: Path) -> int:
    """Total bytes of all files under `path`. Returns 0 if path doesn't exist."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _format_size(n: int) -> str:
    """Human-readable size string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


# ── tree node definition ─────────────────────────────────────────────────

class DiskNode:
    __slots__ = ("name", "path", "size", "children", "label")
    def __init__(self, name: str, path: Path, size: int = 0, children=None, label: str = ""):
        self.name = name
        self.path = path
        self.size = size          # total bytes of this subtree
        self.children = children or []
        self.label = label        # extra info (e.g., original filename)


# ── tree builder ────────────────────────────────────────────────────────

def build_tree(
    darwin_dir: Path,
    registry,                    # your file registry (can be None if not needed)
    model_store: Optional[ModelStore] = None,
    hash_label_fn: Optional[Callable[[str], str]] = None,
) -> DiskNode:
    """
    Build a DiskNode tree of the entire Darwin directory.
    If model_store is None, it will be created with the default models dir.
    """
    if model_store is None:
        model_store = ModelStore()

    paths = StorePaths(darwin_dir)

    # Helper: label for a hash directory (if registry provided)
    def _label_from_hash(hash_name: str) -> str:
        if registry and hash_label_fn:
            return hash_label_fn(hash_name)
        return ""

    # ── Models ────────────────────────────────────────────────────────
    models_node = DiskNode("models", default_models_dir())
    if default_models_dir().exists():
        for key in model_store.list_installed():
            model_path = model_store.model_path(key)
            if model_path is None:
                continue
            size = dir_size(model_path)
            spec = find_spec(key)
            label = f"{spec.model_id} ({spec.tier})" if spec else key
            child = DiskNode(key, model_path, size, label=label)
            models_node.children.append(child)
        models_node.size = sum(c.size for c in models_node.children)

    # ── Indexes (content-hashed directories) ──────────────────────────
    indexes_node = DiskNode("indexes", paths.indexes_dir)
    if paths.indexes_dir.exists():
        for hash_dir in paths.indexes_dir.iterdir():
            if not hash_dir.is_dir():
                continue
            total = dir_size(hash_dir)
            label = _label_from_hash(hash_dir.name)
            child = DiskNode(hash_dir.name, hash_dir, total, label=label)

            # Enumerate artifacts from the canonical manifest
            for stage, spec in ARTIFACT_MANIFEST.items():
                for key, rel_path in spec.items():
                    artifact_path = hash_dir / rel_path
                    if artifact_path.exists():
                        art_size = dir_size(artifact_path)
                        child.children.append(
                            DiskNode(
                                artifact_path.name,
                                artifact_path,
                                art_size,
                                label=stage,
                            )
                        )

            indexes_node.children.append(child)
        indexes_node.size = sum(c.size for c in indexes_node.children)

    # ── Checkpoints ───────────────────────────────────────────────────
    checkpoints_node = DiskNode("checkpoints", paths.checkpoints_dir)
    if paths.checkpoints_dir.exists():
        checkpoints_node.size = dir_size(paths.checkpoints_dir)

    # ── Conversations ─────────────────────────────────────────────────
    conv_node = DiskNode("conversations", paths.conversations_dir)
    if paths.conversations_dir.exists():
        conv_node.size = dir_size(paths.conversations_dir)

    # ── Logs ──────────────────────────────────────────────────────────
    logs_dir = darwin_dir / "logs"
    logs_node = DiskNode("logs", logs_dir)
    if logs_dir.exists():
        logs_node.size = dir_size(logs_dir)

    # ── Root node ─────────────────────────────────────────────────────
    root = DiskNode(darwin_dir.name, darwin_dir)
    root.children = [models_node, indexes_node, checkpoints_node,
                     conv_node, logs_node]
    root.size = sum(c.size for c in root.children)
    return root


# ── JSON export for React treemap ────────────────────────────────────────

def disk_node_to_dict(node: DiskNode) -> dict:
    """Convert a DiskNode into a nested dict suitable for a React treemap component."""
    result = {
        "name": node.name,
        "size": node.size,
    }
    if node.label:
        result["label"] = node.label
    if node.children:
        result["children"] = [disk_node_to_dict(c) for c in node.children]
    return result


# ── Path traversal inside the tree ──────────────────────────────────────

def find_node(root: DiskNode, parts: list) -> Optional[DiskNode]:
    """Return the node reachable by following `parts` down from `root`."""
    current = root
    for part in parts:
        found = None
        for child in current.children:
            if child.name == part:
                found = child
                break
        if found is None:
            return None
        current = found
    return current


# ── Terminal tree printer ────────────────────────────────────────────────

def print_tree(node: DiskNode, indent: str = "", is_last: bool = True):
    """Pretty-print a tree to the console with sizes."""
    connector = "└── " if is_last else "├── "
    name = node.name
    if node.label:
        name += f"  ({node.label})"
    size_str = _format_size(node.size) if node.size else "-"
    print(f"{indent}{connector}{name}  [{size_str}]")

    if not node.children:
        return

    new_indent = indent + ("    " if is_last else "│   ")
    sorted_children = sorted(node.children, key=lambda c: c.size, reverse=True)
    for i, child in enumerate(sorted_children):
        last = (i == len(sorted_children) - 1)
        print_tree(child, new_indent, last)


def print_treemap(root: DiskNode):
    """Currently an alias for print_tree; will become a true ASCII treemap later."""
    print_tree(root)