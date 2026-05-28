"""
Disk usage tree (JSON exportable) for the Darwin data directory.
Labels hashed directories with original filenames from the file registry.
"""

from pathlib import Path
from typing import Optional

from src.core.Storage.StorePaths import StorePaths
from src.core.Storage.FileRegistry import FileRegistry
from src.core.Models.ModelRegistry import REGISTRY
from src.core.Models.ModelStore import get_model_store

# ── size helpers ─────────────────────────────────────────────────────────

def dir_size(path: Path) -> int:
    """Total bytes of all files under `path`. Returns 0 if path doesn’t exist."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _format_size(n: int) -> str:
    """Human‑readable size string."""
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


# ── tree builder (always returns the full Darwin root) ───────────────────

def build_tree(darwin_dir: Path, registry: FileRegistry, model_store) -> DiskNode:
    paths = StorePaths(darwin_dir)

    # Helper: given a hash directory name, try to find the original filename
    def _label_from_hash(hash_name: str) -> str:
        for rec in registry.list_all():
            if rec.file_hash == hash_name:
                return f"{rec.original_name} ({hash_name[:8]}…)"
        return ""

    # ── Models ────────────────────────────────────────────────────────
    models_node = DiskNode("models", paths.models_dir)
    if paths.models_dir.exists():
        for key in model_store.list_installed():
            model_path = model_store.model_path(key)
            size = dir_size(model_path)
            spec = REGISTRY[key]
            label = f"{spec.model_id} ({spec.tier})"
            child = DiskNode(key, model_path, size, label=label)
            models_node.children.append(child)
        models_node.size = sum(c.size for c in models_node.children)

    # ── Indexes (content‑hashed directories) ──────────────────────────
    indexes_node = DiskNode("indexes", paths.indexes_dir)
    if paths.indexes_dir.exists():
        for hash_dir in paths.indexes_dir.iterdir():
            if not hash_dir.is_dir():
                continue
            total = dir_size(hash_dir)
            label = _label_from_hash(hash_dir.name)
            child = DiskNode(hash_dir.name, hash_dir, total, label=label)
            # Optionally add sub-children for major artifacts
            for artifact in ["chunks.json", "meta.json", "markdown.md",
                              "embeddings_*.npz", "bm25.npz", "vector.usearch",
                              "graph", "wiki", "images"]:
                for found in hash_dir.glob(artifact):
                    art_size = dir_size(found)
                    child.children.append(DiskNode(found.name, found, art_size))
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

    # ── Root config files ─────────────────────────────────────────────
    root_files = []
    for root_file in ["config.json", "file_registry.json"]:
        fp = darwin_dir / root_file
        if fp.exists():
            root_files.append(DiskNode(root_file, fp, fp.stat().st_size))
    configs_node = DiskNode("config", darwin_dir, size=sum(f.size for f in root_files),
                            children=root_files)

    # ── Root node ─────────────────────────────────────────────────────
    root = DiskNode(darwin_dir.name, darwin_dir)
    root.children = [models_node, indexes_node, checkpoints_node,
                     conv_node, logs_node, configs_node]
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
    """Pretty‑print a tree to the console with sizes."""
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