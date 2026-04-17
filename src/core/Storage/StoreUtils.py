from pathlib import Path
import hashlib
import re
from datetime import datetime
from dataclasses import dataclass, asdict, field
import json


# ── Constants ──────────────────────────────────────────────────────────────────

DARWIN_DIR  = ".darwin"
CONFIG_FILE = "config.json"
FILE_META   = "meta.json"

# ── Default config ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "version":        "0.1.0",
    "embedding_tier": "light",        # light | medium | large
    "embedding_model": "intfloat/e5-small-v2",
    "llm_mode":       "none",        # local | api | none
    "llm_model":      "none",            # e.g. "ollama/llama3.2" or "gpt-4o" or none
    "reranker":       "ms-marco-TinyBERT-L-2-v2",
    "parser":         "light",        # light | high
    "top_k":          5,
    "graph_threshold": 0.40,          # default zoom level
    "created_at":     None,           # set on init
    "updated_at":     None,           # set on each update
    "updates" : {},              # tracks when each component was last updated
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def sanitize(filename: str) -> str:
    """
    Turn "biology.pdf" into "biology_pdf".
    Replaces dots, spaces, and special chars with underscores.
    Folder names with dots cause occasional headaches on some systems.
    """
    name = Path(filename).name          # strip any directory prefix
    name = re.sub(r"[^\w]", "_", name) # replace non-alphanumeric with _
    name = re.sub(r"_+", "_", name)    # collapse multiple underscores
    return name.strip("_").lower()


def file_hash(path: Path) -> str:
    """
    MD5 hash of file contents.
    Used to detect if a file has changed since last processing.
    If hash matches stored hash → skip reprocessing.
    If hash differs            → file changed, reprocess.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()



def now() -> str:
    return datetime.now().isoformat()




# ── JSON helpers ───────────────────────────────────────────────────────────

def write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}








# ── File store ─────────────────────────────────────────────────────────────────


@dataclass
class FileMeta:
    filename:    str
    file_hash:   str
    chunk_count: int
    page_count:  int
    parser:      str
    version:     str = "1.0"
    created_at:  str = ""
    updates:     dict[str, str] = field(default_factory=dict)
    updated_at:  str = ""

