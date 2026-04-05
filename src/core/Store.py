"""
darwin.core.store
-----------------
Handles all persistence for Darwin.

Creates and manages the .darwin folder structure inside any directory
the user points Darwin at. One subfolder per file, graph per file.

Structure:
    your-notes/
    ├── biology.pdf
    └── .darwin/
        ├── config.json
        └── biology_pdf/
            ├── meta.json
            ├── chunks.json
            ├── embeddings.npy
            ├── bm25.pkl
            ├── stopwords.json
            └── graph/
                ├── base_graph.json
                └── meta.json

Usage:
    store = DarwinStore("/path/to/your/notes")
    store.init()

    # save after processing
    store.save_chunks("biology.pdf", chunks)
    store.save_embeddings("biology.pdf", embeddings)
    store.save_bm25("biology.pdf", bm25_index)
    store.save_graph("biology.pdf", graph_data)

    # load on next startup
    if store.is_fresh("biology.pdf"):         # file unchanged?
        chunks = store.load_chunks("biology.pdf")
        ...
    else:
        # file changed — reprocess
"""

import json
import pickle
import hashlib
import shutil
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np

from core.Chunker import Chunk


# ── Constants ──────────────────────────────────────────────────────────────────

DARWIN_DIR  = ".darwin"
CONFIG_FILE = "config.json"


# ── Default config ─────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "version":        "0.1.0",
    "embedding_tier": "light",        # light | medium | large
    "embedding_model": "intfloat/e5-small-v2",
    "llm_mode":       "local",        # local | api | none
    "llm_model":      "ollama/llama3.2",
    "reranker":       "ms-marco-TinyBERT-L-2-v2",
    "parser":         "light",        # light | high
    "top_k":          5,
    "graph_threshold": 0.40,          # default zoom level
    "created_at":     None,           # set on init
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sanitize(filename: str) -> str:
    """
    Turn "biology.pdf" into "biology_pdf".
    Replaces dots, spaces, and special chars with underscores.
    Folder names with dots cause occasional headaches on some systems.
    """
    name = Path(filename).name          # strip any directory prefix
    name = re.sub(r"[^\w]", "_", name) # replace non-alphanumeric with _
    name = re.sub(r"_+", "_", name)    # collapse multiple underscores
    return name.strip("_").lower()


def _file_hash(path: Path) -> str:
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


def _now() -> str:
    return datetime.utcnow().isoformat()


# ── File store ─────────────────────────────────────────────────────────────────

@dataclass
class FileMeta:
    filename:    str      # original filename e.g. "biology.pdf"
    file_hash:   str      # md5 of file contents
    chunk_count: int      # how many chunks were produced
    page_count:  int      # how many pages the PDF has
    parser:      str      # which parser was used: light | high
    created_at:  str      # when first processed
    updated_at:  str      # when last processed


@dataclass
class GraphMeta:
    filename:       str    # which file this graph belongs to
    concept_count:  int    # how many unique concept nodes
    edge_count:     int    # edges at default threshold
    threshold:      float  # threshold used when graph was built
    built_at:       str    # when graph was built


# ── Main store class ───────────────────────────────────────────────────────────

class DarwinStore:
    """
    Manages the .darwin folder for a project directory.

    One instance per project directory. All save/load operations
    go through here — nothing else touches the filesystem directly.

    Args:
        project_dir: path to the folder containing the user's documents
                     e.g. "/home/user/biology-notes"
    """

    def __init__(self, project_dir: str | Path):
        self.project_dir = Path(project_dir).resolve()
        self.darwin_dir  = self.project_dir / DARWIN_DIR

    # ── Setup ──────────────────────────────────────────────────────────────────

    def init(self) -> bool:
        """
        Initialize the .darwin folder for this project.
        Safe to call multiple times — does nothing if already initialized.

        Returns:
            True if freshly initialized, False if already existed
        """
        if self.darwin_dir.exists():
            return False   # already initialized

        self.darwin_dir.mkdir(parents=True)

        # write default config
        config = dict(DEFAULT_CONFIG)
        config["created_at"] = _now()
        self._write_json(self.darwin_dir / CONFIG_FILE, config)

        return True

    def is_initialized(self) -> bool:
        return self.darwin_dir.exists()

    def _require_init(self):
        if not self.is_initialized():
            raise RuntimeError(
                f"Darwin not initialized in {self.project_dir}\n"
                f"Run: darwin init"
            )

    # ── Directory helpers ──────────────────────────────────────────────────────

    def _file_dir(self, filename: str) -> Path:
        """Returns .darwin/biology_pdf/ for "biology.pdf"."""
        return self.darwin_dir / _sanitize(filename)

    def _graph_dir(self, filename: str) -> Path:
        """Returns .darwin/biology_pdf/graph/"""
        return self._file_dir(filename) / "graph"

    def _ensure_file_dir(self, filename: str) -> Path:
        d = self._file_dir(filename)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ensure_graph_dir(self, filename: str) -> Path:
        d = self._graph_dir(filename)
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── JSON helpers ───────────────────────────────────────────────────────────

    def _write_json(self, path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _read_json(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Config ─────────────────────────────────────────────────────────────────

    def load_config(self) -> dict:
        self._require_init()
        return self._read_json(self.darwin_dir / CONFIG_FILE)

    def save_config(self, config: dict) -> None:
        self._require_init()
        self._write_json(self.darwin_dir / CONFIG_FILE, config)

    def update_config(self, **kwargs) -> None:
        """Update specific config keys without overwriting everything."""
        config = self.load_config()
        config.update(kwargs)
        self.save_config(config)

    # ── File freshness check ───────────────────────────────────────────────────

    def is_fresh(self, filename: str) -> bool:
        """
        Returns True if the file has not changed since last processing.
        Compares current MD5 hash against stored hash in meta.json.

        If True  → safe to load cached chunks, embeddings, etc.
        If False → file changed or never processed, must reprocess.
        """
        self._require_init()
        meta_path = self._file_dir(filename) / "meta.json"

        if not meta_path.exists():
            return False   # never processed

        file_path = self.project_dir / filename
        if not file_path.exists():
            return False   # source file gone

        stored_hash  = self._read_json(meta_path).get("file_hash", "")
        current_hash = _file_hash(file_path)
        return stored_hash == current_hash

    def has_graph(self, filename: str) -> bool:
        """Returns True if a graph has been built for this file."""
        return (self._graph_dir(filename) / "base_graph.json").exists()

    def list_processed_files(self) -> list[str]:
        """
        Returns list of original filenames that have been processed.
        Reads from meta.json in each subfolder.
        """
        self._require_init()
        result = []
        for d in self.darwin_dir.iterdir():
            if d.is_dir() and d.name != "graph":
                meta_path = d / "meta.json"
                if meta_path.exists():
                    meta = self._read_json(meta_path)
                    result.append(meta.get("filename", d.name))
        return result

    # ── Chunks ─────────────────────────────────────────────────────────────────

    def save_chunks(
        self,
        filename:   str,
        chunks:     list[Chunk],
        file_hash:  str  = "",
        page_count: int  = 0,
        parser:     str  = "light",
    ) -> None:
        """
        Serialize and save chunks to .darwin/biology_pdf/chunks.json
        Also writes meta.json with file hash for freshness checks.
        """
        self._require_init()
        d = self._ensure_file_dir(filename)

        # serialize chunks — dataclass → dict
        chunks_data = [asdict(c) for c in chunks]
        self._write_json(d / "chunks.json", chunks_data)

        # write meta
        meta = FileMeta(
            filename    = filename,
            file_hash   = file_hash or _file_hash(self.project_dir / filename),
            chunk_count = len(chunks),
            page_count  = page_count,
            parser      = parser,
            created_at  = _now(),
            updated_at  = _now(),
        )
        self._write_json(d / "meta.json", asdict(meta))

    def load_chunks(self, filename: str) -> list[Chunk]:
        """
        Load chunks from disk and reconstruct Chunk dataclass objects.
        """
        self._require_init()
        path = self._file_dir(filename) / "chunks.json"

        if not path.exists():
            raise FileNotFoundError(
                f"No chunks found for {filename}. "
                f"Has this file been processed? Run: darwin add {filename}"
            )

        raw = self._read_json(path)
        return [Chunk(**c) for c in raw]

    # ── Embeddings ─────────────────────────────────────────────────────────────

    def save_embeddings(self, filename: str, embeddings: np.ndarray) -> None:
        """
        Save embedding matrix as .npy (numpy binary format).
        Much faster to load than JSON — a 1000-chunk matrix loads
        in milliseconds vs seconds.
        """
        self._require_init()
        d    = self._ensure_file_dir(filename)
        path = d / "embeddings.npy"
        np.save(path, embeddings)

    def load_embeddings(self, filename: str) -> np.ndarray:
        """Load embedding matrix from .npy file."""
        self._require_init()
        path = self._file_dir(filename) / "embeddings.npy"

        if not path.exists():
            raise FileNotFoundError(
                f"No embeddings found for {filename}."
            )

        return np.load(path)

    # ── BM25 index ─────────────────────────────────────────────────────────────

    def save_bm25(self, filename: str, bm25_index) -> None:
        """
        Pickle the BM25 index object to disk.
        Pickle is the right tool here — BM25Okapi is not JSON serializable
        and rebuilding it from scratch on every startup is wasteful.
        """
        self._require_init()
        d    = self._ensure_file_dir(filename)
        path = d / "bm25.pkl"
        with open(path, "wb") as f:
            pickle.dump(bm25_index, f)

    def load_bm25(self, filename: str):
        """Load pickled BM25 index."""
        self._require_init()
        path = self._file_dir(filename) / "bm25.pkl"

        if not path.exists():
            raise FileNotFoundError(
                f"No BM25 index found for {filename}."
            )

        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Domain stopwords ───────────────────────────────────────────────────────

    def save_stopwords(self, filename: str, stopwords: set[str]) -> None:
        """
        Save auto-discovered domain stopwords for this file.
        Stored as a sorted list in JSON (sets aren't JSON serializable).
        Sorted so the file is human-readable and diff-friendly.
        """
        self._require_init()
        d    = self._ensure_file_dir(filename)
        path = d / "stopwords.json"
        self._write_json(path, {"stopwords": sorted(stopwords)})

    def load_stopwords(self, filename: str) -> set[str]:
        """Load domain stopwords as a set."""
        self._require_init()
        path = self._file_dir(filename) / "stopwords.json"

        if not path.exists():
            return set()   # no stopwords saved yet → return empty set

        data = self._read_json(path)
        return set(data.get("stopwords", []))

    # ── Graph ──────────────────────────────────────────────────────────────────

    def save_graph(
        self,
        filename:   str,
        graph_data: dict,
        threshold:  float = 0.40,
    ) -> None:
        """
        Save the base graph data for a file.

        graph_data should contain:
            {
                "concepts": [
                    {
                        "id":           0,
                        "label":        "chlorophyll absorbs red and blue light",
                        "weight":       8,
                        "source_chunks": [3, 7],
                        "page":         12
                    },
                    ...
                ],
                "similarity_matrix": [[1.0, 0.82, ...], ...]
            }

        The similarity matrix is the full n×n float matrix.
        The slider reads from this — never recomputed after build.
        """
        self._require_init()
        d = self._ensure_graph_dir(filename)

        self._write_json(d / "base_graph.json", graph_data)

        # write graph meta
        n_concepts = len(graph_data.get("concepts", []))
        matrix     = graph_data.get("similarity_matrix", [])

        # count edges at the default threshold
        edge_count = 0
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                if matrix[i][j] > threshold:
                    edge_count += 1

        meta = GraphMeta(
            filename      = filename,
            concept_count = n_concepts,
            edge_count    = edge_count,
            threshold     = threshold,
            built_at      = _now(),
        )
        self._write_json(d / "meta.json", asdict(meta))

    def load_graph(self, filename: str) -> dict:
        """
        Load base graph data from disk.
        Returns the full dict with concepts + similarity matrix.
        """
        self._require_init()
        path = self._graph_dir(filename) / "base_graph.json"

        if not path.exists():
            raise FileNotFoundError(
                f"No graph found for {filename}. "
                f"Run: darwin graph {filename}"
            )

        return self._read_json(path)

    def load_graph_meta(self, filename: str) -> dict:
        """Load graph metadata without loading the full similarity matrix."""
        self._require_init()
        path = self._graph_dir(filename) / "meta.json"

        if not path.exists():
            raise FileNotFoundError(f"No graph meta found for {filename}.")

        return self._read_json(path)

    # ── File removal ───────────────────────────────────────────────────────────

    def remove_file(self, filename: str) -> bool:
        """
        Remove all cached data for a file.
        Called when user removes a document from the project.

        Returns True if data existed and was removed, False if nothing to remove.
        """
        self._require_init()
        d = self._file_dir(filename)

        if not d.exists():
            return False

        shutil.rmtree(d)
        return True

    # ── Project status ─────────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Returns a summary of the current project state.
        Used by the CLI splash screen to show progress bars.

        Returns:
            {
                "project_dir":  "/home/user/notes",
                "files": [
                    {
                        "filename":    "biology.pdf",
                        "is_fresh":    True,
                        "has_chunks":  True,
                        "has_embeddings": True,
                        "has_bm25":    True,
                        "has_graph":   True,
                        "chunk_count": 42,
                        "concept_count": 18,
                        "page_count":  24,
                    }
                ],
                "config": { ... }
            }
        """
        self._require_init()

        files_status = []

        for d in sorted(self.darwin_dir.iterdir()):
            if not d.is_dir():
                continue

            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue

            meta     = self._read_json(meta_path)
            filename = meta.get("filename", d.name)

            # check what's been built
            has_chunks     = (d / "chunks.json").exists()
            has_embeddings = (d / "embeddings.npy").exists()
            has_bm25       = (d / "bm25.pkl").exists()
            has_graph      = (d / "graph" / "base_graph.json").exists()

            graph_meta     = {}
            if has_graph:
                try:
                    graph_meta = self._read_json(d / "graph" / "meta.json")
                except Exception:
                    pass

            files_status.append({
                "filename":      filename,
                "sanitized":     d.name,
                "is_fresh":      self.is_fresh(filename),
                "has_chunks":    has_chunks,
                "has_embeddings": has_embeddings,
                "has_bm25":      has_bm25,
                "has_graph":     has_graph,
                "chunk_count":   meta.get("chunk_count", 0),
                "page_count":    meta.get("page_count", 0),
                "concept_count": graph_meta.get("concept_count", 0),
                "parser":        meta.get("parser", "light"),
                "processed_at":  meta.get("updated_at", ""),
            })

        return {
            "project_dir": str(self.project_dir),
            "files":       files_status,
            "config":      self.load_config(),
        }