"""
darwin.core.Storage.DarwinStore
=================

Persistence layer for Darwin.

This module defines the DarwinStore, which is responsible for managing
all on-disk state inside a project’s `.darwin/` directory. It acts as the
single source of truth for saving, loading, and validating processed data.

Core Responsibilities
--------------------
- Initialize and maintain the `.darwin/` workspace
- Persist intermediate and derived artifacts per file:
    - parsed markdown
    - chunks
    - embeddings
    - BM25 indices
    - stopwords
    - graph data
- Track processing state via metadata (`meta.json`)
- Detect file changes using content hashing
- Provide fast reload paths for previously processed files
- Maintain a lightweight audit log (`updates`) for pipeline stages

Directory Layout
----------------
For a project directory:

    your-notes/
    ├── biology.pdf
    └── .darwin/
        ├── config.json
        └── biology_pdf/
            ├── meta.json
            ├── chunks.json
            ├── biology_pdf.md
            ├── embeddings.npy
            ├── bm25.pkl
            ├── stopwords.json
            └── graph/
                ├── base_graph.json
                └── meta.json

Design Notes
------------
- One subdirectory per source file (sanitized filename)
- All filesystem access is centralized in DarwinStore
- Metadata is append-only in spirit (audit-style updates tracking)
- Expensive computations (embeddings, graphs) are cached and reused
- File freshness is determined via MD5 hash comparison
- Methods are safe to call repeatedly (idempotent where possible)

Typical Usage
-------------
Initialize a project:

    store = DarwinStore("/path/to/notes")
    store.init()

Processing pipeline:

    store.save_chunks("biology.pdf", chunks)
    store.save_markdown("biology.pdf", parsed_doc)
    store.save_embeddings("biology.pdf", embeddings)
    store.save_bm25("biology.pdf", bm25_index)
    store.save_graph("biology.pdf", graph_data)

Reloading cached data:

    if store.is_fresh("biology.pdf"):
        chunks = store.load_chunks("biology.pdf")
        embeddings = store.load_embeddings("biology.pdf")
    else:
        # file changed → reprocess pipeline
        ...

Inspection:

    store.status()              # project-level overview
    store.load_meta_file(...)    # raw metadata
    store.get_component_status(...)  # pipeline health snapshot

Notes
-----
- This module does not implement parsing, embedding, or graph logic.
  It strictly handles persistence and state tracking.
- Higher-level orchestration should live outside this module (CLI/services).
"""



import json
import pickle
import hashlib
import shutil
import re
from datetime import datetime
from pathlib import Path
from dataclasses import  asdict
import numpy as np

from src.core.Chunker import Chunk
from src.core.Parsing.ParserUtils import ParsedDocument
from src.lib.Terminal import TerminalUI

from src.core.Storage.StoreUtils import (
    DEFAULT_CONFIG,DARWIN_DIR,DEFAULT_CONFIG,CONFIG_FILE,FILE_META, FileMeta, GraphMeta,file_hash,now,sanitize
)



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
        self.darwin_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.darwin_dir / CONFIG_FILE

        if not config_path.exists():
            config = dict(DEFAULT_CONFIG)
            config["created_at"] = now()
            self._write_json(config_path, config)
            return True

        return False
    

    def has_markdown(self, filename: str) -> bool:
        """
        Checks if a processed markdown version of the source file exists.
        Returns True if .darwin/sanitized_name/sanitized_name.md exists.
        """
        sanitized_name = sanitize(filename)
        md_path = self._file_dir(sanitized_name) / f"{sanitized_name}.md"
        
        return md_path.exists()

    def has_chunks(self, filename: str) -> bool:
        """
        Checks if processed chunks of the source file exist.
        Returns True if .darwin/sanitized_name/sanitized_name.chunks exists.
        """
        sanitized_name = sanitize(filename)
        chunks_path = self._file_dir(sanitized_name) / f"{sanitized_name}.chunks"
        
        return chunks_path.exists()
    

    
    def has_embeddings(self, filename: str) -> bool:
        """
        Checks if embeddings of the source file exist.
        Returns True if .darwin/sanitized_name/sanitized_name.embeddings exists.
        """
        sanitized_name = sanitize(filename)
        config = self.load_config()
        embeddings_path =self._file_dir(sanitized_name) / f"{sanitize(config["embedding_model"])}_{sanitize(filename)}_embeddings.npy"

        return embeddings_path.exists()

    def has_bm25(self, filename: str) -> bool:
        """
        Checks if a BM25 index of the source file exists.
        Returns True if .darwin/sanitized_name/bm25.pkl exists.
        """
        sanitized_name = sanitize(filename)
        bm25_path = self._file_dir(sanitized_name) / "bm25.pkl"
        
        return bm25_path.exists()

    def has_stopwords(self, filename: str) -> bool:
        """
        Checks if domain stopwords for the source file exist.
        Returns True if .darwin/sanitized_name/stopwords.json exists.
        """
        sanitized_name = sanitize(filename)
        stopwords_path = self._file_dir(sanitized_name) / "stopwords.json"
        
        return stopwords_path.exists()

    def has_graph(self, filename: str) -> bool:
        """Returns True if a graph has been built for this file."""
        return (self._graph_dir(filename) / "base_graph.json").exists()




    def _update_component(self, filename: str, component: str, extras: dict = None) -> None:
        """
        Atomically updates meta.json. 
        Creates the file with defaults if it doesn't exist.
        """
        meta_path = self._file_dir(filename) / "meta.json"
        meta = self._read_json(meta_path)
        
        # 1. THE INITIALIZATION CHECK
        if not meta:
            # This is the "First Time Setup" for this specific file
            meta = {
                "filename": filename,
                "created_at": now(),
                "updates": {},
                "updated_at": now(),
                "version": "1.0" # Good for future-proofing!
            }
        
        # 2. Ensure nested structures exist (defensive)
        if "updates" not in meta:
            meta["updates"] = {}
        
        # 3. Update the Audit Log
        timestamp = now()
        meta["updates"][component] = timestamp
        meta["updated_at"] = timestamp

        # 4. Merge the extras (the ParsedDocument fields)
        if extras:
            # .update() merges dictionaries without wiping existing keys
            meta.update(extras)

        # 5. Commit to disk
        self._write_json(meta_path, meta)

    def get_component_status(self, filename: str) -> dict:
        """
        Returns a health check of the file's processing pipeline. 
        Just a snapshot of which components have been updated and when, based on meta.json.
        """
        meta = self.load_meta_file(filename)
        updates = meta.get("updates", {})
        
        return {
            "is_parsed": "parser" in updates,
            "is_vectorized": "embeddings" in updates,
            "has_search_index": "bm25" in updates,
            "has_graph": "graph" in updates,
            "last_activity": meta.get("updated_at", "never"),
            "has_markdown": self.has_markdown(filename),
            "has_chunks": self.has_chunks(filename),
            "has_embeddings": self.has_embeddings(filename),
        }



    def is_initialized(self) -> bool:
        return self.darwin_dir.exists()
    

    def _require_init(self):
        if not self.darwin_dir.exists():
            self.init()

        config_path = self.darwin_dir / CONFIG_FILE

        if not config_path.exists():
            config = dict(DEFAULT_CONFIG)
            config["created_at"] = now()
            self._write_json(config_path, config)
        # ── Directory helpers ──────────────────────────────────────────────────────

    def _file_dir(self, filename: str) -> Path:
        """Returns .darwin/biology_pdf/ for "biology.pdf"."""
        return self.darwin_dir / sanitize(filename)
    
    def image_dir(self, filename: str) -> Path:
        """Returns the path for extracted images for a given file."""
        return self._file_dir(filename) / "extracted_images"

    def ensure_image_dir(self, filename: str) -> Path:
        """
        Ensure the image directory exists. 
        Just creates it if missing.
        """
        img_dir = self.image_dir(filename)
        img_dir.mkdir(parents=True, exist_ok=True)
        return img_dir

    def ensure_image_dir_and_check(self, filename: str) -> tuple[Path, bool]:
        """
        Ensure the image directory exists and check if it contains PNG images.
        Returns: (directory path, has_png_images)
        """
        img_dir = self.ensure_image_dir(filename)
        png_files = list(img_dir.glob("*.png"))
        has_images = len(png_files) > 0
        return img_dir, has_images


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
        if not path.exists():
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
   
    # ── Config ─────────────────────────────────────────────────────────────────

    def load_config(self) -> dict:
        self._require_init()
        return self._read_json(self.darwin_dir / CONFIG_FILE)
    
    

    def load_meta_file(self, filename: str) -> dict:
        self._require_init()
        metaJsonFIle = self._read_json(self._file_dir(filename) / FILE_META)
        if not metaJsonFIle:
            return {}
        return metaJsonFIle


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
        meta_path = self._file_dir(filename) / FILE_META

        if not meta_path.exists():
            return False   # never processed

        file_path = self.project_dir / filename
        if not file_path.exists():
            return False   # source file gone

        stored_hash  = self._read_json(meta_path).get("file_hash", "")
        current_hash = file_hash(file_path)
        return stored_hash == current_hash


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
            Serialize and save chunks to .darwin/sanitized_name/chunks.json
            Also writes meta.json using the update_component logic for consistency.
            """
            self._require_init()
            d = self._ensure_file_dir(filename)

            if self.is_fresh(filename) and self.has_chunks(filename):
                TerminalUI.bold("Skipping chunk saving (cached chunks found and file is fresh)")
                return

            # 1. Serialize and save the actual data
            chunks_data = [asdict(c) for c in chunks]
            self._write_json(d / "chunks.json", chunks_data)

            # 2. Check for existing meta to preserve the 'created_at' and 'updates' log
            meta_path = d / FILE_META
            existing_meta = self._read_json(meta_path)
            
            # 3. Construct the FileMeta object
            # We use existing values if they exist, otherwise defaults.
            meta = FileMeta(
                filename    = filename,
                file_hash   = file_hash or file_hash(self.project_dir / filename),
                chunk_count = len(chunks),
                page_count  = existing_meta.get("page_count", page_count),
                parser  = existing_meta.get("parser", parser),
                created_at  = existing_meta.get("created_at", now()),
                updates     = existing_meta.get("updates", {}), # Preserve existing audit log
                updated_at  = now(),
            )

            # 4. Save the base meta file
            self._write_json(meta_path, asdict(meta))

            # 5. Use your new update_component method to mark 'chunks' as updated
            # This ensures the 'chunks' key inside 'updates' is set/refreshed
            self._update_component(filename, "chunks")



    def load_chunks(self, filename: str) -> list[Chunk]:
        path = self._file_dir(filename) / "chunks.json"
        raw = self._read_json(path)
        
        # Get valid field names from the Chunk dataclass
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(Chunk)}

        processed = []
        for c in raw:
            # 1. Backwards Compatibility Defaults
            c.setdefault("clean",    c.get("text", ""))
            c.setdefault("page",     0)
            c.setdefault("page_end", 0)
            c.setdefault("heading",  "")
            
            # 2. Filter out any keys that aren't in the current Chunk class
            # This prevents TypeError: __init__() got an unexpected keyword argument
            filtered_c = {k: v for k, v in c.items() if k in valid_fields}
            
            processed.append(Chunk(**filtered_c))

        return processed

        # ─ Markdown ────────────────────────────────────────────────────────────────
    def save_markdown(
        self,
        filename: str,
        mdDocument: ParsedDocument,
        rerun: bool = False
    ) -> None:
        """
        Saves parsed Markdown and metadata. 
        Handles rerun logic if the user changes settings (like images/strategy).
        """
        # 1. Immediate Safety Check
        if mdDocument.error:
            TerminalUI.info(
                label="Markdown Error", 
                value=f"⚠️ Skipping {filename}: {mdDocument.error}"
            )
            return

        self._require_init()
        d = self._ensure_file_dir(filename)
        md_path = d / f"{sanitize(filename)}.md"

        # 2. The "Should I Skip?" Logic
        # We skip ONLY if it's not a rerun AND the file is fresh AND the file exists
        if not rerun:
            if self.is_fresh(filename) and md_path.exists():
                TerminalUI.bold(f"Skipping: Fresh markdown found for {filename}")
                return
        else:
            TerminalUI.bold(f"Rerunning Markdown generation for {filename}...")

        # 3. Execution
        try:
            # Save the actual text content
            md_path.write_text(mdDocument.text, encoding="utf-8")

            # 4. Meta Cleanup (Keeping meta.json light)
            all_fields = asdict(mdDocument)
            
            
            for key in ["text", "error", "filename"]:
                all_fields.pop(key, None)



            self._update_component(filename, "markdown", extras=all_fields)
            
            TerminalUI.success(f"Markdown saved: {md_path.name}")

        except Exception as e:
            TerminalUI.error(f"❌ Failed to save markdown for {filename}: {e}")
   
    def load_markdown(self, filename: str) -> str | None:
        """
        Retrieves the full Markdown content for a file.
        Returns an empty string if the file doesn't exist.
        """
        

        if not self.has_markdown(filename):
            TerminalUI.error(f"❌ Markdown not found for {filename}")
            return None
        
        md_path = self._file_dir(sanitize(filename)) / f"{sanitize(filename)}.md"
        try:
            return md_path.read_text(encoding="utf-8")
        except Exception as e:
            # Handle potential OS-level read errors
            TerminalUI.error(f"❌ Error reading markdown for {filename}: {e}")
            return None


    # ── Embeddings ────────────────────────────────────────────────────────────────

    def save_embeddings(self, filename: str, embeddings: np.ndarray, rerun: bool = False) -> None:
        """Save the embedding matrix to a .npy file. Versioned by model name to allow multiple sets of embeddings."""
        self._require_init()
        d = self._ensure_file_dir(filename)
        
        # 1. Always load config first so it's available everywhere in the method
        config = self.load_config()
        model_name = sanitize(config.get('embedding_model', 'default'))
        file_name_clean = sanitize(filename)
        path = d / f"{model_name}_{file_name_clean}_embeddings.npy"

        # 2. Check for "Skip" conditions
        if not rerun:
            if self.is_fresh(filename) and path.exists():
                TerminalUI.bold(f"Skipping: Fresh embeddings already exist for {filename}")
                return
        else:
            # If rerunning, we only skip if the specific model's file already exists 
            # (This matches your logic of not overwriting if already done)
            if path.exists():
                TerminalUI.bold(f"Model-specific embedding exists, skipping overwrite.")
                return
            TerminalUI.bold(f"Rerunning Embeddings for {filename}...")

        # 3. Single Save Point
        try:
            np.save(path, embeddings)
            self._update_component(filename, "embeddings")
            TerminalUI.success(f"Saved embeddings: {path.name}")
        except Exception as e:
            TerminalUI.error(f"Failed to save embeddings: {e}")

    def load_embeddings(self, filename: str) -> np.ndarray:
        """Load embedding matrix from .npy file."""
        self._require_init()
        
        config = self.load_config()
        path = self._file_dir(filename) / f"{sanitize(config.get('embedding_model',''))}_{sanitize(filename)}_embeddings.npy"
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
        if self.is_fresh(filename) and self.has_bm25(filename):
            TerminalUI.bold("Skipping BM25 saving (cached index found and file is fresh)")
            return
        
        d    = self._ensure_file_dir(filename)
        path = d / "bm25.pkl"
        with open(path, "wb") as f:
            pickle.dump(bm25_index, f)
        self._update_component(filename, "bm25")


    def load_bm25(self, filename: str):
        """Load pickled BM25 index."""
        if not self.has_bm25(filename):
            TerminalUI.error(f"❌ BM25 index not found for {filename}")
            return None
    
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
        if self.is_fresh(filename) and self.has_stopwords(filename):
            TerminalUI.bold("Skipping stopwords saving (cached stopwords found and file is fresh)")
            return
        self._require_init()
        d    = self._ensure_file_dir(filename)
        path = d / "stopwords.json"
        self._write_json(path, {"stopwords": sorted(stopwords)})
        self._update_component(filename, "stopwords")

    def load_stopwords(self, filename: str) -> set[str]:
        """Load domain stopwords as a set."""
        if not self.has_stopwords(filename):
            TerminalUI.error(f"❌ Stopwords not found for {filename}")
            return set()
    
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
        if self.is_fresh(filename) and self.has_graph(filename):
            TerminalUI.bold("Skipping graph saving (cached graph found and file is fresh)")
            return
        
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
            built_at      = now(),
        )
        self._write_json(d / FILE_META, asdict(meta))
        self._update_component(filename, "graph")

    def load_graph(self, filename: str) -> dict:
        """
        Load base graph data from disk.
        Returns the full dict with concepts + similarity matrix.
        """
        if not self.has_graph(filename):
            TerminalUI.error(f"❌ Graph not found for {filename}")
            return {}
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
        path = self._graph_dir(filename) / FILE_META

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

