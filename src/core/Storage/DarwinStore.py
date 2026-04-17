"""
darwin.core.Storage.DarwinStore
=================

Persistence layer for Darwin.

This module defines the DarwinStore, which is responsible for managing
all on-disk state inside a project's `.darwin/` directory. It acts as the
single source of truth for saving, loading, and validating processed data.

Core Responsibilities
--------------------
- Initialize and maintain the `.darwin/` workspace
- Persist intermediate and derived artifacts per file:
    - parsed markdown
    - chunks
    - embeddings (bundled with chunk metadata in .npz)
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
    ├── biology_pdf (sanitized from "biology.pdf")
    └── .darwin/
        ├── config.json
        └── biology_pdf/
            ├── meta.json
            ├── chunks.json
            ├── biology_pdf.md
            ├── e5_small_biology_pdf.npz     ← vectors + chunk metadata bundled
            ├── bm25.pkl
            ├── stopwords.json
            └── graph/
                ├── base_graph.json
                └── meta.json

Embedding storage (.npz)
------------------------
Vectors and chunk metadata are stored in a single compressed numpy archive
per embedding model. This keeps them permanently in sync — if the file
exists, both vectors and metadata exist. No separate files to drift apart.

    data = np.load("e5_small_biology_pdf.npz", allow_pickle=True)
    vectors  = data["vectors"]    # shape (n_chunks, 384)
    text     = data["text"]       # original text per chunk
    clean    = data["clean"]      # stripped text per chunk
    headings = data["headings"]   # nearest heading per chunk
    pages    = data["pages"]      # page numbers
    source   = data["source"]     # filename repeated n times

Design Notes
------------
- One subdirectory per source file (sanitized filename)
- All filesystem access is centralized in DarwinStore
- Metadata is append-only in spirit (audit-style updates tracking)
- Expensive computations (embeddings, graphs) are cached and reused
- File freshness is determined via MD5 hash comparison
- Methods are safe to call repeatedly (idempotent where possible)
"""


import pickle
import shutil
from pathlib import Path
from dataclasses import asdict
from typing import Optional
import numpy as np

from src.core.Chunking.ChunkerUtils import Chunk
from src.core.Parsing.ParserUtils import ParsedDocument
from src.lib.Terminal import TerminalUI

from src.core.Storage.StoreUtils import (
    DEFAULT_CONFIG, DARWIN_DIR, CONFIG_FILE, FILE_META,
    FileMeta, file_hash, now, sanitize, read_json, write_json
)
from src.core.Knowledge.Graph.GraphUtils import(
    GraphData
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
            True if freshly initialized, False if already existed.
        """
        self.darwin_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.darwin_dir / CONFIG_FILE

        if not config_path.exists():
            config = dict(DEFAULT_CONFIG)
            config["created_at"] = now()
            write_json(config_path, config)
            return True

        return False

    def has_markdown(self, filename: str) -> bool:
        sanitized_name = sanitize(filename)
        md_path = self._file_dir(sanitized_name) / f"{sanitized_name}.md"
        return md_path.exists()

    def has_chunks(self, filename: str) -> bool:
        sanitized_name = sanitize(filename)
        chunks_path = self._file_dir(sanitized_name) / "chunks.json"
        return chunks_path.exists()

    def has_embeddings(self, filename: str) -> bool:
        """Check for .npz bundle (vectors + chunk metadata)."""
        return self._npz_path(filename).exists()

    def has_bm25(self, filename: str) -> bool:
        sanitized_name = sanitize(filename)
        bm25_path = self._file_dir(sanitized_name) / "bm25.pkl"
        return bm25_path.exists()

    def has_stopwords(self, filename: str) -> bool:
        sanitized_name = sanitize(filename)
        stopwords_path = self._file_dir(sanitized_name) / "stopwords.json"
        return stopwords_path.exists()

    def has_graph(self, filename: str) -> bool:
        g_fname =  f"{sanitize(filename)}_graph.json"
        return (self._graph_dir(filename) / g_fname).exists()
    def has_graph_html(self, filename: str) -> bool:
        g_html_name =  f"{sanitize(filename)}_graph.html"
        return (self._graph_dir(filename) / g_html_name).exists()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _npz_path(self, filename: str) -> Path:
        """
        Returns the .npz path for a given file and the current embedding model.
        e.g. .darwin/biology_pdf/e5_small_biology_pdf.npz
        """
        config     = self.load_config()
        model_name = sanitize(config.get("embedding_model", "intfloat/e5-small-v2"))
        file_clean = sanitize(filename)
        return self._file_dir(filename) / f"{model_name}_{file_clean}.npz"

    def _update_component(self, filename: str, component: str, extras: dict = None) -> None:
        meta_path = self._file_dir(filename) / "meta.json"
        meta = read_json(meta_path)

        if not meta:
            meta = {
                "filename": filename,
                "created_at": now(),
                "updates": {},
                "updated_at": now(),
                "version": "1.0",
            }

        if "updates" not in meta:
            meta["updates"] = {}

        timestamp = now()
        meta["updates"][component] = timestamp
        meta["updated_at"] = timestamp

        if extras:
            for key, value in extras.items():
                if isinstance(value, dict) and isinstance(meta.get(key), dict):
                    meta[key].update(value) 
                else:
                    meta[key] = value
        write_json(meta_path, meta)



    # ------ multi processing shenanigans
    # In DarwinStore class, add these methods:

    def get_markdown_path(self, filename: str) -> Path:
        sanitized_name = sanitize(filename)
        return self._file_dir(sanitized_name) / f"{sanitized_name}.md"
    def get_image_dir(self,filename: str) -> Path:
        sanitized_name = sanitize(filename)
        return self._file_dir(sanitized_name) / "images"
    def get_temp_directory(self) -> Optional[Path]:
        """Get the temporary directory for this store"""
        return self.project_dir / ".temp"

    def cleanup_partial_parsing(self, filename: str):
        """
        Clean up partial/incomplete results for a file.
        Called when parsing is cancelled mid-way.
        """
        # Remove partial markdown file
        md_path = self.get_markdown_path(filename)
        if md_path.exists():
            md_path.unlink()
        
        # Remove partial images directory
        img_dir = self.get_image_dir(filename)
        if img_dir and img_dir.exists():
            import shutil
            shutil.rmtree(img_dir)
        
        # Remove any lock files
        lock_file = self.project_dir / f".{filename}.lock"
        if lock_file.exists():
            lock_file.unlink()
        
        # Remove any temporary directory
        temp_dir = self.get_temp_directory()
        if temp_dir and temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            

    def get_component_status(self, filename: str) -> dict:
        meta    = self.load_meta_file(filename)
        updates = meta.get("updates", {})

        return {
            "is_parsed":        "parser"     in updates,
            "is_vectorized":    "embeddings" in updates,
            "has_search_index": "bm25"       in updates,
            "has_graph":       self.has_graph(filename) ,
            "last_activity":    meta.get("updated_at", "never"),
            "has_markdown":     self.has_markdown(filename),
            "has_chunks":       self.has_chunks(filename),
            "has_embeddings":   self.has_embeddings(filename),
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
            write_json(config_path, config)
    

    # ── Directory helpers ──────────────────────────────────────────────────────




    def _file_dir(self, filename: str) -> Path:
        return self.darwin_dir / sanitize(filename)

    def image_dir(self, filename: str) -> Path:
        return self._file_dir(filename) / "extracted_images"

    def ensure_image_dir(self, filename: str) -> Path:
        img_dir = self.image_dir(filename)
        img_dir.mkdir(parents=True, exist_ok=True)
        return img_dir

    def ensure_image_dir_and_check(self, filename: str) -> tuple[Path, bool]:
        img_dir   = self.ensure_image_dir(filename)
        has_images = len(list(img_dir.glob("*.png"))) > 0
        return img_dir, has_images

    def _graph_dir(self, filename: str) -> Path:
        return self._file_dir(filename) / "graph"

    def _ensure_file_dir(self, filename: str) -> Path:
        d = self._file_dir(filename)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ensure_graph_dir(self, filename: str) -> Path:
        d = self._graph_dir(filename)
        d.mkdir(parents=True, exist_ok=True)
        return d


    def wiki_dir(self, filename: str) -> Path:
        return self._file_dir(filename)/"wiki"
    def _ensure_wiki_dir(self, filename: str) -> Path:
        """Ensure wiki directory exists."""
        wiki_dir = self.wiki_dir(filename)
        wiki_dir.mkdir(parents=True, exist_ok=True)
        return wiki_dir

    def _wiki_dir(self, filename: str) -> Path:
        """Get wiki directory for document."""
        return self._ensure_wiki_dir(filename)


    # ── Config ─────────────────────────────────────────────────────────────────

    def load_config(self) -> dict:
        self._require_init()
        return read_json(self.darwin_dir / CONFIG_FILE)

    def load_meta_file(self, filename: str) -> dict:
        self._require_init()
        return read_json(self._file_dir(filename) / FILE_META) or {}

    def save_config(self, config: dict) -> None:
        self._require_init()
        write_json(self.darwin_dir / CONFIG_FILE, config)

    def update_config(self, **kwargs) -> None:
        config = self.load_config()
        config.update(kwargs)
        self.save_config(config)

    # ── File freshness ─────────────────────────────────────────────────────────

    def is_fresh(self, filename: str) -> bool:
        """
        Returns True if the file has not changed since last processing.
        Compares current MD5 hash against stored hash in meta.json.
        """
        self._require_init()
        meta_path = self._file_dir(filename) / FILE_META

        if not meta_path.exists():
            return False

        file_path = self.project_dir / filename
        if not file_path.exists():
            return False

        stored_hash  = read_json(meta_path).get("file_hash", "")
        current_hash = file_hash(file_path)
        return stored_hash == current_hash

    # ── Chunks ─────────────────────────────────────────────────────────────────

    def save_chunks(
        self,
        filename:       str,
        chunks:         list[Chunk],
        source_hash:    str = "",       # ← was `file_hash` — shadowed the import
        page_count:     int = 0,
        parser:         str = "light",
        max_tokens:     int = 0,        # ← new: chunking settings for staleness check
        chunk_model:    str = "",       # ← new: which embedding model these chunks target
    ) -> None:
        """
        Serialize and save chunks to .darwin/sanitized_name/chunks.json.

        max_tokens and chunk_model are stored in meta so we can detect when
        chunks need regenerating due to settings changes, not just file changes.
        """
        self._require_init()
        d = self._ensure_file_dir(filename)

        if self.is_fresh(filename) and self.has_chunks(filename):
            TerminalUI.bold("Skipping chunk saving (cached chunks found and file is fresh)")
            return

        chunks_data = [asdict(c) for c in chunks]
        write_json(d / "chunks.json", chunks_data)

        meta_path     = d / FILE_META
        existing_meta = read_json(meta_path)

        # resolve hash — parameter name no longer shadows the imported function
        resolved_hash = source_hash or file_hash(self.project_dir / filename)

        meta = FileMeta(
            filename    = filename,
            file_hash   = resolved_hash,
            chunk_count = len(chunks),
            page_count  = existing_meta.get("page_count", page_count),
            parser      = existing_meta.get("parser", parser),
            created_at  = existing_meta.get("created_at", now()),
            updates     = existing_meta.get("updates", {}),
            updated_at  = now(),
        )

        write_json(meta_path, asdict(meta))

        self._update_component(filename, "chunks", extras={
            "chunk_max_tokens": max_tokens,
            "chunk_model":      chunk_model,
        })

    def load_chunks(self, filename: str) -> list[Chunk]:
        path = self._file_dir(filename) / "chunks.json"
        raw  = read_json(path)

        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(Chunk)}

        processed = []
        for c in raw:
            c.setdefault("clean",    c.get("text", ""))
            c.setdefault("page",     0)
            c.setdefault("page_end", 0)
            c.setdefault("heading",  "")
            filtered = {k: v for k, v in c.items() if k in valid_fields}
            processed.append(Chunk(**filtered))

        return processed

    # ── Markdown ───────────────────────────────────────────────────────────────

    def save_markdown(
        self,
        filename:   str,
        mdDocument: ParsedDocument,
        rerun:      bool = False,
    ) -> None:
        if mdDocument.error:
            TerminalUI.info(
                label="Markdown Error",
                value=f"Skipping {filename}: {mdDocument.error}",
            )
            return

        self._require_init()
        d       = self._ensure_file_dir(filename)
        md_path = d / f"{sanitize(filename)}.md"

        if not rerun:
            if self.is_fresh(filename) and md_path.exists():
                TerminalUI.bold(f"Skipping: Fresh markdown found for {filename}")
                return
        else:
            TerminalUI.bold(f"Rerunning Markdown generation for {filename}...")

        try:
        
            md_path.write_text(mdDocument.embedding_text, encoding="utf-8")

            all_fields = asdict(mdDocument)
            for key in ["text", "error", "filename", "images", "embedding_text", "display_text"]:
                all_fields.pop(key, None)

            self._update_component(filename, "markdown", extras=all_fields)
            TerminalUI.success(f"Markdown saved: {md_path.name}")

        except Exception as e:
            TerminalUI.error(f"Failed to save markdown for {filename}: {e}")

    def load_markdown(self, filename: str) -> str | None:
        if not self.has_markdown(filename):
            TerminalUI.error(f"Markdown not found for {filename}")
            return None

        md_path = self._file_dir(sanitize(filename)) / f"{sanitize(filename)}.md"
        try:
            return md_path.read_text(encoding="utf-8")
        except Exception as e:
            TerminalUI.error(f"Error reading markdown for {filename}: {e}")
            return None

    # ── Embeddings (.npz bundle) ───────────────────────────────────────────────

    def save_embeddings(
        self,
        filename:  str,
        vectors:   np.ndarray,
        chunks:    list[Chunk],
        rerun:     bool = False,
    ) -> None:
        """
        Save embedding vectors + chunk metadata as a single .npz bundle.

        Storing both together guarantees they're always in sync —
        one file exists or neither does. No separate JSON that could
        drift out of step with the vectors.

        Bundle contents:
            vectors   shape (n_chunks, embedding_dim)  float32
            text      shape (n_chunks,)                original text per chunk
            clean     shape (n_chunks,)                stripped text per chunk
            headings  shape (n_chunks,)                nearest heading per chunk
            pages     shape (n_chunks,)                int page numbers
            source    shape (n_chunks,)                filename repeated

        Args:
            filename:  source document filename
            vectors:   numpy array from your embedding model
            chunks:    list of Chunk objects — must be same length as vectors
            rerun:     if True, overwrite existing bundle
        """
        self._require_init()
        self._ensure_file_dir(filename)

        if len(vectors) != len(chunks):
            raise ValueError(
                f"vectors length ({len(vectors)}) != chunks length ({len(chunks)}). "
                f"They must match — one vector per chunk."
            )

        path = self._npz_path(filename)

        if not rerun and self.is_fresh(filename) and path.exists():
            TerminalUI.bold(f"Skipping: Fresh embeddings already exist for {filename}")
            return

        try:
            np.savez_compressed(
                path,
                vectors  = vectors.astype(np.float32),
                text     = np.array([c.text     for c in chunks], dtype=object),
                clean    = np.array([c.clean    for c in chunks], dtype=object),
                headings = np.array([c.heading  for c in chunks], dtype=object),
                pages    = np.array([c.page     for c in chunks], dtype=np.int32),
                source   = np.array([c.source   for c in chunks], dtype=object),
            )
            self._update_component(filename, "embeddings")
            TerminalUI.success(f"Saved embeddings bundle: {path.name}")

        except Exception as e:
            TerminalUI.error(f"Failed to save embeddings: {e}")

    def load_embeddings(self, filename: str) -> dict:
        """
        Load the .npz bundle and return everything as a plain dict.

        Returns:
            {
                "vectors":   np.ndarray  shape (n_chunks, dim)
                "text":      list[str]   original chunk text
                "clean":     list[str]   stripped chunk text
                "headings":  list[str]   nearest heading per chunk
                "pages":     list[int]   page numbers
                "source":    list[str]   filename repeated
            }

        Raises FileNotFoundError if no bundle exists for this file.
        """
        self._require_init()
        path = self._npz_path(filename)

        if not path.exists():
            raise FileNotFoundError(
                f"No embeddings found for '{filename}'. "
                f"Run the embedding pipeline first."

            )

        data = np.load(path, allow_pickle=True)

        return {
            "vectors":  data["vectors"],
            "text":     data["text"].tolist(),
            "clean":    data["clean"].tolist(),
            "headings": data["headings"].tolist(),
            "pages":    data["pages"].tolist(),
            "source":   data["source"].tolist(),
        }

    # ── BM25 ───────────────────────────────────────────────────────────────────

    def save_bm25(self, filename: str, bm25_index, rerun: bool = False) -> None:
        self._require_init()
        if self.is_fresh(filename) and self.has_bm25(filename) and not rerun:
            TerminalUI.bold("Skipping BM25 saving (cached index found and file is fresh)")
            return

        d    = self._ensure_file_dir(filename)
        path = d / "bm25.pkl"
        with open(path, "wb") as f:
            pickle.dump(bm25_index, f)
        self._update_component(filename, "bm25")

    def load_bm25(self, filename: str):
        if not self.has_bm25(filename):
            TerminalUI.error(f"BM25 index not found for {filename}")
            return None

        self._require_init()
        path = self._file_dir(filename) / "bm25.pkl"

        if not path.exists():
            raise FileNotFoundError(f"No BM25 index found for {filename}.")

        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Stopwords ──────────────────────────────────────────────────────────────

    def save_stopwords(self, filename: str, stopwords: set[str]) -> None:
        if self.is_fresh(filename) and self.has_stopwords(filename):
            TerminalUI.bold("Skipping stopwords saving (cached stopwords found and file is fresh)")
            return

        self._require_init()
        d    = self._ensure_file_dir(filename)
        path = d / "stopwords.json"
        write_json(path, {"stopwords": sorted(stopwords)})
        self._update_component(filename, "stopwords")

    def load_stopwords(self, filename: str) -> set[str]:
        if not self.has_stopwords(filename):
            TerminalUI.error(f"Stopwords not found for {filename}")
            return set()

        self._require_init()
        path = self._file_dir(filename) / "stopwords.json"

        if not path.exists():
            return set()

        data = read_json(path)
        return set(data.get("stopwords", []))

    # ── Graph ──────────────────────────────────────────────────────────────────

    def save_graph(
        self,
        filename:   str,
        graph_data: GraphData,
        rerun : bool = False,
    ) -> None:
        if not rerun and self.is_fresh(filename) and self.has_graph(filename):
            TerminalUI.bold("Skipping graph saving (cached graph found)")
            return

        self._require_init()
        d = self._ensure_graph_dir(filename)

        grap_dict = graph_data.to_dict()
        
        g_fname =f"{sanitize(filename)}_graph.json"
        write_json(d / g_fname, grap_dict)


        n_nodes = len(graph_data.nodes)
        n_edges = len(graph_data.edges)
        chunk_count = graph_data.chunk_count
        total_concepts_extracted = graph_data.total_concepts_extracted

        meta = {
            "graph" : {
            "concept_count" : n_nodes,
            "edge_count"    : n_edges,
            "chunk_count"   : chunk_count,
            "total_concepts_extracted" : total_concepts_extracted,
            "built_at"      : now()}}
        

        
        self._update_component(filename, "graph", extras=meta)



   
   
    def save_graph_html(
        self,
        filename:   str,
        html_graph: str,
        rerun : bool = False,
    ) -> None:
        
        if html_graph is None:
            TerminalUI.error(f"Graph not found for {filename}")
            return
        
        if not rerun and self.is_fresh(filename) and self.has_graph_html(filename):
            TerminalUI.bold("Skipping graph saving (cached graph found)")
            return

        self._require_init()

        d = self._ensure_graph_dir(filename)
        g_html_name = d / f"{sanitize(filename)}_graph.html"
        
        with open(g_html_name, "w", encoding="utf-8") as f:
            f.write(html_graph)
        
        self._update_component(filename, "graph_vis")



    def load_graph(self, filename: str) -> dict:
        if not self.has_graph(filename):
            TerminalUI.error(f"Graph not found for {filename}")
            return {}

        self._require_init()
        g_fname =f"{sanitize(filename)}_graph.json"

        path = self._graph_dir(filename) / g_fname

        if not path.exists():
            raise FileNotFoundError(f"No graph found for {filename}.")

        return read_json(path)




    def save_wiki(self, filename: str, wiki, rerun: bool = False) -> None:
        """Save generated wiki to .darwin directory."""
        import json
        
        self._require_init()
        wiki_dir = self._ensure_wiki_dir(filename)
        
        wiki_path = wiki_dir / f"{sanitize(filename)}_wiki.json"
        
        if not rerun and wiki_path.exists():
            TerminalUI.dim("Skipping wiki save (cached wiki found)")
            return
        
        # Convert to serializable
        wiki_dict = {
            "filename": wiki.filename,
            "mode": wiki.mode,
            "concept_count": wiki.concept_count,
            "pages": [
                {
                    "title": p.title,
                    "content": p.content,
                    "concept": p.concept,
                    "chunk_count": p.chunk_count,
                    "related": p.related,
                }
                for p in wiki.pages
            ]
        }

        wiki_update_dict ={
            
            "mode": wiki.mode,
            "concept_count": wiki.concept_count,
            "pages": [
                {
                    "title": p.title,
                    "content": p.content,
                    "concept": p.concept,
                    "chunk_count": p.chunk_count,
                    "related": p.related,
                }
                for p in wiki.pages
            ],
            "page_num" : len(wiki.pages)
        }
        
        write_json(wiki_path, wiki_dict)
        
        self._update_component(filename, "wiki", extras=wiki_update_dict)


    def has_wiki(self, filename: str) -> bool:
        """Check if wiki exists for document."""
        self._require_init()
        wiki_path = self._wiki_dir(filename) / f"{sanitize(filename)}_wiki.json"
        return wiki_path.exists()




    # ── File removal ───────────────────────────────────────────────────────────

    def remove_file(self, filename: str) -> bool:
        self._require_init()
        d = self._file_dir(filename)

        if not d.exists():
            return False

        shutil.rmtree(d)
        return True

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        self._require_init()

        result = {
            "project_dir": str(self.project_dir),
            "initialized": self.is_initialized(),
            "files":       {},
        }

        source_files = [
            f for f in self.project_dir.iterdir()
            if f.is_file() and f.name != DARWIN_DIR
        ]

        for file_path in source_files:
            filename = file_path.name
            meta     = self.load_meta_file(filename) or {}

            components = {
                "markdown":   self.has_markdown(filename),
                "chunks":     self.has_chunks(filename),
                "embeddings": self.has_embeddings(filename),
                "bm25":       self.has_bm25(filename),
                "stopwords":  self.has_stopwords(filename),
                "graph":      self.has_graph(filename),
            }

            pipeline = self.get_component_status(filename)

            result["files"][filename] = {
                "exists":     file_path.exists(),
                "fresh":      self.is_fresh(filename),
                "components": components,
                "pipeline":   pipeline,
                "meta": {
                    "chunk_count":      meta.get("chunk_count"),
                    "page_count":       meta.get("page_count"),
                    "parser":           meta.get("parser"),
                    "chunk_max_tokens": meta.get("chunk_max_tokens"),
                    "chunk_model":      meta.get("chunk_model"),
                    "updated_at":       meta.get("updated_at"),
                    "created_at":       meta.get("created_at"),
                },
            }

        return result

    def file_status(self, filename: str) -> dict:
        self._require_init()

        file_path  = self.project_dir / filename
        meta       = self.load_meta_file(filename) or {}

        components = {
            "markdown":   self.has_markdown(filename),
            "chunks":     self.has_chunks(filename),
            "embeddings": self.has_embeddings(filename),
            "bm25":       self.has_bm25(filename),
            "stopwords":  self.has_stopwords(filename),
            "graph":      self.has_graph(filename),
        }

        pipeline = self.get_component_status(filename)

        return {
            "filename":  filename,
            "exists":    file_path.exists(),
            "fresh":     self.is_fresh(filename),
            "components": components,
            "pipeline":   pipeline,
            "meta": {
                "chunk_count":      meta.get("chunk_count"),
                "page_count":       meta.get("page_count"),
                "parser":           meta.get("parser"),
                "chunk_max_tokens": meta.get("chunk_max_tokens"),
                "chunk_model":      meta.get("chunk_model"),
                "updated_at":       meta.get("updated_at"),
                "created_at":       meta.get("created_at"),
            },
            "ready_for_search": components["chunks"] and components["embeddings"],
            "fully_processed":  all(components.values()),
        }