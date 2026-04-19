"""
src/core/Storage/StoreArtifacts.py
-----------------------------------
Save / load operations for every artifact type Darwin persists.

Grouped into one mixin per artifact family so each section is easy to
find and test in isolation.  DarwinStore inherits all of them.

Mixin classes here must NOT define __init__ — they rely on the attributes
set by DarwinStore.__init__:
    self.paths   : StorePaths
    self.project_dir : Path

All public methods follow one of two signatures:
    save_X(filename, data, rerun=False) -> None
    load_X(filename) -> data | raises FileNotFoundError
"""

import dataclasses
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.core.Chunking.ChunkerUtils import Chunk
from src.core.Parsing.ParserUtils import ParsedDocument
from src.core.Storage.StoreUtils import file_hash, now, read_json, write_json
from src.lib.Terminal import TerminalUI


# ── Internal shared helpers ────────────────────────────────────────────────────

def _chunk_from_dict(raw: dict) -> Chunk:
    """Safely reconstruct a Chunk, back-filling fields added in later versions."""
    valid_fields = {f.name for f in dataclasses.fields(Chunk)}
    raw.setdefault("clean",    raw.get("text", ""))
    raw.setdefault("page",     0)
    raw.setdefault("page_end", 0)
    raw.setdefault("heading",  "")
    raw.setdefault("links",    [])
    return Chunk(**{k: v for k, v in raw.items() if k in valid_fields})


# ── Mixin: Meta / freshness ────────────────────────────────────────────────────

class MetaMixin:
    """Handles meta.json read/write and file freshness checks."""

    def _update_meta(self, filename: str, component: str, extras: dict | None = None) -> None:
        """
        Record that `component` was just processed for `filename`.
        All pipeline stages call this so meta.json is an audit trail.
        """
        meta_path = self.paths.meta_path(filename)
        meta      = read_json(meta_path) or {
            "filename":   filename,
            "created_at": now(),
            "updates":    {},
        }

        meta.setdefault("updates", {})
        timestamp              = now()
        meta["updates"][component] = timestamp
        meta["updated_at"]     = timestamp

        if extras:
            for key, value in extras.items():
                if isinstance(value, dict) and isinstance(meta.get(key), dict):
                    meta[key].update(value)
                else:
                    meta[key] = value

        write_json(meta_path, meta)

    def load_meta(self, filename: str) -> dict:
        return read_json(self.paths.meta_path(filename)) or {}

    def is_fresh(self, filename: str) -> bool:
        """
        True when the source file's MD5 hash matches the stored hash.
        A stale result means the file changed since last processing.
        """
        meta_path = self.paths.meta_path(filename)
        if not meta_path.exists():
            return False

        source = self.project_dir / filename
        if not source.exists():
            return False

        return read_json(meta_path).get("file_hash", "") == file_hash(source)


# ── Mixin: Config ──────────────────────────────────────────────────────────────

class ConfigMixin:
    """Handles config.json (project-level settings)."""

    def load_config(self) -> dict:
        return read_json(self.paths.config_path) or {}

    def save_config(self, config: dict) -> None:
        write_json(self.paths.config_path, config)

    def update_config(self, **kwargs) -> None:
        config = self.load_config()
        config.update(kwargs)
        self.save_config(config)

    def _embedding_model(self) -> str:
        return self.load_config().get("embedding_model", "intfloat/e5-small-v2")


# ── Mixin: Markdown ────────────────────────────────────────────────────────────

class MarkdownMixin:

    def has_markdown(self, filename: str) -> bool:
        return self.paths.markdown_path(filename).exists()

    def save_markdown(self, filename: str, doc: ParsedDocument, rerun: bool = False) -> None:
        if doc.error:
            TerminalUI.info("Markdown Error", f"Skipping {filename}: {doc.error}")
            return

        self.paths.ensure_file_dir(filename)
        md_path = self.paths.markdown_path(filename)

        if not rerun and self.is_fresh(filename) and md_path.exists():
            TerminalUI.dim(f"Skipping markdown (fresh): {filename}")
            return

        try:
            md_path.write_text(doc.embedding_text, encoding="utf-8")



            skip_fields = {"text", "error", "filename", "images", "embedding_text", "display_text"}
            extras      = {k: v for k, v in dataclasses.asdict(doc).items() if k not in skip_fields}
            self._update_meta(filename, "markdown", extras=extras)

            TerminalUI.success(f"Markdown saved: {md_path.name}")
        except Exception as e:
            TerminalUI.error(f"Failed to save markdown for {filename}: {e}")

    def load_markdown(self, filename: str) -> str | None:
        md_path = self.paths.markdown_path(filename)
        if not md_path.exists():
            TerminalUI.error(f"Markdown not found for {filename}")
            return None
        try:
            return md_path.read_text(encoding="utf-8")
        except Exception as e:
            TerminalUI.error(f"Error reading markdown for {filename}: {e}")
            return None


# ── Mixin: Links ───────────────────────────────────────────────────────────────

class LinksMixin:
    """Hyperlinks extracted during parsing — saved alongside markdown."""

    def has_links(self, filename: str) -> bool:
        return self.paths.links_path(filename).exists()

    def load_links(self, filename: str) -> list[dict]:
        """
        Returns raw link dicts (url, page, text, rect).
        Pass directly to assign_links_to_chunks() in the chunker.
        Returns [] if no links file exists (non-PDF sources, old parsed files).
        """
        path = self.paths.links_path(filename)
        if not path.exists():
            return []
        return read_json(path) or []


# ── Mixin: Chunks ──────────────────────────────────────────────────────────────

class ChunksMixin:

    def has_chunks(self, filename: str) -> bool:
        return self.paths.chunks_path(filename).exists()

    def save_chunks(self, filename: str, chunks: list[Chunk], chunk_meta: dict | None = None, rerun: bool = False) -> None:
        if not rerun and self.is_fresh(filename) and self.has_chunks(filename):
            TerminalUI.dim(f"Skipping chunks (fresh): {filename}")
            return

        self.paths.ensure_file_dir(filename)
        write_json(self.paths.chunks_path(filename), [dataclasses.asdict(c) for c in chunks])

        extras = {**(chunk_meta or {}), "chunk_count": len(chunks)}
        self._update_meta(filename, "chunks", extras=extras)

    def load_chunks(self, filename: str) -> list[Chunk]:
        path = self.paths.chunks_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"No chunks found for '{filename}'.")
        return [_chunk_from_dict(c) for c in (read_json(path) or [])]


# ── Mixin: Embeddings ──────────────────────────────────────────────────────────

class EmbeddingsMixin:

    def has_embeddings(self, filename: str) -> bool:
        return self.paths.npz_path(filename, self._embedding_model()).exists()

    def save_embeddings(
        self,
        filename: str,
        vectors:  np.ndarray,
        chunks:   list[Chunk],
        rerun:    bool = False,
    ) -> None:
        """
        Save vectors + chunk metadata as a single .npz bundle.

        Storing both together guarantees permanent sync — if the file
        exists, both vectors and metadata exist.

        Bundle arrays (all length n_chunks):
            vectors   float32  (n_chunks, embedding_dim)
            text      object   original chunk text
            clean     object   stripped text for embedder
            headings  object   nearest heading
            pages     int32    page numbers
            source    object   filename repeated
        """
       
        if not rerun and self.is_fresh(filename) and path.exists():
            TerminalUI.dim(f"Skipping embeddings (fresh): {filename}")
            return


        if len(vectors) != len(chunks):
            raise ValueError(
                f"vectors ({len(vectors)}) and chunks ({len(chunks)}) must be the same length."
            )

        path = self.paths.npz_path(filename, self._embedding_model())

       

        self.paths.ensure_file_dir(filename)

        try:
            np.savez_compressed(
                path,
                vectors  = vectors.astype(np.float32),
                text     = np.array([c.text    for c in chunks], dtype=object),
                clean    = np.array([c.clean   for c in chunks], dtype=object),
                headings = np.array([c.heading for c in chunks], dtype=object),
                pages    = np.array([c.page    for c in chunks], dtype=np.int32),
                source   = np.array([c.source  for c in chunks], dtype=object),
            )
            self._update_meta(filename, "embeddings")
            TerminalUI.success(f"Saved embeddings: {path.name}")
        except Exception as e:
            TerminalUI.error(f"Failed to save embeddings for {filename}: {e}")

    def load_embeddings(self, filename: str) -> dict:
        """
        Returns:
            {
                "vectors":  np.ndarray  shape (n_chunks, dim)
                "text":     list[str]
                "clean":    list[str]
                "headings": list[str]
                "pages":    list[int]
                "source":   list[str]
            }
        """
        path = self.paths.npz_path(filename, self._embedding_model())
        if not path.exists():
            raise FileNotFoundError(f"No embeddings found for '{filename}'. Run the embedding pipeline first.")

        data = np.load(path, allow_pickle=True)
        return {
            "vectors":  data["vectors"],
            "text":     data["text"].tolist(),
            "clean":    data["clean"].tolist(),
            "headings": data["headings"].tolist(),
            "pages":    data["pages"].tolist(),
            "source":   data["source"].tolist(),
        }


# ── Mixin: BM25 ────────────────────────────────────────────────────────────────

class BM25Mixin:

    def has_bm25(self, filename: str) -> bool:
        return self.paths.bm25_path(filename).exists()

    def save_bm25(self, filename: str, bm25_index: Any, rerun: bool = False) -> None:
        if not rerun and self.is_fresh(filename) and self.has_bm25(filename):
            TerminalUI.dim(f"Skipping BM25 (fresh): {filename}")
            return

        self.paths.ensure_file_dir(filename)
        with open(self.paths.bm25_path(filename), "wb") as f:
            pickle.dump(bm25_index, f)
        self._update_meta(filename, "bm25")

    def load_bm25(self, filename: str) -> Any:
        path = self.paths.bm25_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"No BM25 index found for '{filename}'.")
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Mixin: Stopwords ───────────────────────────────────────────────────────────

class StopwordsMixin:

    def has_stopwords(self, filename: str) -> bool:
        return self.paths.stopwords_path(filename).exists()

    def save_stopwords(self, filename: str, stopwords: set[str], rerun: bool = False) -> None:
        if not rerun and self.is_fresh(filename) and self.has_stopwords(filename):
            TerminalUI.dim(f"Skipping stopwords (fresh): {filename}")
            return

        self.paths.ensure_file_dir(filename)
        write_json(self.paths.stopwords_path(filename), {"stopwords": sorted(stopwords)})
        self._update_meta(filename, "stopwords")

    def load_stopwords(self, filename: str) -> set[str]:
        path = self.paths.stopwords_path(filename)
        if not path.exists():
            return set()
        return set(read_json(path).get("stopwords", []))


# ── Mixin: Graph ───────────────────────────────────────────────────────────────

class GraphMixin:

    def has_graph(self, filename: str) -> bool:
        return self.paths.graph_json_path(filename).exists()

    def has_graph_html(self, filename: str) -> bool:
        return self.paths.graph_html_path(filename).exists()

    def save_graph(self, filename: str, graph_data: Any, rerun: bool = False) -> None:
        if not rerun and self.is_fresh(filename) and self.has_graph(filename):
            TerminalUI.dim(f"Skipping graph (fresh): {filename}")
            return

        self.paths.ensure_graph_dir(filename)
        write_json(self.paths.graph_json_path(filename), graph_data.to_dict())

        self._update_meta(filename, "graph", extras={
            "graph": {
                "concept_count":            len(graph_data.nodes),
                "edge_count":               len(graph_data.edges),
                "chunk_count":              graph_data.chunk_count,
                "total_concepts_extracted": graph_data.total_concepts_extracted,
                "built_at":                 now(),
            }
        })

    def save_graph_html(self, filename: str, html: str, rerun: bool = False) -> None:
        if html is None:
            TerminalUI.error(f"No graph HTML to save for {filename}")
            return

        if not rerun and self.is_fresh(filename) and self.has_graph_html(filename):
            TerminalUI.dim(f"Skipping graph HTML (fresh): {filename}")
            return

        self.paths.ensure_graph_dir(filename)
        self.paths.graph_html_path(filename).write_text(html, encoding="utf-8")
        self._update_meta(filename, "graph_vis")

    def load_graph(self, filename: str) -> dict:
        path = self.paths.graph_json_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"No graph found for '{filename}'.")
        return read_json(path) or {}


# ── Mixin: Wiki ────────────────────────────────────────────────────────────────

class WikiMixin:

    def has_wiki(self, filename: str) -> bool:
        return self.paths.wiki_json_path(filename).exists()

    def save_wiki(self, filename: str, wiki: Any, rerun: bool = False) -> None:
        path = self.paths.wiki_json_path(filename)
        if not rerun and path.exists():
            TerminalUI.dim(f"Skipping wiki (cached): {filename}")
            return

        self.paths.ensure_wiki_dir(filename)

        wiki_dict = {
            "filename":      wiki.filename,
            "mode":          wiki.mode,
            "concept_count": wiki.concept_count,
            "pages": [
                {
                    "title":       p.title,
                    "content":     p.content,
                    "concept":     p.concept,
                    "chunk_count": p.chunk_count,
                    "related":     p.related,
                }
                for p in wiki.pages
            ],
        }
        write_json(path, wiki_dict)

        self._update_meta(filename, "wiki", extras={
            **{k: wiki_dict[k] for k in ("mode", "concept_count", "pages")},
            "page_num": len(wiki.pages),
        })

    def load_wiki(self, filename: str) -> dict:
        path = self.paths.wiki_json_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"No wiki found for '{filename}'.")
        return read_json(path) or {}