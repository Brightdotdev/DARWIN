
"""
src/core/Storage/StoreArtifacts.py
-----------------------------------
Save / load operations for every artifact type Darwin persists.

All mixins inherit _PathResolverMixin which resolves a bare filename
string → file hash → index directory via the FileRegistry. This is the
bridge between the calling code (which only knows filenames) and the
content-addressed StorePaths (which only knows hashes).

Mixin classes here must NOT define __init__ — they rely on the attributes
set by DarwinStore.__init__:
    self.paths    : StorePaths
    self.registry : FileRegistry
"""

import dataclasses
from typing import Any, Optional
import numpy as np

from src.core.Chunking.ChunkerUtils import Chunk
from src.core.Parsing.ParserUtils import ParsedDocument
from src.core.Storage.StoreUtils import file_hash, now, read_json, write_json
from src.lib.Terminal import TerminalUI
from src.core.BM25.BM25Utils import BM25Data


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _chunk_from_dict(raw: dict) -> Chunk:
    """Safely reconstruct a Chunk, back-filling fields added in later versions."""
    valid_fields = {f.name for f in dataclasses.fields(Chunk)}
    raw.setdefault("clean",    raw.get("text", ""))
    raw.setdefault("page",     0)
    raw.setdefault("page_end", 0)
    raw.setdefault("heading",  "")
    raw.setdefault("links",    [])
    return Chunk(**{k: v for k, v in raw.items() if k in valid_fields})


def _normalize_model_name(name: str) -> str:
    """
    Convert any model identifier to the canonical model_id.
    Handles both registry keys and model_ids.
    """
    from src.core.Models.ModelRegistry import REGISTRY
    
    # If it's a registry key, convert to model_id
    if name in REGISTRY:
        return REGISTRY[name].model_id
    
    # If it's already a model_id, check if any spec uses it
    id_to_key = {spec.model_id: key for key, spec in REGISTRY.items()}
    if name in id_to_key:
        return name
    
    # Unknown - return as-is
    return name

def _get_current_model_name() -> str:
    """Resolve the currently active embedding model name (full model_id)."""
    try:
        from src.core.Models import get_loader, ModelCategory
        loader = get_loader()
        return loader.get_model_key(ModelCategory.EMBEDDING)
    except Exception as e:
        TerminalUI.error(f"Failed to detect current model: {e}")
        return "unknown"
    

def _get_current_model_dimensions() -> int:
    """Expected embedding dimensions for the active model. Safe fallback."""
    try:
        from src.core.Models import get_loader, ModelCategory
        loader = get_loader()
        key = loader.get_model_key(ModelCategory.EMBEDDING)
        dims = {
            "intfloat/e5-small-v2": 384,
            "BAAI/bge-m3":      1024,
            "GritLM/GritLM-7B":   4096,
        }
        return dims.get(key, 0)
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# Base mixin — filename → hash → directory resolution
# ═══════════════════════════════════════════════════════════════════════════

class _PathResolverMixin:
    """
    Bridge between calling code (bare filenames) and StorePaths (file hashes).

    All artifact mixins inherit this. The registry is the source of truth —
    a file must be registered before any artifact can be saved for it.
    """

    def _record_for(self, filename: str):
        """Return the FileRecord for a bare filename. Raises KeyError if missing."""
        record = self.registry._find_by_name(filename)
        if record is None:
            TerminalUI.error(f"File '{filename}' is not registered. Register it before saving artifacts.")
            return None
        return record

    def _hash_for(self, filename: str) -> str:
        """Resolve a bare filename to its file hash."""
        record = self._record_for(filename)
        return record.file_hash if record else ""

    def _dir_for(self, filename: str):
        """Return (and create) the index directory for a bare filename."""
        return self.paths.dir_for_hash(self._hash_for(filename))

    def _source_path_for(self, filename: str):
        """Return the last-known source Path for a file (may not exist if moved)."""
        from pathlib import Path
        record = self._record_for(filename)
        if record.last_seen_path:
            return Path(record.last_seen_path)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Meta / freshness
# ═══════════════════════════════════════════════════════════════════════════

class MetaMixin(_PathResolverMixin):
    """Handles meta.json read/write and file freshness checks."""

    def _update_meta(self, filename: str, component: str, extras: dict | None = None) -> None:
        """
        Record that `component` was just processed for `filename`.
        All pipeline stages call this so meta.json is an audit trail.
        """
        meta_path = self._dir_for(filename) / "meta.json"
        meta = read_json(meta_path) or {
            "filename":   filename,
            "created_at": now(),
            "updates":    {},
        }

        meta.setdefault("updates", {})
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

    def load_meta(self, filename: str) -> dict:
        return read_json(self._dir_for(filename) / "meta.json") or {}

    def is_fresh(self, filename: str) -> bool:
        """
        True when the source file's MD5 hash matches the stored hash.
        A stale result means the file changed since last processing.
        """
        meta_path = self._dir_for(filename) / "meta.json"
        if not meta_path.exists():
            return False

        source = self._source_path_for(filename)
        if source is None or not source.exists():
            return False

        return read_json(meta_path).get("file_hash", "") == file_hash(source)


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Config
# ═══════════════════════════════════════════════════════════════════════════

class ConfigMixin:
    """
    Handles config.json (global user preferences).

    This stores what the user WANTS — not what actually happened per file.
    Per-file state lives in meta.json, embeddings.meta.json, and vector.json.
    """

    def load_config(self) -> dict:
        return read_json(self.paths.config_path) or {}

    def save_config(self, config: dict) -> None:
        write_json(self.paths.config_path, config)

    def update_config(self, **kwargs) -> None:
        config = self.load_config()
        config.update(kwargs)
        self.save_config(config)

    def get_user_embedding_model(self) -> str:
        """The model the user selected during onboarding/settings."""
        return self.load_config().get("embedding_model", "unknown")

    def set_user_embedding_model(self, model_name: str) -> None:
        """Called when the user changes their embedding model preference."""
        self.update_config(embedding_model=model_name)
        TerminalUI.info(
            label="Embedding model updated",
            value=f"New files will use {model_name}. Existing files may need re-embedding."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Markdown
# ═══════════════════════════════════════════════════════════════════════════

class MarkdownMixin(_PathResolverMixin):

    def has_markdown(self, filename: str) -> bool:
        return (self._dir_for(filename) / "markdown.md").exists()

    def save_markdown(self, filename: str, doc: ParsedDocument) -> None:
        
        if doc.error:
            TerminalUI.info("Markdown Error", f"Skipping {filename}: {doc.error}")
            return
        

        d = self._dir_for(filename)
        md_path = d / "markdown.md"


        try:
            md_path.write_text(doc.embedding_text, encoding="utf-8")

            skip_fields = {"text", "error", "filename", "images", "embedding_text", "display_text"}
            extras = {k: v for k, v in dataclasses.asdict(doc).items() if k not in skip_fields}
            
            self._update_meta(filename, "markdown", extras=extras)
           
            record = self._record_for(filename)
            if record:
                self.registry.update_pipeline(
                    record.last_seen_path or filename,
                    "markdown",
                    **extras
                )

            TerminalUI.success(f"Markdown saved: {md_path.name}")
        except Exception as e:
            TerminalUI.error(f"Failed to save markdown for {filename}: {e}")

    def load_markdown(self, filename: str) -> str | None:
        md_path = self._dir_for(filename) / "markdown.md"
        if not md_path.exists():
            TerminalUI.error(f"Markdown not found for {filename}")
            return None
        try:
            return md_path.read_text(encoding="utf-8")
        except Exception as e:
            TerminalUI.error(f"Error reading markdown for {filename}: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Chunks
# ═══════════════════════════════════════════════════════════════════════════

class ChunksMixin(_PathResolverMixin):

    def has_chunks(self, filename: str) -> bool:
        return (self._dir_for(filename) / "chunks.json").exists()

    def save_chunks(
        self,
        filename: str,
        chunks: list[Chunk],
        chunk_meta: dict | None = None,
    ) -> None:
        if not chunks:
            TerminalUI.error(f"No chunks to save for {filename}")
            return
        
        d = self._dir_for(filename)
        write_json(d / "chunks.json", [dataclasses.asdict(c) for c in chunks])

        extras = {**(chunk_meta or {}), "chunk_count": len(chunks)}
        self._update_meta(filename, "chunks", extras=extras)
        record = self._record_for(filename)
        if record:
                self.registry.update_pipeline(
                    record.last_seen_path or filename,
                    "chunks",
                    **extras
                )


    def load_chunks(self, filename: str) -> list[Chunk]:
        path = self._dir_for(filename) / "chunks.json"
        if not path.exists():
            return []
        return [_chunk_from_dict(c) for c in (read_json(path) or [])]


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Vector (USearch)
# ═══════════════════════════════════════════════════════════════════════════

class VectorMixin(_PathResolverMixin):

    def has_vector(self, filename: str) -> bool:
        return (self._dir_for(filename) / "vector.usearch").exists()

    def save_vector(
        self,
        filename: str,
        model_name: str,
        vectors: np.ndarray,
        chunk_ids: list[str],
    ) -> None:
        """Build and save USearch index with per-file model metadata."""
        
        if len(vectors) != len(chunk_ids):
            raise ValueError(
                f"vectors ({len(vectors)}) and chunk_ids ({len(chunk_ids)}) must be the same length."
            )

        d = self._dir_for(filename)

        from usearch.index import Index
        model_name = _normalize_model_name(model_name)
        dimension = vectors.shape[1]
        index = Index(ndim=dimension, metric="cos", dtype="f32")
        index.add(
            keys=np.arange(len(vectors), dtype=np.int64),
            vectors=vectors.astype(np.float32),
        )

        index.save(str(d / "vector.usearch"))

        meta_data = {
            "dimension":     dimension,
            "chunk_ids":     chunk_ids,
            "total_vectors": len(vectors),
            "embedding_model":         model_name,
            "created_at":    now(),
        }
        write_json(d / "vector.json", meta_data)

        self._update_meta(filename, "vector", extras={
            "vector_model":     model_name,
            "vector_dimension": dimension,
        })

        record = self._record_for(filename)
        if record:
                self.registry.update_pipeline(
                    record.last_seen_path or filename,
                    "vector")

        TerminalUI.success(f"Vector index saved: {len(vectors)} vectors")

    def load_vector(self, filename: str):
        """Load USearch index and return (index, chunk_ids)."""
        d = self._dir_for(filename)
        index_path = d / "vector.usearch"
        meta_path  = d / "vector.json"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No vector index found for '{filename}'")

        meta = read_json(meta_path)
        stored_model = meta.get("embedding_model", "unknown")
        current_model = _get_current_model_name()
        
        TerminalUI.info("Model Comparison", 
            f"Stored: '{stored_model}' | Current: '{current_model}' | "
            f"Match: {stored_model == current_model}"
        )

        if stored_model != current_model and stored_model != "unknown":
            TerminalUI.warn(
                f"⚠️  Vector index for '{filename}' was built with '{stored_model}' "
                f"but your current embedding model is '{current_model}'.\n"
                f"    Search results will be wrong. Run 'darwin reindex' to rebuild."
            )
            return None, None
        
        expected_dim = _get_current_model_dimensions()
        if expected_dim and meta.get("dimension") != expected_dim:
            TerminalUI.warn(
                f"⚠️  Vector dimension mismatch: index has {meta['dimension']}d "
                f"but current model expects {expected_dim}d. Re-embedding required."
            )
            return None, None

        from usearch.index import Index
        index = Index.restore(str(index_path))
        return index, meta["chunk_ids"]

    def search_vector(
        self,
        filename: str,
        query_vec: np.ndarray,
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search vector index and return (chunk_id, score) pairs."""
        index, chunk_ids = self.load_vector(filename)

        if index is None or chunk_ids is None:
            return []

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        matches = index.search(query_vec.astype(np.float32), k)

        results: list[tuple[str, float]] = []
        if matches is None:
            return results

        if hasattr(matches, "keys") and hasattr(matches, "distances"):
            for key, dist in zip(matches.keys, matches.distances):
                idx = int(key)
                score = 1.0 - float(dist)
                if 0 <= idx < len(chunk_ids):
                    results.append((chunk_ids[idx], score))
        else:
            for match in matches:
                idx = int(match.key)
                score = 1.0 - float(match.distance)
                if 0 <= idx < len(chunk_ids):
                    results.append((chunk_ids[idx], score))

        return results

    def load_vector_metadata(self, filename: str) -> Optional[dict]:
        """Return per-file vector metadata (model, dimension, chunk_ids)."""
        path = self._dir_for(filename) / "vector.json"
        if not path.exists():
            return None
        return read_json(path)

    def is_vector_compatible(self, filename: str) -> bool:
        """Quick check: can we search this file's index with the current model?"""
        try:
            index, chunk_ids = self.load_vector(filename)
            return index is not None and chunk_ids is not None
        except FileNotFoundError:
            return False


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Embeddings (NPZ bundles)
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingsMixin(_PathResolverMixin):

    def has_embeddings(self, filename: str) -> bool:
        current_model = _get_current_model_name()
        fh = self._hash_for(filename)
        return self.paths.npz_path(fh, current_model).exists()

    def save_embeddings(
        self,
        filename: str,
        vectors: np.ndarray,
        chunks: list[Chunk],
        model_name: str = None,
    ) -> None:
        """
        Save vectors + chunk metadata as a single .npz bundle.
        Per-file model metadata is saved alongside for mismatch detection.
        """
        if chunks is None or len(chunks) == 0:
            TerminalUI.error(f"No chunks provided for embedding save of {filename}")
            return

        if model_name is None:
            model_name = _get_current_model_name()
        
        model_name = _normalize_model_name(model_name)

        fh   = self._hash_for(filename)
        path = self.paths.npz_path(fh, model_name)

        if len(vectors) != len(chunks):
            raise ValueError(
                f"vectors ({len(vectors)}) and chunks ({len(chunks)}) must be the same length."
            )

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

            meta_path = path.with_suffix(".meta.json")
            write_json(meta_path, {
                "embedding_model":  model_name,
                "dimensions":  vectors.shape[1],
                "chunk_count": len(chunks),
                "created_at":  now(),
            })

            self._update_meta(filename, "embeddings", extras={
                "embedding_model":      model_name,
                "embedding_dimensions": vectors.shape[1],
            })
            record = self._record_for(filename)
            if record:
                self.registry.update_pipeline(
                record.last_seen_path or filename,
                "embeddings")
                
            TerminalUI.success(f"Saved embeddings: {path.name}")

        except Exception as e:
            TerminalUI.error(f"Failed to save embeddings for {filename}: {e}")

    def load_embeddings(self, filename: str) -> dict:
        """
        Load the NPZ bundle. Warns if the stored model differs from the
        user's current preference.
        """
        current_model = _get_current_model_name()
        fh   = self._hash_for(filename)
        path = self.paths.npz_path(fh, current_model)

        if not path.exists():
            meta_path = self._dir_for(filename) / "meta.json"
            if meta_path.exists():
                file_meta    = read_json(meta_path)
                stored_model = file_meta.get("embedding_model", "unknown")
                if stored_model != "unknown" and stored_model != current_model:
                    raise FileNotFoundError(
                        f"Embeddings for '{filename}' were built with '{stored_model}', "
                        f"but your current model is '{current_model}'. "
                        f"Run the embedding pipeline to re-index."
                    )
            raise FileNotFoundError(
                f"No embeddings found for '{filename}'. Run the embedding pipeline first."
            )

        meta_path = path.with_suffix(".meta.json")
        
        if meta_path.exists():
            stored_meta  = read_json(meta_path)
            stored_model = stored_meta.get("embedding_model", "unknown")

            if stored_model != current_model and stored_model != "unknown":
                TerminalUI.warn(
                    f"⚠️  Embeddings were created with '{stored_model}' "
                    f"but current model is '{current_model}'. "
                    f"Re-embedding recommended."
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

    def get_embedding_metadata(self, filename: str) -> Optional[dict]:
        current_model = _get_current_model_name()
        fh        = self._hash_for(filename)
        path      = self.paths.npz_path(fh, current_model)
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            return None
        return read_json(meta_path)

    def is_embedding_compatible(self, filename: str) -> bool:
        try:
            self.load_embeddings(filename)
            return True
        except FileNotFoundError:
            return False


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: BM25
# ═══════════════════════════════════════════════════════════════════════════

class BM25Mixin(_PathResolverMixin):

    def has_bm25(self, filename: str) -> bool:
        return (self._dir_for(filename) / "bm25.npz").exists()

    def save_bm25(self, filename: str, data: BM25Data) -> None:
        """
        Persist BM25Data as two files:
            bm25.npz        — tokens (object array) + doc_lengths (int32)
            bm25_meta.json  — everything else
        """
  
        if not data.tokens or not data.doc_lengths:
            TerminalUI.error(f"No BM25 data to save for {filename}")
            return
        d         = self._dir_for(filename)
        npz_path  = d / "bm25.npz"
        meta_path = d / "bm25_meta.json"

        try:
            tokens_arr = np.empty(len(data.tokens), dtype=object)
            for i, tok_list in enumerate(data.tokens):
                tokens_arr[i] = tok_list

            np.savez_compressed(
                npz_path,
                tokens      = tokens_arr,
                doc_lengths = np.array(data.doc_lengths, dtype=np.int32),
            )

            write_json(meta_path, {
                "avgdl":            data.avgdl,
                "chunk_count":      data.chunk_count,
                "vocab_size":       data.vocab_size,
                "threshold":        data.threshold,
                "idf_smoothing":    data.idf_smoothing,
                "domain_stopwords": data.domain_stopwords,
                "doc_freqs":        data.doc_freqs,
                "precomputed_idfs": data.precomputed_idfs or {},
            })

            self._update_meta(filename, "bm25")
            record = self._record_for(filename)
            if record:
                    self.registry.update_pipeline(
                        record.last_seen_path or filename,
                        "bm25")
            TerminalUI.success(f"Saved BM25 index: {npz_path.name}")

        except Exception as e:
            TerminalUI.error(f"Failed to save BM25 for {filename}: {e}")

    def load_bm25(self, filename: str) -> BM25Data | None:
        d         = self._dir_for(filename)
        npz_path  = d / "bm25.npz"
        meta_path = d / "bm25_meta.json"

        if not npz_path.exists() or not meta_path.exists():
            return None

        raw  = np.load(npz_path, allow_pickle=True)
        meta = read_json(meta_path)

        tokens = [list(tok_list) for tok_list in raw["tokens"]]

        return BM25Data(
            tokens           = tokens,
            doc_freqs        = meta["doc_freqs"],
            avgdl            = meta["avgdl"],
            doc_lengths      = raw["doc_lengths"].tolist(),
            domain_stopwords = meta["domain_stopwords"],
            threshold        = meta["threshold"],
            chunk_count      = meta["chunk_count"],
            vocab_size       = meta["vocab_size"],
            idf_smoothing    = meta["idf_smoothing"],
            precomputed_idfs = meta.get("precomputed_idfs") or None,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Stopwords
# ═══════════════════════════════════════════════════════════════════════════

class StopwordsMixin(_PathResolverMixin):

    def has_stopwords(self, filename: str) -> bool:
        return (self._dir_for(filename) / "stopwords.json").exists()

    def save_stopwords(self, filename: str, stopwords: set[str]) -> None:

        if not stopwords:
            TerminalUI.error(f"No stopwords to save for {filename}")
            return
        d = self._dir_for(filename)
        write_json(d / "stopwords.json", {"stopwords": sorted(stopwords)})
        self._update_meta(filename, "stopwords")
        record = self._record_for(filename)
        if record:
                    self.registry.update_pipeline(
                        record.last_seen_path or filename,
                        "stopwords")


    def load_stopwords(self, filename: str) -> set[str]:
        path = self._dir_for(filename) / "stopwords.json"
        if not path.exists():
            return set()
        return set(read_json(path).get("stopwords", []))


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Graph
# ═══════════════════════════════════════════════════════════════════════════

class GraphMixin(_PathResolverMixin):

    def has_graph(self, filename: str) -> bool:
        return (self._dir_for(filename) / "graph" / "graph.json").exists()

    def has_graph_html(self, filename: str) -> bool:
        return (self._dir_for(filename) / "graph" / "graph.html").exists()

    def save_graph(self, filename: str, graph_data: Any) -> None:
        if not graph_data:
            TerminalUI.error(f"No graph data to save for {filename}")
            return

        d = self._dir_for(filename) / "graph"
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "graph.json", graph_data.to_dict())

        self._update_meta(filename, "graph", extras={
            "graph": {
                "concept_count":            len(graph_data.nodes),
                "edge_count":               len(graph_data.edges),
                "chunk_count":              graph_data.chunk_count,
                "total_concepts_extracted": graph_data.total_concepts_extracted,
                "built_at":                 now(),
            }
        })
        record = self._record_for(filename)
        if record:
            self.registry.update_pipeline(
            record.last_seen_path or filename,
            "graph",
            concept_count=len(graph_data.nodes),
            edge_count=len(graph_data.edges),
        )
                            
                    

    def save_graph_html(self, filename: str, html: str, ) -> None:
        if html is None:
            TerminalUI.error(f"No graph HTML to save for {filename}")
            return



        d = self._dir_for(filename) / "graph"
        d.mkdir(parents=True, exist_ok=True)
        (d / "graph.html").write_text(html, encoding="utf-8")
        self._update_meta(filename, "graph_vis")
        record = self._record_for(filename)
        if record:
                    self.registry.update_pipeline(
                        record.last_seen_path or filename,
                        "graph_vis")

        

    def load_graph(self, filename: str) -> dict:
        path = self._dir_for(filename) / "graph" / "graph.json"
        if not path.exists():
            raise FileNotFoundError(f"No graph found for '{filename}'.")
        return read_json(path) or {}


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Wiki
# ═══════════════════════════════════════════════════════════════════════════

class WikiMixin(_PathResolverMixin):

    def has_wiki(self, filename: str) -> bool:
        return (self._dir_for(filename) / "wiki" / "wiki.json").exists()

    def save_wiki(self, filename: str, wiki: Any,) -> None:
        if wiki is None:
            TerminalUI.error(f"No wiki data to save for {filename}")
            return
        
        d = self._dir_for(filename) / "wiki"
        d.mkdir(parents=True, exist_ok=True)
        path = d / "wiki.json"


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

        record = self._record_for(filename)
        if record:
                    self.registry.update_pipeline(
                        record.last_seen_path or filename,
                        "wiki",
                        page_num=len(wiki.pages),
                        concept_count=wiki.concept_count,
                    )


    def load_wiki(self, filename: str) -> dict:
        path = self._dir_for(filename) / "wiki" / "wiki.json"
        if not path.exists():
            raise FileNotFoundError(f"No wiki found for '{filename}'.")
        return read_json(path) or {}
