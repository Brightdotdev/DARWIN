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
from src.core.BM25.BM25Utils import BM25Data
from src.core.Logger.DarwinLogger import darwin, LogType


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
    """Convert any model identifier to the canonical model_id."""
    from src.core.Models.ModelRegistry import REGISTRY
    
    if name in REGISTRY:
        return REGISTRY[name].model_id
    
    id_to_key = {spec.model_id: key for key, spec in REGISTRY.items()}
    if name in id_to_key:
        return name
    
    return name


def _get_current_model_name() -> str:
    """Resolve the currently active embedding model name (full model_id)."""
    try:
        from src.core.Models import get_loader, ModelCategory
        loader = get_loader()
        return loader.get_model_key(ModelCategory.EMBEDDING)
    except Exception as e:
        darwin.log(f"Failed to detect current model: {e}", LogType.ERROR)
        return "unknown"
    

def _get_current_model_dimensions() -> int:
    """Expected embedding dimensions for the active model. Safe fallback."""
    try:
        from src.core.Models import get_loader, ModelCategory
        loader = get_loader()
        key = loader.get_model_key(ModelCategory.EMBEDDING)
        dims = {
            "intfloat/e5-small-v2": 384,
            "BAAI/bge-m3": 1024,
            "GritLM/GritLM-7B": 4096,
        }
        return dims.get(key, 0)
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# Base mixin — filename → hash → directory resolution
# ═══════════════════════════════════════════════════════════════════════════

class _PathResolverMixin:
    """Bridge between calling code (bare filenames) and StorePaths (file hashes)."""

    def _record_for(self, filename: str):
        """Return the FileRecord for a bare filename. Raises KeyError if missing."""
        record = self.registry._find_by_name(filename)
        if record is None:
            darwin.log(f"File '{filename}' is not registered. Register it before saving artifacts.", LogType.ERROR)
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
    """Handles config.json (global user preferences)."""

    def load_config(self) -> dict:
        return read_json(self.paths.config_path) or {}

    def save_config(self, config: dict) -> None:
        write_json(self.paths.config_path, config)

    def update_config(self, **kwargs) -> None:
        config = self.load_config()
        config.update(kwargs)
        self.save_config(config)

    def get_user_embedding_model(self) -> str:
        return self.load_config().get("embedding_model", "unknown")

    def set_user_embedding_model(self, model_name: str) -> None:
        self.update_config(embedding_model=model_name)
        darwin.log(f"Embedding model updated — new files will use {model_name}. Existing files may need re-embedding.", LogType.INFO)


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Markdown
# ═══════════════════════════════════════════════════════════════════════════

class MarkdownMixin(_PathResolverMixin):

    def has_markdown(self, filename: str) -> bool:
        return (self._dir_for(filename) / "markdown.md").exists()

    def save_markdown(self, filename: str, doc: ParsedDocument) -> None:
        if doc.error:
            darwin.log(f"Skipping markdown save for {filename}: {doc.error}", LogType.INFO)
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

            darwin.log(f"Markdown saved: {md_path.name}", LogType.SUCCESS)
        except Exception as e:
            darwin.log(f"Failed to save markdown for {filename}: {e}", LogType.ERROR)

    def load_markdown(self, filename: str) -> str | None:
        md_path = self._dir_for(filename) / "markdown.md"
        if not md_path.exists():
            darwin.log(f"Markdown not found for {filename}", LogType.ERROR)
            return None
        try:
            return md_path.read_text(encoding="utf-8")
        except Exception as e:
            darwin.log(f"Error reading markdown for {filename}: {e}", LogType.ERROR)
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
            darwin.log(f"No chunks to save for {filename}", LogType.ERROR)
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
            "dimension": dimension,
            "chunk_ids": chunk_ids,
            "total_vectors": len(vectors),
            "embedding_model": model_name,
            "created_at": now(),
        }
        write_json(d / "vector.json", meta_data)

        self._update_meta(filename, "vector", extras={
            "vector_model": model_name,
            "vector_dimension": dimension,
        })

        record = self._record_for(filename)
        if record:
            self.registry.update_pipeline(record.last_seen_path or filename, "vector")

        darwin.log(f"Vector index saved: {len(vectors)} vectors", LogType.SUCCESS)

    def load_vector(self, filename: str):
        d = self._dir_for(filename)
        index_path = d / "vector.usearch"
        meta_path = d / "vector.json"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No vector index found for '{filename}'")

        meta = read_json(meta_path)
        stored_model = meta.get("embedding_model", "unknown")
        current_model = _get_current_model_name()

        if stored_model != current_model and stored_model != "unknown":
            darwin.log(
                f"Vector index for '{filename}' was built with '{stored_model}' "
                f"but current model is '{current_model}'. Search results will be wrong. "
                f"Run 'darwin reindex' to rebuild.",
                LogType.WARNING
            )
            return None, None
        
        expected_dim = _get_current_model_dimensions()
        if expected_dim and meta.get("dimension") != expected_dim:
            darwin.log(
                f"Vector dimension mismatch: index has {meta['dimension']}d "
                f"but current model expects {expected_dim}d. Re-embedding required.",
                LogType.WARNING
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
        path = self._dir_for(filename) / "vector.json"
        if not path.exists():
            return None
        return read_json(path)

    def is_vector_compatible(self, filename: str) -> bool:
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
        if chunks is None or len(chunks) == 0:
            darwin.log(f"No chunks provided for embedding save of {filename}", LogType.ERROR)
            return

        if model_name is None:
            model_name = _get_current_model_name()
        
        model_name = _normalize_model_name(model_name)

        fh = self._hash_for(filename)
        path = self.paths.npz_path(fh, model_name)

        if len(vectors) != len(chunks):
            raise ValueError(
                f"vectors ({len(vectors)}) and chunks ({len(chunks)}) must be the same length."
            )

        try:
            import json
            np.savez_compressed(
                path,
                vectors=vectors.astype(np.float32),
                text=np.array([c.text for c in chunks], dtype=object),
                clean=np.array([c.clean for c in chunks], dtype=object),
                headings=np.array([c.heading for c in chunks], dtype=object),
                pages=np.array([c.page for c in chunks], dtype=np.int32),
                pages_end=np.array([c.page_end for c in chunks], dtype=np.int32),
                overlap_direction=np.array([c.overlap_direction.value for c in chunks], dtype=object),
                token_counts=np.array([c.token_count for c in chunks], dtype=np.int32),
                source=np.array([c.source for c in chunks], dtype=object),
                # links are more complex - need JSON serialization
                links_json=np.array([json.dumps([{"url": l.url, "text": l.text} for l in c.links]) for c in chunks], dtype=object),
)
            meta_path = path.with_suffix(".meta.json")
            write_json(meta_path, {
                "embedding_model": model_name,
                "dimensions": vectors.shape[1],
                "chunk_count": len(chunks),
                "created_at": now(),
            })

            self._update_meta(filename, "embeddings", extras={
                "embedding_model": model_name,
                "embedding_dimensions": vectors.shape[1],
            })
            record = self._record_for(filename)
            if record:
                self.registry.update_pipeline(record.last_seen_path or filename, "embeddings")
                
            darwin.log(f"Saved embeddings: {path.name}", LogType.SUCCESS)

        except Exception as e:
            darwin.log(f"Failed to save embeddings for {filename}: {e}", LogType.ERROR)

    

    def load_embeddings(self, filename: str) -> tuple[np.ndarray, list[Chunk]]:
        """Load embeddings and return (vectors, chunks)."""
        current_model = _get_current_model_name()
        fh = self._hash_for(filename)
        path = self.paths.npz_path(fh, current_model)

        if not path.exists():
            meta_path = self._dir_for(filename) / "meta.json"
            if meta_path.exists():
                file_meta = read_json(meta_path)
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
            stored_meta = read_json(meta_path)
            stored_model = stored_meta.get("embedding_model", "unknown")
            if stored_model != current_model and stored_model != "unknown":
                darwin.log(
                    f"Embeddings were created with '{stored_model}' "
                    f"but current model is '{current_model}'. Re-embedding recommended.",
                    LogType.WARNING
                )

        data = np.load(path, allow_pickle=True)
        vectors = data["vectors"]
        
        # Reconstruct chunks
        from src.core.Chunking.ChunkerUtils import Chunk, ChunkLink, OverlapDirection
        import json
        
        chunks = []
        n_chunks = len(data['text'])
        
        for i in range(n_chunks):
            # Parse links from JSON
            links_json = data.get('links_json', [None])[i] if 'links_json' in data else None
            if links_json and links_json != 'null' and links_json is not None:
                links = [ChunkLink(url=l['url'], text=l.get('text', '')) 
                        for l in json.loads(links_json)]
            else:
                links = []
            
            # Get overlap direction with default
            overlap_dir_raw = data.get('overlap_direction', ['none'])[i] if 'overlap_direction' in data else 'none'
            if isinstance(overlap_dir_raw, str):
                overlap_dir = OverlapDirection(overlap_dir_raw)
            else:
                overlap_dir = OverlapDirection.NONE
            
            chunk = Chunk(
                text=str(data['text'][i]),
                clean=str(data['clean'][i]),
                index=i,
                source=str(data['source'][i]),
                token_count=int(data.get('token_counts', [0])[i]) if 'token_counts' in data else 0,
                heading=str(data['headings'][i]),
                page=int(data['pages'][i]),
                page_end=int(data.get('pages_end', [data['pages'][i]])[i]) if 'pages_end' in data else int(data['pages'][i]),
                overlap_direction=overlap_dir,
                links=links,
            )
            chunks.append(chunk)
        
        return vectors, chunks

    def get_embedding_metadata(self, filename: str) -> Optional[dict]:
        current_model = _get_current_model_name()
        fh = self._hash_for(filename)
        path = self.paths.npz_path(fh, current_model)
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
        if not data.tokens or not data.doc_lengths:
            darwin.log(f"No BM25 data to save for {filename}", LogType.ERROR)
            return
        d = self._dir_for(filename)
        npz_path = d / "bm25.npz"
        meta_path = d / "bm25_meta.json"

        try:
            tokens_arr = np.empty(len(data.tokens), dtype=object)
            for i, tok_list in enumerate(data.tokens):
                tokens_arr[i] = tok_list

            np.savez_compressed(
                npz_path,
                tokens=tokens_arr,
                doc_lengths=np.array(data.doc_lengths, dtype=np.int32),
            )

            write_json(meta_path, {
                "avgdl": data.avgdl,
                "chunk_count": data.chunk_count,
                "vocab_size": data.vocab_size,
                "threshold": data.threshold,
                "idf_smoothing": data.idf_smoothing,
                "domain_stopwords": data.domain_stopwords,
                "doc_freqs": data.doc_freqs,
                "precomputed_idfs": data.precomputed_idfs or {},
            })

            self._update_meta(filename, "bm25")
            record = self._record_for(filename)
            if record:
                self.registry.update_pipeline(record.last_seen_path or filename, "bm25")
            darwin.log(f"Saved BM25 index: {npz_path.name}", LogType.SUCCESS)

        except Exception as e:
            darwin.log(f"Failed to save BM25 for {filename}: {e}", LogType.ERROR)

    def load_bm25(self, filename: str) -> BM25Data | None:
        d = self._dir_for(filename)
        npz_path = d / "bm25.npz"
        meta_path = d / "bm25_meta.json"

        if not npz_path.exists() or not meta_path.exists():
            return None

        raw = np.load(npz_path, allow_pickle=True)
        meta = read_json(meta_path)

        tokens = [list(tok_list) for tok_list in raw["tokens"]]

        return BM25Data(
            tokens=tokens,
            doc_freqs=meta["doc_freqs"],
            avgdl=meta["avgdl"],
            doc_lengths=raw["doc_lengths"].tolist(),
            domain_stopwords=meta["domain_stopwords"],
            threshold=meta["threshold"],
            chunk_count=meta["chunk_count"],
            vocab_size=meta["vocab_size"],
            idf_smoothing=meta["idf_smoothing"],
            precomputed_idfs=meta.get("precomputed_idfs") or None,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Term Bank (NEW — Query Expansion without LLM)
# ═══════════════════════════════════════════════════════════════════════════

class TermBankMixin(_PathResolverMixin):
    """
    Per-document term bank for query expansion.

    Storage
    -------
    term_bank.json          — raw terms + frequencies (model-agnostic, tiny)
    term_embeddings.npz     — embedded terms (model-specific, ~3-6MB)
    term_embedding_meta.json — which model produced the embeddings

    Separation means model switches only require re-embedding the vocabulary,
    not rebuilding it from scratch.
    """

    def has_term_bank(self, filename: str) -> bool:
        return (self._dir_for(filename) / "term_bank.json").exists()

    def has_term_embeddings(self, filename: str) -> bool:
        current_model = _get_current_model_name()
        path = self._dir_for(filename) / f"term_embeddings_{current_model.replace('/', '_')}.npz"
        return path.exists()

    def save_term_bank(
        self,
        filename: str,
        terms: list[str],
        term_freqs: dict[str, int],
        chunk_count: int,
    ) -> None:
        """
        Save the raw vocabulary as a model-agnostic JSON file.
        Called once during BM25 indexing — never needs rebuilding.
        """
        if not terms:
            darwin.log(f"No terms to save for term bank: {filename}", LogType.WARNING)
            return

        d = self._dir_for(filename)
        path = d / "term_bank.json"

        write_json(path, {
            "terms": terms,
            "term_freqs": term_freqs,
            "vocab_size": len(terms),
            "chunk_count": chunk_count,
            "created_at": now(),
        })

        self._update_meta(filename, "term_bank")
        darwin.log(f"Term bank saved: {len(terms)} terms", LogType.SUCCESS)

    def load_term_bank(self, filename: str) -> dict | None:
        """Load raw terms + frequencies (no embeddings)."""
        path = self._dir_for(filename) / "term_bank.json"
        if not path.exists():
            return None
        return read_json(path)

    def save_term_embeddings(
        self,
        filename: str,
        term_embeddings: np.ndarray,
        terms: list[str],
        model_name: str | None = None,
    ) -> None:
        """
        Save embedded terms for a specific model.
        Called during BM25 indexing (if embedder available) or on model switch.
        """
        if model_name is None:
            model_name = _get_current_model_name()
        model_name = _normalize_model_name(model_name)

        if len(term_embeddings) != len(terms):
            raise ValueError(
                f"term_embeddings ({len(term_embeddings)}) and terms ({len(terms)}) must match"
            )

        d = self._dir_for(filename)
        safe_name = model_name.replace("/", "_")
        npz_path = d / f"term_embeddings_{safe_name}.npz"
        meta_path = d / f"term_embedding_meta_{safe_name}.json"

        np.savez_compressed(
            npz_path,
            embeddings=term_embeddings.astype(np.float16),
            terms=np.array(terms, dtype=object),
        )

        write_json(meta_path, {
            "model": model_name,
            "dim": term_embeddings.shape[1],
            "dtype": "float16",
            "term_count": len(terms),
            "created_at": now(),
        })

        darwin.log(f"Term embeddings saved: {len(terms)} terms @ {term_embeddings.shape[1]}d", LogType.SUCCESS)

    def load_term_embeddings(self, filename: str) -> tuple[list[str], np.ndarray] | tuple[None, None]:
        """
        Load term embeddings for the current model.
        Returns (terms, embeddings) or (None, None) if not found.
        """
        current_model = _get_current_model_name()
        d = self._dir_for(filename)
        safe_name = current_model.replace("/", "_")
        npz_path = d / f"term_embeddings_{safe_name}.npz"

        if not npz_path.exists():
            return None, None

        data = np.load(npz_path, allow_pickle=True)
        return data["terms"].tolist(), data["embeddings"].astype(np.float32)

    def get_term_embedding_model(self, filename: str) -> str | None:
        """Return the model name used for the current term embeddings, or None."""
        current_model = _get_current_model_name()
        d = self._dir_for(filename)
        safe_name = current_model.replace("/", "_")
        meta_path = d / f"term_embedding_meta_{safe_name}.json"

        if not meta_path.exists():
            return None
        return read_json(meta_path).get("model")

    def expand_query(
        self,
        filename: str,
        query: str,
        top_n: int = 5,
        min_similarity: float = 0.5,
    ) -> str:
        """
        Expand a query with top-N most similar terms from the document's vocabulary.

        Only used for keyword (BM25) search — semantic search doesn't need expansion.
        Terms already in the query are excluded. Terms below min_similarity are filtered.
        """
        terms, embeddings = self.load_term_embeddings(filename)

        if terms is None or embeddings is None:
            darwin.log(f"No term embeddings for {filename} — query not expanded", LogType.INFO)
            return query

        try:
            from src.core.Embedding.Embedder import embed_query
            query_vec = embed_query(query)

            if query_vec is None:
                return query

            # Cosine similarity (both are normalized)
            sims = embeddings @ query_vec

            # Get top-N, filtering by similarity and dedup
            top_indices = np.argsort(sims)[::-1]
            query_lower = query.lower()
            expansion_terms = []

            for idx in top_indices:
                if len(expansion_terms) >= top_n:
                    break
                term = terms[idx]
                if sims[idx] < min_similarity:
                    continue
                if term.lower() in query_lower:
                    continue
                expansion_terms.append(term)

            if expansion_terms:
                expanded = f"{query} {' '.join(expansion_terms)}"
                darwin.log(f"Query expanded: '{query}' → '{expanded}'", LogType.INFO)
                return expanded

        except Exception as e:
            darwin.log(f"Query expansion failed: {e}", LogType.WARNING)

        return query


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Stopwords
# ═══════════════════════════════════════════════════════════════════════════

class StopwordsMixin(_PathResolverMixin):

    def has_stopwords(self, filename: str) -> bool:
        return (self._dir_for(filename) / "stopwords.json").exists()

    def save_stopwords(self, filename: str, stopwords: set[str]) -> None:
        if not stopwords:
            darwin.log(f"No stopwords to save for {filename}", LogType.ERROR)
            return
        d = self._dir_for(filename)
        write_json(d / "stopwords.json", {"stopwords": sorted(stopwords)})
        self._update_meta(filename, "stopwords")
        record = self._record_for(filename)
        if record:
            self.registry.update_pipeline(record.last_seen_path or filename, "stopwords")

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
            darwin.log(f"No graph data to save for {filename}", LogType.ERROR)
            return

        d = self._dir_for(filename) / "graph"
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "graph.json", graph_data.to_dict())

        self._update_meta(filename, "graph", extras={
            "graph": {
                "concept_count": len(graph_data.nodes),
                "edge_count": len(graph_data.edges),
                "chunk_count": graph_data.chunk_count,
                "total_concepts_extracted": graph_data.total_concepts_extracted,
                "built_at": now(),
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

    def save_graph_html(self, filename: str, html: str) -> None:
        if html is None:
            darwin.log(f"No graph HTML to save for {filename}", LogType.ERROR)
            return

        d = self._dir_for(filename) / "graph"
        d.mkdir(parents=True, exist_ok=True)
        (d / "graph.html").write_text(html, encoding="utf-8")
        self._update_meta(filename, "graph_vis")
        record = self._record_for(filename)
        if record:
            self.registry.update_pipeline(record.last_seen_path or filename, "graph_vis")

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

    def save_wiki(self, filename: str, wiki: Any) -> None:
        if wiki is None:
            darwin.log(f"No wiki data to save for {filename}", LogType.ERROR)
            return
        
        d = self._dir_for(filename) / "wiki"
        d.mkdir(parents=True, exist_ok=True)
        path = d / "wiki.json"

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