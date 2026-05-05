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
from pathlib import Path
import numpy as np

from src.features.Ingestion.Chunking.lib import Chunk
from src.features.Ingestion.Parsing.lib import ParsedDocument
from src.features.Storage.StoreUtils import (
    file_hash,
    now,
    read_json,
    write_json,
    get_current_model_name,
)
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
    from src.Models.ModelRegistry import REGISTRY
    
    if name in REGISTRY:
        return REGISTRY[name].model_id
    
    id_to_key = {spec.model_id: key for key, spec in REGISTRY.items()}
    if name in id_to_key:
        return name
    
    return name


def _get_current_model_dimensions() -> int:
    """Expected embedding dimensions for the active model. Safe fallback."""
    try:
        from src.Models import ModelManager, ModelCategory
        loader = ModelManager.get_loader()
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

    def _resolve(self, filename: str) -> tuple[str, Path]:
        """Resolve a bare filename to (file_hash, index_dir) in one registry lookup."""
        record = self._record_for(filename)
        if record is None:
            return "", None
        fh = record.file_hash
        return fh, self.paths.dir_for_hash(fh)

    def _hash_for(self, filename: str) -> str:
        """Resolve a bare filename to its file hash."""
        fh, _ = self._resolve(filename)
        return fh

    def _dir_for(self, filename: str):
        """Return (and create) the index directory for a bare filename."""
        _, d = self._resolve(filename)
        return d

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
        fh, d = self._resolve(filename)
        if d is None:
            return
        meta_path = d / "meta.json"
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

    def _batch_meta_update(self, filename: str, updates: dict[str, dict]) -> None:
        """Write meta.json once for multiple component updates."""
        fh, d = self._resolve(filename)
        if d is None:
            return
        meta_path = d / "meta.json"
        meta = read_json(meta_path) or {
            "filename":   filename,
            "created_at": now(),
            "updates":    {},
        }
        meta.setdefault("updates", {})
        timestamp = now()

        for component, extras in updates.items():
            meta["updates"][component] = timestamp
            if extras:
                for key, value in extras.items():
                    if isinstance(value, dict) and isinstance(meta.get(key), dict):
                        meta[key].update(value)
                    else:
                        meta[key] = value

        meta["updated_at"] = timestamp
        write_json(meta_path, meta)

    def load_meta(self, filename: str) -> dict:
        return read_json(self._dir_for(filename) / "meta.json") or {}

    def is_fresh(self, filename: str) -> bool:
        fh, d = self._resolve(filename)
        if d is None:
            return False
        meta_path = d / "meta.json"
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
                dir_path = self._dir_for(filename)
                if dir_path is None:
                    return False
                return (dir_path / "vector.usearch").exists()


    def save_markdown(self, filename: str, doc: ParsedDocument) -> None:
        if doc.error:
            darwin.log(f"Skipping markdown save for {filename}: {doc.error}", LogType.INFO)
            return

        d = self._dir_for(filename)
        md_path = d / "markdown.md"

        try:
            md_path.write_text(doc.display_text, encoding="utf-8")

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

            self.invalidate(filename)
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

        dir_path = self._dir_for(filename)
        if dir_path is None:
            return False
        return (dir_path / "chunks.json").exists()

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

        self.invalidate(filename)

    def load_chunks(self, filename: str) -> list[Chunk]:
        cache_key = f"chunks:{filename}"
        if hasattr(self, '_artifact_cache') and cache_key in self._artifact_cache:
            return self._artifact_cache[cache_key]

        path = self._dir_for(filename) / "chunks.json"
        if not path.exists():
            result = []
        else:
            result = [_chunk_from_dict(c) for c in (read_json(path) or [])]

        if hasattr(self, '_artifact_cache'):
            self._artifact_cache[cache_key] = result
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Vector (USearch)
# ═══════════════════════════════════════════════════════════════════════════

class VectorMixin(_PathResolverMixin):

    def has_vector(self, filename: str) -> bool:
                dir_path = self._dir_for(filename)
                if dir_path is None:
                    return False
                return (dir_path / "vector.usearch").exists()

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

        self.invalidate(filename)
        darwin.log(f"Vector index saved: {len(vectors)} vectors", LogType.SUCCESS)

    def load_vector(self, filename: str):
        d = self._dir_for(filename)
        index_path = d / "vector.usearch"
        meta_path = d / "vector.json"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No vector index found for '{filename}'")

        meta = read_json(meta_path)
        stored_model = meta.get("embedding_model", "unknown")
        current_model = get_current_model_name()

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


class EmbeddingsMixin(_PathResolverMixin):
    """Mixin for saving/loading embeddings with full chunk metadata."""

    def has_embeddings(self, filename: str) -> bool:
        fh, d = self._resolve(filename)
        if d is None:
            return False
        current_model = get_current_model_name()
        return self.paths.npz_path(fh, current_model).exists()

    def _get_overlap_value(self, chunk) -> str:
        """Safely extract overlap direction as string."""
        if not hasattr(chunk, 'overlap_direction'):
            return "none"
        
        od = chunk.overlap_direction
        if hasattr(od, 'value'):
            return od.value
        return str(od)

    def _serialize_links(self, chunk) -> str:
        """Serialize chunk links to JSON string."""
        import json
        if not hasattr(chunk, 'links') or not chunk.links:
            return "[]"
        
        links_data = []
        for link in chunk.links:
            if hasattr(link, 'url'):
                # ChunkLink object
                links_data.append({
                    "url": link.url,
                    "text": getattr(link, 'text', '')
                })
            elif isinstance(link, dict):
                # Already a dict
                links_data.append({
                    "url": link.get('url', ''),
                    "text": link.get('text', '')
                })
            else:
                # Unknown type, try to get string representation
                links_data.append({
                    "url": str(link),
                    "text": ""
                })
        
        return json.dumps(links_data)

    def save_embeddings(
        self,
        filename: str,
        vectors: np.ndarray,
        chunks: list,
        model_name: str = None,
    ) -> None:
        """Save embeddings with full chunk metadata."""
        if chunks is None or len(chunks) == 0:
            darwin.log(f"No chunks provided for embedding save of {filename}", LogType.ERROR)
            return

        if model_name is None:
            model_name = get_current_model_name()
        
        model_name = _normalize_model_name(model_name)

        fh, d = self._resolve(filename)
        if d is None:
            return
        path = self.paths.npz_path(fh, model_name)

        if len(vectors) != len(chunks):
            raise ValueError(
                f"vectors ({len(vectors)}) and chunks ({len(chunks)}) must be the same length."
            )

        try:
            # Prepare overlap direction strings
            overlap_values = [self._get_overlap_value(c) for c in chunks]
            
            # Prepare links JSON
            links_json_list = [self._serialize_links(c) for c in chunks]
            
            np.savez_compressed(
                path,
                vectors=vectors.astype(np.float32),
                text=np.array([c.text for c in chunks], dtype=object),
                clean=np.array([c.clean for c in chunks], dtype=object),
                headings=np.array([c.heading for c in chunks], dtype=object),
                pages=np.array([c.page for c in chunks], dtype=np.int32),
                pages_end=np.array([c.page_end for c in chunks], dtype=np.int32),
                overlap_direction=np.array(overlap_values, dtype=object),
                token_counts=np.array([c.token_count for c in chunks], dtype=np.int32),
                source=np.array([c.source for c in chunks], dtype=object),
                links_json=np.array(links_json_list, dtype=object),
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
            
            self.invalidate(filename)
            darwin.log(f"Saved embeddings: {path.name}", LogType.SUCCESS)

        except Exception as e:
            import traceback
            darwin.log(f"Failed to save embeddings for {filename}: {e}", LogType.ERROR)
            darwin.log(traceback.format_exc(),)

    def load_embeddings(self, filename: str) -> tuple[np.ndarray, list]:
        """Load embeddings and return (vectors, chunks)."""
        import json
        from src.features.Ingestion.Chunking.ChunkerUtils import Chunk, ChunkLink, OverlapDirection

        cache_key = f"embeddings:{filename}"
        if hasattr(self, '_artifact_cache') and cache_key in self._artifact_cache:
            return self._artifact_cache[cache_key]

        fh, d = self._resolve(filename)
        if d is None:
            return None, None
        
        current_model = get_current_model_name()
        path = self.paths.npz_path(fh, current_model)

        if not path.exists():
            meta_path = d / "meta.json"
            if meta_path.exists():
                file_meta = read_json(meta_path)
                stored_model = file_meta.get("embedding_model", "unknown")
                if stored_model != "unknown" and stored_model != current_model:
                    darwin.log(
                        f"Embeddings for '{filename}' were built with '{stored_model}', "
                        f"but your current model is '{current_model}'. "
                        f"Run the embedding pipeline to re-index.", LogType.ERROR
                    )
                    return None, None
            darwin.log(
                f"No embeddings found for '{filename}'. Run the embedding pipeline first.", LogType.ERROR)
            return None, None

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

        data = np.load(path, allow_pickle=True, mmap_mode='r')
        vectors = data["vectors"]
        n_chunks = len(data['text'])

        darwin.log(f"Reconstructing {n_chunks} chunks...", LogType.INFO)

        # Convert object arrays to lists upfront
        texts = data['text'].tolist()
        cleans = data['clean'].tolist()
        sources = data['source'].tolist()
        headings = data['headings'].tolist()
        pages = data['pages'].tolist()
        pages_end = data['pages_end'].tolist() if 'pages_end' in data else pages
        token_counts = data['token_counts'].tolist() if 'token_counts' in data else [0] * n_chunks
        overlap_raw = data['overlap_direction'].tolist()
        links_raw = data['links_json'].tolist()
        
        overlap_dirs = []
        for raw in overlap_raw:
            try:
                overlap_dirs.append(OverlapDirection(str(raw)))
            except (ValueError, TypeError):
                overlap_dirs.append(OverlapDirection.NONE)

        all_links = []
        for links_json in links_raw:
            links = []
            if links_json and links_json not in ('null', '[]', ''):
                try:
                    parsed = json.loads(links_json if isinstance(links_json, str) else links_json)
                    if isinstance(parsed, list):
                        for l in parsed:
                            links.append(ChunkLink(url=l.get('url', ''), text=l.get('text', '')))
                except (json.JSONDecodeError, TypeError):
                    pass
            all_links.append(links)

        chunks = [
            Chunk(
                text=str(texts[i]),
                clean=str(cleans[i]),
                index=i,
                source=str(sources[i]),
                token_count=token_counts[i],
                heading=str(headings[i]),
                page=pages[i],
                page_end=pages_end[i],
                overlap_direction=overlap_dirs[i],
                links=all_links[i],
            )
            for i in range(n_chunks)
        ]

        result = (vectors, chunks)
        if hasattr(self, '_artifact_cache'):
            self._artifact_cache[cache_key] = result
        darwin.log(f"Loaded {len(chunks)} chunks", LogType.INFO)
        return result

    def get_embedding_metadata(self, filename: str) -> Optional[dict]:
        """Get embedding metadata for a file."""
        fh, d = self._resolve(filename)
        if d is None:
            return None
        current_model = get_current_model_name()
        path = self.paths.npz_path(fh, current_model)
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            return None
        return read_json(meta_path)

    def is_embedding_compatible(self, filename: str) -> bool:
        """Check if embeddings exist and are compatible with current model."""
        try:
            result = self.load_embeddings(filename)
            return result is not None and result[0] is not None
        except FileNotFoundError:
            return False

# ═══════════════════════════════════════════════════════════════════════════
# Mixin: BM25
# ═══════════════════════════════════════════════════════════════════════════

class BM25Mixin(_PathResolverMixin):

    from src.features.Indexing.BM25.lib import BM25Data

    def has_bm25(self, filename: str) -> bool:
                dir_path = self._dir_for(filename)
                if dir_path is None:
                    return False
                return (dir_path / "bm25.npz").exists()


    def save_bm25(self, filename: str, data) -> None:
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
            }, compact=True)

            self._update_meta(filename, "bm25")
            record = self._record_for(filename)
            if record:
                self.registry.update_pipeline(record.last_seen_path or filename, "bm25")
            self.invalidate(filename)
            darwin.log(f"Saved BM25 index: {npz_path.name}", LogType.SUCCESS)

        except Exception as e:
            darwin.log(f"Failed to save BM25 for {filename}: {e}", LogType.ERROR)

    def load_bm25(self, filename: str) -> BM25Data | None:
        from src.features.Indexing.BM25.lib import BM25Data

        cache_key = f"bm25:{filename}"
        if hasattr(self, '_artifact_cache') and cache_key in self._artifact_cache:
            return self._artifact_cache[cache_key]

        d = self._dir_for(filename)
        npz_path = d / "bm25.npz"
        meta_path = d / "bm25_meta.json"

        if not npz_path.exists() or not meta_path.exists():
            return None

        raw = np.load(npz_path, allow_pickle=True)
        meta = read_json(meta_path)

        tokens = [list(tok_list) for tok_list in raw["tokens"]]

        result = BM25Data(
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

        if hasattr(self, '_artifact_cache'):
            self._artifact_cache[cache_key] = result
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Mixin: Stopwords
# ═══════════════════════════════════════════════════════════════════════════

class StopwordsMixin(_PathResolverMixin):

    def has_stopwords(self, filename: str) -> bool:
                dir_path = self._dir_for(filename)
                if dir_path is None:
                    return False
                return (dir_path / "stopwords.json").exists()


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
        self.invalidate(filename)

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
        
                dir_path = self._dir_for(filename)
                if dir_path is None:
                    return False
                return (dir_path / "graph" / "graph.json").exists()


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
        self.invalidate(filename)

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
            self.registry.update_pipeline(
                record.last_seen_path or filename, "graph_vis"
            )
        self.invalidate(filename)


    def save_graph_centroids(self, filename: str, centroids: list) -> None:
        d = self._dir_for(filename) / "graph"
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "centroids.json", centroids)
        self._update_meta(filename, "graph_centroids")
        record = self._record_for(filename)
        if record:
            self.registry.update_pipeline(
                record.last_seen_path or filename, "graph_centroids"
            )
        self.invalidate(filename)

    def load_graph_centroids(self, filename: str) -> list:
        path = self._dir_for(filename) / "graph" / "centroids.json"
        if not path.exists():
            return []
        return read_json(path) or []

    def load_graph(self, filename: str) -> dict:
        
        cache_key = f"graph:{filename}"
        if hasattr(self, '_artifact_cache') and cache_key in self._artifact_cache:
            return self._artifact_cache[cache_key]

        path = self._dir_for(filename) / "graph" / "graph.json"
        if not path.exists():
            None
        result = read_json(path) or {}

        if hasattr(self, '_artifact_cache'):
            self._artifact_cache[cache_key] = result
        return result


class WikiMixin(_PathResolverMixin):

    def has_wiki(self, filename: str) -> bool:
                
        
                dir_path = self._dir_for(filename)
                if dir_path is None:
                    return False
                return (dir_path / "wiki" / "wiki.json").exists()

    def save_wiki(self, filename: str, wiki: Any) -> None:
        if wiki is None:
            darwin.log(f"No wiki data to save for {filename}", LogType.ERROR)
            return

        d = self._dir_for(filename) / "wiki"
        d.mkdir(parents=True, exist_ok=True)

        wiki_dict = {
            "filename": wiki.filename,
            "manifest": wiki.manifest,
            "pages": [
                {
                    "concept":       p.concept,
                    "chunk_indices": p.chunk_indices,
                    "related":       p.related,
                    "centroid":      p.centroid,
                    "frequency":     p.frequency,
                    "summary":       p.summary,
                    "keywords":      p.keywords,
                }
                for p in wiki.pages
            ],
        }
        write_json(d / "wiki.json", wiki_dict)

        record = self._record_for(filename)
        if record:
            self.registry.update_pipeline(
                record.last_seen_path or filename,
                "wiki",
                page_num=len(wiki.pages),
            )
        self.invalidate(filename)

    def load_wiki(self, filename: str) -> dict:
        path = self._dir_for(filename) / "wiki" / "wiki.json"
                
        if not path.exists():
            return None
        return read_json(path) or {}