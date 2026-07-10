from __future__ import annotations

from typing import List, Optional, Dict, Any

import numpy as np

from src.core.Pipeline.Indexing.Embedding.EmbeddingEngine import EmbeddingEngine
from src.core.Models.ModelBatchSizing import get_model_batch_size
from src.core.Models.EmbeddingConfig import (
    ModelPreset,
    get_preset,
)

from src.core.Models.ModelManager import modelManager
from src.core.Models.Lib.ModelSpec import ModelCategory
from src.core.Logger.DarwinLogger import darwin


# ═══════════════════════════════════════════════════════════════════════════
# Text preparation — extracted to focused functions
# ═══════════════════════════════════════════════════════════════════════════

def prepare_passage_text(text: str, preset: ModelPreset) -> str:
    """Add passage prefix / instruction template to a chunk of text."""
    if preset.needs_prefix:
        return f"{preset.passage_prefix}{text}"
    return text


def prepare_query_text(query: str, preset: ModelPreset) -> str:
    """Add query prefix / instruction template to a search query."""
    if preset.needs_prefix:
        return f"{preset.query_prefix}{query}"
    return query


# ═══════════════════════════════════════════════════════════════════════════
# Core embedding — delegates ALL heavy lifting to EmbeddingEngine
# ═══════════════════════════════════════════════════════════════════════════

def embed_chunks(
    embedder: Any,
    chunks: List[Dict[str, Any]],
    file_name: str,
    preset: ModelPreset,
    batch_size: Optional[int] = None,
    use_length_batching: bool = True,
    token_budget: int = 2048,
) -> Optional[np.ndarray]:
    """
    Embed chunks with model-specific prompting.

    Args:
        embedder: Loaded embedding model (already acquired).
        chunks: List of chunk dicts with 'clean' text field.
        file_name: Identifier for progress tracking.
        preset: Resolved ModelPreset for prefixing.
        batch_size: Override batch size. If None, uses hardware-aware default.
        use_length_batching: Group by token budget, not item count.
        token_budget: Max tokens per batch (if length_batching=True).

    Returns:
        np.ndarray of shape (N, D) or None on failure.
    """
    if not chunks:
        darwin.warning("No chunks to embed")
        return np.array([], dtype=np.float32)

    texts = [prepare_passage_text(c.get("clean", ""), preset) for c in chunks]
    optimal_batch = batch_size or get_model_batch_size(preset)

    darwin.info(
        f"Embedding {len(texts)} texts with {preset.model_name} "
        f"(batch_size={optimal_batch}, length_batching={use_length_batching})"
    )

    engine = EmbeddingEngine()
    vectors = engine.encode_stream(
        embedder,
        texts,
        base_batch_size=optimal_batch,
        file_name=file_name,
        expected_dim=preset.dimensions if preset.dimensions > 0 else None,
        use_length_batching=use_length_batching,
        token_budget=token_budget or preset.max_tokens,
    )

    if vectors is None or len(vectors) == 0:
        darwin.error("Embedding returned empty result — possible OOM or encoding failure")
        return None

    model_short = preset.model_name.split("/")[-1]
    darwin.success(
        f"Embedded {len(chunks)} chunks → shape {vectors.shape} "
        f"(model: {model_short}, batch_size={optimal_batch})"
    )

    return vectors


def embed_query(
    embedder: Any,
    query: str,
    preset: ModelPreset,
) -> Optional[np.ndarray]:
    """
    Embed a single query string.

    Args:
        embedder: Loaded embedding model (already acquired).
        query: The search query text.
        preset: Resolved ModelPreset for prefixing.

    Returns:
        np.ndarray of shape (1, D) or None on failure.
    """
    prepared = prepare_query_text(query, preset)
    return embedder.encode(prepared).embeddings.astype(np.float32)


def embed_text(
    embedder: Any,
    text: str,
    preset: ModelPreset,
) -> Optional[np.ndarray]:
    """
    Embed a single passage text.

    Args:
        embedder: Loaded embedding model (already acquired).
        text: The passage text to embed.
        preset: Resolved ModelPreset for prefixing.

    Returns:
        np.ndarray of shape (1, D) or None on failure.
    """
    prepared = prepare_passage_text(text, preset)
    return embedder.encode(prepared).embeddings.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline entry point — pre-filter structural chunks, embed, save
# ═══════════════════════════════════════════════════════════════════════════

def run_embedding_pipeline(
    filename: str,
    chunks: List[Dict[str, Any]],
    store,
    embedder: Any,
    preset: ModelPreset,
    rerun: bool = False,
    batch_size: Optional[int] = None,
    use_length_batching: bool = True,
) -> Optional[np.ndarray]:
    """
    Full embedding pipeline: filter structural chunks, embed, save.

    Args:
        filename: Document identifier.
        chunks: List of chunk dicts.
        store: DarwinStore instance for persistence.
        embedder: Pre-loaded embedding model.
        preset: Resolved ModelPreset.
        rerun: If True, re-embed even if fresh index exists.
        batch_size: Override batch size.
        use_length_batching: Enable token-budget batch grouping.

    Returns:
        np.ndarray of shape (N, D) or None.
    """
    if not chunks:
        darwin.warning(f"No chunks to embed for {filename}")
        return None

    if not rerun and store.has_embeddings(filename):
        darwin.warning(f"Skipping {filename} — embeddings already exist")
        return None


    model_name = preset.model_name
    darwin.info(
        f"Embedding {filename} ({len(chunks)} chunks) with {model_name}"
    )

    vectors = embed_chunks(
        embedder=embedder,
        chunks=chunks,
        file_name=filename,
        preset=preset,
        batch_size=batch_size,
        use_length_batching=use_length_batching,
    )

    if vectors is None or len(vectors) == 0:
        return None

    
    store.save_embeddings(filename, vectors, indices=[c["index"] for c in chunks])
    store.save_vector(filename, vectors)
    return vectors


# ═══════════════════════════════════════════════════════════════════════════
# Context manager for explicit lifecycle control
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingSession:
    """
    Explicit embedder lifecycle context manager.
    Loads model ONCE on enter, unloads ONCE on exit.

    Usage:
        with EmbeddingSession() as session:
            for file in files:
                vec = session.embed_file(file, chunks)
        # model guaranteed released here
    """

    def __init__(self, model_key: Optional[str] = None):
        self.model_key = model_key
        self._preset: Optional[ModelPreset] = None
        self._embedder: Optional[Any] = None
        self._acquired = False

    def __enter__(self):
        self._preset = get_preset(self.model_key)
        self._embedder = modelManager.get(ModelCategory.EMBEDDING)
        self._acquired = self._embedder is not None
        if self._acquired:
            darwin.info(f"Embedding session acquired: {self._preset.model_name}")
        else:
            darwin.error("Failed to load embedding model")
            darwin.info("Run: python -m src.core.Models.cli setup --profile required")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            modelManager.unload(ModelCategory.EMBEDDING)
            darwin.info("Embedding model unloaded")
        return False

    @property
    def embedder(self) -> Optional[Any]:
        return self._embedder

    @property
    def preset(self) -> Optional[ModelPreset]:
        return self._preset

    def embed_file(
        self,
        filename: str,
        chunks: List[Dict[str, Any]],
        store,
        rerun: bool = False,
        batch_size: Optional[int] = None,
        use_length_batching: bool = True,
    ) -> Optional[np.ndarray]:
        """Embed a single file's chunks using the held model."""
        if not self._acquired:
            darwin.error("Embedding session not acquired")
            return None
        return run_embedding_pipeline(
            filename=filename,
            chunks=chunks,
            store=store,
            embedder=self._embedder,
            preset=self._preset,
            rerun=rerun,
            batch_size=batch_size,
            use_length_batching=use_length_batching,
        )

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        if not self._acquired:
            darwin.error("Embedding session not acquired")
            return None
        prepared = prepare_query_text(query, self._preset)
        return self._embedder.encode(prepared).embeddings.astype(np.float32)

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if not self._acquired:
            darwin.error("Embedding session not acquired")
            return None
        prepared = prepare_passage_text(text, self._preset)
        return self._embedder.encode(prepared).embeddings.astype(np.float32)