

# ═══════════════════════════════════════════════════════════════════════════════
# src/core/Embedding/Embedder.py
# ═══════════════════════════════════════════════════════════════════════════════
"""
Thin orchestrator. All heavy lifting (singleton cache, adaptive batching,
stream encoding, OOM recovery) is delegated to EmbeddingEngine.

Preset handling is EXPLICIT — no ambiguous str|ModelPreset unions.
Callers must resolve the active model key before calling embed functions.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any

import numpy as np

from src.features.Indexing.Embedding.EmbeddingEngine import EmbeddingEngine
from src.core.Models.ModelBatchSizing import get_model_batch_size
from src.core.Models.EmbeddingConfig import (
    ModelPreset,
    get_preset,
)
from src.features.Ingestion.Chunking.StructuralClassifier import is_structural_for_embedding
from src.core.Models.ModelManager import modelManager
from src.core.Models.Lib.ModelSpec import ModelCategory
from src.core.Logger.DarwinLogger import darwin, LogType


# ─────────────────────────────────────────────────────────────────────────────
# Preset resolution — single source of truth
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Text preparation — always uses resolved preset
# ─────────────────────────────────────────────────────────────────────────────

def prepare_passage_text(text: str, preset: ModelPreset) -> str:
    """Add passage prefix / instruction template to a chunk of text."""
 
    if preset.needs_prefix:
        return f"{preset.passage_prefix}{text}"
    else:
        return text


def prepare_query_text(query: str, preset: ModelPreset) -> str:
    """Add query prefix / instruction template to a search query."""
    
    if preset.needs_prefix:
        return f"{preset.query_prefix}{query}"
    else:
        return query


# ─────────────────────────────────────────────────────────────────────────────
# Core embedding — delegates all heavy lifting to EmbeddingEngine
# ─────────────────────────────────────────────────────────────────────────────

def embed_chunks(
    chunks: List[Dict[str, Any]],
    file_name: str,
    batch_size: Optional[int] = None,
    unload_after: bool = False,
    model_key: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Embed chunks with model-specific prompting.

    Args:
        chunks: List of chunk dicts with 'clean' text field.
        file_name: Identifier for progress tracking.
        batch_size: Override batch size. If None, uses hardware-aware default.
        unload_after: If True, unload model after embedding.
        model_key: Explicit registry key. If None, uses active embedding model.

    Returns:
        np.ndarray of shape (N, D) or None on failure.
    """
    if not chunks:
        darwin.log("No chunks found to embed", LogType.WARNING)
        return np.array([], dtype=np.float32)

    # ── Resolve preset ONCE ──────────────────────────────────────────
    

    embedder = modelManager.get(ModelCategory.EMBEDDING)
    preset = get_preset(model_key)

    if embedder is None:
        darwin.log("Failed to load embedding model", LogType.ERROR)
        darwin.log("[HELP] Run: python -m src.core.Models.cli setup --profile required")
        return None


    texts = [prepare_passage_text(c.get("clean", ""), preset) for c in chunks]
    optimal_batch = batch_size or get_model_batch_size(preset, model_key=model_key)

    darwin.log(
        f"Embedding with {preset.model_name} ({preset.category}, {preset.max_tokens} token limit) "
        f"[batch_size={optimal_batch}]",
        LogType.INFO,
    )

    # Delegate stream encoding to engine — memory-aware, OOM recovery, progress tracking
    engine = EmbeddingEngine()
    vectors = engine.encode_stream(
        embedder,
        texts,
        base_batch_size=optimal_batch,
        file_name=file_name,
    )

    if vectors is None or len(vectors) == 0:
        darwin.log("Embedding returned empty result; possible OOM or encoding failure.", LogType.ERROR)
        return None

    model_short = preset.model_name.split("/")[-1]
    darwin.log(
        f"Embedded {len(chunks)} chunks → {vectors.shape} "
        f"(model: {model_short}, batch_size={optimal_batch})",
        LogType.SUCCESS,
    )

    if unload_after:
        modelManager.unload(ModelCategory.EMBEDDING)

    return vectors


def embed_query(
    query: str,
    model_key: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Embed a single query string.

    Args:
        query: The search query text.
        model_key: Explicit registry key. If None, uses active embedding model.

    Returns:
        np.ndarray of shape (1, D) or None on failure.
    """
    preset = get_preset(model_key)
    
    embedder = modelManager.get(ModelCategory.EMBEDDING)

    if embedder is None:
        darwin.log("Failed to load embedding model", LogType.ERROR)
        darwin.log("[HELP] Run: python -m src.core.Models.cli setup --profile required")
        return None

    prepared = prepare_query_text(query, preset)
    return embedder.encode(
        prepared,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def embed_text(
    text: str,
    model_key: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Embed a single passage text.

    Args:
        text: The passage text to embed.
        model_key: Explicit registry key. If None, uses active embedding model.

    Returns:
        np.ndarray of shape (1, D) or None on failure.
    """
    preset = get_preset(model_key)
    
    embedder = modelManager.get(ModelCategory.EMBEDDING)

    if embedder is None:
        darwin.log("Failed to load embedding model", LogType.ERROR)
        darwin.log("[HELP] Run: python -m src.core.Models.cli setup --profile required")
        return None

    prepared = prepare_passage_text(text, preset)
    return embedder.encode(
        prepared,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry point — pre-filter structural chunks
# ─────────────────────────────────────────────────────────────────────────────

def run_embedding_pipeline(
    filename: str,
    chunks: List[Dict[str, Any]],
    store,
    rerun: bool = False,
    unload_after: bool = False,
    model_key: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Full embedding pipeline: filter structural chunks, embed, save.

    Args:
        filename: Document identifier.
        chunks: List of chunk dicts.
        store: DarwinStore instance for persistence.
        rerun: If True, re-embed even if fresh index exists.
        unload_after: If True, unload model after embedding.
        model_key: Explicit registry key. If None, uses active embedding model.
    """
    if not chunks:
        darwin.log(f"No chunks to embed for {filename}", LogType.WARNING)
        return None

    if not rerun and store.has_embeddings(filename):
        darwin.log(f"Skipping embedding: fresh index exists for {filename}", LogType.INFO)
        return None

    # Pre-filter: exclude structural chunks BEFORE any model work
    content_chunks = [
        c for c in chunks if not is_structural_for_embedding(c.get("structural_type"))
    ]
    n_structural = len(chunks) - len(content_chunks)

    if n_structural > 0:
        darwin.log(
            f"{filename}: {n_structural}/{len(chunks)} structural chunks excluded from embedding",
            LogType.INFO,
        )

    if not content_chunks:
        darwin.log(f"No content chunks to embed for {filename}", LogType.WARNING)
        return None

    preset = get_preset(model_key)
    model_name = preset.model_name

    darwin.log(
        f"Embedding {filename} ({len(content_chunks)} content chunks) with {model_name}",
        LogType.INFO,
    )

    vectors = embed_chunks(
        content_chunks,
        filename,
        unload_after=unload_after,
        model_key=model_key,
    )

    if vectors is None or len(vectors) == 0:
        return None

    # Save with chunk index alignment so retrieval maps correctly
    store.save_embeddings(filename, vectors, indices=[c["index"] for c in content_chunks])
    store.save_vector(filename, vectors)

    return vectors


# ─────────────────────────────────────────────────────────────────────────────
# Context manager for explicit lifecycle control
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingSession:
    """
    Explicit embedder lifecycle context manager.

    Usage:
        with EmbeddingSession() as session:
            vec = session.embed_query("what is RAG?")
        # model guaranteed released (ref-counted) here
    """

    def __init__(self, model_key: Optional[str] = None):
        self.model_key = model_key
        self._preset: Optional[ModelPreset] = None
        self._loader = None
        self._embedder = None
        self._acquired = False

    def __enter__(self):
        self._preset = get_preset(self.model_key)
        self._loader = modelManager
        self._embedder = self._loader.get(ModelCategory.EMBEDDING)
        self._acquired = self._embedder is not None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            self._loader.unload(ModelCategory.EMBEDDING)
        return False

    def embed_query(self, query: str) -> Optional[np.ndarray]:
        if not self._acquired:
            darwin.log("EmbeddingSession not acquired", LogType.ERROR)
            return None
        prepared = prepare_query_text(query, self._preset)
        return self._embedder.encode(
            prepared,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if not self._acquired:
            darwin.log("EmbeddingSession not acquired", LogType.ERROR)
            return None
        prepared = prepare_passage_text(text, self._preset)
        return self._embedder.encode(
            prepared,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)