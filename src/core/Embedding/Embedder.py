"""
src/core/Embedder.py
--------------------
Converts chunks into embedding vectors using e5-small (default).

e5-small specifics
------------------
intfloat/e5-small-v2 was trained with task prefixes — skipping them
tanks retrieval quality significantly:

    passage: {chunk text}   → used when embedding document chunks
    query:   {query text}   → used when embedding a search query

Always use _prefix_passage() before embedding chunks.
Always use _prefix_query() before embedding a query.

Both are exported so the search layer can use _prefix_query directly.

Pipeline
--------
    chunks → batch → encode → L2 normalize → np.float32 array

L2 normalization means cosine similarity at search time is just a
dot product — no division needed, faster on CPU.

Batching
--------
Batch size is configurable. On CPU:
    batch_size=32  → safe for 8GB RAM
    batch_size=64  → fine for 16GB+ RAM
    batch_size=16  → conservative for low-end hardware

The model is lazy-loaded on first call and cached for the session.
Not reloaded between documents.

Config integration
------------------
Reads embedding_model from DarwinStore config (default: e5-small).
Model name stored in config so DarwinStore can version the .npz files
correctly per model.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional

from src.core.Chunker import Chunk
from src.lib.Terminal import TerminalUI


# ============================================================================
# Constants
# ============================================================================

DEFAULT_MODEL      = "intfloat/e5-small-v2"
DEFAULT_BATCH_SIZE = 32

# model registry — maps short config name → HuggingFace model id
MODEL_REGISTRY = {
    "e5-small":  "intfloat/e5-small-v2",
    "e5-small-v2": "intfloat/e5-small-v2",
}


# ============================================================================
# e5 prefix helpers
# These must be used consistently — passages during indexing,
# queries during search. Asymmetric by design.
# ============================================================================

def prefix_passage(text: str) -> str:
    """Prepend passage prefix for e5 models. Use when embedding chunks."""
    return f"passage: {text}"


def prefix_query(text: str) -> str:
    """Prepend query prefix for e5 models. Use when embedding search queries."""
    return f"query: {text}"


# ============================================================================
# Model loader — lazy, cached per session
# ============================================================================

_model_cache: dict = {}


def _load_model(model_id: str):
    """
    Lazy load a SentenceTransformer model.
    Cached by model_id so it's only loaded once per session.
    """
    if model_id in _model_cache:
        return _model_cache[model_id]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embedding. "
            "Install with: pip install sentence-transformers"
        )

    t = TerminalUI.timer(f"load_model:{model_id.split('/')[-1]}")
    TerminalUI.dim(f"Loading embedding model: {model_id}")

    model = SentenceTransformer(model_id)
    _model_cache[model_id] = model

    t.done()
    return model


def _resolve_model_id(model_name: str) -> str:
    """
    Resolve a short config name to a HuggingFace model id.
    Falls back to treating the name as a full model id if not in registry.
    """
    return MODEL_REGISTRY.get(model_name.lower(), model_name)


# ============================================================================
# Core embedding function
# ============================================================================

def embed_chunks(
    chunks:     List[Chunk],
    model_name: str = "e5-small",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Embed a list of chunks using e5-small (or configured model).

    Uses chunk.clean (stripped plain text) for embedding — not chunk.text.
    Applies e5 passage prefix before encoding.
    Returns L2-normalized float32 vectors.

    Args:
        chunks:     list of Chunk objects from chunk_document()
        model_name: short name from MODEL_REGISTRY or full HF model id
        batch_size: chunks per batch — lower if running out of RAM

    Returns:
        np.ndarray shape (len(chunks), embedding_dim)
        e5-small-v2 embedding_dim = 384
    """
    if not chunks:
        return np.array([], dtype=np.float32)

    model_id = _resolve_model_id(model_name)
    model    = _load_model(model_id)

    # apply e5 passage prefix to clean text
    texts = [prefix_passage(c.clean) for c in chunks]

    t = TerminalUI.timer("embed_chunks")
    TerminalUI.dim(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")

    all_vectors = []
    n_batches   = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch      = texts[i : i + batch_size]
        batch_num  = i // batch_size + 1

        if batch_num % 5 == 0 or batch_num == 1 or batch_num == n_batches:
            TerminalUI.dim(f"  batch {batch_num}/{n_batches}")

        vecs = model.encode(
            batch,
            batch_size        = len(batch),
            show_progress_bar = False,
            convert_to_numpy  = True,
            normalize_embeddings = True,   # L2 normalize in-model
        )
        all_vectors.append(vecs)

    t.split("encoding")

    vectors = np.vstack(all_vectors).astype(np.float32)

    t.done()
    TerminalUI.success(f"Embedded {len(chunks)} chunks → shape {vectors.shape}")

    return vectors


def embed_query(
    query:      str,
    model_name: str = "e5-small",
) -> np.ndarray:
    """
    Embed a single search query.

    Uses query prefix (not passage prefix) — asymmetric by design for e5.
    Returns L2-normalized float32 vector, shape (embedding_dim,).

    Args:
        query:      raw query string from user
        model_name: must match the model used to embed the chunks

    Returns:
        np.ndarray shape (384,) for e5-small-v2
    """
    model_id = _resolve_model_id(model_name)
    model    = _load_model(model_id)

    prefixed = prefix_query(query)

    vec = model.encode(
        prefixed,
        convert_to_numpy     = True,
        normalize_embeddings = True,
    )

    return vec.astype(np.float32)


# ============================================================================
# Pipeline entry point
# Chains chunk → embed → store in one call.
# ============================================================================

def run_embedding_pipeline(
    filename:   str,
    chunks:     List[Chunk],
    store,                      # DarwinStore instance
    model_name: str = "e5-small",
    batch_size: int = DEFAULT_BATCH_SIZE,
    rerun:      bool = False,
) -> Optional[np.ndarray]:
    """
    Full embedding pipeline for one document.

    Takes chunks from the chunker, embeds them, saves the .npz bundle
    to DarwinStore. Returns the vectors (useful for testing/inspection).

    Args:
        filename:   source document filename
        chunks:     list of Chunk objects from chunk_document()
        store:      DarwinStore instance
        model_name: embedding model — must match what's in config
        batch_size: batch size for encoding
        rerun:      if True, re-embed even if fresh embeddings exist

    Returns:
        np.ndarray of vectors, or None if skipped (fresh + not rerun)
    """
    if not chunks:
        TerminalUI.warn(f"No chunks to embed for {filename}")
        return None

    # skip if already fresh and not forced
    if not rerun and store.is_fresh(filename) and store.has_embeddings(filename):
        TerminalUI.dim(f"Skipping embedding: fresh index exists for {filename}")
        return None

    TerminalUI.bold(f"Embedding {filename} ({len(chunks)} chunks)")

    vectors = embed_chunks(chunks, model_name=model_name, batch_size=batch_size)

    store.save_embeddings(filename, vectors, chunks, rerun=rerun)

    # update config so DarwinStore versions future .npz files correctly
    store.update_config(embedding_model=model_name)

    return vectors