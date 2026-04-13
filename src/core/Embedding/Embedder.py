
"""
src/core/Embedding/Embedder.py
------------------------------
Converts chunked document text into embedding vectors using
SentenceTransformers models (default: e5-small-v2).

Designed for Retrieval-Augmented Generation (RAG) pipelines where
chunk-level semantic search is required.

Model behavior (e5 family)
-------------------------
e5 models are trained with asymmetric prefixes. Using them correctly
is critical for retrieval quality:

    passage: {chunk text}   → used when embedding document chunks
    query:   {query text}   → used when embedding search queries

This module enforces that:
- chunk.clean is always prefixed with "passage:"
- queries are prefixed with "query:"

Failing to apply these prefixes will significantly degrade search accuracy.

Pipeline
--------
    chunks (Chunk.clean)
        → prefix_passage()
        → batch encoding (SentenceTransformer)
        → L2 normalization
        → float32 numpy array

L2 normalization ensures cosine similarity reduces to a dot product,
which improves retrieval speed and simplifies scoring.

Batching
--------
Batch size is configurable depending on hardware:

    batch_size=32  → safe default (≈8GB RAM)
    batch_size=64  → faster on higher memory systems
    batch_size=16  → conservative fallback

Model Loading
-------------
Models are:
- lazily loaded on first use
- cached per session (no reload between documents)

Model resolution supports:
- short names via MODEL_REGISTRY (e.g. "e5-small")
- direct HuggingFace model IDs

Outputs
-------
- Chunk embeddings: np.ndarray (N, D)
- Query embedding: np.ndarray (D,)

For e5-small-v2:
    D = 384

Storage Integration
------------------
Embeddings are saved via DarwinStore as .npz bundles alongside chunks.

The embedding model name is stored in config to:
- version embeddings correctly
- prevent mixing incompatible vector spaces

Entry Points
------------
embed_chunks(chunks)
    → embeds document chunks

embed_query(query)
    → embeds search query

run_embedding_pipeline(...)
    → full pipeline: chunk → embed → store

Notes
-----
- Always embed chunk.clean, not chunk.text (clean removes markdown noise)
- Embeddings are normalized in-model (normalize_embeddings=True)
- Output is cast to float32 to reduce memory footprint
"""

import numpy as np
from pathlib import Path
from typing import List, Optional

from src.core.Chunking.Chunker import Chunk
from src.lib.Terminal import TerminalUI
from src.core.Embedding.EmbedderUtils import (
    DEFAULT_BATCH_SIZE,
    prefix_passage,
    resolve_model_id,
    load_model,
    prefix_query,
)
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

    model_id = resolve_model_id(model_name)
    model    = load_model(model_id)

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
    model_id = resolve_model_id(model_name)
    model    = load_model(model_id)

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