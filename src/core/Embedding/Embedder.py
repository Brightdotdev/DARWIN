
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
    prefix_passage,
    prefix_query,
)
from src.core.Embedding.EmbeddingModel import (
    DEFAULT_BATCH_SIZE,
    load_model,
    get_optimal_batch_size,
    DEFAULT_MODEL

)
# ============================================================================
# Core embedding function
# ============================================================================


def embed_chunks(
    chunks:     List[Chunk],
    file_name : str,
    model_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)


    model    = load_model(model_name)

    # 1. Prepare all texts at once
    texts = [prefix_passage(c.clean) for c in chunks]
    n_batches = (len(texts) + batch_size - 1) // batch_size
    TerminalUI.dim(f"The optimal batch size is: {get_optimal_batch_size(batch_size)}")

    # 2. Use our cool new Live Tracker
    with TerminalUI.track_embedding(n_batches, model_name=model_name.split('/')[-1],file_name=file_name) as update_progress:
        
        # We still loop so we can update the UI, but we let 
        # SentenceTransformers handle the internal optimization
        all_vectors = []
        
        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch = texts[i : i + batch_size]
            
            update_progress(batch_num, msg=f"Encoding batch {batch_num}...")

            vecs = model.encode(
                batch,
                batch_size           = len(batch),
                show_progress_bar    = False, # We have our own cool bar now
                convert_to_numpy     = True,
                normalize_embeddings = True,
            )
            all_vectors.append(vecs)

    # 3. Finalize
    vectors = np.vstack(all_vectors).astype(np.float32)
    TerminalUI.success(f"Embedded {len(chunks)} chunks -> {vectors.shape}")
    
    return vectors



def embed_query(
    query:      str,
    model_name: str,
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
    
    model    = load_model(model_name)

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
    store,
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
    
    config = store.load_config()
    model_name = config.get("embedding_model", DEFAULT_MODEL)

    TerminalUI.bold(f"Embedding {filename} ({len(chunks)} chunks)")

    vectors = embed_chunks(chunks,filename, model_name=model_name, batch_size=batch_size)

    store.save_embeddings(filename, vectors, chunks, rerun=rerun)

    return vectors