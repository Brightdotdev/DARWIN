"""
src/core/Embedding/Embedder.py
------------------------------
Multi-model embedding system with auto-detected model config.

All model-specific behavior (prefixes, token limits, batch sizes, instruction
templates) comes from ChunkConfig.get_preset() — no hardcoded model names.

Model lifecycle
───────────────
- embed_chunks: loads the embedder lazily via ModelLoader, unloads if requested
- embed_query:  never unloads — queries are too frequent for reload overhead
- run_embedding_pipeline: orchestrates embed + save, passes unload_after through
"""

import numpy as np
from typing import List, Optional,Dict,Any

from src.core.Models.EmbeddingConfig import (
    ModelPreset,
    get_preset,
    get_model_batch_size,
)
from src.features.Ingestion.Chunking.lib import Chunk
from src.core.Models import ModelCategory, ModelManager
from src.core.Logger.DarwinLogger import darwin, LogType


def prepare_passage_text(text: str, preset: ModelPreset | str | None = None) -> str:
    """
    Prepare chunk text for embedding with model-specific prompting.
    
    Args:
        text:   Raw chunk text
        preset: ModelPreset, model key string, or None (auto-detect)
    
    Returns:
        Formatted text ready for the embedder
    """
        
    if preset is None or isinstance(preset, str):
        preset = get_preset(preset)  

    if preset.is_instruction_tuned and preset.instruction_template:
        return preset.instruction_template.format(text=text)
    elif preset.needs_prefix:
        return f"{preset.passage_prefix}{text}"
    else:
        return text


def prepare_query_text(query: str, preset: ModelPreset | str | None = None) -> str:
    """
    Prepare query text with model-specific formatting.
    
    Args:
        query:  Raw search query
        preset: ModelPreset, model key string, or None (auto-detect)
    
    Returns:
        Formatted query ready for the embedder
    """
    
    if preset is None or isinstance(preset, str):
        preset = get_preset(preset)  

    if preset.is_instruction_tuned:
        return f"<|user|>\nGiven a query, retrieve relevant passages: {query}\n<|assistant|>\n"
    elif preset.needs_prefix:
        return f"{preset.query_prefix}{query}"
    else:
        return query


# ═══════════════════════════════════════════════════════════════════════════
# Core embedding
# ═══════════════════════════════════════════════════════════════════════════

def embed_chunks(
    chunks: List[Dict[str, Any]],
    file_name: str,
    batch_size: Optional[int] = None,
    unload_after: bool = False,
) -> Optional[np.ndarray]:
    """
    Embed chunks with model-specific prompting.

    Args:
        chunks:       List of chunk dictionaries to embed
        file_name:    Display name (used for UI only)
        batch_size:   Override batch size — auto-detected if None
        unload_after: Free the embedding model from RAM after encoding

    Returns:
        numpy array of shape (n_chunks, embedding_dim) or None on failure
    """
    import torch


    if not chunks:
        darwin.log("No chunks found to embed", LogType.WARNING)
        return np.array([], dtype=np.float32)

    # ── Resolve preset ──
    preset = get_preset()
    darwin.log("Loading embedding model...")
    loader = ModelManager.get_loader()
    
    embedder = loader.get(ModelCategory.EMBEDDING)
    darwin.log("Loaded embedding model...")
  
  




    texts = [prepare_passage_text(c.get("clean", ""), preset) for c in chunks]

    optimal_batch = batch_size or get_model_batch_size(preset=preset)
    n_batches = (len(texts) + optimal_batch - 1) // optimal_batch

    darwin.log(
        f"Embedding with {preset.model_name} ({preset.category}, {preset.max_tokens} token limit) [batch_size={optimal_batch}]",
        LogType.INFO
    )

    all_vectors: list = []
    
    # Progress tracking via event system
    progress_id = f"embed_{file_name}"
    darwin.progress_start(progress_id, n_batches, f"Embedding {file_name}")

    try:
        for batch_num in range(0, len(texts), optimal_batch):
            batch_texts = texts[batch_num:batch_num + optimal_batch]
            progress_num = batch_num // optimal_batch + 1

            darwin.progress_update(
                progress_id, 
                progress_num,
                f"Encoding batch {progress_num}/{n_batches}"
            )

            encode_kwargs = {
                "batch_size": len(batch_texts),
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
            }

            if "bge-m3" in preset.model_name.lower():
                encode_kwargs["return_dense"] = True

            try:
                vecs = embedder.encode(batch_texts, **encode_kwargs)
                all_vectors.extend(vecs)

            except torch.cuda.OutOfMemoryError:
                darwin.log(f"OOM at batch_size={optimal_batch}, falling back to 16", LogType.WARNING)
                darwin.progress_end(progress_id)
                return embed_chunks(
                    chunks,
                    file_name,
                    batch_size=32,
                    unload_after=unload_after,
                )

            except KeyboardInterrupt:
                darwin.log("Embedding interrupted", LogType.WARNING)
                darwin.progress_end(progress_id)
                if unload_after:
                    loader.unload(ModelCategory.EMBEDDING)
                raise

    finally:
        darwin.progress_end(progress_id)

    vectors = np.vstack(all_vectors).astype(np.float32)

    if unload_after:
        loader.unload(ModelCategory.EMBEDDING)
        darwin.log("Embedding model unloaded from memory", LogType.INFO)

    model_short = preset.model_name.split('/')[-1]
    darwin.log(
        f"Embedded {len(chunks)} chunks → {vectors.shape} (model: {model_short}, batch_size={optimal_batch})",
        LogType.SUCCESS
    )

    return vectors


def embed_query(
    query: str,
    preset: ModelPreset | str | None = None,
) -> np.ndarray:
    """
    Embed a search query with model-specific formatting.
    
    Never unloads the embedder — queries are too frequent for that overhead.
    The embedder stays cached in the loader for the lifetime of the session.

    Args:
        query:  Raw search query string
        preset: ModelPreset, model key string, or None (auto-detect)

    Returns:
        numpy array of shape (embedding_dim,) — float32, normalized
    """
    preset = get_preset()

    loader = ModelManager.get_loader()
    
    model = loader.get(ModelCategory.EMBEDDING)
    
    
    prepared = prepare_query_text(query, preset)

    encode_kwargs = {
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    if "bge-m3" in preset.model_name.lower():
        encode_kwargs["return_dense"] = True

    return model.encode(prepared, **encode_kwargs).astype(np.float32)





# ═══════════════════════════════════════════════════════════════════════════
# Pipeline entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_embedding_pipeline(
    filename: str,
    chunks: List[Dict[str, Any]],
    store,
    rerun: bool = False,
    unload_after: bool = False,
) -> Optional[np.ndarray]:
    """
    Full embedding pipeline with model auto-detection and resume support.

    1. Checks for existing fresh embeddings — skips if found
    2. Detects active model from ModelLoader
    3. Embeds all chunks with model-specific prompting
    4. Saves NPZ bundle + USearch index with per-file model metadata
    5. Optionally unloads the embedder from RAM

    Args:
        filename:     Document filename (for store paths)
        chunks:       Chunk objects from the chunker
        store:       
        rerun:        Force re-embed even if fresh embeddings exist
        unload_after: Free embedding model from RAM after completion

    Returns:
        numpy array of shape (n_chunks, embedding_dim) or None
    """
    if not chunks:
        darwin.log(f"No chunks to embed for {filename}", LogType.WARNING)
        return None

    if not rerun and store.has_embeddings(filename):
        darwin.log(f"Skipping embedding: fresh index exists for {filename}", LogType.INFO)
        return None

    # ── Auto-detect model ──
    preset = get_preset()
    model_name = preset.model_name

    darwin.log(
        f"Embedding {filename} ({len(chunks)} chunks) with {model_name}",
        LogType.INFO
    )

    vectors = embed_chunks(
        chunks,
        filename,
        unload_after=unload_after,
    )

    if vectors is None:
        return None

    # ── Save NPZ bundle (text + vectors) ──
    store.save_embeddings(
        filename,
        vectors,
    )

    # ── Save USearch index (vectors only, for fast search) ──
    chunk_ids = [str(c.get("index", "")) for c in chunks]
    store.save_vector(
        filename,
        vectors,
    )

    return vectors