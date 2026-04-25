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
from typing import List, Optional
import torch

from src.core.Models.EmbeddingConfig import (
    ModelPreset,
    get_preset,
    get_model_batch_size,

)
from src.core.Chunking.ChunkerUtils import Chunk
from src.core.Models import get_loader, ModelCategory
from src.lib.Terminal import TerminalUI


# ═══════════════════════════════════════════════════════════════════════════
# Text preparation
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_preset(preset: ModelPreset | str | None) -> ModelPreset:
    """
    Accept a ModelPreset object, a model key string, or None.
    Always returns a valid ModelPreset.
    """
    if preset is None:
        return get_preset()
    if isinstance(preset, str):
        # Someone passed a model name string — resolve it
        from src.core.Chunking.ChunkConfig import PRESETS, DEFAULT_PRESET
        return PRESETS.get(preset, DEFAULT_PRESET)
    return preset


def prepare_passage_text(text: str, preset: ModelPreset | str | None = None) -> str:
    """
    Prepare chunk text for embedding with model-specific prompting.
    
    Args:
        text:   Raw chunk text
        preset: ModelPreset, model key string, or None (auto-detect)
    
    Returns:
        Formatted text ready for the embedder
    """
    preset = _resolve_preset(preset)

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
    preset = _resolve_preset(preset)

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
    chunks: List[Chunk],
    file_name: str,
    store=None,
    preset: ModelPreset | str | None = None,
    batch_size: Optional[int] = None,
    unload_after: bool = False,
) -> Optional[np.ndarray]:
    """
    Embed chunks with model-specific prompting and resume support.
    
    For large document sets (300+ chunks), saves checkpoints after each batch
    so interrupted embeddings can resume from where they left off.

    Args:
        chunks:       List of Chunk objects to embed
        file_name:    Display name and checkpoint key
        store:        DarwinStore — required for checkpointing large sets
        preset:       ModelPreset, model key string, or None (auto-detect)
        batch_size:   Override batch size — auto-detected if None
        unload_after: Free the embedding model from RAM after encoding

    Returns:
        numpy array of shape (n_chunks, embedding_dim) or None on failure
    """
    if not chunks:
        TerminalUI.warn("No chunks found to embed")
        return np.array([], dtype=np.float32)

    # ── Resolve preset (handles None, string, or ModelPreset) ──
    preset = _resolve_preset(preset)

    loader = get_loader()
    embedder = loader.get(ModelCategory.EMBEDDING)

    total_chunks = len(chunks)
    use_checkpoint = total_chunks >= 300 and store is not None

    # ── Resume from checkpoint if available ──
    completed_indices: set = set()
    all_vectors: list = []

    if use_checkpoint:
        checkpoint = store.load_embed_checkpoint(file_name) if hasattr(store, 'load_embed_checkpoint') else None
        if checkpoint:
            completed_indices = checkpoint.get("completed_indices", set())
            all_vectors = checkpoint.get("vectors", [])

            remaining = total_chunks - len(completed_indices)
            if remaining == 0:
                TerminalUI.success(
                    f"   ✅ {file_name} — all {total_chunks} chunks already embedded (from checkpoint)"
                )
                if unload_after:
                    loader.unload(ModelCategory.EMBEDDING)
                return np.vstack(all_vectors).astype(np.float32)

            TerminalUI.info(
                label="   🔄 Resuming embedding",
                value=f"{file_name} — {len(completed_indices)}/{total_chunks} chunks done, "
                      f"{remaining} remaining"
            )

    # ── Filter remaining chunks ──
    if completed_indices:
        remaining_chunks = [(i, c) for i, c in enumerate(chunks) if i not in completed_indices]
        texts = [prepare_passage_text(c.clean, preset) for _, c in remaining_chunks]
    else:
        texts = [prepare_passage_text(c.clean, preset) for c in chunks]

    optimal_batch = batch_size or get_model_batch_size(preset=preset)
    n_batches = (len(texts) + optimal_batch - 1) // optimal_batch

    if not completed_indices:
        TerminalUI.info(
            label=f"🎯 Embedding with {preset.model_name}",
            value=f"({preset.category}, {preset.max_tokens} token limit) [batch_size={optimal_batch}]"
        )

    with TerminalUI.track_embedding(
        n_batches,
        model_name=preset.model_name.split('/')[-1],
        file_name=file_name
    ) as update_progress:

        for batch_num in range(0, len(texts), optimal_batch):
            batch_texts = texts[batch_num:batch_num + optimal_batch]
            progress_num = batch_num // optimal_batch + 1

            update_progress(
                progress_num,
                msg=f"Encoding batch {progress_num}/{n_batches} "
                    f"({len(completed_indices) + batch_num}/{total_chunks} total)"
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

                # ── Save checkpoint after every batch ──
                if use_checkpoint:
                    batch_indices = list(range(batch_num, min(batch_num + optimal_batch, len(texts))))
                    completed_indices.update(batch_indices)
                    if hasattr(store, 'save_embed_checkpoint'):
                        store.save_embed_checkpoint(
                            file_name,
                            np.vstack(all_vectors),
                            completed_indices,
                        )

            except torch.cuda.OutOfMemoryError:
                TerminalUI.warn(f"   ⚠️  OOM at batch_size={optimal_batch}, falling back to 16")
                return embed_chunks(
                    chunks, file_name,
                    store=store,
                    preset=preset,
                    batch_size=16,
                    unload_after=unload_after,
                )

            except KeyboardInterrupt:
                TerminalUI.warn(
                    f"\n   ⚠️  Embedding interrupted at batch {progress_num}/{n_batches}. "
                    f"Progress saved — run again to resume."
                )
                if unload_after:
                    loader.unload(ModelCategory.EMBEDDING)
                raise

    vectors = np.vstack(all_vectors).astype(np.float32)

    # ── Clean up checkpoint on success ──
    if use_checkpoint and hasattr(store, 'clear_embed_checkpoint'):
        store.clear_embed_checkpoint(file_name)
        TerminalUI.dim("   🧹 Embed checkpoint cleaned up")

    # ── Unload if requested ──
    if unload_after:
        loader.unload(ModelCategory.EMBEDDING)
        TerminalUI.dim("   🧹 Embedding model unloaded from memory")

    TerminalUI.success(
        f"   ✅ Embedded {len(chunks)} chunks → {vectors.shape} "
        f"(model: {preset.model_name.split('/')[-1]}, batch_size={optimal_batch})"
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
    preset = _resolve_preset(preset)

    loader = get_loader()
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
    chunks: List[Chunk],
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
        store:        DarwinStore instance
        rerun:        Force re-embed even if fresh embeddings exist
        unload_after: Free embedding model from RAM after completion

    Returns:
        numpy array of shape (n_chunks, embedding_dim) or None
    """
    if not chunks:
        TerminalUI.warn(f"No chunks to embed for {filename}")
        return None

    if not rerun and store.has_embeddings(filename):
        TerminalUI.dim(f"   ⏭️  Skipping embedding: fresh index exists for {filename}")
        return None

    # ── Auto-detect model ──
    preset = get_preset()
    model_name = preset.model_name

    TerminalUI.bold(
        f" Embedding {filename} ({len(chunks)} chunks) "
        f"with {model_name}"
    )

    vectors = embed_chunks(
        chunks,
        filename,
        store=store,
        preset=preset,
        unload_after=unload_after,
    )

    if vectors is None:
        return None

    # ── Save NPZ bundle (text + vectors) ──
    store.save_embeddings(
        filename,
        vectors,
        chunks,
        model_name=model_name,
    )

    # ── Save USearch index (vectors only, for fast search) ──
    chunk_ids = [str(c.index) for c in chunks]
    store.save_vector(
        filename,
        model_name,
        vectors,
        chunk_ids,
    )

    return vectors