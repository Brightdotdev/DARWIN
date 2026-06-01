"""
src/core/Embedding/Embedder.py
------------------------------
Multi-model embedding system with auto-detected model config.
No torch dependency in core — uses generic memory handling.
"""

import numpy as np
from typing import List, Optional, Dict, Any

from src.core.Models.EmbeddingConfig import (
    ModelPreset,
    get_preset,
    get_model_batch_size,
)
from src.features.Ingestion.Chunking.lib import Chunk
from src.core.Models import ModelCategory, ModelManager
from src.core.Logger.DarwinLogger import darwin, LogType


def prepare_passage_text(text: str, preset: ModelPreset | str | None = None) -> str:
    if preset is None or isinstance(preset, str):
        preset = get_preset(preset)

    if preset.is_instruction_tuned and preset.instruction_template:
        return preset.instruction_template.format(text=text)
    elif preset.needs_prefix:
        return f"{preset.passage_prefix}{text}"
    else:
        return text


def prepare_query_text(query: str, preset: ModelPreset | str | None = None) -> str:
    if preset is None or isinstance(preset, str):
        preset = get_preset(preset)

    if preset.is_instruction_tuned:
        return f"|<|user|>\nGiven a query, retrieve relevant passages: {query}\n<|<|assistant|>\n"
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
    No torch dependency — handles OOM via generic MemoryError or psutil.
    """
    import psutil

    if not chunks:
        darwin.log("No chunks found to embed", LogType.WARNING)
        return np.array([], dtype=np.float32)

    preset = get_preset()
    darwin.log("Loading embedding model...")
    loader = ModelManager.get_loader()

    embedder = loader.get(ModelCategory.EMBEDDING)
    darwin.log("Loaded embedding model...")

    texts = [prepare_passage_text(c.get("clean", ""), preset) for c in chunks]

    optimal_batch = batch_size or get_model_batch_size(preset=preset)
    darwin.log(f"Embedding in {optimal_batch} batches")
    n_batches = (len(texts) + optimal_batch - 1) // optimal_batch

    darwin.log(
        f"Embedding with {preset.model_name} ({preset.category}, {preset.max_tokens} token limit) [batch_size={optimal_batch}]",
        LogType.INFO
    )

    all_vectors: list = []

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

            except MemoryError:
                darwin.log(f"OOM at batch_size={optimal_batch}, falling back to 16", LogType.WARNING)
                darwin.progress_end(progress_id)
                return embed_chunks(
                    chunks,
                    file_name,
                    batch_size=32,
                    unload_after=unload_after,
                )
            except Exception as e:
                # Check if it's a low-memory condition via psutil
                mem = psutil.virtual_memory()
                if mem.percent > 95:
                    darwin.log(f"System memory critical ({mem.percent}%), retrying with smaller batch", LogType.WARNING)
                    darwin.progress_end(progress_id)
                    return embed_chunks(
                        chunks,
                        file_name,
                        batch_size=max(1, optimal_batch // 2),
                        unload_after=unload_after,
                    )
                raise

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


def embed_text(
    text: str,
    preset: ModelPreset | str | None = None,
) -> np.ndarray:
    preset = get_preset()
    loader = ModelManager.get_loader()
    model = loader.get(ModelCategory.EMBEDDING)
    prepared = prepare_passage_text(text, preset)

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
    if not chunks:
        darwin.log(f"No chunks to embed for {filename}", LogType.WARNING)
        return None

    if not rerun and store.has_embeddings(filename):
        darwin.log(f"Skipping embedding: fresh index exists for {filename}", LogType.INFO)
        return None

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

    store.save_embeddings(filename, vectors)
    store.save_vector(filename, vectors)

    return vectors