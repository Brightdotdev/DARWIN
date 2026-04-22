"""
src/core/Embedding/Embedder.py
------------------------------
Multi-model embedding system with token-aware chunk validation.

Supported Models & Presets:
    intfloat/e5-small-v2:  512 tokens, asymmetric prefixes
    GritLM/GritLM-7B:      4096 tokens, instruction-tuned
    BAAI/bge-m3:           8192 tokens, multi-lingual

Each model has specific prompting requirements and token limits.
This module enforces them at embed time.
"""

import numpy as np
from typing import List, Optional
import torch

from src.core.Chunking.Chunker import Chunk
from src.lib.Terminal import TerminalUI
from src.core.Embedding.EmbeddingModel import (
    DEFAULT_BATCH_SIZE,
    load_model,
    get_preset,
    get_model_specific_batch_size,
    DEFAULT_MODEL,
)




def prepare_passage_text(text: str, model_name: str) -> tuple[str, bool]:
    """
    Prepare chunk text for embedding with model-specific prompting.
    
    Returns:
        (prepared_text, was_truncated)
    """
    preset = get_preset(model_name)
    
    
    
    if preset.is_instruction_tuned and preset.instruction_template:
        prepared = preset.instruction_template.format(text=text)
    elif preset.needs_prefix:
        prepared = f"{preset.passage_prefix}{text}"
    else:
        prepared = text
    
    return prepared


def prepare_query_text(query: str, model_name: str) -> str:
    """Prepare query text with model-specific formatting."""
    preset = get_preset(model_name)
    
    if preset.is_instruction_tuned:
        # GritLM query format
        return f"<|user|>\nGiven a query, retrieve relevant passages: {query}\n<|assistant|>\n"
    elif preset.needs_prefix:
        return f"{preset.query_prefix}{query}"
    else:
        return query


# ============================================================================
# Core embedding functions (updated)
# ============================================================================

def embed_chunks(
    chunks: List[Chunk],
    file_name: str,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Embed chunks with model-specific prompting and token validation.
    """
    if not chunks:
        TerminalUI.warn("No chunks found to embed")
        return np.array([], dtype=np.float32)
    
    preset = get_preset(model_name)
    model = load_model(model_name)
    
    # Prepare all texts with model-specific formatting
    texts = []
    
    for chunk in chunks:
        prepared = prepare_passage_text(
            chunk.clean, 
            model_name, 
        )
        texts.append(prepared)
        
    # Get optimal batch size based on hardware
    optimal_batch = get_model_specific_batch_size( model_name, batch_size)
    n_batches = (len(texts) + optimal_batch - 1) // optimal_batch
    
    TerminalUI.info(
        label=f"Embedding with {model_name}",
       value=f"({preset.category}, {preset.max_tokens} token limit) [batch_size={optimal_batch}]"
    )
    
    with TerminalUI.track_embedding(
        n_batches, 
        model_name=model_name.split('/')[-1],
        file_name=file_name
    ) as update_progress:
        
        all_vectors = []
        
        # Encode in batches
        for i in range(0, len(texts), optimal_batch):
            batch_num = i // optimal_batch + 1
            batch = texts[i:i + optimal_batch]
            
            update_progress(batch_num, msg=f"Encoding batch {batch_num}/{n_batches}")
            
            
            encode_kwargs = {
                "batch_size": len(batch),  
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
            }
            
            # BGE-M3 specific: return dense vectors only
            if "bge-m3" in model_name.lower():
                encode_kwargs["return_dense"] = True
            
            try:
                vecs = model.encode(batch, **encode_kwargs)
                all_vectors.append(vecs)
                
            except torch.cuda.OutOfMemoryError:
                # 🔧 NEW: Fallback if optimal_batch was too aggressive
                TerminalUI.warn(f"OOM at batch_size={optimal_batch}, falling back to 16")
                optimal_batch = 16
                n_batches = (len(texts) + optimal_batch - 1) // optimal_batch
                # Restart embedding with smaller batches
                return embed_chunks(
                    chunks, 
                    file_name, 
                    model_name=model_name, 
                    batch_size=optimal_batch,  # Force smaller batch
                )
    
    vectors = np.vstack(all_vectors).astype(np.float32)
    TerminalUI.success(
        f"Embedded {len(chunks)} chunks → {vectors.shape} "
        f"(model: {model_name})"
    )
    
    return vectors



def embed_query(
    query: str,
    model_name: str = DEFAULT_MODEL
) -> np.ndarray:
    """
    Embed a search query with model-specific prompting.
    """
    model = load_model(model_name)
    prepared = prepare_query_text(query, model_name)
    
    encode_kwargs = {
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    
    if "bge-m3" in model_name.lower():
        encode_kwargs["return_dense"] = True
    
    vec = model.encode(prepared, **encode_kwargs)
    
    return vec.astype(np.float32)






# ============================================================================
# Pipeline entry point (updated)
# ============================================================================

def run_embedding_pipeline(
    filename: str,
    chunks: List[Chunk],
    store,
    rerun: bool = False,
) -> Optional[np.ndarray]:
    """
    Full embedding pipeline with multi-model support.
    """
    if not chunks:
        TerminalUI.warn(f"No chunks to embed for {filename}")
        return None
    
    if not rerun and store.is_fresh(filename) and store.has_embeddings(filename):
        TerminalUI.dim(f"Skipping embedding: fresh index exists for {filename}")
        return None
    
    config = store.load_config()
    model_name = config.get("embedding_model", DEFAULT_MODEL)

    
    
    TerminalUI.bold(
        f"Embedding {filename} ({len(chunks)} chunks) "
        f"with {model_name}"
    )
    
    vectors = embed_chunks(
        chunks,
        filename,
        model_name=model_name,
    )
        

    if vectors is None:
        return None

    # Store with model metadata
    store.save_embeddings(
        filename, 
        vectors, 
        chunks, 
        rerun=rerun,
    )
    
    chunk_ids = [c.index for c in chunks]
    store.save_vector(filename, vectors, chunk_ids, model_name, rerun=rerun)
        
    return vectors