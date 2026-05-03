"""
src/core/Chunker.py
-------------------
Converts parser output (embedding_text) into overlapping chunks
ready for embedding.

All model-specific parameters are auto-detected from the active
embedding model via ChunkConfig. No model names passed anywhere.
"""

from typing import List, Tuple

from src.features.Chunking.ChunkerUtils import (
    Chunk,
    split_into_sections,
    count_page_seps,
    merge_small,
    section_to_chunks,
    get_dynamic_chunk_params_for_text,
)
from src.features.Chunking.ChunkConfig import get_chunk_config


def chunk_document(
    text:   str,
    source: str,
) -> List[Chunk]:
    """
    Convert parser embedding_text into chunks ready for embedding.

    All parameters (max_tokens, min_tokens, overlap) are auto-detected
    from the currently loaded embedding model.

    Args:
        text:   embedding_text from ParsedDocument
        source: filename — attached to every chunk for citations

    Returns:
        List[Chunk] ordered and re-indexed from 0.
    """
    if not text or not text.strip():
        return []

    config = get_chunk_config()

    sections    = split_into_sections(text)
    all_chunks: List[Chunk] = []
    current_page = 1

    for heading, body, _level in sections:
        new_chunks = section_to_chunks(
            heading    = heading,
            body       = body,
            source     = source or "unknown",
            index      = len(all_chunks),
            page       = current_page,
            max_tokens = config.max_tokens,
            min_tokens = config.min_tokens,
            overlap    = config.overlap,
        )
        all_chunks.extend(new_chunks)
        current_page += count_page_seps(heading) + count_page_seps(body)

    all_chunks = merge_small(all_chunks, config.min_tokens, config.max_tokens)

    for i, chunk in enumerate(all_chunks):
        chunk.index = i

    return all_chunks


def chunk_for_model(
    text:   str,
    source: str = "unknown",
) -> Tuple[List[Chunk], dict]:
    """
    Convert parser embedding_text into chunks using auto-detected parameters.

    Args:
        text:   embedding_text from ParsedDocument
        source: filename for citations

    Returns:
        (chunks, chunk_metadata)
    """
    chunk_metadata = get_dynamic_chunk_params_for_text(text)

    chunks = chunk_document(
        text   = text,
        source = source,
    )

    return chunks, chunk_metadata