from typing import List

from src.core.Pipeline.Ingestion.Chunking.lib import OverlapDirection
from src.core.Pipeline.Ingestion.Chunking.ChunkerUtils import (
    Chunk,
    split_into_sections,
    count_page_seps,
    merge_small,
    section_to_chunks,
    tokens,   
)
from src.core.Pipeline.Ingestion.Chunking.ChunkConfig import get_chunk_config


def chunk_document(
    text:   str,
    source: str,
) -> List[Chunk]:
    """
    Convert parser embedding_text into chunks ready for embedding.
    """
    if not text or not text.strip():
        return []

    config = get_chunk_config()

    # NEW: estimate total pages for front/back matter detection
    total_pages = max(1, count_page_seps(text) + 1)

    sections    = split_into_sections(text)
    all_chunks: List[Chunk] = []
    current_page = 1

    for heading, body, _level in sections:
        new_chunks = section_to_chunks(
            heading     = heading,
            body        = body,
            source      = source or "unknown",
            index       = len(all_chunks),
            page        = current_page,
            max_tokens  = config.max_tokens,
            min_tokens  = config.min_tokens,
            overlap     = config.overlap,
            total_pages = total_pages,  # NEW
        )
        all_chunks.extend(new_chunks)
        current_page += count_page_seps(heading) + count_page_seps(body)

    all_chunks = merge_small(all_chunks, config.min_tokens, config.max_tokens)

    # CHANGED: Hard ceiling enforcement — split any chunk still over max_tokens
    final_chunks: List[Chunk] = []
    for chunk in all_chunks:
        if chunk.token_count > int(config.max_tokens * 1.1):
            # Force-split by character boundary
            char_limit = config.max_tokens * 3
            pieces = []
            remaining = chunk.clean
            while remaining:
                if tokens(remaining) <= config.max_tokens:
                    pieces.append(remaining)
                    break
                split_point = remaining[:char_limit].rsplit(" ", 1)[0] if len(remaining) > char_limit else remaining
                if not split_point:
                    split_point = remaining[:char_limit]
                pieces.append(split_point)
                remaining = remaining[len(split_point):].lstrip()
            
            # NEW: preserve structural_type across forced splits
            st = chunk.structural_type or "body"
            for i, piece in enumerate(pieces):
                final_chunks.append(Chunk(
                    text=piece,
                    clean=piece,
                    index=len(final_chunks),
                    source=chunk.source,
                    token_count=tokens(piece),
                    heading=chunk.heading,
                    page=chunk.page,
                    page_end=chunk.page_end,
                    overlap_direction=OverlapDirection.NONE,
                    links=chunk.links if i == 0 else [],
                    structural_type=st,  # NEW
                ))
        else:
            chunk.index = len(final_chunks)
            final_chunks.append(chunk)

    # CHANGED: Absolute floor — drop anything below 20 tokens
    final_chunks = [c for c in final_chunks if c.token_count >= 20]

    for i, chunk in enumerate(final_chunks):
        chunk.index = i

    return final_chunks

