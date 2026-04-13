"""
src/core/Chunker.py
-------------------
Converts parser output (embedding_text) into overlapping chunks
ready for embedding.

Strategy — heading-first, not paragraph-first
----------------------------------------------
The parser already bakes structure signals into embedding_text:
  ## / ###     section boundaries
  [Table]      structured data block
  [Caption:]   figure caption
  ---          page separator

We exploit those signals directly instead of blindly splitting on \\n\\n
and trying to fix size problems after the fact.

Pipeline:
  1. Split text into sections at ## / ### boundaries
  2. Each section = heading + its body
  3. Section fits in token budget  → one chunk, heading attached
  4. Section too big               → split into paragraphs, heading inherited
  5. Paragraph still too big       → sentence split with overlap
  6. Section too small             → merge with next same-level section
  7. [Table] blocks                → always their own chunk, never split
  8. No headings found             → fall back to paragraph splitting

Two text fields per chunk:
  text   → original text shown to user / sent to LLM
  clean  → stripped plain text used for embedding

Token budget is a parameter — caller sets it based on their embedding model:
  e5-small        →  350  (512 window, leave headroom)
  BGE-M3          →  800  (8192 window)
  GritLM-7B       →  512  (4096 window)
"""

from typing import List
from src.core.Chunking.ChunkerUtils import (Chunk,DEFAULT_MAX_TOKENS,DEFAULT_MIN_TOKENS,DEFAULT_OVERLAP,split_into_sections,count_page_seps,merge_small,section_to_chunks)



# ============================================================================
# Public API
# ============================================================================

def chunk_document(
    text:       str,
    source:     str = "unknown",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    overlap:    int = DEFAULT_OVERLAP,
) -> List[Chunk]:
    """
    Convert parser embedding_text into chunks ready for embedding.

    Args:
        text:       embedding_text from ParsedDocument
        source:     filename — attached to every chunk for citations
        max_tokens: maximum chunk size — set based on your embedding model
                    e5-small → 350, BGE-M3 → 800, GritLM → 512
        min_tokens: chunks below this get merged with neighbours
        overlap:    sentences of overlap between chunks when sentence-splitting

    Returns:
        List[Chunk] — ordered, re-indexed from 0
        Each chunk has:
            .text        original text (send to LLM)
            .clean       stripped text (feed to embedder)
            .heading     nearest section heading (use for citations)
            .page        page number from parser (0 if unknown)
            .token_count estimated token count of .clean
    """
    if not text or not text.strip():
        return []

    # track page number by counting --- separators as we go
    page = 1
    sections = split_into_sections(text)

    all_chunks: List[Chunk] = []

    for heading, body, _level in sections:
        # update page estimate from separators in this section
        page += count_page_seps(body)

        new_chunks = section_to_chunks(
            heading    = heading,
            body       = body,
            source     = source,
            index      = len(all_chunks),
            page       = page,
            max_tokens = max_tokens,
            min_tokens = min_tokens,
            overlap    = overlap,
        )
        all_chunks.extend(new_chunks)

    # final merge pass for any remaining small chunks
    all_chunks = merge_small(all_chunks, min_tokens, max_tokens)

    # re-index cleanly from 0
    for i, chunk in enumerate(all_chunks):
        chunk.index = i

    return all_chunks


# ============================================================================
# Convenience — model presets
# ============================================================================

def chunk_for_model(
    text:   str,
    source: str   = "unknown",
    model:  str   = "e5-small",
    overlap: int  = DEFAULT_OVERLAP,
    
) -> List[Chunk]:
    """
    Convenience wrapper that sets token budget based on embedding model name.

    chunk_for_model(text, source="biology.pdf", model="e5-small")
    chunk_for_model(text, source="biology.pdf", model="bge-m3")
    chunk_for_model(text, source="biology.pdf", model="gritlm")
    """
    presets = {
        "e5-small":  350,
        "bge-m3":    800,
        "gritlm":    512,
        "gritlm-7b": 512,
    }
    max_tokens = presets.get(model.lower(), DEFAULT_MAX_TOKENS)
    return chunk_document(text, source, max_tokens=max_tokens, overlap=overlap)