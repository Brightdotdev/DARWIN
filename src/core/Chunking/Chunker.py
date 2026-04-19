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

Link assignment:
  Pass raw_links (from ParsedDocument.links via links_to_dict()) to
  chunk_document() or chunk_for_model() and each chunk will have its
  .links field populated with the hyperlinks whose page falls within
  that chunk's [page, page_end] range.
"""

from typing import List, Optional, Tuple

from src.core.Chunking.ChunkerUtils import (
    Chunk,
    split_into_sections,
    count_page_seps,
    merge_small,
    section_to_chunks,
    get_dynamic_chunk_params_for_text,
)


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_document(
    text:       str,
    source:     str,
    max_tokens: int,
    min_tokens: int,
    overlap:    int,
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
        raw_links:  optional list of link dicts from ParsedDocument.links
                    (use links_to_dict() from ParserUtils to serialise them).
                    When supplied, each chunk's .links field is populated with
                    hyperlinks whose page falls within that chunk's page range.

    Returns:
        List[Chunk] — ordered, re-indexed from 0.
        Each chunk has:
            .text        original text (send to LLM)
            .clean       stripped text (feed to embedder)
            .heading     nearest section heading (use for citations)
            .page        first page of this chunk (1-based, 0 if unknown)
            .page_end    last page of this chunk
            .token_count estimated token count of .clean
            .links       hyperlinks scoped to this chunk's page range
    """
    if not text or not text.strip():
        return []

    page     = 1
    sections = split_into_sections(text)

    all_chunks: List[Chunk] = []

    for heading, body, _level in sections:
        page += count_page_seps(body)

        new_chunks = section_to_chunks(
            heading    = heading,
            body       = body,
            source     = source if source else "unknown",
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


# ── Convenience — auto model preset ───────────────────────────────────────────

def chunk_for_model(
    text:      str,
    source:    str = "unknown",
) -> Tuple[List[Chunk], dict]:
    """
    Convert parser embedding_text into chunks using auto-detected parameters.

    Model-specific parameters (max_tokens, min_tokens, overlap) are resolved
    from the user's config via get_dynamic_chunk_params_for_text().

    Args:
        text:      embedding_text from ParsedDocument
        source:    filename for citations
        raw_links: optional link dicts — see chunk_document() for details

    Returns:
        (chunks, chunk_metadata) tuple.
        chunk_metadata contains the resolved params + doc stats and is
        suitable for passing to store.save_chunks(..., chunk_meta=...).
    """
    chunk_metadata = get_dynamic_chunk_params_for_text(text)
    chunks = chunk_document(
        text       = text,
        source     = source,
        max_tokens = chunk_metadata["max_tokens"],
        min_tokens = chunk_metadata["min_tokens"],
        overlap    = chunk_metadata["overlap"],
    )
    return chunks, chunk_metadata