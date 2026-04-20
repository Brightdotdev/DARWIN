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
  [Page N]     page boundary (new extractor)
  ---          page boundary (legacy pymupdf4llm)

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

Token budget comes from model_name via get_model_config():
  e5-small   →  480  tokens
  BGE-M3     →  8000 tokens
  GritLM-7B  →  3800 tokens
"""

from typing import List, Tuple

from src.core.Chunking.ChunkerUtils import (
    Chunk,
    DEFAULT_MODEL,
    split_into_sections,
    count_page_seps,
    merge_small,
    section_to_chunks,
    get_dynamic_chunk_params_for_text,
)


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_document(
    text:        str,
    source:      str,
    max_tokens:  int,
    min_tokens:  int,
    overlap:     int,
    model_name:  str = DEFAULT_MODEL,
) -> List[Chunk]:
    """
    Convert parser embedding_text into chunks ready for embedding.

    Args:
        text:       embedding_text from ParsedDocument
        source:     filename — attached to every chunk for citations
        max_tokens: maximum chunk size (tokens)
        min_tokens: chunks below this are merged with neighbours
        overlap:    sentences of overlap when sentence-splitting large paragraphs
        model_name: embedding model key — controls cleaning mode inside
                    section_to_chunks (plain_text vs semantic_markdown).
                    Defaults to DEFAULT_MODEL if not supplied.

    Returns:
        List[Chunk] ordered and re-indexed from 0.
        Each chunk has:
            .text        original text (send to LLM)
            .clean       stripped text (feed to embedder)
            .heading     nearest section heading (for citations)
            .page        first page this chunk touches
            .page_end    last page this chunk touches
            .token_count estimated token count of .clean
            .links       always [] — populate externally if needed
    """
    if not text or not text.strip():
        return []

    sections    = split_into_sections(text)
    all_chunks: List[Chunk] = []

    # Page tracking: walk section bodies in order, counting boundary markers
    # *as we finish each section* so each section's start page is the page
    # we were on when we entered it, not after counting its own separators.
    current_page = 1

    for heading, body, _level in sections:
        new_chunks = section_to_chunks(
            heading    = heading,
            body       = body,
            source     = source or "unknown",
            index      = len(all_chunks),
            page       = current_page,
            max_tokens = max_tokens,
            min_tokens = min_tokens,
            overlap    = overlap,
            model_name = model_name,
        )
        all_chunks.extend(new_chunks)

        # Advance page counter after processing the section so the *next*
        # section starts on the correct page.
        current_page += count_page_seps(body)

    all_chunks = merge_small(all_chunks, min_tokens, max_tokens)

    # Re-index cleanly from 0 after merging may have collapsed chunks
    for i, chunk in enumerate(all_chunks):
        chunk.index = i

    return all_chunks


# ── Convenience — auto model preset ───────────────────────────────────────────

def chunk_for_model(
    text:       str,
    source:     str = "unknown",
    model_name: str = DEFAULT_MODEL,
) -> Tuple[List[Chunk], dict]:
    """
    Convert parser embedding_text into chunks using model-aware auto parameters.

    Resolves max_tokens / min_tokens / overlap from the model preset and
    scales them to the document length via get_dynamic_chunk_params_for_text().

    Args:
        text:       embedding_text from ParsedDocument
        source:     filename for citations
        model_name: embedding model key — drives both parameter selection and
                    the cleaning mode used when building each chunk's .clean field

    Returns:
        (chunks, chunk_metadata) where chunk_metadata contains the resolved
        params plus doc stats (doc_tokens, estimated_chunks, strategy).
        Pass chunk_metadata to store.save_chunks(..., chunk_meta=...).
    """
    chunk_metadata = get_dynamic_chunk_params_for_text(text, model_name=model_name)

    chunks = chunk_document(
        text       = text,
        source     = source,
        max_tokens = chunk_metadata["max_tokens"],
        min_tokens = chunk_metadata["min_tokens"],
        overlap    = chunk_metadata["overlap"],
        model_name = model_name,
    )

    return chunks, chunk_metadata