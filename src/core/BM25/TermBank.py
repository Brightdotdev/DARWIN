"""
src/core/Search/TermBank.py
----------------------------
Build per-document term banks from BM25 vocabulary for query expansion.

The term bank is a frequency-filtered, deduplicated vocabulary extracted
from document chunks. Terms are embedded using the same model as chunk
embeddings so similarity search works natively.

No LLM involved — just cosine similarity between query embedding and term
embeddings to find related vocabulary.
"""

from collections import Counter
from typing import Optional
import numpy as np

from src.core.Chunking.ChunkerUtils import Chunk
from src.core.BM25.BM25Utils import tokenize_text, STANDARD_STOPS
from src.core.Logger.DarwinLogger import darwin, LogType


def build_term_bank(
    chunks: list[Chunk],
    domain_stopwords: set[str],
    min_freq: int = 5,
    max_terms: int = 2000,
) -> tuple[list[str], dict[str, int], int]:
    """
    Extract a frequency-filtered vocabulary from document chunks.

    Args:
        chunks: Chunk objects from the chunker
        domain_stopwords: Terms already removed as document-level noise
        min_freq: Minimum number of chunks a term must appear in
        max_terms: Maximum vocabulary size

    Returns:
        (terms, term_freqs, chunk_count) ready for store.save_term_bank()
        terms: Sorted list of unique terms (most frequent first)
        term_freqs: term → chunk_count mapping
        chunk_count: Total chunks processed
    """
    if not chunks:
        darwin.log("No chunks provided for term bank", LogType.WARNING)
        return [], {}, 0

    all_stops = STANDARD_STOPS | domain_stopwords
    freq: Counter = Counter()

    for chunk in chunks:
        tokens = tokenize_text(chunk.clean, domain_stops=all_stops)
        # Document frequency: count each term once per chunk
        freq.update(set(tokens))

    # Filter: must appear in at least min_freq chunks
    filtered = {term: count for term, count in freq.items() if count >= min_freq}

    if not filtered:
        darwin.log(f"No terms met minimum frequency threshold ({min_freq})", LogType.INFO)
        return [], {}, len(chunks)

    # Sort by frequency (most common first), cap at max_terms
    sorted_terms = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:max_terms]

    terms = [t for t, _ in sorted_terms]
    term_freqs = dict(sorted_terms)

    darwin.log(f"Term bank built: {len(terms)} terms from {len(chunks)} chunks", LogType.INFO)
    return terms, term_freqs, len(chunks)


def embed_terms(
    terms: list[str],
    model_name: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Embed a list of terms using the current embedding model.

    Args:
        terms: List of vocabulary terms to embed
        model_name: Optional model override (auto-detects if None)

    Returns:
        numpy array of shape (len(terms), embedding_dim) or None on failure
    """
    if not terms:
        return None

    try:
        from src.core.Embedding.Embedder import embed_chunks

        # Create lightweight Chunk objects just for embedding
        # The embedder only uses chunk.clean
        class _TermChunk:
            def __init__(self, text: str, idx: int):
                self.clean = text
                self.index = idx
                self.text = text
                self.page = 0
                self.heading = ""
                self.links = []
                self.is_overlap = False
                self.source = "term_bank"

        term_chunks = [_TermChunk(term, i) for i, term in enumerate(terms)]
        darwin.log("Loading embedding model to embed terms")
        embeddings = embed_chunks(
            chunks=term_chunks,
            file_name="term_bank",
            batch_size=64,
            unload_after=False, 
        )

        return embeddings

    except Exception as e:
        darwin.log(f"Failed to embed terms: {e}", LogType.WARNING)
        return None