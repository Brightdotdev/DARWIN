"""
src/core/BM25/BM25Indexer.py
----------------------------
BM25 sparse indexing for keyword retrieval.

This module builds BM25 indexes from chunked documents and saves
them to DarwinStore for later retrieval. It handles tokenization,
domain stopword detection, and term frequency calculation.

Indexing Pipeline
-----------------
chunks
    → detect domain stopwords
    → tokenize each chunk (standard + domain stops removed)
    → compute term frequencies and document frequencies
    → calculate avgdl and doc_lengths
    → save via store.save_bm25()

Why save frequencies not scores?
--------------------------------
BM25 scoring requires:
    - IDF: log((N - df + 0.5) / (df + 0.5) + 1)
    - TF: (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * len / avgdl))

We save the raw counts (doc_freqs, doc_lengths, avgdl) so scores
can be computed dynamically at search time with any k1, b parameters.

Integration
-----------
Used by:
    - CLI: `darwin index-bm25 --file biology.pdf`
    - Hybrid search: loads .bm25 alongside .npz embeddings
"""

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi

from src.core.Chunking.Chunker import Chunk
from src.core.BM25.BM25Utils import (
    detect_domain_stopwords,
    tokenize_chunks,
    BM25Data,
    DOMAIN_STOP_THRESHOLD,
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI


# ============================================================================
# Core indexing function
# ============================================================================

def index_chunks_for_bm25(
    chunks: List[Chunk],
    filename: str,
    threshold: float = DOMAIN_STOP_THRESHOLD,
) -> BM25Data:
    """
    Build BM25 index data from chunks.
    
    Args:
        chunks: List of Chunk objects
        filename: Source document name (for logging)
        threshold: Domain stopword threshold (0-1)
    
    Returns:
        BM25Data with all indexing information
    """
    if not chunks:
        TerminalUI.warn(f"No chunks to index for {filename}")
        return None
    
    TerminalUI.bold(f"Building BM25 index for {filename} ({len(chunks)} chunks)")
    
    # 1. Detect domain stopwords
    t = TerminalUI.timer("domain_stopwords")
    domain_stops = detect_domain_stopwords(chunks, threshold)
    TerminalUI.dim(f"Discovered {len(domain_stops)} domain stopwords")
    t.done()
    
    # 2. Tokenize all chunks
    t = TerminalUI.timer("tokenization")
    tokenized = tokenize_chunks(chunks, domain_stops=domain_stops)
    
    # Filter out empty chunks
    valid_chunks = []
    valid_tokens = []
    for i, tokens in enumerate(tokenized):
        if tokens:
            valid_chunks.append(chunks[i])
            valid_tokens.append(tokens)
    
    TerminalUI.dim(f"Tokenized {len(valid_tokens)} valid chunks")
    t.done()
    
    # 3. Compute document frequencies
    t = TerminalUI.timer("frequencies")
    doc_freqs: Counter = Counter()
    doc_lengths: List[int] = []
    
    for tokens in valid_tokens:
        doc_lengths.append(len(tokens))
        # Unique terms per document for DF
        doc_freqs.update(set(tokens))
    
    avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
    vocab_size = len(doc_freqs)
    t.done()
    
    # 4. Package for saving
    data = BM25Data(
        tokens=valid_tokens,
        doc_freqs=dict(doc_freqs),
        avgdl=avgdl,
        doc_lengths=doc_lengths,
        domain_stopwords=sorted(domain_stops),
        threshold=threshold,
        chunk_count=len(valid_tokens),
        vocab_size=vocab_size,
    )
    
    TerminalUI.success(
        f"BM25 index built: {data.chunk_count} chunks, "
        f"{data.vocab_size} terms, {len(domain_stops)} domain stops"
    )
    
    return data


# ============================================================================
# BM25 scorer (for inspection, not search)
# ============================================================================

def create_bm25_scorer(data: BM25Data, k1: float = 1.5, b: float = 0.75) -> BM25Okapi:
    """
    Create a BM25Okapi scorer from saved data.
    
    Used for inspection and testing — not for production search.
    Production search loads data and computes scores directly.
    
    Args:
        data: BM25Data from index_chunks_for_bm25()
        k1: Term frequency saturation parameter
        b: Length normalization parameter
    
    Returns:
        BM25Okapi instance ready for scoring
    """
    scorer = BM25Okapi(
        corpus=data.tokens,
        k1=k1,
        b=b,
    )
    
    # Inject pre-computed frequencies (so we don't recompute)
    # Note: This is a hack for inspection. Real search uses raw counts.
    
    return scorer


# ============================================================================
# Pipeline entry point
# ============================================================================

def run_bm25_indexing_pipeline(
    filename: str,
    chunks: List[Chunk],
    store: DarwinStore,
    threshold: float = DOMAIN_STOP_THRESHOLD,
    rerun: bool = False,
) -> Optional[BM25Data]:
    """
    Full BM25 indexing pipeline for one document.
    
    Takes chunks from the chunker, builds BM25 index, saves to DarwinStore.
    Returns the BM25Data (useful for testing/inspection).
    
    Args:
        filename: Source document filename
        chunks: List of Chunk objects from chunk_document()
        store: DarwinStore instance
        threshold: Domain stopword threshold
        rerun: If True, re-index even if fresh index exists
    
    Returns:
        BM25Data or None if skipped
    """
    if not chunks:
        TerminalUI.warn(f"No chunks to index for {filename}")
        return None
    
    # Skip if fresh and not forced
    if not rerun and store.is_fresh(filename) and store.has_bm25(filename):
        TerminalUI.dim(f"Skipping BM25 indexing: fresh index exists for {filename}")
        return None
    
    # Build index
    data = index_chunks_for_bm25(chunks, filename, threshold)
    
    if data:
        # Save to store
        store.save_bm25(filename, data, rerun=rerun)
        store.save_stopwords(filename, data.domain_stopwords)
        TerminalUI.success(f"Saved BM25 index for {filename}")
    
    return data