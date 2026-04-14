"""
src/core/Search/KeywordSearcher.py
----------------------------------
Pure BM25 (sparse) retrieval over chunked document text.

This module provides exact keyword matching using BM25Okapi with
pre-computed indexes from the BM25 module. It loads .pkl bundles
and performs term-based retrieval.

Search Pipeline
---------------
query string
    → tokenize (same as indexing)
    → BM25 scoring against all chunks
    → top-k indices
    → return SearchResult objects with metadata

BM25 Formula
------------
score(D, Q) = Σ IDF(qi) × TF(qi, D)

Where:
    IDF(qi) = log((N - df + 0.5) / (df + 0.5) + 1)
    TF(qi, D) = (freq × (k1 + 1)) / (freq + k1 × (1 - b + b × len/avgdl))

Parameters:
    k1 = 1.5 (default) — term frequency saturation
    b  = 0.75 (default) — length normalization

Integration
-----------
Used by:
    - CLI: `darwin keyword "query"`
    - Hybrid search: combined with semantic results via RRF
"""

import numpy as np
from typing import List, Optional, Dict

from src.core.Chunking.Chunker import Chunk
from src.core.Search.Keyword.KeywordUtilities import (
    tokenize_query,
    DEFAULT_TOP_K,
    DEFAULT_K1,
    DEFAULT_B,
    KeywordSearchResult,
    compute_bm25_scores
)
from src.core.Storage.DarwinStore import DarwinStore
from src.lib.Terminal import TerminalUI



# ============================================================================
# Core search function
# ============================================================================

def keyword_search(
    query: str,
    data,  # BM25Data object
    chunks: List[Chunk],
    top_k: int = DEFAULT_TOP_K,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> List[KeywordSearchResult]:
    """
    Pure BM25 keyword search over pre-computed index.
    
    Args:
        query: Raw user query string
        data: BM25Data object from store.load_bm25()
        chunks: List of Chunk objects
        top_k: Number of results to return
        k1: BM25 k1 parameter
        b: BM25 b parameter
    
    Returns:
        List[KeywordSearchResult] sorted by descending BM25 score
    """
    if not data or not chunks:
        return []
    
    # Tokenize query
    query_tokens = tokenize_query(query)
    
    if not query_tokens:
        TerminalUI.warn("Query contains no indexable terms after tokenization")
        return []
    
    TerminalUI.dim(f"Query tokens: {query_tokens}")
    TerminalUI.dim(f"Searching {data.chunk_count} chunks, {data.vocab_size} terms")
    
    # Compute scores
    scores = compute_bm25_scores(query_tokens, data, k1, b)
    
    # Get top-k indices
    if top_k >= len(scores):
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    
    # Build results
    results = []
    for idx in top_indices:
        score = float(scores[idx])
        
        if score <= 0:
            continue
            
        chunk = chunks[idx]
        results.append(KeywordSearchResult(
            text=chunk.text,
            clean=chunk.clean,
            score=score,
            source=chunk.source,
            page=chunk.page,
            heading=chunk.heading,
            chunk_index=chunk.index,
        ))
    
    return results

# ============================================================================
# Single document search
# ============================================================================

def search_single_document_keyword(
    query: str,
    filename: str,
    store: DarwinStore,
    top_k: int = DEFAULT_TOP_K,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> List[KeywordSearchResult]:
    """
    BM25 keyword search within a single document.
    
    Args:
        query: User query
        filename: Document to search
        store: DarwinStore instance
        top_k: Number of results
        k1: BM25 k1 parameter
        b: BM25 b parameter
    
    Returns:
        List[KeywordSearchResult] from this document
    """
    try:
        data = store.load_bm25(filename)
        chunks = store.load_chunks(filename)
        
        return keyword_search(
            query=query,
            data=data,
            chunks=chunks,
            top_k=top_k,
            k1=k1,
            b=b,
        )
        
    except FileNotFoundError:
        TerminalUI.error(f"No BM25 index found for {filename}")
        return []
    except Exception as e:
        TerminalUI.error(f"Keyword search failed: {e}")
        return []


# ============================================================================
# Multi-document search
# ============================================================================

def search_across_documents_keyword(
    query: str,
    store: DarwinStore,
    filenames: Optional[List[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> List[KeywordSearchResult]:
    """
    BM25 search across multiple documents.
    
    Args:
        query: User query
        store: DarwinStore instance
        filenames: List of documents to search (None = all with BM25)
        top_k: Total results to return
        k1: BM25 k1 parameter
        b: BM25 b parameter
    
    Returns:
        Combined and re-ranked results from all documents
    """
    if filenames is None:
        status = store.status()
        filenames = [
            fname for fname, info in status.get("files", {}).items()
            if info["components"]["bm25"]
        ]
    
    if not filenames:
        TerminalUI.warn("No documents with BM25 index found")
        return []
    
    all_results = []
    
    for filename in filenames:
        try:
            results = search_single_document_keyword(
                query=query,
                filename=filename,
                store=store,
                top_k=top_k,
                k1=k1,
                b=b,
            )
            all_results.extend(results)
        except Exception as e:
            TerminalUI.error(f"Failed to search {filename}: {e}")
            continue
    
    # Re-rank combined results by BM25 score
    all_results.sort(key=lambda r: r.score, reverse=True)
    
    return all_results[:top_k]